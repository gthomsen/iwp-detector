#!/usr/bin/env python3

import getopt
import os
import sys

import matplotlib.cm

import iwp.data_loader
import iwp.labels
import iwp.quantization
import iwp.rendering
import iwp.statistics
import iwp.utilities

# use a sensible default colormap.  we aim for something perceptually uniform
# and is widely accessible.
DEFAULT_COLORMAP_NAME    = "viridis"

# emphasize the 95th percentile of data by default.
DEFAULT_QUANT_TABLE_NAME = "build_two_sigma_quantization_table"

# magenta is a high contrast color relative to most colormaps used (viridis,
# bwr, seismic, inferno, etc).
DEFAULT_LABEL_COLOR = (1.0, 0.0, 1.0)

def _get_dask_cluster_manager( cluster_address, cluster_parameters ):
    """
    Acquires a Dask cluster context manager from the supplied parameters.  This
    either connects to an existing cluster or starts a new local cluster.  The
    returned context manager releases its resources (either disconnecting, or
    tearing down the local cluster) when its context has exited.

    If neither a existing cluster address nor a local cluster's parameters are
    supplied, a null context manager is returned.

    Takes 2 arguments:

      cluster_address    - URI to an existing Dask cluster.  May be specified as None
                           if a local cluster is to be created from cluster_parameters.
      cluster_parameters - Tuple comprised of three elements specifying parameters
                           to start a local Dask cluster.  The parameters provided
                           are as follows:

                             1. number_workers     Integral number of worker processes
                                                   to spawn.  Must be positive.
                             2. memory_limit       Optional integral number of MiB
                                                   to restrict each worker's memory
                                                   footprint to.  If omitted, defaults
                                                   to (system memory / number_workers).
                             3. dashboard_address  Optional URI to the dashboard.
                                                   If omitted, defaults to "localhost:0"
                                                   which picks a random port on the
                                                   loopback adapter.

    Returns 1 value:

      cluster_manager - Context manager representing the cluster resources acquired.
                        Will be contextlib.nullcontext() when both cluster_address
                        and cluster_parameters are specified as None.

    """

    #
    # NOTE: we don't unconditionally import the modules found in this routine at
    #       the top of the script as they're not frequently used and don't need
    #       to slow down the common use case.
    #
    if cluster_address is not None:
        import dask.distributed

        # attempt to connect to an existing cluster.
        try:
            cluster_manager = dask.distributed.Client( cluster_address )
        except Exception as e:
            raise RuntimeError( "Failed to connect to '{:s}' - {:s}".format(
                cluster_address,
                str( e ) ) )
    elif cluster_parameters is not None:
        import dask.distributed

        # attempt to create a new local cluster.
        (number_workers,
         memory_limit,
         dashboard_address) = cluster_parameters

        # our memory limits are provided in MiB per process.
        if memory_limit is not None:
            memory_limit = "{:d}M".format( memory_limit )

        # default to a locally-accessible dashboard with a randomly selected
        # port.  the port selected is part of the URI available from the
        # .dashboard_address attribute on the returned context manager.
        if dashboard_address is None:
            dashboard_address = "localhost:0"

        try:
            cluster_manager = dask.distributed.LocalCluster( n_workers=number_workers,
                                                             memory_limit=memory_limit,
                                                             dashboard_address=dashboard_address )
        except Exception as e:
            raise RuntimeError( "Failed to launch a local cluster (number_workers={:d}, "
                                "memory_limit={}, dashboard_address={:s}) - {:s}".format(
                                    number_workers,
                                    memory_limit,
                                    dashboard_address,
                                    str( e ) ) )
    else:
        import contextlib

        # common case: we don't have a Dask cluster to distribute our work.
        cluster_manager = contextlib.nullcontext()

    return cluster_manager

def print_usage( program_name, file_handle=sys.stdout ):
    """
    Prints the script's usage to standard output.

    Takes 2 arguments:

      program_name - Name of the program currently executing.
      file_handle  - File handle to print to.  If omitted, defaults to standard output.

    Returns nothing.

    """

    usage_str = \
"""{program_name:s} [-C <cluster_address>] [-c <colormap>] [-F] [-h] [-L] [-l <labels_path>[,<color>]] [-n <cluster_spec>] [-q <quant_table>] [-S <statistics_path>] [-s <min>:<max>:<std>[,...]] [-T] [-t <time_start>:<time_stop>] [-v] [-z <z_start>:<z_stop>] <netcdf_pattern> <output_root> <experiment> <variables>

    Renders subsets of IWP datasets into a tree of PNG images suitable for analysis
    or labeling.  The netCDF4 files at <netcdf_pattern> are read and a subset of time
    steps and XY slices, for one or more <variables>, are written to disk beneath
    <output_root> with a directory structure of:

       <output_root>/<experiment_name>/<variable>/

    Individual image paths are of the form:

       <experiment>-<variable>-z=<z_index>-Nt=<time_index>.png

    Data for each <variable> in the comma separated <variables> are processed, while
    all other variables in the dataset are ignored.  This allows portions of a dataset
    to be rendered.

    Images are normalized according to each variable's global statistics.  These are
    computed from the underlying dataset unless specified via a command line option
    (see below).  Normalized data are quantized and then colorized prior to saving
    to disk.  Quantization tables and colormaps may be specified via a command line
    option (see below).

    The command line options shown above are described below:

        -c <colormap>                Use <colormap> when rendering images.  Must be a
                                     the name of a valid Matplotlib colormap that is
                                     found at "matplotlib.cm.<colormap>".  If omitted,
                                     defaults to "{colormap:s}".
        -C <cluster_address>         Use the Dask distributed workers located at
                                     <cluster_address>.  One connection attempt will be
                                     made before giving up and exiting the script.
                                     If omitted, all work is done in the local Python
                                     instance.  May not be specified when a local
                                     cluster is requested (see '-n' below).
        -F                           Rendered images are decorated Matplotlib figures
                                     instead of converting XY slices directly.  If
                                     omitted, XY slices are converted directly.
                                     of undecorated array visualizations.
        -h                           Print this help message and exit.
        -L                           Render images using local statistics of each XY
                                     slice instead of global statistics across all
                                     XY slices for a given variable. If omitted,
                                     global statistics are used.
        -l <labels_path>[,<color>]   Path to IWP labels to overlay on the images
                                     rendered.  Optional <color> to create the labels'
                                     bounding boxes with.  <color> may be specified
                                     as PIL or Matplotlib colorspec, or as a
                                     colon-delimited sequence of floating point values
                                     in the range of [0, 1].  If both are omitted,
                                     no labels are overlaid.  If <color> is omitted,
                                     a high contrast default is selected.
        -n <cluster_spec>            Dask local cluster specification to distribute
                                     label data creation across.  <cluster_spec> has
                                     the form:

                                       <number_workers>[:<memory_limit>[:<dashboard_port>]]

                                     <number_workers> specifies the number of processes
                                     to start locally.  If <memory_limit> is provided,
                                     it sets a hard memory limit in MiB for each worker
                                     process to adhere to.  If <dashboard_port> is
                                     specified, it is provided to the local cluster
                                     configuration.  If the cluster cannot be created
                                     for any reason (including <dashboard_port> being
                                     in use), the script will exit.

                                     If <memory_limit> is omitted, no constrain per
                                     worker is imposed which may exhaust the system's
                                     resources for large jobs.  If <dashboard_port>
                                     is omitted, a random port is selected internally.
                                     May not be specified when an existing cluster
                                     is specified (see '-C' above).
        -q <quant_table>             Use <quant_table> for quantizing XY slice data
                                     into image data.  Must be a valid IWP quantization
                                     table that is found at "iwp.quantization.<quant_table>".
                                     If omitted, defaults to "{quant_table:s}".
        -s <min>:<max>:<std>[,...]   Use the statistics provided (<min>, <max>, and <std>),
                                     instead of computing local statistics (-L) or using
                                     pre-computed statistics (-S).  This is useful for
                                     cases where the quantization tables available are
                                     not sufficient to visualize features of interest
                                     and a more direct approach is required (typically
                                     in addition to using a linear quantization table).
                                     Cannot be specified in conjunction with either
                                     local statistics or pre-computed statistics.  Must
                                     be specified once per variable in <variables>.
                                     If omitted, either local or pre-computed statistics
                                     are used.
        -S <statistics_path>         Specifies pre-computed statistics for the dataset.
                                     Variable statistics contained in <statistics_path>
                                     are used instead of computing them on the fly,
                                     unless local statistics are requested.  If omitted,
                                     all statistics, local and global, are computed.
        -T                           Prevents XY slice titles from being burned into
                                     rendered images.  By default, each XY slice image
                                     has metadata text overlaid to ease identification
                                     and analysis, though the rendered text may obscure
                                     some features in the data.  If specified, metadata
                                     titles are omitted.
        -t <time_start>:<time_stop>  Specifies a range of time step indices to render
                                     images for.  All time steps in [<time_start>, <time_stop>]
                                     must be present in the data found in <netcdf_pattern>.
                                     If omitted, defaults to all time steps available.
        -v                           Enable verbose execution.  Status messages about
                                     progress are written to standard output.  If
                                     omitted, defaults to normal execution.
        -z <z_start>:<z_stop>        Specifies a range of XY slice indices to render
                                     images for.  All XY slices in [<z_start>, <z_stop>]
                                     must be present in the data found in <netcdf_pattern>.
                                     If omitted, defaults to all XY slices available.

""".format(
    program_name=program_name,
    colormap=DEFAULT_COLORMAP_NAME,
    quant_table=DEFAULT_QUANT_TABLE_NAME
)

    print( usage_str, file=file_handle )

def parse_command_line( argv ):
    """
    Parses command line arguments for generating on-disk labeling data.

    See print_usage() for the structure of the options and arguments parsed.

    Raises ValueError if there is an issue parsing arguments.  This may occur when
    an incorrect number of arguments are supplied, if an invalid argument is supplied,
    or if an unknown option is parsed.

    Takes 1 argument:

      argv - List of strings representing the command line to parse.  Assumes the
             first string is the name of the script executing.

    Returns 2 values:

      options   - Object whose attributes represent the optional flags parsed.  Contains
                  at least the following:

                      .colormap_name            - String specifying the name of a
                                                  Matplotlib colormap.
                      .dask_cluster_address     - Address of an existing Dask cluster
                                                  to connect to for distributing
                                                  work to.  None if a cluster was
                                                  not provided.  At most, one of
                                                  .dask_cluster_address and
                                                  .dask_cluster_parameters may be
                                                  non-None.
                      .dask_cluster_parameters  - Parameters for a local Dask cluster
                                                  to create and distribute work to.
                                                  When specified, is a tuple of three
                                                  elements specifying: 1) number
                                                  workers, 2) memory limit in MiB
                                                  per worker, and 3) dashboard
                                                  address to use.  At most, one of
                                                  .dask_cluster_address and
                                                  .dask_cluster_parameters may be
                                                  non-None.
                      .iwp_labels_path          - Path to IWP labels to overlay on
                                                  rendered images.  None if a labels
                                                  file was not specified.
                      .label_color              - Sequence specifying an RGB(A) color
                                                  for overlaid labels.  Will have
                                                  either three or four components (for
                                                  RGB and RGBA, respectively) in the
                                                  range of [0, 255] or [0.0, 1.0]
                                                  (for PIL- and Matplotlib-rendered
                                                  images, respectively - see
                                                  .render_figure_flag).
                      .local_statistics_flag    - Flag specifying whether variable
                                                  statistics are computed per-XY
                                                  slice or across all XY slices.
                                                  If True, per-slice statistics are
                                                  used.
                      .quantization_table_name  - String specifying the name of a
                                                  IWP quantization table.
                      .render_figure_flag       - Flag specifying whether XY slices
                                                  should be rendered to Matplotlib
                                                  figures or not.  If False, each
                                                  XY slice is directly translated
                                                  into an image.
                      .slice_index_range        - Range object specifying the
                                                  XY indices to process.
                      .statistics_override_list - List of triplets, one per variable
                                                  in arguments.variable_names,
                                                  specifying variable statistics to
                                                  use.
                      .time_index_range         - Range object specifying the time
                                                  indices to process.
                      .title_images_flag        - Flag specifying whether metadata
                                                  titles are burned into each image.
                      .variable_statistics_path - Path to a JSON file containing
                                                  pre-computed variable statistics.
                      .verbose_flag             - Flag specifying verbose execution.

                  NOTE: Will be None if execution is not required.

      arguments - Object whose attributes represent the positional arguments parsed.
                  Contains at least the following:

                      .experiment_name     - Name of the experiment.
                      .netcdf_path_pattern - List of path patterns representing
                                             the IWP dataset.
                      .output_root         - Path to directory where images are
                                             written to.
                      .variable_names      - List of variable names whose data
                                             will be imaged.

                  NOTE: Will be None if execution is not required.

    """

    # indices into argv's positional arguments specifying each of the required
    # arguments.
    ARG_NETCDF_PATH_PATTERN = 0
    ARG_OUTPUT_ROOT         = 1
    ARG_EXPERIMENT_NAME     = 2
    ARG_VARIABLE_NAMES      = 3
    NUMBER_ARGUMENTS        = ARG_VARIABLE_NAMES + 1

    # empty class designed to hold name values.
    #
    # XXX: this is a kludge until we sort out how to replace everything with
    #      argparse.
    #
    class _Arguments( object ):
        pass

    options, arguments = _Arguments(), _Arguments()

    # set defaults for each of the options.
    #
    # use reasonable defaults for color maps and quantization tables and process
    # all available data (time step and XY slices), unless otherwise specified.
    # metadata titles are burned into images generated and statistics are
    # computed globally prior to rendering images.
    options.colormap_name            = DEFAULT_COLORMAP_NAME
    options.dask_cluster_address     = None
    options.dask_cluster_parameters  = None
    options.iwp_labels_path          = None
    options.label_color              = DEFAULT_LABEL_COLOR
    options.local_statistics_flag    = False
    options.quantization_table_name  = DEFAULT_QUANT_TABLE_NAME
    options.render_figure_flag       = False
    options.slice_index_range        = None
    options.statistics_override_list = []
    options.time_index_range         = None
    options.title_images_flag        = True
    options.variable_statistics_path = None
    options.verbose_flag             = False

    # parse our command line options.
    try:
        option_flags, positional_arguments = getopt.getopt( argv[1:], "C:c:Fhl:Ln:q:s:S:Tt:vz:" )
    except getopt.GetoptError as error:
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ) )

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-c":
            options.colormap_name = option_value
        if option == "-C":
            options.dask_cluster_address = option_value
        elif option == "-F":
            options.render_figure_flag = True
        elif option == "-h":
            print_usage( argv[0] )
            return (None, None)
        elif option == "-L":
            options.local_statistics_flag = True
        elif option == "-l":
            option_components = option_value.split( "," )

            # deal with the fact that we take the default color if only a label
            # path was provided.
            if len( option_components ) == 1:
                options.iwp_labels_path = option_components[0]
            elif len( option_components ) == 2:
                options.iwp_labels_path = option_components[0]
                options.label_color     = option_components[1]
            else:
                raise ValueError( "Invalid label specification received.  Must be of "
                                  "the form <labels_path>[,<colorspec>] ({:s}).\n"
                                  "\n"
                                  "<colorspec> may be of the form:\n"
                                  "\n"
                                  "   <PIL or Matplotlib name>\n"
                                  "   <R:G:B>                    (where R, G, and B are in [0, 1])\n".format(
                    option_value ) )
        elif option == "-n":
            options.dask_cluster_parameters = option_value.split( ":" )

            if not (0 < len( options.dask_cluster_parameters ) <= 3):
                raise ValueError( "Local Dask cluster specification must be of the form  "
                                  "'<number_workers>[:<memory_limit>[:<dashboard_address>]]' but "
                                  "received '{:s}'.".format(
                                      option_value ) )

            # ensure that we have three parameters, even if some are
            # placeholders.
            options.dask_cluster_parameters.extend( [None] *
                                                    (3 - len( options.dask_cluster_parameters )) )

        elif option == "-q":
            options.quantization_table_name = option_value
        elif option == "-S":
            options.variable_statistics_path = option_value
        elif option == "-s":
            # this will always return a list of lists, though we validate that
            # it is numeric and of the right shape below.
            options.statistics_override_list = list( map( lambda x: x.split( ":" ),
                                                          option_value.split( "," ) ) )
        elif option == "-T":
            options.title_images_flag = False
        elif option == "-t":
            options.time_index_range = iwp.utilities.parse_range( option_value )

            if options.time_index_range is None:
                raise ValueError( "Invalid time index range specified ({:s}).".format(
                    option_value ) )
        elif option == "-v":
            options.verbose_flag = True
        elif option == "-z":
            options.slice_index_range = iwp.utilities.parse_range( option_value )

            if options.slice_index_range is None:
                raise ValueError( "Invalid XY slice range specified ({:s}).".format(
                    option_value ) )

    # ensure we have the correct number of arguments.
    if len( positional_arguments ) != NUMBER_ARGUMENTS:
        raise ValueError( "Incorrect number of arguments.  Expected {:d}, received {:d}.".format(
            NUMBER_ARGUMENTS,
            len( positional_arguments ) ) )

    # map the positional arguments to named variables.
    arguments.experiment_name     = positional_arguments[ARG_EXPERIMENT_NAME]
    arguments.netcdf_path_pattern = positional_arguments[ARG_NETCDF_PATH_PATTERN].split( "," )
    arguments.output_root         = positional_arguments[ARG_OUTPUT_ROOT]
    arguments.variable_names      = positional_arguments[ARG_VARIABLE_NAMES].split( "," )

    # validate that the statistics overrides are well formed: 1) minimum,
    # maximum, and standard deviation, 2) each quantity is a floating point
    # value, and 3) each minimum is strictly less than the maximum.
    for override_index, variable_override in enumerate( options.statistics_override_list ):
        if len( variable_override ) != 3:
            raise ValueError( "Statistics override #{:d} does not have a minimum, "
                              "maximum, and standard deviation.  Instead, received {:d} component{:s}.".format(
                                  override_index + 1,
                                  len( variable_override ),
                                  "" if len( variable_override ) == 1 else "s" ) )

        try:
            variable_override = list( map( lambda x: float( x ),
                                           variable_override ) )
            options.statistics_override_list[override_index] = variable_override
        except ValueError:
            raise ValueError( "Failed to convert statistics override #{:d} to "
                              "floating point values ({}).".format(
                                  override_index + 1,
                                  variable_override ) )

        if not (variable_override[0] < variable_override[1]):
            raise ValueError( "Statistics override #{:d} has a minimum that is "
                              "not less than its maximum ({:f} >= {:f}).".format(
                                  override_index + 1,
                                  variable_override[0],
                                  variable_override[1] ) )

    # ensure that we have one override per variable rendered.
    if (len( options.statistics_override_list ) > 0 and
        len( options.statistics_override_list ) != len( arguments.variable_names )):
        raise ValueError( "Invalid number of statistics overrides provided.  "
                          "Expected {:d} but received {:d}.".format(
                              len( arguments.variable_names ),
                              len( options.statistics_override_list ) ) )

    # ensure that we were only requested to do one of the three statistics
    # approaches.
    statistics_approach_count = 0
    if options.local_statistics_flag:
        statistics_approach_count += 1
    if len( options.statistics_override_list ) > 0:
        statistics_approach_count += 1
    if options.variable_statistics_path is not None:
        statistics_approach_count += 1

    if statistics_approach_count > 1:
        raise RuntimeError( "Cannot specify more than one of 1) local statistics computation, "
                            "2) statistics overrides, or 3) pre-computed statistics.  {:d} "
                            "were requested.".format(
                                statistics_approach_count ) )

    # ensure that labels exist if we're overlaying them.
    if ((options.iwp_labels_path is not None) and
        not os.path.isfile( options.iwp_labels_path )):
        raise ValueError( "Label overlay was requested with labels in '{:s}' but "
                          "it does not exist.".format(
                          options.iwp_labels_path ) )

    # attempt to validate the color specification was correct based on how
    # we're rendering the XY slices.  PIL and Matplotlib allow different
    # color specifications, so we try and find invalid ones up front rather
    # than at the bottom of a 45+ deep call stack.
    if options.render_figure_flag:
        validator_type = iwp.utilities.ColorSystemType.MATPLOTLIB
    else:
        validator_type = iwp.utilities.ColorSystemType.PIL

    try:
        options.label_color = iwp.utilities.normalize_color_like( options.label_color,
                                                                  validator_type )
    except ValueError:
        raise ValueError( "Invalid {:s} color specification provided ({}).".format(
            "Matplotlib" if options.render_figure_flag else "PIL",
            options.label_color ) )

    # we can only use one cluster at a time, so the caller must be clear in
    # their specification.
    if ((options.dask_cluster_address is not None) and
        (options.dask_cluster_parameters is not None)):
        raise ValueError( "Must specify either an existing Dask cluster to connect "
                          "to or parameters to create a local cluster, but not both." )

    # check that the parameters to create a local cluster make sense.
    #
    # NOTE: we don't have to validate a cluster address as that is done
    #       implicitly when we attempt to connect to it.
    #
    if options.dask_cluster_parameters is not None:
        (number_workers,
         memory_limit,
         dashboard_address) = options.dask_cluster_parameters

        # ensure that we got 1 or more workers.
        try:
            number_workers = int( number_workers )

            if number_workers <= 0:
                raise ValueError()

        except ValueError as e:
            #
            # NOTE: we don't specify this as a string type since we may leave
            #       the try block either as a string or as an integer.
            #
            raise ValueError( "Invalid number of Dask cluster workers specified ({}).".format(
                number_workers ) )
        else:
            options.dask_cluster_parameters[0] = number_workers

        # ensure that we got an positive number of MiB.
        if memory_limit is not None:
            try:
                memory_limit = int( memory_limit )

                if memory_limit <= 0:
                    raise ValueError()

            except ValueError as e:
                #
                # NOTE: we don't specify this as a string type since we may leave
                #       the try block either as a string or as an integer.
                #
                raise ValueError( "Invalid Dask cluster worker memory limit specified ({}).".format(
                    memory_limit ) )
            else:
                options.dask_cluster_parameters[1] = memory_limit

        #
        # NOTE: we don't validate the dashboard address.  this will be
        #       implicitly checked when the cluster is created.
        #

    return options, arguments

def main( argv ):
    """
    Parses the supplied command line arguments and renders IWP data as on-disk
    images for labeling via an external tool.

    See print_usage() for details on execution and command line arguments.

    Takes 1 argument:

      argv - List of command line arguments and options, including the executing script.

    Returns 1 value:

      return_value - Integer status code.  Zero if execution was successful, non-zero
                     otherwise.

    """

    try:
        options, arguments = parse_command_line( argv )
    except Exception as e:
        print( str( e ), file=sys.stderr )
        return 1

    # return success in the case where normal execution is not required, but
    # we're not in an exceptional situation (e.g. requested help).
    if (options is None) and (arguments is None):
        return 0

    # figure out our Dask cluster configuration first.  this ensures that any
    # heavyweight setup operations benefit from its existence.
    try:
        cluster_manager = _get_dask_cluster_manager( options.dask_cluster_address,
                                                     options.dask_cluster_parameters )

        # announce the cluster's dashboard if we launched/connected properly.
        # this lets us connect our browser and view work as it is distributed.
        if options.verbose_flag and hasattr( cluster_manager, "dashboard_link" ):
            print( "Dask cluster dashboard is at '{:s}'.".format(
                cluster_manager.dashboard_link ) )
    except Exception as e:
        print( "Failed to get a cluster manager ({:s}).".format(
            str( e ) ),
               file=sys.stderr )
        return 1

    with cluster_manager:
        # default to None specifies local statistics computation.
        variable_statistics = None

        # load statistics from disk if provided.
        if (not options.local_statistics_flag) and (options.variable_statistics_path is not None):
            try:
                loaded_statistics = iwp.statistics.load_statistics( options.variable_statistics_path )
            except Exception as e:
                print( "Failed to load statistics from '{:s}' ({:s}).".format(
                    options.variable_statistics_path,
                    str( e ) ),
                       file=sys.stderr )
                return 1

            # pack the serialized statistics into lists that match the
            # iwp.quantization interface.
            variable_statistics = {}
            for variable_name in loaded_statistics.keys():
                # attempt to unpack the statistics of interest.  if these don't all
                # exist, then we pretend we didn't get any.
                try:
                    variable_statistics[variable_name] = [
                        loaded_statistics[variable_name]["minimum"],
                        loaded_statistics[variable_name]["maximum"],
                        loaded_statistics[variable_name]["stddev"]
                    ]
                except Exception:
                    pass
                else:
                    if options.verbose_flag:
                        print( "Loaded statistics for '{:s}'.".format(
                            variable_name ) )

        # acquire a quantization table.
        quantization_table_builder = iwp.utilities.lookup_module_function( iwp.quantization,
                                                                           options.quantization_table_name )

        if quantization_table_builder is None:
            print( "Invalid quantization table builder specified ('{:s}').".format(
                options.quantization_table_name ),
                   file=sys.stderr )
            return 1

        # acquire a color map.
        colormap = iwp.utilities.lookup_module_function( matplotlib.cm,
                                                         options.colormap_name )

        if colormap is None:
            print( "Invalid colormap specified ('{:s}').".format(
                options.colormap_name ),
                   file=sys.stderr )
            return 1

        # acquire the IWP labels.
        iwp_labels = []
        if options.iwp_labels_path is not None:
            try:
                iwp_labels = iwp.labels.load_iwp_labels( options.iwp_labels_path )

                if options.verbose_flag:
                    print( "Loaded {:d} label{:s}.".format(
                        len( iwp_labels ),
                        "" if len( iwp_labels ) == 1 else "s" ) )
            except Exception as e:
                print( "Failed to load IWP labels from '{:s}' ({:s}).".format(
                    options.iwp_labels_path,
                    str( e ) ),
                       file=sys.stderr )
                return 1

        # open the dataset.
        try:
            xarray_dataset = iwp.data_loader.open_xarray_dataset( arguments.netcdf_path_pattern )
        except Exception as e:
            print( "Failed to load a dataset from '{}' ({:s}).".format(
                arguments.netcdf_path_pattern,
                str( e ) ),
                   file=sys.stderr )
            return 1

        # default to the entirety of each dimension if the caller has not specified
        # ranges of interest.
        if options.time_index_range is None:
            options.time_index_range = list( xarray_dataset.coords["Cycle"].values )
        if options.slice_index_range is None:
            options.slice_index_range = range( len( xarray_dataset.coords["z"] ) )

        # ensure that accessing the dataset will not cause any issues in the middle
        # of our data generation.  better to bail now with a sensible error message.
        try:
            iwp.utilities.validate_variables_and_ranges( xarray_dataset,
                                                         arguments.variable_names,
                                                         options.time_index_range,
                                                         options.slice_index_range )
        except ValueError as e:
            print( "Failed to validate the request ({:s}).".format(
                str( e ) ) )
            return 1

        # handle computing global statistics or using the statistics overrides.
        if not options.local_statistics_flag:
            # handle the case where we did not load statistics from disk.
            if variable_statistics is None:
                variable_statistics = {}

            if len( options.statistics_override_list ) > 0:
                # map the overrides to their variable names.
                for variable_index, variable_name in enumerate( arguments.variable_names ):
                    variable_statistics[variable_name] = options.statistics_override_list[variable_index]
            else:
                # compute global statistics for each variable that doesn't have them
                # pre-loaded.
                for variable_name in arguments.variable_names:
                    if variable_name not in variable_statistics:

                        if options.verbose_flag:
                            print( "Computing statistics for '{:s}'.".format(
                                variable_name ) )

                        variable_statistics[variable_name] = iwp.statistics.compute_statistics( xarray_dataset[variable_name] )

        # render each of the requested XY slices as images.
        iwp.rendering.ds_write_xy_slice_images( xarray_dataset,
                                                arguments.output_root,
                                                arguments.experiment_name,
                                                arguments.variable_names,
                                                options.time_index_range,
                                                options.slice_index_range,
                                                variable_statistics,
                                                colormap,
                                                quantization_table_builder,
                                                render_figure_flag=options.render_figure_flag,
                                                title_flag=options.title_images_flag,
                                                iwp_labels=iwp_labels,
                                                label_color=options.label_color,
                                                verbose_flag=options.verbose_flag ),

        # now we leave the cluster context manager.

    # we don't have a return code from the rendering, so assume if we got here
    # that everything is okay.
    return 0

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
