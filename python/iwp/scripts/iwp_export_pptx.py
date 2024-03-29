#!/usr/bin/env python3

import getopt
import os
import sys

import matplotlib.cm

import iwp.data_loader
import iwp.pptx
import iwp.quantization
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

def print_usage( program_name, file_handle=sys.stdout ):
    """
    Prints the script's usage to standard output.

    Takes 2 arguments:

      program_name - Name of the program currently executing.
      file_handle  - File handle to print to.  If omitted, defaults to standard output.

    Returns nothing.

    """

    usage_str = \
"""{program_name:s} [-c <colormap>] [-h] [-l <labels_path>[,<color>]] [-q <quant_table>] [-S <input_statistics_path>] <netcdf_pattern> <pptx_path> <experiment> <variable>[,<variable>[,<variable>]] <time_step_index>,<xy_slice_index> [...]

    Exports one or more XY slices from <netcdf_pattern> into an Powerpoint slide deck
    with one slide per XY slice, written to <pptx_path>.

    Each slide contains the data from up to three of the IWP dataset's <variable>s
    for comparison and review.  Data are normalized, quantized, and colorized according
    to the variable statistics, quantization table, and colormaps provided, respectively.
    Labels corresponding to each of the XY slices may be overlaid on the data if
    provided.

    The IWP dataset's pattern, <netcdf_pattern>, must be specified as a format string
    template that is instantiated with each slice's <time_step_index>.  For example,
    the pattern below assumes the dataset is indexed by zero padded, six digit integers:

       /path/to/iwp/data/dataset-{{:06d}}.nc

    Wildcard patterns are not supported.

    Slices are identified by (time, xy slice) pairs and are specified as one or more
    comma-delimeted pairs, <time_step_index>,<xy_slice_index>.  Slides are generated
    in the order specified on the command line.

    The command line options shown above are described below:

        -c <colormap>                Use <colormap> when rendering images.  Must be a
                                     the name of a valid Matplotlib colormap that is
                                     found at "matplotlib.cm.<colormap>".  If omitted,
                                     defaults to "{colormap:s}".
        -h                           Print this help message and exit.
        -l <labels_path>[,<color>]   Path to serialized IWP labels to overlay on XY
                                     slices.  Optional <color> to create the labels'
                                     bounding boxes with.  <color> may be specified
                                     as a Matplotlib colorspec or as a
                                     colon-delimited sequence of floating point
                                     values in the range of [0, 1].  If both are
                                     omitted, no labels are overlaid.  If <color> is
                                     omitted, a high contrast default is selected.
        -q <quant_table>             Use <quant_table> for quantizing XY slice data
                                     into image data.  Must be a valid IWP quantization
                                     table that is found at "iwp.quantization.<quant_table>".
                                     If omitted, defaults to "{quant_table:s}".
        -S <input_statistics_path>   Specifies pre-computed statistics for the dataset.
                                     Variable statistics contained in <input_statistics_path>
                                     are loaded prior to computation allowing for
                                     building dataset statistics in stages.  Must contain
                                     statistics for all <variable>s specified. If omitted,
                                     all statistics are computed on the fly on a per-slice
                                     basis.
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

                      .colormap_names           - Sequence of strings specifying the
                                                  names of Matplotlib colormaps to
                                                  apply to the rendered variables.
                                                  Will have the same length as
                                                  arguments.variable_names.
                      .input_statistics_path    - Path to a JSON file containing
                                                  pre-computed variable statistics.
                                                  None if not specified which denotes
                                                  on-demand, local variable statistics.
                      .iwp_labels_path          - Path to IWP labels to use during
                                                  playlist creation.  If omitted,
                                                  defaults to None specifying no
                                                  labels are available.
                      .label_color              - Sequence specifying an RGB color
                                                  for overlaid labels.  Will have
                                                  three components (for RGB) in the
                                                  range of [0, 255].
                      .quantization_table_names - Sequence of strings specifying the
                                                  name of IWP quantization tables to
                                                  apply to the rendered variables.
                                                  Will have the same length as
                                                  arguments.variable_names.

                  NOTE: Will be None if execution is not required.

      arguments - Object whose attributes represent the positional arguments parsed.
                  Contains at least the following:

                      .experiment_name     - Name of the experiment associated with
                                             the IWP dataset.
                      .netcdf_path_pattern - List of path patterns representing the
                                             IWP dataset.
                      .pptx_path           - Path to write the generated PowerPoint
                                             file to.
                      .variable_names      - List of variable names whose data will
                                             be imaged.
                      .slice_pairs         - List of tuples, (time_step_index, xy_slice_index),
                                             of XY slices to image.

                  NOTE: Will be None if execution is not required.

    """

    # indices into argv's positional arguments specifying each of the required
    # arguments.
    ARG_NETCDF_PATH_PATTERN  = 0
    ARG_PPTX_PATH            = 1
    ARG_EXPERIMENT_NAME      = 2
    ARG_VARIABLE_NAMES       = 3
    ARG_SLICE_PAIRS          = 4
    MINIMUM_NUMBER_ARGUMENTS = ARG_SLICE_PAIRS + 1

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
    # use reasonable defaults for color maps and quantization tables, and
    # process each XY slice according to global statistics.  labels or
    # pre-computed statistics must be provided to be utilized.
    options.colormap_names           = [DEFAULT_COLORMAP_NAME]
    options.input_statistics_path    = None
    options.iwp_labels_path          = None
    options.label_color              = DEFAULT_LABEL_COLOR
    options.quantization_table_names = [DEFAULT_QUANT_TABLE_NAME]

    # parse our command line options.
    try:
        option_flags, positional_arguments = getopt.getopt( argv[1:], "c:hl:q:S:" )
    except getopt.GetoptError as error:
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ) )

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-c":
            options.colormap_names = option_value.split( "," )
        elif option == "-h":
            print_usage( argv[0] )
            return (None, None)
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
                raise ValueError( "Invalid label specification received ({:s}).".format(
                    option_value ) )
        elif option == "-q":
            options.quantization_table_names = option_value.split( "," )
        elif option == "-S":
            options.input_statistics_path = option_value

    # ensure we have the correct number of arguments.
    if len( positional_arguments ) < MINIMUM_NUMBER_ARGUMENTS:
        raise ValueError( "Incorrect number of arguments.  Expected at least {:d}, received {:d}.".format(
            MINIMUM_NUMBER_ARGUMENTS,
            len( positional_arguments ) ) )

    # map the positional arguments to named variables.
    arguments.netcdf_path_pattern = positional_arguments[ARG_NETCDF_PATH_PATTERN].split( "," )
    arguments.pptx_path           = positional_arguments[ARG_PPTX_PATH]
    arguments.experiment_name     = positional_arguments[ARG_EXPERIMENT_NAME]
    arguments.variable_names      = positional_arguments[ARG_VARIABLE_NAMES].split( "," )

    try:
        arguments.slice_pairs = list( map( lambda x: tuple( map( lambda y: int( y ), x.split( "," ) ) ),
                                           positional_arguments[ARG_SLICE_PAIRS:] ) )
    except ValueError:
        raise ValueError( "Failed to parse the slice pairs ({:s}).".format(
            " ".join( positional_arguments[ARG_SLICE_PAIRS:] ) ) )

    # bail if we have too few or too many variables to review.  we have limited
    # space on each slide and cannot fit more than three XY slices before
    # running out of room.
    if not (0 < len( arguments.variable_names ) < 4):
        raise ValueError( "Must have between 1 and 3 variables to generate "
                          "review data for, received {:d}.".format(
                              len( arguments.variable_names ) ) )
    if any( map( lambda name: name.strip() == "", arguments.variable_names ) ):
        raise ValueError( "Variable names cannot be empty ({:s}).".format(
            positional_arguments[ARG_VARIABLE_NAMES] ) )

    # ensure that we got pairs of non-negative integers.  we don't have a dataset
    # available to know what the upper bounds are so we only make a basic sanity
    # check here.
    #
    # NOTE: we could pass negative integers through to the underlying dataset
    #       object, though it is difficult for users to understand which slice
    #       they actually represent.  disallowing them reduces flexibility
    #       slightly though, in reality, does not reduce functionality all that
    #       much.
    #
    for pair_index, (time_index, xy_slice_index) in enumerate( arguments.slice_pairs ):
        if time_index < 0:
            raise ValueError( "Slice pair #{:d} has a negative time index ({:d}).".format(
                pair_index + 1,
                time_index ) )
        elif xy_slice_index < 0:
            raise ValueError( "Slice pair #{:d} has a negative XY slice index ({:d}).".format(
                pair_index + 1,
                time_index ) )

    # ensure that labels exist if we're overlaying them.
    if ((options.iwp_labels_path is not None) and
        not os.path.isfile( options.iwp_labels_path )):
        raise ValueError( "Label overlay was requested with labels in '{:s}' but "
                          "it does not exist.".format(
                          options.iwp_labels_path ) )

    # validate the color specification provided.
    try:
        options.label_color = iwp.utilities.normalize_color_like( options.label_color,
                                                                  iwp.utilities.ColorSystemType.MATPLOTLIB )

        # map the components into [0, 255] to be compatible with python-pptx.
        options.label_color = tuple( map( lambda x: int( x * 255 ),
                                          options.label_color ) )
    except ValueError:
        raise ValueError( "Invalid Matplotlib color specification provided ({:s}).".format(
            options.label_color ) )

    # ensure that we either got one colormap or one per variable.  replicate
    # either so that we have one per variable, regardless.
    if (len( options.colormap_names ) != 1 and
        len( options.colormap_names ) != len( arguments.variable_names )):
        raise ValueError( "Must have either 1 or number_variables ({:d}) color maps, "
                          "received {:d}.".format(
                              len( arguments.variable_names ),
                              len( options.colormap_names ) ) )
    elif any( map( lambda name: name.strip() == "",
                   options.colormap_names ) ):
        raise ValueError( "Colormap names cannot be empty ({:s}).".format(
            ",".join( map( lambda name: "'" + name + "'",
                           options.colormap_names ) ) ) )
    elif len( options.colormap_names ) == 1:
        # we were only supplied a single colormap to apply to all variables.
        # replicate it as if we were given N-many colormaps, one per variable.
        options.colormap_names *= len( arguments.variable_names )

    # ensure that we either got one quantization table or one per variable.
    # replicate either so that we have one per variable, regardless.
    if (len( options.quantization_table_names ) != 1 and
        len( options.quantization_table_names ) != len( arguments.variable_names )):
        raise ValueError( "Must have either 1 or number_variables ({:d}) quantization "
                          "tables, received {:d}.".format(
                              len( arguments.variable_names ),
                              len( options.quantization_table_names ) ) )
    elif any( map( lambda name: name.strip() == "",
                   options.quantization_table_names ) ):
        raise ValueError( "Quantization table names cannot be empty ({:s}).".format(
            ",".join( map( lambda name: "'" + name + "'",
                           options.quantization_table_names ) ) ) )
    elif len( options.quantization_table_names ) == 1:
        # we were only supplied a single quantization table to apply to all
        # variables.  replicate it as if we were given N-many quantization
        # tables, one per variable.
        options.quantization_table_names *= len( arguments.variable_names )

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

    # load statistics from disk if provided.  otherwise start with None to
    # signal local statistics computation.
    variable_statistics = None
    if options.input_statistics_path is not None:
        variable_statistics = iwp.statistics.load_statistics( options.input_statistics_path )

        # verify that we have statistics for each of the variables requested.
        # we don't support computing global statistics, so bail if we don't
        # have everything we need.
        for variable_name in arguments.variable_names:
            if (variable_name not in variable_statistics or
                "minimum" not in variable_statistics[variable_name] or
                "maximum" not in variable_statistics[variable_name] or
                "stddev" not in variable_statistics[variable_name]):
                print( "Global statistics were provided in '{:s}' but "
                       "does not contain statistics for '{:s}'.".format(
                           options.input_statistics_path,
                           variable_name ),
                       file=sys.stderr )
                return 1

    # acquire quantization tables.
    quantization_table_builders = []
    for quantization_table_name in options.quantization_table_names:
        quantization_table_builder = iwp.utilities.lookup_module_function( iwp.quantization,
                                                                           quantization_table_name )

        if quantization_table_builder is None:
            print( "Invalid quantization table name specified ('{:s}').".format(
                quantization_table_name ),
                   file=sys.stderr )
            return 1

        quantization_table_builders.append( quantization_table_builder )

    # acquire color maps.
    colormaps = []
    for colormap_name in options.colormap_names:
        colormap = iwp.utilities.lookup_module_function( matplotlib.cm,
                                                         colormap_name )

        if colormap is None:
            print( "Invalid colormap specified ('{:s}').".format(
                colormap_name ),
                   file=sys.stderr )
            return 1

        colormaps.append( colormap )

    # open the dataset.
    try:
        time_step_indices = set( map( lambda slice_pair: slice_pair[0],
                                      arguments.slice_pairs ) )

        iwp_dataset = iwp.data_loader.IWPDataset( arguments.netcdf_path_pattern[0],
                                                  list( time_step_indices ),
                                                  variables=arguments.variable_names )
    except ValueError as e:
        print( "Failed to open the dataset at '{:s}' with times [{:s}] and "
               "variables {:s} ({:s}).".format(
            arguments.netcdf_path_pattern,
                   ", ".join( sorted( time_step_indices ) ),
                   ", ".join( map( lambda x: "'" + x + "'",
                                   arguments.variable_names ) ),
                   str( e ) ) )
        return 1

    # ensure that accessing the dataset will not cause any issues in the middle
    # of our slide generation.  better to bail now with a sensible error message.
    #
    # NOTE: we've already implicitly validated the time step indices and
    #       variable names by virtue that we could open the dataset.  we ignore
    #       the inefficiency since this only happens once during setup.
    #
    try:
        time_step_indices, xy_slice_indices = zip( *arguments.slice_pairs )

        # only validate the unique indices.  presumably the work to deduplicate
        # indices here is more efficient than repeatedly validating the same
        # indices.
        unique_time_step_indices = list( set( time_step_indices ) )
        unique_xy_slice_indices  = list( set( xy_slice_indices ) )

        iwp.utilities.validate_variables_and_ranges( iwp_dataset,
                                                     arguments.variable_names,
                                                     unique_time_step_indices,
                                                     unique_xy_slice_indices )
    except ValueError as e:
        print( "Failed to validate the request ({:s}).".format(
            str( e ) ) )
        return 1

    # load IWP labels to overlay on the slices.
    iwp_labels = []
    if options.iwp_labels_path is not None:
        try:
            iwp_labels = iwp.labels.load_iwp_labels( options.iwp_labels_path )
        except ValueError as e:
            print( "Failed to load IWP labels from '{:s}' ({:s}).".format(
                options.iwp_labels_path,
                str( e ) ) )

    presentation = iwp.pptx.create_data_review_presentation( iwp_dataset,
                                                             arguments.experiment_name,
                                                             arguments.variable_names,
                                                             arguments.slice_pairs,
                                                             variable_statistics,
                                                             colormaps,
                                                             quantization_table_builders,
                                                             iwp_labels=iwp_labels,
                                                             label_color=options.label_color )

    # write the presentation to disk.
    try:
        presentation.save( arguments.pptx_path )
    except Exception as e:
        print( "Failed to save slides to '{:s}' ({:s}).".format(
            arguments.pptx_path,
            str( e ) ) )

    return 0

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
