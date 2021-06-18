#!/usr/bin/env python3

import getopt
import sys

import matplotlib.cm

import iwp.data_loader
import iwp.statistics
import iwp.utilities

def print_usage( program_name, file_handle=sys.stdout ):
    """
    Prints the script's usage to standard output.

    Takes 2 arguments:

      program_name - Name of the program currently executing.
      file_handle  - File handle to print to.  If omitted, defaults to standard output.

    Returns nothing.

    """

    usage_str = \
"""{program_name:s} [-f] [-h] [-S <input_statistics_path>] [-s <z_start>:<z_stop>] [-t <time_start>:<time_stop>] [-v] <netcdf_pattern> <output_statistics_path> [<variable>[,<variable>[...]]]

    Computes summary statistics of grid variables in the netCDF4 dataset specified
    by <netcdf_pattern> and serializes them into a JSON file at <output_statistics_path>.
 .  This pre-computation is useful to accelerate computations and rendering IWP data
    in various workflows (e.g. generating labeling images, pre-processing training data,
    etc).

    Existing statistics can be loaded from disk and updated with new, and or, additional
    statistics to support incremental workflows.  Subsets of an IWP dataset volume,
    in both time and XY slices, can be specified to compute localized statistics.

    Currently computes the following statistics:

      minimum
      maximum
      standard deviation

    If a comma-separated list of <variable>'s is omitted, all grid variables in the
    underlying dataset are summarized.

    The command line options shown above are described below:

        -f                           Force overwriting of <output_statistics_path>
                                     when input statistics are requested from the same
                                     file (see -S below).  If omitted, a warning is
                                     issued and execution is halted to prevent data
                                     loss.
        -h                           Print this help message and exit.
        -S <input_statistics_path>   Specifies pre-computed statistics for the dataset.
                                     Variable statistics contained in <input_statistics_path>
                                     are loaded prior to computation allowing for
                                     building dataset statistics in stages.  If omitted,
                                     all statistics are computed on the fly.
        -s <z_start>:<z_stop>        Specifies a range of XY slice indices to compute
                                     statistics for.  All XY slices in [<z_start>, <z_stop>]
                                     must be present in the data found in <netcdf_pattern>.
                                     If omitted, defaults to all XY slices available.
        -t <time_start>:<time_stop>  Specifies a range of time step indices to
                                     compute statistics for.  All time steps in
                                     [<time_start>, <time_stop>] must be present in
                                     the data found in <netcdf_pattern>.  If omitted,
                                     defaults to all time steps available.
        -v                           Enable verbose execution.  Status messages about
                                     progress are written to standard output.  If
                                     omitted, defaults to normal execution.
""".format(
    program_name=program_name
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

                      .force_flag            - Flag specifying forced execution.
                      .slice_index_range     - Range object specifying the XY indices
                                               o process.
                      .time_index_range      - Range object specifying the time
                                               indices to process.
                      .input_statistics_path - Path to a JSON file containing
                                               pre-computed variable statistics.
                      .verbose_flag          - Flag specifying verbose execution.

                  NOTE: Will be None if execution is not required.

      arguments - Object whose attributes represent the positional arguments parsed.
                  Contains at least the following:

                      .netcdf_path_pattern    - List of path patterns representing
                                                the IWP dataset.
                      .output_statistics_path - Path to directory where images are
                                                written to.
                      .variable_names         - List of variable names whose data
                                                will be imaged.

                  NOTE: Will be None if execution is not required.

    """

    # indices into argv's positional arguments specifying each of the required
    # arguments.
    ARG_NETCDF_PATH_PATTERN    = 0
    ARG_OUTPUT_STATISTICS_PATH = 1
    ARG_VARIABLE_NAMES         = 2
    NUMBER_MINIMUM_ARGUMENTS   = ARG_OUTPUT_STATISTICS_PATH + 1
    NUMBER_MAXIMUM_ARGUMENTS   = ARG_VARIABLE_NAMES + 1

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
    # compute statistics for all available data (time step and XY slices),
    # unless otherwise specified.  existing statistics are not overwritten
    # unless requested and execution is quiet by default.
    options.force_flag            = False
    options.input_statistics_path = None
    options.slice_index_range     = None
    options.time_index_range      = None
    options.verbose_flag          = 0

    # parse our command line options.
    try:
        option_flags, positional_arguments = getopt.getopt( argv[1:], "fhS:s:t:v" )
    except getopt.GetoptError as error:
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ) )

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-f":
            options.force_flag = True
        elif option == "-h":
            print_usage( argv[0] )
            return (None, None)
        elif option == "-S":
            options.input_statistics_path = option_value
        elif option == "-s":
            options.slice_index_range = iwp.utilities.parse_range( option_value )

            if options.slice_index_range is None:
                raise ValueError( "Invalid XY slice range specified ({:s}).".format(
                    option_value ) )
        elif option == "-t":
            options.time_index_range = iwp.utilities.parse_range( option_value )

            if options.time_index_range is None:
                raise ValueError( "Invalid time index range specified ({:s}).".format(
                    option_value ) )
        elif option == "-v":
            options.verbose_flag = True

    # ensure we have the correct number of arguments.
    if not (NUMBER_MINIMUM_ARGUMENTS <= len( positional_arguments ) <= NUMBER_MAXIMUM_ARGUMENTS):
        raise ValueError( "Incorrect number of arguments.  Expected at least {:d}, "
                          "but no more than {:d}, received {:d}.".format(
                              NUMBER_MINIMUM_ARGUMENTS,
                              NUMBER_MAXIMUM_ARGUMENTS,
                              len( positional_arguments ) ) )

    # map the positional arguments to named variables.
    arguments.netcdf_path_pattern    = positional_arguments[ARG_NETCDF_PATH_PATTERN].split( "," )
    arguments.output_statistics_path = positional_arguments[ARG_OUTPUT_STATISTICS_PATH]

    if len( positional_arguments ) >= (ARG_VARIABLE_NAMES + 1):
        arguments.variable_names = positional_arguments[ARG_VARIABLE_NAMES].split( "," )
    else:
        arguments.variable_names = None

    # provide a safety net to accidentally overwriting the input statistics.
    if (not options.force_flag and
        (options.input_statistics_path == arguments.output_statistics_path)):
        raise ValueError( "Input and output statistics paths are the same ({:s}).  "
                          "Cowardly refusing to update the file without being forced.".format(
                          options.input_statistics_path ) )

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

    # load statistics from disk if provided.  otherwise start with an empty
    # dictionary of statistics.
    if options.input_statistics_path is not None:
        variable_statistics = iwp.statistics.load_statistics( options.input_statistics_path )
    else:
        variable_statistics = {}

    # open the dataset.
    xarray_dataset = iwp.data_loader.open_xarray_dataset( arguments.netcdf_path_pattern )

    # default to the entirety of each dimension if the caller has not specified
    # ranges of interest.
    if options.time_index_range is None:
        options.time_index_range = list( xarray_dataset.coords["Cycle"].values )
    if options.slice_index_range is None:
        options.slice_index_range = range( len( xarray_dataset.coords["z"] ) )
    if arguments.variable_names is None:
        # identify all grid variables that have a floating point data type.
        # this ignores support variables that have the right shape but aren't
        # outputs of the simulation (e.g. MPI rank and subdomain identifiers).
        arguments.variable_names = list( filter( lambda variable: xarray_dataset[variable].dtype.kind == "f",
                                                 xarray_dataset.data_vars ) )

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

    # compute statistics for each of the variables requested.
    for variable_name in arguments.variable_names:
        if options.verbose_flag:
            print( "Computing statistics for '{:s}' over T={:d}:{:d} and Z={:d}:{:d}.".format(
                variable_name,
                options.time_index_range[0],
                options.time_index_range[-1],
                options.slice_index_range[0],
                options.slice_index_range[-1] ) )

        da = iwp.utilities.get_xarray_view( xarray_dataset,
                                            variable_name,
                                            options.time_index_range,
                                            options.slice_index_range )

        local_statistics = iwp.statistics.compute_statistics( da )

        variable_statistics[variable_name] = {
            "minimum": float( local_statistics[0] ),
            "maximum": float( local_statistics[1] ),
            "stddev":  float( local_statistics[2] )
            }

    try:
        if options.verbose_flag:
            print( "Writing statistics to '{:s}'.".format(
                arguments.output_statistics_path ) )

        iwp.statistics.save_statistics( arguments.output_statistics_path,
                                        variable_statistics )
    except Exception as e:
        print( "Failed to write out statistics to '{:s}' ({:s}).".format(
            arguments.output_statistics_path,
            str( e ) ) )
        return 1

    return 0

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
