#!/usr/bin/env python3

import getopt
import os
import sys

import iwp.labels

# sort methods supported.  these map one to one to the sort types in
# iwp.labels.IWPLabelSortType.
SORT_METHOD_NONE     = "none"
SORT_METHOD_SPATIAL  = "spatial"
SORT_METHOD_TEMPORAL = "temporal"

def print_usage( program_name, file_handle=sys.stdout ):
    """
    Prints the script's usage to standard output.

    Takes 2 arguments:

      program_name - Name of the program currently executing.
      file_handle  - File handle to print to.  If omitted, defaults to standard output.

    Returns nothing.

    """

    usage_str = \
"""{program_name:s} [-f] [-h] [-s <sort_method>] <output_labels> <input_labels> [<input_labels>] [...]]

    Combines one or more files containing IWP labels into a single file.  The contents
    of <input_labels> are merged together, possibly sorted (see '-s' below), and written
    out to <output_labels>.  This is used to interact with utilities that expect a single
    file of IWP labels.

    The command line options shown above are described below:

        -f                 Overwrite <output_labels> if it exists.  By default, label
                           merging will only create new files to avoid data loss.
        -h                 Print this help message and exit.
        -s <sort_method>   Specifies how labels are combined.  By default labels from
                           <input_labels> are simply concatenated together, though may
                           be sorted if desired.  <sort_method> must be one of the
                           following:

                              {method_none:12s}   Do not sort.
                              {method_temporal:12s}   Sort so that time steps are sorted
                                             before XY slices.
                              {method_spatial:12s}   Sort spatially so that XY slices are
                                             sorted before time steps.
""".format(
    program_name=program_name,
    method_none=SORT_METHOD_NONE,
    method_spatial=SORT_METHOD_SPATIAL,
    method_temporal=SORT_METHOD_TEMPORAL
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

                      .force_flag    - Boolean flag specifying whether overwriting the
                                       output file should be forced.
                      .sort_method   - String specifying the sort method.  One of:
                                       "none" (no sorting), "xy_slices" (temporal sorting),
                                       or "z_stacks" (spatial sorting).  If omitted,
                                       defaults to "xy_slices".

                  NOTE: Will be None if execution is not required.

      arguments - Object whose attributes represent the positional arguments parsed.
                  Contains at least the following:

                      .input_paths   - List of IWP label paths to parse and merge together.
                      .output_path   - IWP label path to write the merged IWP labels to.

                  NOTE: Will be None if execution is not required.

    """

    # indices into argv's positional arguments specifying each of the required
    # arguments.
    ARG_OUTPUT_PATH          = 0
    ARG_INPUT_PATHS          = 1
    MINIMUM_NUMBER_ARGUMENTS = ARG_INPUT_PATHS + 1

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
    # by default we prevent accidental overwrite of existing data and sort
    # IWP labels temporally.
    options.force_flag  = False
    options.sort_method = SORT_METHOD_TEMPORAL

    # parse our command line options.
    try:
        option_flags, positional_arguments = getopt.getopt( argv[1:], "fhs:" )
    except getopt.GetoptError as error:
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ) )

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-f":
            options.force_flag = True
        elif option == "-h":
            print_usage( argv[0] )
            return (None, None)
        elif option == "-s":
            options.sort_method = option_value

    # ensure we have the correct number of arguments.
    if len( positional_arguments ) < MINIMUM_NUMBER_ARGUMENTS:
        raise ValueError( "Incorrect number of arguments.  Expected at least {:d}, received {:d}.".format(
            MINIMUM_NUMBER_ARGUMENTS,
            len( positional_arguments ) ) )

    # map the positional arguments to named variables.
    arguments.input_paths = positional_arguments[ARG_INPUT_PATHS:]
    arguments.output_path = positional_arguments[ARG_OUTPUT_PATH]

    # validate the sort method requested.
    VALID_SORT_METHODS = [SORT_METHOD_NONE,
                          SORT_METHOD_SPATIAL,
                          SORT_METHOD_TEMPORAL]
    if not( options.sort_method.lower() in VALID_SORT_METHODS ):
        raise ValueError( "Invalid sort method specified (\"{:s}\").  Must be one "
                          "of {:s}.".format(
            options.sort_method,
            ", ".join( list( map( lambda x: "\"{:s}\"".format( x ),
                                  VALID_SORT_METHODS ) ) ) ) )

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

    # check if the output path already exists and we've been requested to overwrite (early out).
    if os.path.exists( arguments.output_path ):
        if os.path.isdir( arguments.output_path ):
            print( "Output path is a directory not a file ({:s}).".format(
                arguments.output_path ),
                   file=sys.stderr )
            return 1
        if not options.force_flag:
            print( "Output path exists but we're cowardly avoiding overwriting "
                   "its contents ({:s}).".format(
                   arguments.output_path ),
                   file=sys.stderr )
            return 1

    # ensure that the input paths exist.
    for input_path in arguments.input_paths:
        if os.path.exists( input_path ) and not os.path.isfile( input_path ):
            print( "IWP labels file \"{:s}\" exists but is not a file.".format(
                input_path ),
                   file=sys.stderr )

            return 1

    # load each of the IWP label files and concatenate.
    iwp_labels = []
    for input_path in arguments.input_paths:
        try:
            current_iwp_labels = iwp.labels.load_iwp_labels( input_path )
        except Exception as e:
            print( "Failed to load IWP labels from \"{:s}\" - {:s}.".format(
                input_path,
                str( e ) ),
                   file=sys.stderr )

            return 1

        iwp_labels.extend( current_iwp_labels )

    # sort if requested.
    if options.sort_method != SORT_METHOD_NONE:
        if options.sort_method == SORT_METHOD_SPATIAL:
            sort_type = iwp.labels.IWPLabelSortType.SPATIAL
        elif options.sort_method == SORT_METHOD_TEMPORAL:
            sort_type = iwp.labels.IWPLabelSortType.TEMPORAL

        # note that we work in place since there are no other users of the
        # labels.
        _ = iwp.labels.sort_iwp_labels( iwp_labels,
                                        sort_type,
                                        in_place_flag=True )

    # serialized the merged labels.
    try:
        iwp.labels.save_iwp_labels( arguments.output_path, iwp_labels )
    except Exception as e:
        print( "Failed to write IWP labels to \"{:s}\" - {:s}".format(
            arguments.output_path,
            str( e ) ),
               file=sys.stderr )

        return 1

    return 0

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
