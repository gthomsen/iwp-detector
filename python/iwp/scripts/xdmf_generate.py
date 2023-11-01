#!/usr/bin/env python3

import getopt
import sys

import iwp.xdmf

def print_usage( program_name, file_handle=sys.stdout ):
    """
    Prints the script's usage to standard output.

    Takes 2 arguments:

      program_name - Name of the program currently executing.
      file_handle  - File handle to print to.  If omitted, defaults to standard output.

    Returns nothing.

    """

    usage_str = \
"""{program_name:s} [-h] <xmdf_path> <netcdf_path_template> <sequence_start>:<sequence_stop> <dataset> [<dataset> ...]

Generates XDMF to describe a sequence of netCDF files.  Creates an XDMF file, <xdmf_path>,
that describes the netCDF files found when <netcdf_path_template> is instantiated with
the integers in the range of <sequence_start>:<sequence_stop>.  Each <dataset> specified
is exposed from the underlying netCDF files.

<dataset>s are accessed relative to the file root and do not need to have a leading
slash.

The command line options shown above are described below:

    -h                  Print this help message and exit.
""".format(
    program_name=program_name
)

    print( usage_str, file=file_handle )

def parse_command_line( argv ):
    """
    Parses the script's command line into two objects whose attributes contain
    the script's execution parameters.

    Takes 1 argument:

      argv - List of strings representing the command line to parse.  Assumes the
             first string is the name of the script executing.

    Returns 2 values:

      options   - Object whose attributes represent the optional flags parsed.

                  NOTE: Currently no options are handled, so options will always
                        be None.

                  NOTE: Will be None if execution is not required.

      arguments - Object whose attributes represent the positional arguments parsed.
                  Contains at least the following:

                      .datasets             - List of dataset names to expose.
                      .netcdf_path_template - String template describing the netCDF
                                              file family's path.
                      .sequence             - Range object specifying the sequence
                                              numbers to instantiate
                                              .netcdf_path_template with.
                      .xdmf_path            - Path to the XDMF generated.

    """

    # indicies into argv's positional arguments specifying each of the required
    # arguments.
    ARG_XDMF_PATH            = 0
    ARG_NETCDF_PATH_TEMPLATE = 1
    ARG_SEQUENCE             = 2
    ARG_DATASET_NAME         = 3
    NUMBER_MINIMUM_ARGUMENTS = ARG_DATASET_NAME + 1

    # empty class designed to hold name values.
    #
    # XXX: this is a kludge until we sort out how to replace everything with
    #      argparse.
    #
    class _Arguments( object ):
        pass

    options, arguments = _Arguments(), _Arguments()

    # parse our command line options.
    try:
        option_flags, positional_arguments = getopt.getopt( argv[1:], "h" )
    except getopt.GetoptError as error:
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ) )

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-h":
            print_usage( argv[0] )
            return (None, None)

    # ensure we have enough positional arguments.
    if len( positional_arguments ) < NUMBER_MINIMUM_ARGUMENTS:
        raise ValueError( "Incorrect number of arguments.  Expected at least {:d} but "
                          "received {:d}.".format(
                              NUMBER_MINIMUM_ARGUMENTS,
                              len( positional_arguments ) ) )

    # map the positional arguments to named variables.
    arguments.xdmf_path            = positional_arguments[ARG_XDMF_PATH]
    arguments.netcdf_path_template = positional_arguments[ARG_NETCDF_PATH_TEMPLATE]
    arguments.sequence             = iwp.utilities.parse_range( positional_arguments[ARG_SEQUENCE] )
    arguments.datasets             = positional_arguments[ARG_DATASET_NAME:]

    # drop leading slashes on datasets.  attributes are presented relative to
    # the group opened and must be accessed sans leading slash.
    arguments.datasets = list( map( lambda dataset: dataset.removeprefix( "/" ),
                                    arguments.datasets ) )

    # ensure the datasets are unique.
    #
    # NOTE: we do this after partial normalization (stripping leading slashes)
    #       so we catch more common duplications.  this isn't perfect since we
    #       don't resolve links within the file's datasets, but that is highly
    #       unlikely to ever be encountered.
    #
    seen_datasets      = []
    duplicate_datasets = []
    for dataset in arguments.datasets:
        if dataset in seen_datasets:
            if dataset not in duplicate_datasets:
                duplicate_datasets.append( dataset )
        else:
            seen_datasets.append( dataset )

    if len( duplicate_datasets ) > 0:
        raise ValueError( "{:d} dataset{:s} specified multiple times: {:s}".format(
            len( duplicate_datasets ),
            " was" if len( duplicate_datasets ) == 1 else "s were",
            ", ".join( sorted( duplicate_datasets ) )
        ) )

    return options, arguments

def main( argv ):
    """
    Parses the supplied command line arguments and generates an XDMF file describing
    a netCDF file family for access in external tools (e.g. ParaView).

    Takes 1 argument:

      argv - Sequence of strings specifying the script's command line, starting with the
             script's path.

    Returns 1 value

      exit_status - Integer status code to exit the script with.

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

    xdmf_generator = iwp.xdmf.XDMFGenerator( arguments.netcdf_path_template,
                                             arguments.sequence,
                                             arguments.datasets )

    with open( arguments.xdmf_path, "w" ) as xdmf_fp:
        xdmf_size_bytes = xdmf_fp.write( xdmf_generator.generate() )

    print( "Wrote {:d} byte{:s} to '{:s}'.".format(
        xdmf_size_bytes,
        "" if xdmf_size_bytes == 1 else "s",
        arguments.xdmf_path ) )

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
