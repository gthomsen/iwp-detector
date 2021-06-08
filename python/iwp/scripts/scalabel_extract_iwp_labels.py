#!/usr/bin/env python3

import getopt
import sys

import iwp.labels
import iwp.scalabel

def print_usage( program_name, file_handle=sys.stdout ):
    """
    Prints the script's usage to standard output.

    Takes 2 arguments:

      program_name - Name of the program currently executing.
      file_handle  - File handle to print to.  If omitted, defaults to standard output.

    Returns nothing.

    """

    usage_str = \
"""{program_name:s} [-h] <playlist_path> <labels_path>

    Extracts IWP labels from a Scalabel.ai playlist, at <playlist_path>, and writes
    them to disk at <labels_path>.

    The command line options shown above are described below:

        -h                        Print this help message and exit.
""".format(
    program_name=program_name,
)

    print( usage_str, file=file_handle )

def parse_command_line( argv ):
    """
    Parses command line arguments for extracting IWP labels from a Scalabel playlist.

    See print_usage() for the structure of the options and arguments parsed.

    Raises ValueError if there is an issue parsing arguments.  This may occur when
    an incorrect number of arguments are supplied, if an invalid argument is supplied,
    or if an unknown option is parsed.

    Takes 1 argument:

      argv - List of strings representing the command line to parse.  Assumes the
             first string is the name of the script executing.

    Returns 2 values:

      options   - Object whose attributes represent the optional flags parsed.  Currently
                  is an empty object.

                  NOTE: Will be None if execution is not required.

      arguments - Object whose attributes represent the positional arguments parsed.
                  Contains at least the following:

                      .scalabel_playlist_path - Path to the Scalabel playlist to extract
                                                IWP labels from.
                      .iwp_labels_path        - Path to write the extracted IWP labels to.

                  NOTE: Will be None if execution is not required.

    """

    # indices into argv's positional arguments specifying each of the required
    # arguments.
    ARG_SCALABEL_PLAYLIST_PATH = 0
    ARG_IWP_LABELS_PATH        = 1
    NUMBER_ARGUMENTS           = ARG_IWP_LABELS_PATH + 1

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
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ),
                          file=sys.stderr )

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-h":
            print_usage( argv[0] )
            return (None, None)

    # ensure we have the correct number of arguments.
    if len( positional_arguments ) != NUMBER_ARGUMENTS:
        raise ValueError( "Incorrect number of arguments.  Expected {:d}, received {:d}.".format(
            NUMBER_ARGUMENTS,
            len( positional_arguments ) ) )

    # map the positional arguments to named variables.
    arguments.scalabel_playlist_path = positional_arguments[ARG_SCALABEL_PLAYLIST_PATH]
    arguments.iwp_labels_path        = positional_arguments[ARG_IWP_LABELS_PATH]

    return options, arguments

def main( argv ):
    """
    Parses the supplied command line arguments and extracts IWP labels from a Scalabel
    playlist.

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

    # load the Scalabel frames and all of their labels.
    try:
        scalabel_frames = iwp.scalabel.load_scalabel_frames( arguments.scalabel_playlist_path )
    except Exception as e:
        print( "Failed to load the Scalabel playlist from to '{:s}' ({:s}).".format(
            arguments.scalabel_playlist_path,
            str( e ) ),
               file=sys.stderr )

        return 1

    # convert the Scalabel labels into IWP labels.
    iwp_labels = iwp.scalabel.extract_iwp_labels_from_frames( scalabel_frames )

    # serialize the labels to disk.
    try:
        iwp.labels.save_iwp_labels( arguments.iwp_labels_path, iwp_labels )
    except Exception as e:
        print( "Failed to write the IWP labels to '{:s}' ({:s}).".format(
            arguments.iwp_labels_path,
            str( e ) ),
               file=sys.stderr )

        return 1

    return 0

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
