#!/usr/bin/env python3

import getopt
import json
import sys

import iwp.scalabel
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
"""{program_name:s} [-f] [-h] [-l <labels_path>] [-L <labeling_strategy>] <playlist_path> <experiment> <variable>[,<variable>[...]] <time_start>:<time_stop> <z_start>:<z_stop> <data_root> <url_prefix> <component_count>

    Creates a JSON playlist suitable for importing into a new Scalabel.ai labeling project
    at <playlist_path>.  The playlist generated contains a sequence of "video frames"
    representing XY slices from an IWP data set.  Existing IWP labels may be incorporated
    into the playlist to support label refinement and review, rather than initial label
    creation.

    The sequence of XY slices generated depends on the <labeling_strategy> requested.
    Depending on the strategy specified, each frame's video name is constructed such
    that Scalabel.ai's video labeling project partitions a playlist into logical
    chunks that are each reasonably ordered (e.g. all time steps for fixed XY slice
    or all XY slices for a fixed time step).  Supported strategies are the following:

      {no_order:12s}- No order is specified.
      {xy_slices:12s}- Frames are sorted by location within the dataset.  Labelers see
                    all time steps for a specific XY slice.
      {z_stacks:12s}- Frames are sorted by time within the dataset.  Labelers see
                    all XY slices for a specific time step.
      {variables:12s}- Frames are sorted by time and location within the dataset.  Labelers
                    see each of the variables for a XY slice at a specific time step.

    Playlist frames have URLs comprised of <url_prefix> prepended to supplied <data_root>
    with <component_count>-many leading path components removed.  This allows replacement
    of some portion of the labeling data's path with the Scalabel.ai's web server address
    like so:

        <data_root> is /data/iwp/R5F04/
        <component_count> is 2
        Scalabel.ai's web root is /data/iwp/
        <url_prefix> is http://localhost:8686/items

        Generated frames will have URLs starting with:

            http://localhost:8686/items/R5F04/

    To avoid headaches with Scalabel.ai's tool, Each frame's underlying data's path is
    checked for accessibility before writing the playlist JSON.  Frame without data
    generate an error message on standard error and prevent the playlist from being written.

    The generate frames' metadata is structured such that, when exported from the labeling
    tool, the Scalabel labels may be extracted and converted into IWP labels for
    post-processing and configuration management.  Individual frames are programmaticaly
    named using <experiment>, <variable>, <time_index>, and <z_index>, so that it
    survives the round trip into and out of the Scalabel.ai tool.

    The command line options shown above are described below:

        -f                        Force creation of the playlist JSON regardless of whether
                                  the frames' underlying data exists or not.  If a datum
                                  doesn't exist, a warning is printed to standard error.
        -h                        Print this help message and exit.
        -l <labels_path>          Path to serialized IWP labels to incorporate in the created
                                  playlist.
        -L <labeling_strategy>    Strategy for sequencing the generated playlist.  Must
                                  be one of: {no_order:s}, {xy_slices:s}, {z_stacks:s},
                                  {variables:s}.  See the description above for details.
""".format(
    program_name=program_name,
    no_order="'{:s}'".format( iwp.scalabel.LabelingStrategyType.NO_ORDER.name.lower() ),
    xy_slices="'{:s}'".format( iwp.scalabel.LabelingStrategyType.XY_SLICES.name.lower() ),
    z_stacks="'{:s}'".format( iwp.scalabel.LabelingStrategyType.Z_STACKS.name.lower() ),
    variables="'{:s}'".format( iwp.scalabel.LabelingStrategyType.VARIABLES.name.lower() )
)

    print( usage_str, file=file_handle )

def parse_command_line( argv ):
    """
    Parses command line arguments for Scalabel playlist generation.

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

                      .force_flag      - Flag specifying whether playlist creation
                                         should be forced.  If omitted, defaults to
                                         False.
                      .iwp_labels_path - Path to IWP labels to use during playlist
                                         creation.  If omitted, defaults to None
                                         specifying no labels are available.

                  NOTE: Will be None if execution is not required.

      arguments - Object whose attributes represent the positional arguments parsed.
                  Contains at least the following:

                      .playlist_path   - Path to the Scalabel playlist to create.
                      .experiment_name - Name of the experiment represented by
                                         .playlist_path.
                      .variable_names  - List of strings specifying the experiment
                                         variables used in the generated playlist.
                      .time_ranges     - Slice specifying the time step indices to
                                         create a playlist from.
                      .xy_slice_range  - Slice specifying the XY slice indices to
                                         create a playlist from.
                      .data_root       - Path to the data represented in the
                                         generated playlist.
                      .url_prefix      - Prefix of the server's URL hosting the
                                         data in .data_root.
                      .component_count - Number of leading path components to strip
                                         from .data_root when creating URLs with
                                         .url_prefix.  Must be non-negative.

                  NOTE: Will be None if execution is not required.

    """

    # indices into argv's positional arguments specifying each of the required
    # arguments.
    ARG_PLAYLIST_PATH   = 0
    ARG_EXPERIMENT_NAME = 1
    ARG_VARIABLES_LIST  = 2
    ARG_TIME_RANGE      = 3
    ARG_XY_SLICE_RANGE  = 4
    ARG_DATA_ROOT       = 5
    ARG_URL_PREFIX      = 6
    ARG_COMPONENT_COUNT = 7
    NUMBER_ARGUMENTS    = ARG_COMPONENT_COUNT + 1

    # empty class designed to hold name values.
    #
    # XXX: this is a kludge until we sort out how to replace everything with
    #      argparse.
    #
    class _Arguments( object ):
        pass

    options, arguments = _Arguments(), _Arguments()

    # abort if we are creating a Scalabel frame that references non-existent
    # data.
    options.force_flag = False

    # create Scalabel frames without labels.
    options.iwp_labels_path = None

    # create playlists without any particular ordering of the frames by default.
    options.labeling_strategy = iwp.scalabel.LabelingStrategyType.NO_ORDER

    # parse our command line options.
    try:
        option_flags, positional_arguments = getopt.getopt( argv[1:], "fhl:L:" )
    except getopt.GetoptError as error:
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ) )

    valid_labeling_strategies = {
        iwp.scalabel.LabelingStrategyType.NO_ORDER.name:  iwp.scalabel.LabelingStrategyType.NO_ORDER,
        iwp.scalabel.LabelingStrategyType.XY_SLICES.name: iwp.scalabel.LabelingStrategyType.XY_SLICES,
        iwp.scalabel.LabelingStrategyType.Z_STACKS.name:  iwp.scalabel.LabelingStrategyType.Z_STACKS,
        iwp.scalabel.LabelingStrategyType.VARIABLES.name: iwp.scalabel.LabelingStrategyType.VARIABLES
        }

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-f":
            options.force_flag = True
        elif option == "-h":
            print_usage( argv[0] )
            return (None, None)
        elif option == "-l":
            options.iwp_labels_path = option_value
        elif option == "-L":
            if option_value.upper() in valid_labeling_strategies:
                options.labeling_strategy = valid_labeling_strategies[option_value.upper()]
            else:
                raise ValueError( "Unknown labeling strategy '{:s}'.  Must be one of: {:s}.".format(
                    option_value,
                    ", ".join( map( lambda x: x.name.lower() ) ) ) )

    # ensure we have the correct number of arguments.
    if len( positional_arguments ) != NUMBER_ARGUMENTS:
        raise ValueError( "Incorrect number of arguments.  Expected {:d}, received {:d}.".format(
            NUMBER_ARGUMENTS,
            len( positional_arguments ) ) )

    # map the positional arguments to named variables.
    arguments.playlist_path   = positional_arguments[ARG_PLAYLIST_PATH]
    arguments.experiment_name = positional_arguments[ARG_EXPERIMENT_NAME]
    arguments.variable_names  = positional_arguments[ARG_VARIABLES_LIST].split( "," )
    arguments.time_range      = iwp.utilities.parse_range( positional_arguments[ARG_TIME_RANGE] )
    arguments.xy_slice_range  = iwp.utilities.parse_range( positional_arguments[ARG_XY_SLICE_RANGE] )
    arguments.data_root       = positional_arguments[ARG_DATA_ROOT]
    arguments.url_prefix      = positional_arguments[ARG_URL_PREFIX]
    arguments.component_count = positional_arguments[ARG_COMPONENT_COUNT]

    # validate the ranges supplied are sensible
    if arguments.time_range is None:
        raise ValueError( "Failed to parse a time range from \"{:s}\"".format(
            positional_arguments[ARG_TIME_RANGE] ) )
    if arguments.xy_slice_range is None:
        raise ValueError( "Failed to parse a xy slice range from \"{:s}\"".format(
            positional_arguments[ARG_XY_SLICE_RANGE] ) )

    # ensure we have a non-negative component count.
    try:
        arguments.component_count = int( arguments.component_count )
    except ValueError:
        raise ValueError( "Could not parse an integral component count from '{:s}'.".format(
            positional_arguments[ARG_COMPONENT_COUNT] ) )

    if arguments.component_count < 0:
        raise ValueError( "Component count must be non-negative ({:d}).".format(
            arguments.component_count ) )

    return options, arguments

def main( argv ):
    """
    Parses the supplied command line arguments and builds a Scalabel playlist for a
    subset of an IWP dataset.

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

    # generate playlist's frames based on the parameters.
    try:
        scalabel_frames = iwp.scalabel.build_scalabel_frames( arguments.experiment_name,
                                                              arguments.variable_names,
                                                              arguments.time_range,
                                                              arguments.xy_slice_range,
                                                              arguments.data_root,
                                                              ".png",
                                                              arguments.url_prefix,
                                                              arguments.component_count,
                                                              labeling_strategy=options.labeling_strategy,
                                                              check_data_flag=options.force_flag )
    except FileNotFoundError as e:
        print( "Could not build Scalabel playlist - {:s}".format(
            str( e ) ),
               file=sys.stderr )

        return 1

    # merge the IWP labels into the frames.  this is a no-op if we were not
    # supplied labels by the caller.
    scalabel_frames = iwp.scalabel.set_iwp_labels( scalabel_frames,
                                                   options.iwp_labels_path )

    # serialize playlist.
    try:
        with open( arguments.playlist_path, "w" ) as playlist_path:
            json.dump( scalabel_frames, playlist_path, indent=4 )
    except Exception as e:
        print( "Failed to write the Scalabel playlist to '{:s}' ({:s}).".format(
            arguments.playlist_path,
            str( e ) ),
               file=sys.stderr )

        return 1

    return 0

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
