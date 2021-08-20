#!/bin/sh

# wrapper script around ffmpeg to create XY slice animations.  intended to hide
# the complexities of ffmpeg from the end user and make it trivial to create
# animations that either scan through time or Z for further review and analysis.
# care is taken to validate the inputs so that common errors (e.g. missing paths)
# do not result in cryptic messages from ffmpeg.
#
# currently this creates H.264 encoded movies (.MP4) with a fixed quality factor
# and pixel encoding (YUV 420p) to maximize compatibility with movie players.
# these could be exposed as knobs, though the defaults are intended to be
# suitable for almost all use cases.  IWP simulations do not create incredibly
# long outputs (Nt ~ O(100) or Z ~ O(1000)) which means file sizes do not
# necessitate high compression or fancy codecs, and the features simulated
# typically do not have large frame-to-frame dynamics necessitating lossless
# encodings.  exposing options for movie encoding is straight forward if it
# becomes necessary.
#
# NOTE: one should be careful with changing the output frame rate as small
#       frame rates (< 10fps) causes odd behavior that has not been fully
#       understood and dealt with.  in particular, low frame rates cause some
#       movie players to have problems (e.g. VLC with ~5 fps) while others
#       work but do not allow frame accurate step throughs (e.g. Quicktime
#       Player with ~10 fps).
#
#       support for low frame rates can be implemented, though will likely
#       need to support the animation frame rate (-framerate N) separately
#       from the video frame rate (-r M) so that videos can advance at M fps
#       while the animation advances at a slower N fps.  care will have to
#       be taken in specifying the number of frames (-frames:v P) as this
#       is set in units of the video frame rate (M) but also governs which
#       images are read from disk.
#
#       have fun.
#

print_usage()
{
    echo "${0} [-f] [-F <ffmpeg>] [-h] [-r <fps>] [-t] [-v] <output file> <images root> <experiment> <variable> <z range> <Nt range>"
    echo
    echo "Wrapper around ffmpeg to create animations from a series of XY slice images"
    echo "on disk.  Animations can either be fixing on a single Z height and varying"
    echo "time steps, or fixing on a single time step and varying the Z height, and"
    echo "are specified by <z range> and <Nt range>, respectively.  Parameters are"
    echo "validated to avoid simple mistakes and the ensuing debugging efforts."
    echo
    echo "Creates an H.264 encoded movie as MP4s at <output file> for maximum compatibility"
    echo "with video players."
    echo
    echo "Source images for the animation are located beneath <images root> with paths to"
    echo "individual images having the form of:"
    echo
    echo "    <images root>/<experiment>-<variable>-z=<Z>-Nt=<T>.png"
    echo
    echo "The above path structure matches that found in iwp_create_labeling_data.py to"
    echo "make it trivial to create XY slice animations for a given IWP dataset."
    echo
    echo "One (and only one) of <z range> and <Nt range> must be of the format:"
    echo
    echo "    <start>:<stop>"
    echo
    echo "while the other is simply an index.  A fixed time, varied Z height animation"
    echo "is created when <z range> is of the form <start>:<stop>.  A fixed Z height,"
    echo "varied time steps animation is created when <Nt range> is of the form"
    echo "<start>:<stop>.  Ranges must be non-decreasing, such that <start> >= <stop>."
    echo
    echo "The command line options shown above are described below:"
    echo
    echo "    -f                       Force overwriting <output file> if it exists."
    echo "                             By default the script will stop if <output file>"
    echo "                             exists to prevent accidental data loss."
    echo "    -F <ffmpeg>              Use the FFmpeg binary at <ffmpeg> instead of the"
    echo "                             one in the path.  If omitted, defaults to"
    echo "                             \"${FFMPEG}\"."
    echo "    -h                       Display this help message and exit."
    echo "    -r <fps>                 Render the animation at <fps> frames per second."
    echo "                             May be fractional.  If omitted, defaults to"
    echo "                             ${ANIMATION_FRAME_RATE} fps."
    echo "    -t                       Enable testing mode.  If specified, the ffmpeg"
    echo "                             command to create the animation will not be"
    echo "                             executed but rather printed to standard output"
    echo "                             instead."
    echo "    -v                       Increases verbosity during execution and may be"
    echo "                             specified multiple times.  Specify once to see"
    echo "                             script decisions, twice to see FFmpeg diagnostics"
    echo "                             during encoding."
    echo
}

# predicate that determine if the supplied argument is a colon-delimited range.
# returns "yes" if it is a range, "no" otherwise.
#
# NOTE: this is nowhere near a comprehensive check.  no validation is done on
#       the start or the stop values (start before stop, integral values, etc).
#
is_range()
{
    LOCAL_RANGE="$1"

    LOCAL_COMPONENT_COUNT=`echo ${LOCAL_RANGE} | tr ":" " " | wc -w`

    if [ "${LOCAL_COMPONENT_COUNT}" -eq 2 ]; then
        echo "yes"
    else
        echo "no"
    fi
}

# parses a colon-delimited range and outputs "<start> <stop>".  if it encounters
# a set of non-integral start/stop values, outputs an empty string.
parse_range()
{
    LOCAL_RANGE="$1"

    if [ -z "${LOCAL_RANGE}" ]; then
        echo ""
        return
    fi

    # break up a colon-delimited range into start and stop.
    LOCAL_RANGE_COMPONENTS=`echo ${LOCAL_RANGE} | tr ":" " "`

    LOCAL_RANGE_START=`echo ${LOCAL_RANGE_COMPONENTS} | cut -d " " -f 1`
    LOCAL_RANGE_STOP=`echo ${LOCAL_RANGE_COMPONENTS} | cut -d " " -f 2`

    # filter out non-integer starts and stop.
    #
    # NOTE: there is likely a better way to do this.  the first printf will
    #       always output a value, though its exit code will indicate whether
    #       it successfully instantiated its template.  if so, we execute it
    #       again and capture the output.
    #
    LOCAL_RANGE_START=`printf %d ${LOCAL_RANGE_START} >/dev/null 2>&1 && printf "%03d" ${LOCAL_RANGE_START} 2>/dev/null`
    LOCAL_RANGE_STOP=`printf %d ${LOCAL_RANGE_STOP} >/dev/null 2>&1 && printf "%03d" ${LOCAL_RANGE_STOP} 2>/dev/null`

    if [ -n "${LOCAL_RANGE_START}" -a -n "${LOCAL_RANGE_STOP}" ]; then
        echo "${LOCAL_RANGE_START} ${LOCAL_RANGE_STOP}"
    else
        echo ""
    fi
}

# takes a path template along with a range of indices and verifies that each
# path in that range exists.  echoes the paths missing to standard output.
all_range_paths_exist()
{
    LOCAL_PATH_TEMPLATE="$1"
    LOCAL_START_INDEX="$2"
    LOCAL_STOP_INDEX="$3"

    LOCAL_MISSING_FILES=

    for LOCAL_INDEX in `seq ${LOCAL_START_INDEX} ${LOCAL_STOP_INDEX}`; do
        LOCAL_PATH=`printf ${LOCAL_PATH_TEMPLATE} ${LOCAL_INDEX}`

        if [ ! -f "${LOCAL_PATH}" ]; then
            LOCAL_MISSING_FILES="${LOCAL_MISSING_FILES} ${LOCAL_PATH}"
        fi
    done

    echo "${LOCAL_MISSING_FILES}"
}

# flag specifying whether we overwrite an on disk movie.  by default we protect
# existing files.
FORCE_FLAG="no"

# flag specifying whether to print out commands that would be executed rather
# than executing them.  by default, we process the inputs.
TESTING_FLAG="no"

# flag specifying whether status information is printed to standard output
# during execution.  by default, only error messages are printed to standard
# error.
VERBOSE_FLAG="no"
VERBOSITY_LEVEL=0

# keep ffmpeg's verbosity down by default.  a log level of 24 only shows
# warnings and above.
FFMPEG_LOGLEVEL=24

# frames per second for the created movie.  may be fractional.
#
# NOTE: be careful with small frame rates.  not all players are capable of
#       supporting arbitrary rates.
#
ANIMATION_FRAME_RATE=10

# default to the system FFmpeg which is likely better than what is in the PATH
# when run in a Conda environment.
FFMPEG=`which /usr/bin/ffmpeg 2>/dev/null`

while getopts "fF:hr:tv" OPTION;
do
    case ${OPTION} in
        f)
            FORCE_FLAG="yes"
            ;;
        F)
            FFMPEG=`which ${OPTARG} 2>/dev/null`
            ;;
        h)
            print_usage
            exit 0
            ;;
        r)
            ANIMATION_FRAME_RATE=${OPTARG}
            ;;
        t)
            TESTING_FLAG="yes"
            ;;
        v)
            #VERBOSE_FLAG="yes"
            VERBOSITY_LEVEL=`expr ${VERBOSITY_LEVEL} + 1`
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
done
shift `expr ${OPTIND} - 1`

# ensure we received the correct number of arguments.
if [ $# -ne 6 ]; then
    echo "Expected 6 arguments but received $#!" >&2
    exit 1
fi

# map arguments to variable names.
OUTPUT_MOVIE_PATH=$1
INPUT_IMAGES_ROOT=$2
EXPERIMENT_NAME=$3
VARIABLE_NAME=$4
Z_INDEX_RANGE=$5
TIME_STEPS_RANGE=$6

# ensure we have the tools needed to execute.
if [ -z "${FFMPEG}" ]; then
    echo "ffmpeg is not in the PATH!" >&2

    exit 1
fi

# prevent overwriting an existing movie unless we're told to.
if [ -f "${OUTPUT_MOVIE_PATH}" -a "${FORCE_FLAG}" != "yes" ]; then
    echo "'${OUTPUT_MOVIE_PATH}' exists and we're not overwriting files.  Exiting." >&2

    exit 1
# ensure that the source image directory exists.
elif [ ! -d "${INPUT_IMAGES_ROOT}" ]; then
    echo "The input images directory, '${INPUT_IMAGES_ROOT}', does not exist.  Exiting." >&2

    exit 1
fi

# ensure we got non-empty path components.
if [ -z "${EXPERIMENT_NAME}" ]; then
    echo "The experiment name is empty.  Exiting." >&2

    exit 1
elif [ -z "${VARIABLE_NAME}" ]; then
    echo "The variable name is empty.  Exiting." >&2

    exit 1
fi

# ensure we got non-empty ranges.
#
# NOTE: this is needed to avoid corner cases in parsing below.
#
if [ -z "${Z_INDEX_RANGE}" ]; then
    echo "The Z index (range) is empty.  Exiting." >&2

    exit 1
elif [ -z "${TIME_STEPS_RANGE}" ]; then
    echo "The time step (range) is empty.  Exiting." >&2

    exit 1
fi

# determine which of the axes we got a range for.  we can only vary one of them
# at a time.
IS_Z_RANGE=`is_range ${Z_INDEX_RANGE}`
IS_TIME_RANGE=`is_range ${TIME_STEPS_RANGE}`

if [ \( "${IS_Z_RANGE}" = "no" -a "${IS_TIME_RANGE}" = "no" \) -o \
     \( "${IS_Z_RANGE}" = "yes" -a "${IS_TIME_RANGE}" = "yes" \) ]; then
    if [ "${IS_Z_RANGE}" = "yes" ]; then
        WHICH_RANGE="both"
    else
        WHICH_RANGE="neither"
    fi

    echo "Must have either a Z range or a time range.  Received ${WHICH_RANGE} (Z=${Z_INDEX_RANGE}, Nt=${TIME_STEPS_RANGE})." >&2

    exit 1
fi

# build the pattern our image path names have.  regardless of which axes we're
# varying, the path starts the same.
IMAGE_PATH_PATTERN_ROOT="${INPUT_IMAGES_ROOT}/${VARIABLE_NAME}/${EXPERIMENT_NAME}-${VARIABLE_NAME}"

# are we fixing time and varying Z?
if [ "${IS_Z_RANGE}" = "yes" ]; then
    TARGET_RANGE=`parse_range ${Z_INDEX_RANGE}`
    TIME_STEP_INDEX=`printf %03d ${TIME_STEPS_RANGE} 2>/dev/null`
    WHICH_RANGE="Z"

    IMAGE_PATH_PATTERN="${IMAGE_PATH_PATTERN_ROOT}-z=%03d-Nt=${TIME_STEP_INDEX}.png"
# or fixing Z and varying time?
else
    TARGET_RANGE=`parse_range ${TIME_STEPS_RANGE}`
    Z_INDEX=`printf %03d ${Z_INDEX_RANGE} 2>/dev/null`
    WHICH_RANGE="time"

    IMAGE_PATH_PATTERN="${IMAGE_PATH_PATTERN_ROOT}-z=${Z_INDEX}-Nt=%03d.png"
fi

# compute the number of images in the range.  ffmpeg expects a start offset
# and a frame count.
START_INDEX=`echo ${TARGET_RANGE} | cut -d " " -f 1`
STOP_INDEX=`echo ${TARGET_RANGE} | cut -d " " -f 2`

IMAGE_COUNT=`expr ${STOP_INDEX} - ${START_INDEX} + 1`

if [ "${IMAGE_COUNT}" -le 0 ]; then
    if [ "${WHICH_RANGE}" = "Z" ]; then
        ORIGINAL_RANGE="${Z_INDEX_RANGE}"
    else
        ORIGINAL_RANGE="${TIME_STEPS_RANGE}"
    fi
    echo "Invalid ${WHICH_RANGE} range specified (${ORIGINAL_RANGE}).  Cannot have zero length, or descending ranges." >&2

    exit 1
fi

# check to see if all of the files exist on disk before handing them to ffmpeg.
MISSING_PATHS=`all_range_paths_exist ${IMAGE_PATH_PATTERN} ${START_INDEX} ${STOP_INDEX}`

if [ -n "${MISSING_PATHS}" ]; then
    NUMBER_MISSING=`echo ${MISSING_PATHS} | wc -w`

    echo "The image path pattern '${IMAGE_PATH_PATTERN}' is missing ${NUMBER_MISSING} files:" >&2
    echo >&2
    echo -n "   "  >&2
    echo "${MISSING_PATHS}" | sed -e 's/ /\n    /g'  >&2
    echo >&2

    exit 1
fi

# we have validated everything and have all of the files requested.  tell
# the user what we're about to do.
if [ ${VERBOSITY_LEVEL} -gt 0 ]; then
    if [ "${WHICH_RANGE}" = "Z" ]; then
        echo "Creating an animation while fixing time and varying Z height."
    else
        echo "Creating an animation while fixing Z height and varying time."
    fi

    echo "Parameters used during encoding:"
    echo
    echo "    FFmpeg:                 ${FFMPEG}"
    echo "    Frame path template:    ${IMAGE_PATH_PATTERN}"
    echo "    Frame count:            ${IMAGE_COUNT} (${START_INDEX}-${STOP_INDEX})"
    echo "    Frame rate:             ${ANIMATION_FRAME_RATE}"
    echo
fi

# increase the verbosity to informational if requested.
if [ ${VERBOSITY_LEVEL} -gt 1 ]; then
    FFMPEG_LOGLEVEL=32
fi

# explanation of the arcane incantation coaxing ffmpeg to do what we want:
#
#  * (-y) always force overwrite of the output file.  we already check for
#    existence of the output and ensure we're forcing things above.
#
#  * (-loglevel) set the log level to keep ffmpeg from being chattier than
#    desired.
#
#  * (-start_number, -frames:v) specify the range of frames to encode.
#    controls the sequence of integers instantiated with the input path
#    template.
#
#  * (-i) specifies the input path template instantiated to get input
#    frames.
#
#  * (-f mp4, -vcodec libx264) create H.264-encoded movies.  this increases
#    the likelihood that videos can be viewed regardless of the target
#    OS/hardware *without* requiring additional codecs to be installed.
#    we don't anticipate hours of animations, so we do not need more
#    space efficient encoders (e.g. H.256) that aren't as widely supported
#    out of the box.
#
#  * (-pix_fmt yuv420p) specify a coarse (the coarsest?) yuv subsampling
#    to ensure maximum playback compatibility.  without this, some players
#    (*cough* Quicktime Player *cough*) refuse to play the files output.
#
#  * (-crf) specify a constant rate factor.  this is a quality factor for
#    H.264 in a range of [0, 51], lower being better.  a value of 18 usually
#    results in outputs virtually indistinguishable from the input.
#
#      NOTE: this option is not recognized unless ffmpeg has been built with
#            the --enable-libx264 flag.  it should be noted that ffmpeg
#            installed from the pytorch Conda channel does *not* support this.
#
# NOTE: the order of inputs very much matters when working with ffmpeg, as its
#       command line parameters specify a processing pipeline language.
#
MOVIE_BUILDER_COMMAND="${FFMPEG} \
                                 -y \
                                 -loglevel ${FFMPEG_LOGLEVEL} \
                                 -start_number ${START_INDEX} \
                                 -i \"${IMAGE_PATH_PATTERN}\" \
                                 -r ${ANIMATION_FRAME_RATE} \
                                 -frames:v ${IMAGE_COUNT} \
                                 -f mp4 \
                                 -vcodec libx264 \
                                 -pix_fmt yuv420p \
                                 -crf 18 \
                                 \"${OUTPUT_MOVIE_PATH}\""

# do the thing.
if [ "${TESTING_FLAG}" = "yes" ]; then
    echo ${MOVIE_BUILDER_COMMAND}
else
    eval ${MOVIE_BUILDER_COMMAND}
fi
