#!/bin/sh

# script to prepare IWP datasets for analysis and use as ML inputs.  simulation
# outputs may have chunk sizes that aren't conducive to XY slice-based access
# as well as leaving the coordinate systems in meters instead of in unitless
# dimensions (so to be consistent with literature).  this provides a wrapper
# around the NCO utilities, primarily ncap2, which maps common transforms into
# the underlying command line options while guarding against user error.
#
# functionality remaining:
#
#    1. remove duplication XY slices by Z coordinate
#
#       some solvers partition in Z which results in duplicate Z coordinates.
#       these need to be removed so the Z coordinates are unique, and can be
#       done by a) removing one of the XY slices, or b) averaging them together.

print_usage()
{
    echo "${0} [-c] [-D <diameter>] [-f] [-h] [-t] [-v] [-V <var1>[,<var2>[...]]] <output path> <input path>[ <input path>[...]]"
    echo
    echo "Wrapper script around the netCDF Operators (NCO) suite to prepare IWP datasets"
    echo "for use in analysis and machine learning activities.  One or more files, specified"
    echo "by <input path>, are processed into <output path> such that one or more of the"
    echo "following operations is performed:"
    echo
    echo "    1. Chunk variables by XY slice for slice-by-slice access"
    echo "    2. Scale the (x, y, z) coordinate system by the tow-body diameter (D)"
    echo "    3. Filter out unnecessary variables by name"
    echo
    echo "At least one of the above operations must be performed, otherwise the script"
    echo "exits with an error.  Care is taken to validate the inputs so that problems"
    echo "are reported as human-readable messages rather than cryptic failures in the"
    echo "underlying NCO tools."
    echo
    echo "This script is written such that non-fatal errors are flagged on standard error"
    echo "and it moves onto the next input file to process.  It is assumed that processing"
    echo "some files is better than stopping dead in its tracks, particularly for batch"
    echo "operations."
    echo
    echo "The command line options shown above are described below:"
    echo
    echo "    -c                       Output files are re-chunked so that variables are"
    echo "                             stored with multiple XY slices (i.e Z=1, X and Y"
    echo "                             set to the domain extent).  If omitted, variable"
    echo "                             chunk sizes are copied from input."
    echo "    -D <diameter>            Coordinate system scale factor.  If provided, each"
    echo "                             of the (X, Y, Z) coordinates are divided by <diameter>"
    echo "                             to produce a dimension-less coordinate system."
    echo "                             If omitted, the input coordinate systems are copied"
    echo "                             without modification."
    echo "    -f                       Flag forcing overwriting output files beneath"
    echo "                             <output path>.  If omitted, input files that"
    echo "                             would overwrite a file are flagged and skipped."
    echo "    -h                       Display this help message and exit."
    echo "    -t                       Enable testing mode.  If specified, the ncap2"
    echo "                             command to process <input path>s will not be"
    echo "                             executed but rather printed to standard output"
    echo "                             instead."
    echo "    -v                       Enable verbose execution.  If specified, each"
    echo "                             <input path> is printed to standard output when"
    echo "                             processed."
    echo "    -V <var1>[,<var2>[...]]  Specifies a comma-delimited list of variable names"
    echo "                             to copy from each <input path> during operations."
    echo "                             If omitted, or specified as an empty string, all"
    echo "                             variables are copied.  Each variable requested"
    echo "                             must exist, otherwise the file missing it is"
    echo "                             skipped."
    echo
}

# processes a netCDF file based on the caller's parameters.  translates
# chunking, scaling, and variable filtering into ncap2 command line options.
# executes or prints the resulting ncap2 command depending on the flags
# specified.
process_netcdf_file()
{
    LOCAL_INPUT_FILE=$1
    LOCAL_OUTPUT_FILE=$2
    LOCAL_VERBOSE_FLAG=$3
    LOCAL_TESTING_FLAG=$4
    LOCAL_FORCE_FLAG=$5
    LOCAL_RECHUNK_FLAG=$6
    LOCAL_DIAMETER=$7
    LOCAL_VARIABLE_NAMES_LIST=$8

    # query the file to get its coordinate system.  extract each of the
    # coordinates separately.
    LOCAL_COORDINATES=`${NCDUMP} -sh "${LOCAL_INPUT_FILE}" | grep -A3 dimensions`
    LOCAL_X_COORDINATE=`echo ${LOCAL_COORDINATES} | sed -e 's/.*x *= *\([^ ]*\).*/\1/'`
    LOCAL_Y_COORDINATE=`echo ${LOCAL_COORDINATES} | sed -e 's/.*y *= *\([^ ]*\).*/\1/'`
    LOCAL_Z_COORDINATE=`echo ${LOCAL_COORDINATES} | sed -e 's/.*z *= *\([^ ]*\).*/\1/'`

    # query the file to get its variables.
    LOCAL_AVAILABLE_VARIABLE_NAMES=`${NCINFO} "${LOCAL_INPUT_FILE}" |
                                    grep variables |
                                    sed -e 's/.*variables(dimensions): //' \
                                        -e 's#([^)]*)##g' \
                                        -e 's/, /\n/g' |
                                    cut -f2 -d ' '`

    # make sure we have a 3D simulation domain using (x, y, z).
    if [ -z "${LOCAL_X_COORDINATE}" -o \
         -z "${LOCAL_Y_COORDINATE}" -o \
         -z "${LOCAL_Z_COORDINATE}" ]; then
        echo "${LOCAL_INPUT_FILE} does not have a 3D coordinate system with 'x', 'y', 'z' axes!  Skipping." >&2

        return
    fi

    # announce this file if requested.
    if [ "${LOCAL_VERBOSE_FLAG}" = "yes" ]; then
       echo "Processing '${LOCAL_INPUT_FILE}' (${LOCAL_X_COORDINATE}x${LOCAL_Y_COORDINATE}x${LOCAL_Z_COORDINATE})"
    fi

    # we build up ncap2's options based on the callers parameters.
    LOCAL_NCAP_OPTIONS=""

    # convey whether we're overwriting an existing output file.
    if [ "${LOCAL_FORCE_FLAG}" = "yes" ]; then
        LOCAL_NCAP_OPTIONS="${LOCAL_NCAP_OPTIONS} -O"
    fi

    # set each chunk to a single XY slice.
    if [ "${LOCAL_RECHUNK_FLAG}" = "yes" ]; then
        LOCAL_NCAP_OPTIONS="${LOCAL_NCAP_OPTIONS} --cnk_dmn=z,1 --cnk_dmn=x,${LOCAL_X_COORDINATE} --cnk_dmn=y,${LOCAL_Y_COORDINATE}"
    fi

    # scale the coordinates so they're dimensionless.
    if [ "${LOCAL_DIAMETER}" != "0" ]; then
        LOCAL_NCAP_OPTIONS="${LOCAL_NCAP_OPTIONS} -s 'x=float(x/${LOCAL_DIAMETER});y=float(y/${LOCAL_DIAMETER});z=float(z/${LOCAL_DIAMETER});'"
    fi

    # tell ncap2 to only copy variables that are referenced in the algebra
    # scripts.  we verify each variable requested existed in the input and then
    # setup a dummy "x = x * 1" script for each.
    if [ -n "${LOCAL_VARIABLE_NAMES_LIST}" ]; then
        LOCAL_NCAP_OPTIONS="${LOCAL_NCAP_OPTIONS} -v"

        # add the coordinates into the variable list to ensure they're copied
        # over.
        LOCAL_VARIABLE_NAMES_LIST="${LOCAL_VARIABLE_NAMES_LIST} x y z"
    fi

    for LOCAL_VARIABLE_NAME in `echo ${LOCAL_VARIABLE_NAMES_LIST} | tr , ' '`; do

        # verify that each variable requested exists in the file.  if not we can
        # bail with a sensible error message rather than having ncap2 complain
        # along with a wall of its help message.
        LOCAL_FOUND_NAME_FLAG="no"
        for LOCAL_AVAILABLE_VARIABLE_NAME in ${LOCAL_AVAILABLE_VARIABLE_NAMES}; do
            if [ ${LOCAL_VARIABLE_NAME} = ${LOCAL_AVAILABLE_VARIABLE_NAME} ]; then
                LOCAL_FOUND_NAME_FLAG="yes"
                break;
            fi
        done

        if [ ${LOCAL_FOUND_NAME_FLAG} = "no" ]; then
            echo "'${LOCAL_VARIABLE_NAME}' does not exist in '${LOCAL_INPUT_FILE}'.  Skipping this file." >&2

            return
        fi

        # "copy" this variable by scaling it by one.
        LOCAL_NCAP_OPTIONS="${LOCAL_NCAP_OPTIONS} -s '${LOCAL_VARIABLE_NAME}=${LOCAL_VARIABLE_NAME}*1;'"
    done

    LOCAL_NCAP_COMMAND="${NCAP2} ${LOCAL_NCAP_OPTIONS} ${LOCAL_INPUT_FILE} ${LOCAL_OUTPUT_FILE}"

    # spawn a shell to run the command.
    #
    # NOTE: we have to do this because ncap2 is sensitive to the quotes we've
    #       constructed above and I haven't had time (or the desire) to figure
    #       out why.
    #
    if [ "${LOCAL_TESTING_FLAG}" = "yes" ]; then
        echo sh -c "${LOCAL_NCAP_COMMAND}"
    else
        sh -c "${LOCAL_NCAP_COMMAND}"
    fi
}

# flag specifying whether the output should be chunked by XY slices.  by
# default we leave the chunk size as is.
RECHUNK_FLAG="no"

# flag specifying whether we overwrite output files on disk or skip them.
# by default, we skip existing files.
FORCE_FLAG="no"

# flag specifying whether to print out commands that would be executed rather
# than executing them.  by default, we process the inputs.
TESTING_FLAG="no"

# flag specifying whether status information is printed to standard output
# during execution.  by default, only error messages are printed to standard
# error.
VERBOSE_FLAG="no"

# comma-delimited list of variable names to copy into output files.  if
# empty, all variables from the input are copied to the output.
VARIABLE_NAMES_LIST=

# tow body diameter.  this is a positive value, in meters, that the coordinates
# are scaled by to produce dimension-less units (suitable for comparison against
# literature).
#
# NOTE: DIAMETER is set in conjunction with DIAMETER_PRECISE to enable value
#       validation.  DIAMETER_PRECISE is DIAMETER filtered through 'printf %g'
#       so that we can distinguish zero from small values and sanity check
#       that we can create a scaled coordinate system.
#
#       this is a hot mess.  hopefully it is correct and doesn't require someone
#       to deal with it.
#
DIAMETER=
DIAMETER_PRECISE=0

# paths to the netCDF tools needed.
#
# NOTE: we verify these exist after we've parsed command line options and
#       arguments, so that we can always query for help regardless of being
#       on a properly-configured system.
#
NCDUMP=`which ncdump 2>/dev/null`
NCAP2=`which ncap2 2>/dev/null`
NCINFO=`which ncinfo 2>/dev/null`

while getopts "cD:fhtvV:" OPTION;
do
    case ${OPTION} in
        c)
            RECHUNK_FLAG="yes"
            ;;
        D)
            DIAMETER=${OPTARG}

            # get an alternate representation of the diameter so we can check
            # that it is non-zero later.
            #
            # NOTE: this fails safely as empty and non-digit inputs will
            #       produce a 0.
            #
            DIAMETER_PRECISE=`printf %g ${DIAMETER} 2>/dev/null`
            ;;
        f)
            FORCE_FLAG="yes"
            ;;
        h)
            print_usage
            exit 0
            ;;
        t)
            TESTING_FLAG="yes"
            ;;
        v)
            VERBOSE_FLAG="yes"
            ;;
        V)
            VARIABLE_NAMES_LIST=${OPTARG}
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
done
shift `expr ${OPTIND} - 1`

# ensure we received the correct number of arguments.
if [ $# -lt 2 ]; then
    echo "Expected at least 2 arguments but received $#!" >&2
    exit 1
fi

# map arguments to variable names.
OUTPUT_DIRECTORY=$1
shift 1
INPUT_FILE_LIST="$*"

# ensure we have the tools needed to execute.
if [ -z "${NCAP2}" ]; then
    echo "ncap2 is not in the PATH!" >&2

    exit 1
elif [ -z "${NCDUMP}" ]; then
    echo "ncdump is not in the PATH!" >&2

    exit 1
elif [ -z "${NCINFO}" ]; then
    echo "ncinfo is not in the PATH!" >&2

    exit 1
fi

# ensure that our scale factor is valid.  we divide the solver output by
# the diameter, so we avoid a division by zero.
#
# NOTE: DIAMETER_PRECISE is set whenever DIAMETER is, so if the precise
#       version is still zero then we received a zero diameter.  hooray
#       for validating floating point numbers...
#
if [ -n "${DIAMETER}" -a \
     \( "${DIAMETER_PRECISE}" = "0" -o \
        "${DIAMETER_PRECISE}" = "-0" \) ]; then
    echo "Cannot scale coordinates with a zero diameter! (${DIAMETER})" >&2

    exit 1
# it doesn't make sense to scale the coordinates by a negative diameter, so
# prevent the caller from doing something dumb.
elif [ -n "${DIAMETER}" ]; then
    if [ `echo "${DIAMETER} < 0" | bc` = "1" ]; then
        echo "Cannot scale coordinates by a negative diameter!" >&2

        exit 1
    fi
fi

# ensure that we're doing something.  we could be a very expensive no-op
# though that isn't a use case that needs to be supported right now.
if [ "${RECHUNK_FLAG}" = "no" -a \
     \( "${DIAMETER_PRECISE}" = "0" -o \
        "${DIAMETER_PRECISE}" = "-0" \) -a \
     -z "${VARIABLE_NAMES_LIST}" ]; then
    echo "Chunking, coordinate scaling, and variable filtering were not requested.  Refusing" >&2
    echo "to copy input files." >&2

    exit 1
fi

# set our diameter to zero when we haven't been provided one.  the logic
# above prevents the precise diameter from being zero if we were given
# something on the command line.
if [ "${DIAMETER_PRECISE}" = "0" ]; then
    DIAMETER="0"
fi

# ensure we have a directory to create our output files.
mkdir -p ${OUTPUT_DIRECTORY}
if [ $? -ne 0 ]; then
    echo "Failed to create the output directory ('${OUTPUT_DIRECTORY}')." 2>&1

    exit 1
fi

for INPUT_FILE in ${INPUT_FILE_LIST}; do
    # the netCDF tool suite isn't great about error messages, so warn the user
    # if the provided a non-existent path.
    if [ ! -f "${INPUT_FILE}" ]; then
        "'${INPUT_FILE}' does not exist!  Skipping." >&2

        continue
    fi

    OUTPUT_FILE="${OUTPUT_DIRECTORY}/`basename ${INPUT_FILE}`"

    # let the caller directly know they should have forced creation rather than
    # interpreting the failure message from ncap2.
    if [ -f "${OUTPUT_FILE}" -a "${FORCE_FLAG}" != "yes" ]; then
        echo "'${OUTPUT_FILE}' exists and overwriting was not requested!  Skipping." >&2

        continue
    fi

    # do the work.
    process_netcdf_file ${INPUT_FILE} \
                        ${OUTPUT_FILE} \
                        ${VERBOSE_FLAG} \
                        ${TESTING_FLAG} \
                        ${FORCE_FLAG} \
                        ${RECHUNK_FLAG} \
                        ${DIAMETER} \
                        "${VARIABLE_NAMES_LIST}"

done
