#!/bin/sh

# wrapper around Scalabel.ai's image and the container run-time to simplify the
# user's interface.  this hides things like volume mounting syntax and port
# publishing so users (typically) only need to understand where their data live
# to get up and labeling.

print_usage()
{
    echo "${0} [-c <config_path>] [-h] [-i <image>] [-l <launcher>] [-p <port>] [-q] [-t] <items_path> <projects_path>"
    echo
    echo "Script to launch Scalabel.ai's labeling container with a simplified interface."
    echo
    echo "Maps <items_path> and <projects_path> into the launched container to"
    echo "${SCALABEL_CONTAINER_ITEMS_ROOT} and ${SCALABEL_CONTAINER_PROJECT_ROOT}, respectively."
    echo "Supplied paths are validated to avoid issues where the container is launched"
    echo "but is unable to serve data to users.  Configuration parameters important"
    echo "to the user is printed to standard output before launching the Scalable.ai"
    echo "application to allow easy review."
    echo
    echo "Control over the interface and port exposed for the application, the container image"
    echo "launched, and which container run-time to use are available through command line"
    echo "options."
    echo
    echo "The command line options shown above are described below:"
    echo
    echo "    -c <config_path>  Relative path to the Scalabel.ai configuration file, typically"
    echo "                      config.yml.  When supplied, the configuration file's path is"
    echo "                      constructed as <project_path>/<config_path>.  If omitted, defaults"
    echo "                      to \"${CONFIG_CONTAINER_RELATIVE_PATH}\"."
    echo "    -h                Display this help message and exit."
    echo "    -i <image>        Launch the container from <image> instead of the default."
    echo "                      If omitted, defaults to \"${SCALABEL_IMAGE}\"."
    echo "    -l <launcher>     Use <launcher> to launch the container run-time instead of"
    echo "                      the default.  If omitted, defaults to \"${CONTAINER_LAUNCHER}\"."
    echo "    -p <port>         Expose Scalabel on <port>.  May be specified as an interface"
    echo "                      and a port for advanced configurations.  If omitted, defaults"
    echo "                      to \"${SCALABEL_PORT}\"."
    echo "    -q                Execute quietly.  Informational messages will be suppressed"
    echo "                      during execution."
    echo "    -t                Enable testing mode.  Scalabel will not be launched but the"
    echo "                      command to do so will be printed to standard output."
    echo
}

# default to Docker as alternative run-times aren't as ubiquitous.  run
# interactively in a terminal and clean up the container image after we
# shutdown.
CONTAINER_LAUNCHER="docker run -it --rm"

# use the latest image from Scalabel.ai to label with.
SCALABEL_IMAGE="scalabel/www:latest"

# only expose the web application's port on the local machine.  this prevents
# Scalabel.ai from being accessible on the local network, as well as access to
# its Redis instance.
SCALABEL_PORT="127.0.0.1:8686"

# by default, announce what we're doing.
QUIET_FLAG="no"

# by default, launch the labeling application.
TESTING_FLAG="no"

# path to Scalabel's data directory inside of its container.  this should never
# need to be tweaked by labelers or IWP developers, but only updated when the
# Scalabel application makes a significant update.
SCALABEL_CONTAINER_ROOT="/opt/scalabel/local-data"

# path to Scalabel's items and project, respectively.
SCALABEL_CONTAINER_ITEMS_ROOT="${SCALABEL_CONTAINER_ROOT}/items"
SCALABEL_CONTAINER_PROJECT_ROOT="${SCALABEL_CONTAINER_ROOT}/scalabel"

# relative path to the Scalabel application's configuration file.  this is
# appended to the project path specified by the caller.
CONFIG_CONTAINER_RELATIVE_PATH="config.yml"

# optional path to 'realpath'.  used to normalize paths to aide in diagnostics.
REALPATH=`which realpath 2>/dev/null`

while getopts "c:hi:l:p:qt" OPTION;
do
    case ${OPTION} in
       c)
           CONFIG_CONTAINER_RELATIVE_PATH=${OPTARG}
           ;;
       h)
           print_usage
           exit 0
           ;;
       i)
           SCALABEL_IMAGE=${OPTARG}
           ;;
       l)
           CONTAINER_LAUNCHER=${OPTARG}
           ;;
       p)
           SCALABEL_PORT=${OPTARG}
           ;;
       q)
           QUIET_FLAG="yes"
           ;;
       t)
           TESTING_FLAG="yes"
           ;;
       *)
           print_usage
           exit 1
           ;;
    esac
done
shift `expr ${OPTIND} - 1`

# ensure we received the correct number of arguments.
if [ $# -ne 2 ]; then
    echo "Expected 2 arguments but received $#!" >&2
    exit 1
fi

# map arguments to variable names.
ITEMS_HOST_PATH=$1
PROJECTS_HOST_PATH=$2
CONFIG_CONTAINER_PATH="${SCALABEL_CONTAINER_ROOT}/scalabel/${CONFIG_CONTAINER_RELATIVE_PATH}"

# verify the supplied paths exist.  bail if either the items or the project
# paths are invalid as that prevents labeling.
#
# NOTE: we do this before normalization to report errors with the paths
#       provided, rather than the normalized paths.
#
if [ ! -d "${ITEMS_HOST_PATH}" ]; then
    echo "The items path (${ITEMS_HOST_PATH}) does not exist!" >&2

    exit 1
elif [ ! -d "${PROJECTS_HOST_PATH}" ]; then
    echo "The projects path (${PROJECTS_HOST_PATH}) does not exist!" >&2

    exit 1
fi

# normalize the paths so they're identifiable and understanable in process
# listings.
if [ -z "${REALPATH}" ]; then
    echo "Could not find 'realpath' in the PATH.  Supplied paths will not be normalized." >&2
else
    ITEMS_HOST_PATH=`${REALPATH} ${ITEMS_HOST_PATH}`
    PROJECTS_HOST_PATH=`${REALPATH} ${PROJECTS_HOST_PATH}`

    #
    # NOTE: this path exists in the container so we cannot validate it.  have
    #       realpath normalize it without checking its existence.
    #
    CONFIG_CONTAINER_PATH=`${REALPATH} -m ${CONFIG_CONTAINER_PATH}`
fi

# announce how the container will be launched.
if [ ${QUIET_FLAG} != "yes" ]; then
    echo "Items:    ${ITEMS_HOST_PATH}"
    echo "Projects: ${PROJECTS_HOST_PATH}"
    echo "Config:   ${CONFIG_CONTAINER_PATH}"
    echo "Port:     ${SCALABEL_PORT}"
    echo
fi

# assemble our configuration and build the launch command.
#
# this maps data and projects into the container, exposes the application's port
# outside of its container, and launches a specific application image using a
# specific labeling configuration.
SCALABEL_COMMAND="${CONTAINER_LAUNCHER} \
                  -v '${ITEMS_HOST_PATH}:${SCALABEL_CONTAINER_ITEMS_ROOT}' \
                  -v '${PROJECTS_HOST_PATH}:${SCALABEL_CONTAINER_PROJECT_ROOT}' \
                  -p ${SCALABEL_PORT}:8686 \
                  ${SCALABEL_IMAGE} \
                  node app/dist/main.js \
                  --config ${CONFIG_CONTAINER_PATH} \
                  --max-old-space-size=8192"

# echo or execute depending on the caller's request.
if [ ${TESTING_FLAG} = "yes" ]; then
    echo ${SCALABEL_COMMAND}
else
    eval ${SCALABEL_COMMAND}
fi
