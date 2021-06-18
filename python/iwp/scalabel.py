import copy
import enum
import json
import os

import iwp.labels

# module for all things related to Scalabel frames.

# NOTE: we encode slice parameters in the Scalabel frame's name to support a
#       full round trip during labeling.  without this, we do not have a way to
#       convey metadata about the frame that is needed when we convert to/from
#       canonical IWP label format with the following:
#
#         1. build Scalabel frames
#         2. set canonical IWP labels on the associated Scalabel frames
#         3. label using the Scalabel frames
#         4. extract IWP labels from Scalabel frames
#         5. normalize IWP labels to create new canonical labels
#

# enumeration of labeling strategies for Scalabel playlists:
#
#   no_order  - frames have no special ordering and may be presented in any
#               order.
#   xy_slices - frames are sorted such that each location within the data volume
#               is grouped together.  this results in all of one XY slice's data
#               being temporally ordered before another XY slice is visited.
#   z_stacks  - frames are sorted such that XY slices within a single time step
#               are grouped together.  this results in full stacks of XY slices
#               for a single time step being grouped together.
#   variables - frames are sorted such that variables from a single XY slice
#               are grouped together.
#
#                 NOTE: this does not appear to be a useful sort order and will
#                       likely be removed in the future.
#
@enum.unique
class LabelingStrategyType( enum.Enum ):
    NO_ORDER  = 1
    XY_SLICES = 2
    Z_STACKS  = 3
    VARIABLES = 4

def build_slice_name( experiment_name, variable_name, time_index, xy_slice_index ):
    """
    Builds a unique name for a slice based on the experiment, variable, and location
    within the dataset.

    Takes 4 arguments:

      experiment_name - String specifying the experiment that generated the slice.
      variable_name   - String specifying the variable associated with the slice.
      time_index      - Non-negative index specifying the time step associated with
                        the slice.
      xy_slice_index  - Non-negative index specifying the XY slice.

    Returns 1 value:

      slice_name - String containing the constructed name.

    """

    return "{:s}-{:s}-z={:03d}-Nt={:03d}".format(
        experiment_name,
        variable_name,
        xy_slice_index,
        time_index )

def slice_name_to_components( slice_name ):
    """
    Decomposes a slice name into a map of its unique components.  This is the
    inverse of build_slice_name().  Also handles slice_name's which have been
    converted into a path or URL as a prefix.

    Takes 1 argument:

      slice_name - String specifying the slice's name, as generated by build_slice_name().

    Returns 1 value:

      slice_map - Dictionary containing the decomposed metadata from slice_name.
                  Has the following keys:

                    "experiment":       Experiment name
                    "variable":         Variable name
                    "z_index":          XY slice index
                    "time_step_index":  Time step index

    """

    slice_components = slice_name.split( "-" )

    # handle slice names that have been turned into paths with extensions.
    if "." in slice_components[-1]:
        slice_components[-1] = slice_components[-1].split( "." )[0]

    # map the individual components to their names.
    #
    # NOTE: we use negative indexing to handle the case where the experiment
    #       name may contain one or more hyphens.
    #
    slice_map = {
        "experiment":      "-".join( slice_components[:-3] ),
        "variable":        slice_components[-3],
        "z_index":         int( slice_components[-2].split( "=" )[1] ),
        "time_step_index": int( slice_components[-1].split( "=" )[1] )
    }

    return slice_map

def build_slice_video_name( playlist_strategy, experiment_name, variable_name, time_index, xy_slice_index ):
    """
    Builds a video name for a slice based on the experiment variable, and location
    within the dataset.  The structure of the returned name is governed by the
    strategy specified which ultimately controls how the slice is sorted within a
    playlist when loaded into Scalabel.ai.

    The returned names' structures are as follows:

       NO_ORDER  - <experiment>
       XY_SLICES - <experiment>-<variable>-z=<slice_index>
       Z_STACKS  - <experiment>-<variable>-Nt=<time_index>
       VARIABLES - <experiment>-z=<slice_index>-Nt=<time_index>

    Takes 5 arguments:

      playlist_strategy - Enumeration of type iwp.labels.scalabel.LabelingStrategyType
                          that controls the sort order of a frame containing the
                          returned video name.  See the description above for the
                          individual sort strategies.
      experiment_name   - String specifying the experiment that generated the slice.
      variable_name     - String specifying the variable associated with the slice.
      time_index        - Non-negative index specifying the time step associated with
                          the slice.
      xy_slice_index    - Non-negative index specifying the XY slice.

    Returns 1 value:

      video_name - String containing the constructed video name.

    """

    # default to the experiment name which makes all slices equal to each other.
    # this results in playlists retaining their natural ordering.
    video_name = experiment_name

    if playlist_strategy == LabelingStrategyType.XY_SLICES:
        video_name = "{:s}-{:s}-z={:03d}".format(
            experiment_name,
            variable_name,
            xy_slice_index )
    elif playlist_strategy == LabelingStrategyType.Z_STACKS:
        video_name = "{:s}-{:s}-Nt={:03d}".format(
            experiment_name,
            variable_name,
            time_index )
    elif playlist_strategy == LabelingStrategyType.VARIABLES:
        video_name = "{:s}-z={:03d}-Nt={:03d}".format(
            experiment_name,
            time_index,
            xy_slice_index )

    return video_name

def build_slice_path( data_root, data_suffix, experiment_name, variable_name, time_index, xy_slice_index, index_precision=3 ):
    """
    Returns the on-disk path to a specific slice.  The path generated has the following
    form:

       <root>/<variable>/<experiment>-<variable>-z=<slice>-Nt=<time><suffix>

    <slice> and <time> are zero-padded integers formatted according to an
    optional precision parameter.

    Takes 7 arguments:

      data_root       - String specifying the root on-disk path for the slice.
      data_suffix     - String specifying the path suffix for the slice.
      experiment_name - String specifying the experiment that generated the slice.
      variable_name   - String specifying the variable associated with the slice.
      time_index      - Non-negative index specifying the time step associated with
                        the slice.
      xy_slice_index  - Non-negative index specifying the XY slice.
      index_precision - Optional non-negative integer specifying the precision used
                        when formatting "<slice>" and "<time>".  If omitted, defaults
                        to 3.

    Returns 1 value:

      slice_name - String specifying the constructed path.

    """

    return "{:s}/{:s}/{:s}-{:s}-z={:0{index_precision}d}-Nt={:0{index_precision}d}.png".format(
        data_root,
        variable_name,
        experiment_name,
        variable_name,
        xy_slice_index,
        time_index,
        index_precision=index_precision )

def build_slice_url( url_prefix, slice_path, number_components=0 ):
    """
    Returns the URL to a specific slice's on-disk path.  The URL prefix is combined with
    a portion of the specified path to create the slice URL.

    Takes 3 arguments:

      url_prefix        - URL prefix of the slice.
      slice_path        - Path to the slice's item.  May be either absolute or relative.
      number_components - Optional, non-negative integer specifying the number of leading
                          components to remove from slice_path.  No components are
                          removed when number_components is zero.

    Takes 1 value:

      slice_url - Combination of url_prefix and slice_path with number_components-many
                  path components removed from slice_path.

    """

    path_components = slice_path.split( "/" )

    # handle absolute paths.  these generate an empty component which should be
    # ignored to make number_components consistent between absolute and relative
    # paths.
    if slice_path.startswith( "/" ):
        path_components = path_components[1:]

    if number_components < 0:
        raise ValueError( "Invalid number of components specified! ({})".format(
            number_components ) )

    if number_components >= len( path_components ):
        raise IndexError( "Can't remove {:d} components from {:s} - only has {:d}.".format(
            number_components,
            slice_path,
            len( path_components ) ) )

    return "{:s}/{:s}".format(
        url_prefix,
         "/".join( path_components[number_components:] ) )

def build_scalabel_frames( experiment_name,
                           variables_list,
                           time_range,
                           xy_slice_range,
                           data_root,
                           data_suffix,
                           url_prefix,
                           component_count,
                           labeling_strategy=LabelingStrategyType.NO_ORDER,
                           check_data_flag=False ):
    """
    Builds a sequence of minimal, Scalabel frames according to the slice metadata provided.
    Serializing the generated frames is sufficient for an Items list to start a new Scalabel.ai
    video labeling project.

    Frames are constructed in (Z, time, variable) order in the generated structure
    though are sorted by Scalabel.ai when loaded.  The labeling order within the
    application is governed by the labeling strategy specified:

      NO_ORDER  - No particular order is specified.  All frames are from the same
                  "video" and Scalabel.ai sorts by time stamp and frame name.
      XY_SLICES - Frames are sorted by location within the dataset.  Each XY slice
                  is from the same "video" which results in each of its time steps
                  being grouped together.
      Z_STACKS  - Frames are sorted by time within the dataset.  Each time step is
                  from the same "video" which results in each of its XY slices being
                  grouped together (in a stack).
      VARIABLES - Frames are sorted by time and location within the dataset.  Each
                  XY slice, per time step, is from the same "video" which results in
                  each of its variables being grouped together.  There is no guarantee
                  that adjacent slices, either in XY slice or time step order, are
                  consecutive.

    NOTE: None of the frames generated have labels.  These must be set by hand or with
          set_iwp_labels().

    The supplied experiment name is used as the underlying video name, and individual
    frame names are constructed by build_slice_name().  Frame URLs are constructed by
    build_slice_path() and build_slice_url().

    Raises FileNotFoundError if a datum associated with a generated frame does not
    exist and the caller requested verification.

    Takes 10 arguments:

      experiment_name   - Name of the experiment that generated the underlying frame
                          data.
      variables_list    - Sequence of variables to build frames for.
      time_range        - Sequence of time step indices to build frames for.
      xy_slice_range    - Sequence of XY slice indices to build frames for.
      data_root         - Path root to the slice's on-disk storage.
      data_suffix       - Path suffix to the slice's on-disk storage.
      url_prefix        - URL prefix to use for each frame's URL.
      component_count   - Number of components to strip off of the computed slice
                          path when building the frame's URL.
      labeling_strategy - Optional enumeration of type iwp.labels.scalabel.LabelingStrategyType
                          that controls the sort order generated frames.
      check_data_flag   - Optional boolean specifying whether individual frames datum's
                          will be checked for existence.  If True and the underlying
                          datum does not exist, FileNotFoundError is raised.  If
                          omitted, defaults to False.

    Returns 1 value:

      scalabel_frames - List of Scalabel frames created.

    """

    # list of Scalabel frames.
    scalabel_frames = []

    # walk through each XY slice one at a time, visiting each time step in
    # sequence before moving to the next slice.  each variable is visited in
    # sequence within each time step.
    #
    # NOTE: this order doesn't influence Scalabel.ai's tool at all as it sorts
    #       frames by video name and timestamp before showing them to labelers.
    #
    for xy_slice_index in xy_slice_range:
        for time_index in time_range:
            for variable_name in variables_list:
                # construct the video's name.  this influences the order in
                # which frames are presented to labelers.  timestamps are used
                # as a secondary sort key when frames come from the same video.
                video_name = build_slice_video_name( labeling_strategy,
                                                     experiment_name,
                                                     variable_name,
                                                     time_index,
                                                     xy_slice_index )

                # construct the slice's "name".  this is the frame name within
                # the "video".
                slice_name = build_slice_name( experiment_name,
                                               variable_name,
                                               time_index,
                                               xy_slice_index )

                # build the path to the slice on the host system.
                slice_path = build_slice_path( data_root,
                                               data_suffix,
                                               experiment_name,
                                               variable_name,
                                               time_index,
                                               xy_slice_index )

                # build the URL to the slice within the Scalabel application.
                slice_url  = build_slice_url( url_prefix,
                                              slice_path,
                                              component_count )

                if check_data_flag and not os.path.exists( slice_path ):
                    raise FileNotFoundError( "Scalabel frame's datum does not exist! "
                                             "({:s}, {:s}, {:s}, (T={:d}, Z={:d}))".format(
                                                 slice_path,
                                                 experiment_name,
                                                 variable_name,
                                                 time_index,
                                                 xy_slice_index ) )

                # create a frame with minimal metadata.
                scalabel_frame = {
                    "name":       slice_name,
                    "timestamp":  time_index,
                    "url":        slice_url,
                    "videoName":  video_name
                }

                scalabel_frames.append( scalabel_frame )

    return scalabel_frames

def get_scalabel_frame_key( scalabel_frame ):
    """
    Retrieves a key that locates the supplied Scalabel frame within the underlying
    dataset.  The key returned locates the frame both temporarly and spatially.

    Takes 1 argument:

      scalabel_frame - Scalabel frame.  A dictionary describing a single frame within
                       a dataset.

    Returns 1 value:

      frame_key - Tuple identifying scalabel_frame's location within the data
                  volume.  Comprised of (time step index, z index).

    """

    # deconstruct the frame's name and return our (time, z) location.
    components_map = slice_name_to_components( scalabel_frame["name"] )

    return (components_map["time_step_index"],
            components_map["z_index"])

def extract_iwp_labels_from_frames( scalabel_frames, category_filter=[] ):
    """
    Extracts Scalabel labels from Scalabel frames and converts them to IWP labels,
    optionally filtering out unwanted label categories.  No normalization is done on
    the resulting IWP labels.

    Takes 2 arguments:

      scalabel_frames - List of Scalabel frames to extract and convert labels from.
      category_filter - Optional list of label categories to filter for when converting
                        labels.  Only labels whose categories are in category_filter
                        are kept during conversion.  If specified as an empty list,
                        all labels are kept.  If omitted, defaults to an empty list.

    Returns 1 value:

      iwp_labels - List of converted IWP labels.

    """

    iwp_labels = []

    #
    # NOTE: this method combines two logical concepts (Scalabel label extraction
    #       and Scalabel label to IWP label conversion) into a single routine as
    #       Scalabel labels do not contain enough metadata to convert on to IWP
    #       labels on their own.  rather than create yet another data structure
    #       to hold additional metadata along with each Scalabel label, we fuse
    #       the operations into this function.
    #

    # walk through each Scalabel frame looking for labels matching the category
    # filter.
    for scalabel_frame in scalabel_frames:

        frame_components = slice_name_to_components( scalabel_frame["name"] )

        # walk each of this frame's labels and convert each into IWP labels.
        for scalabel_label in scalabel_frame["labels"]:

            # skip labels in categories we're not interested in.
            if (len( category_filter ) > 0) and (scalabel_label["category"] not in category_filter):
                continue

            # create an IWP label and keep track of it.
            iwp_label = {
                    "time_step_index": frame_components["time_step_index"],
                    "z_index":         frame_components["z_index"],
                    "id":              scalabel_label["id"],
                    "category":        scalabel_label["category"],
                    "bbox":            scalabel_label["box2d"]
                }

            iwp_labels.append( iwp_label )

    return iwp_labels

def set_iwp_labels( scalabel_frames, iwp_labels_path=None ):
    """
    Replaces the labels in the Scalabel frames with those found in the supplied IWP
    labels path.  A copy of the frames is made so the originals are unaltered.

    Takes 2 arguments:

      scalabel_frames - List of Scalabel frames.  Each frame is a dictionary describing
                        a single frame within a dataset.
      iwp_labels_path - Optional path to the IWP labels to set in scalabel_frames.  If
                        omitted, defaults to None and all labels are removed from
                        scalabel_frames.

    Returns 1 value:

      scalabel_frames - Updated list of Scalabel frames.  Each frame's labels are set
                        to those found in iwp_labels_path.

    """

    #
    # NOTE: we make a deep copy of the frames so we can modify them in place.
    #
    scalabel_frames = copy.deepcopy( scalabel_frames )

    # load the IWP labels if they were supplied.  otherwise we use an empty set
    # of labels and ultimately remove the Scalabel frames' labels.
    if iwp_labels_path is not None:
        iwp_labels = iwp.labels.load_iwp_labels( iwp_labels_path )
    else:
        iwp_labels = []

    # map from (time step, slice index) to a dictionary containing an identifier
    # ("id") and a list of IWP labels ("labels").  this allows flattening of IWP
    # labels which simplifies finding all labels for a given slice.
    labels_map = {}

    # build an index for the IWP labels so we can easily find all of the IWP
    # labels for a given Scalabel frame.  Since Scalabel frames may be redundant
    # (e.g. one frame per variable within a IWP XY slice) we need quick lookup
    # to the labels of interest.
    for iwp_label in iwp_labels:

        # get the key that locates this IWP in time and space.
        label_key = iwp.labels.get_iwp_label_key( iwp_label )
        label_id  = iwp_label["id"]

        # track this label in our map.  take care to detect malformed labels
        # (e.g. duplicate) so we don't have problems later.
        if label_key not in labels_map:
            # this (time step, slice index) does not have any labels, so we
            # initialize it.
            labels_map[label_key] = {
                "id":     label_id,
                "labels": [iwp_label]
            }
        elif label_id in labels_map[label_key]["id"]:
            # this (time step, slice index) already has a label by this name.
            # sound the alarm that we have a duplicate.
            raise ValueError( "Supplied IWP labels have a duplicate label for "
                              "{:s} at (T={:d}, Z={:d})!".format(
                                  label_id,
                                  iwp_label["time_step_index"],
                                  iwp_label["z_index"] ) )
        else:
            # this (time step, slice index) has at least one label, so we add
            # another to it.
            labels_map[label_key]["labels"].append( iwp_label )

    # walk through the frames and add the all of the IWP labels associated.
    for scalabel_frame in scalabel_frames:

        # get the key for IWP labels associated with this frame.
        frame_key = get_scalabel_frame_key( scalabel_frame )

        # replace the frame's existing labels so it only contains the IWP
        # frames supplied.
        if frame_key not in labels_map:
            scalabel_frame["labels"] = []
        else:
            scalabel_frame["labels"] = iwp.labels.convert_labels_iwp_to_scalabel( labels_map[frame_key]["labels"] )

    return scalabel_frames

def load_scalabel_frames( scalabel_frames_path ):
    """
    Loads Scalabel frames from a file.  Handles both raw sequences of Scalabel frames
    as well as labels exported from Scalabel.ai's application.

    Raises ValueError if the data read isn't of a known type.

    Takes 1 argument:

      scalabel_frames_path - Path to serialized Scalabel frames.

    Returns 1 value:

      scalabel_frames - A list of Scalabel frames.

    """

    with open( scalabel_frames_path, "r" ) as scalabel_frames_fp:
        scalabel_frames = json.load( scalabel_frames_fp )

    # handle the case where we have exported labels from Scalabel.ai itself vs
    # a list of frames.
    if type( scalabel_frames ) == dict:
        if "frames" in scalabel_frames:
            return scalabel_frames["frames"]
    elif type( scalabel_frames ) == list:
        return scalabel_frames

    raise ValueError( "Unknown structure read from '{:s}'.".format(
        scalabel_frames_path ) )
