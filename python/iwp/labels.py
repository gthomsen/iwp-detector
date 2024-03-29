import collections
import copy
import enum
import json
import numpy as np

# module for all things related to labels, IWP or otherwise.
#
# Documentation on Scalabel image list labels:
#
#   https://doc.scalabel.ai/format.html
#

# enumeration of merge strategies for combining multiple IWP labels:
#
#   union        - disparate bounding boxes are replaced with the smallest
#                  bounding box that contains all of the originals.
#   intersection - disparate bounding boxes are replaced with the smallest
#                  bounding box, possibly empty, that is contained in each
#                  of the originals.
#   error        - multiple bounding boxes cannot be combined and it is an
#                  error to attempt to.
@enum.unique
class IWPLabelMergeType( enum.Enum ):
    UNION        = 1
    INTERSECTION = 2
    ERROR        = 3

# enumeration of sort strategies for combining IWP label lists.
#
#   none     - concatenate only.  don't sort anything.
#   spatial  - sort so that time steps are sorted before XY slices.
#   temporal - sort so that XY slices are sorted before time steps.
@enum.unique
class IWPLabelSortType( enum.Enum ):
    NONE         = 1
    SPATIAL      = 2
    TEMPORAL     = 3

def get_iwp_label_key( iwp_label ):
    """
    Retrieves a key that locates the supplied IWP label within the underlying
    dataset.  The key returned locates the label both temporarly and spatially.

    Takes 1 argument:

      iwp_label - IWP label to locate.

    Returns 1 value:

      label_key - Tuple identifying iwp_label's location within a dataset.  Comprised
                  of (time step index, z index).

    """

    return (iwp_label["time_step_index"], iwp_label["z_index"])

def get_iwp_augmented_label_key( iwp_label ):
    """
    Retrieves a key that locates the supplied IWP label within the underlying
    dataset.  The key returned locates the label temporarly, spatially, and uniquely.

    Takes 1 argument:

      iwp_label - IWP label to locate.

    Returns 1 value:

      label_key - Tuple identifying iwp_label's location within a dataset.  Comprised
                  of (time step index, z index, id).

    """

    return (*get_iwp_label_key( iwp_label ), iwp_label["id"])

def get_iwp_label_name( iwp_label, shortened_flag=False ):
    """
    Retrieves a name for the supplied IWP label.  May be a shortened nickname
    for readability or the full label identifier depending on the caller's needs.

    Takes 1 argument:

      iwp_label      - IWP label to extract a name from.
      shortened_flag - Optional flag specifying whether a shortened name is requested.
                       If specified as True, the first six characters of the name are
                       returned, otherwise the entire identifier.  If omitted, defaults
                       to False.

                       NOTE: Shortened names may not necessarily be unique!

    Returns 1 value:

      label_name - Name string associated with iwp_label.

    """

    # the first six characters of an identifier seems fairly unique, since it
    # covers a space of 6^62 combinations, and is short enough to overlay on
    # XY slices without cluttering things.
    #
    # NOTE: no analysis has been done on Scalabel identifiers to understand
    #       their generation.  use the full label when uniqueness must be
    #       guaranteed.
    #
    if shortened_flag == True:
        return iwp_label["id"][:6]

    return iwp_label["id"]

def convert_labels_iwp_to_scalabel( iwp_labels ):
    """
    Converts IWP labels to Scalabel labels.

    NOTE: This create a Scalabel label, not a *frame*.

    Takes 1 argument:

      iwp_labels - List of IWP labels to convert.

    Returns 1 value:

      scalabel_labels - List of converted Scalabel labels.

    """

    scalabel_labels = []

    for iwp_label in iwp_labels:
        scalabel_label = {
            "id":          iwp_label["id"],
            "category":    iwp_label["category"],
            "attributes" : {},
            "manualShape": True,
            "box2d": {
                "x1": iwp_label["bbox"]["x1"],
                "x2": iwp_label["bbox"]["x2"],
                "y1": iwp_label["bbox"]["y1"],
                "y2": iwp_label["bbox"]["y2"]
                },
            "poly2d":  None,
            "box3d":   None,
            "plane3d": None,
            "customs": {}
            }

        scalabel_labels.append( scalabel_label )

    return scalabel_labels

def sort_iwp_labels( iwp_labels, sort_type, in_place_flag=False ):
    """
    Sorts IWP labels according to the ordering requested.

    Takes 3 arguments:

      iwp_labels    - List of IWP labels.
      sort_type     - Enumeration of IWPLabelSortType specifying how labels should be
                      sorted.
      in_place_flag - Optional flag specifying in place update or an update to a
                      copy of the labels.  If omitted, defaults to False and a new
                      list of IWP labels is returned.

    Returns 1 value:

      sorted_iwp_labels - Sorted list of IWP labels.

    """

    # return early if we have nothing to do.
    if sort_type == IWPLabelSortType.NONE:
        return iwp_labels

    # create the sort key lambda.
    if sort_type == IWPLabelSortType.SPATIAL:
        sort_key = lambda label: (label["z_index"], label["time_step_index"], label["id"])
    elif sort_type == IWPLabelSortType.TEMPORAL:
        sort_key = lambda label: (label["time_step_index"], label["z_index"], label["id"])
    else:
        raise ValueError( "Unknown IWP label sort type specified! ({})".format(
            sort_type ) )

    # sort the labels.  respect the request for a copy or not.
    if in_place_flag:
        iwp_labels.sort( key=sort_key )
        sorted_iwp_labels = iwp_labels
    else:
        sorted_iwp_labels = sorted( iwp_labels, key=sort_key )

    return sorted_iwp_labels

def save_iwp_labels( iwp_labels_path, iwp_labels, pretty_flag=True ):
    """
    Saves IWP labels to a file.

    Takes 3 arguments:

      iwp_labels_path - Path to serialize iwp_labels to.
      iwp_labels      - List of IWP labels to serialize.
      pretty_flag     - Optional boolean specifying whether the IWP labels should be
                        pretty printed during serialization.  If omitted, defaults to
                        True so that the contents of iwp_labels_path is human readable.

    Returns nothing.

    """

    with open( iwp_labels_path, "w" ) as iwp_labels_fp:
        #
        # NOTE: we sort the dictionary keys so that it is easier to compare
        #       different labels without custom tools.
        #
        json.dump( iwp_labels, iwp_labels_fp, indent=4, sort_keys=True )

    return

def load_iwp_labels( iwp_labels_path ):
    """
    Loads IWP labels from a file.

    Takes 1 argument:

      iwp_labels_path - Path to serialized IWP labels.

    Returns 1 value:

      iwp_labels - List of IWP labels read from iwp_labels_path.

    """

    with open( iwp_labels_path, "r" ) as iwp_labels_fp:
        iwp_labels = json.load( iwp_labels_fp )

    # ensure that the slice indices are integral regardless of how they were
    # serialized.
    for iwp_label in iwp_labels:
        iwp_label["time_step_index"] = int( iwp_label["time_step_index"] )
        iwp_label["z_index"]         = int( iwp_label["z_index"] )

    return iwp_labels

def union_iwp_label( iwp_label_a, iwp_label_b ):
    """
    Creates a new IWP label whose bounding box is the union of two IWP labels.
    The newly created label has the same metadata from the first label and
    a bounding box that minimally spans both labels' bounding boxes.

    Takes 2 arguments:

      iwp_label_a - First IWP label to union.
      iwp_label_b - Second IWP label to union.

    Returns 1 value:

      unioned_iwp_label - Unioned IWP label.

    """

    unioned_iwp_label = {
        "id":              iwp_label_a["id"],
        "category":        iwp_label_a["category"],
        "time_step_index": iwp_label_a["time_step_index"],
        "z_index":         iwp_label_a["z_index"],
        "bbox":            {
            "x1": min( iwp_label_a["bbox"]["x1"], iwp_label_b["bbox"]["x1"] ),
            "x2": max( iwp_label_a["bbox"]["x2"], iwp_label_b["bbox"]["x2"] ),
            "y1": min( iwp_label_a["bbox"]["y1"], iwp_label_b["bbox"]["y1"] ),
            "y2": max( iwp_label_a["bbox"]["y2"], iwp_label_b["bbox"]["y2"] )
        }
    }

    return unioned_iwp_label

def intersect_iwp_label( iwp_label_a, iwp_label_b ):
    """
    Creates a new IWP label whose bounding box is the intersection of two IWP labels.
    The newly created label has the same metadata from the first label and a bounding
    box that spans the overlap of both labels' bounding boxes.

    Takes 2 arguments:

      iwp_label_a - First IWP label to intersect.
      iwp_label_b - Second IWP label to intersect.

    Returns 1 value:

      intersected_iwp_label - Intersected IWP label.

    """

    intersected_iwp_label = {
        "id":              iwp_label_a["id"],
        "category":        iwp_label_a["category"],
        "time_step_index": iwp_label_a["time_step_index"],
        "z_index":         iwp_label_a["z_index"],
        "bbox":            {
            "x1": max( iwp_label_a["bbox"]["x1"], iwp_label_b["bbox"]["x1"] ),
            "x2": min( iwp_label_a["bbox"]["x2"], iwp_label_b["bbox"]["x2"] ),
            "y1": max( iwp_label_a["bbox"]["y1"], iwp_label_b["bbox"]["y1"] ),
            "y2": min( iwp_label_a["bbox"]["y2"], iwp_label_b["bbox"]["y2"] )
        }
    }

    return intersected_iwp_label

def _build_label_map( iwp_labels, with_id_flag=True ):
    """
    Builds an ordered dictionary mapping labels' keys to a copy of the first
    instance of a matching label.  This is useful for operating on a list labels
    with the intent to modify their contents.

    NOTE: Only the first instance of a label is tracked in the resulting label
          map.  This function is intended as a building block for de-duplication and
          normalization functions that explicitly handle duplicate labels, so
          said logic is not present with the intent that higher level methods
          will provide it.

    Takes 2 arguments:

      iwp_labels   - List of IWP labels to build a map from.
      with_id_flag - Optional boolean specifying whether a normal or augmented label
                     key is used in the computed map.  When specified as True,
                     get_iwp_augmented_label_key() is used to build label keys, otherwise
                     get_iwp_label_key() is used.  If omitted, defaults to True.

    Returns 1 value:

      label_map - A collections.OrderedDict mapping label keys to deep copies of
                  iwp_label's labels.

    """

    label_map = collections.OrderedDict( [] )

    for iwp_label in iwp_labels:
        # get the appropriate label key.
        if with_id_flag:
            label_key = get_iwp_augmented_label_key( iwp_label )
        else:
            label_key = get_iwp_label_key( iwp_label )

        # keep track of the first instance of this label.  second, and
        # additional, instances are ignored.
        if label_key not in label_map:
            #
            # NOTE: we make a deep copy so that the label itself can be merged
            #       without affecting the original.
            #
            label_map[label_key] = copy.deepcopy( iwp_label )

    return label_map

def _normalize_iwp_labels( iwp_labels, merge_type=IWPLabelMergeType.UNION ):
    """
    Normalizes a list of IWP labels so that each unique label is only seen at most
    once per (time, z) slice.  Duplicate labels are removed according to the requested
    merge method.

    Takes 2 arguments:

      iwp_labels - List of IWP labels.
      merge_type - Enumeration of type IWPLabelMergeType specifying how duplicate labels are
                   handled.  Must be one IWPLabelMergeType.UNION or IWPLabelMergeType.INTERSECTION to
                   union or intersect duplicate labels' bounding boxes when de-duplicating.

    Returns 2 values:

      normalized_iwp_labels - List of IWP labels.  These are deep copies of iwp_label's
                              content.
      label_map             - Dictionary mapping augmented IWP label keys (time, z, id)
                              to IWP labels in normalized_iwp_labels.

    """

    # create a map of the labels so we can easily find labels by (time, z, id).
    label_map = _build_label_map( iwp_labels, with_id_flag=True )

    # walk through each of the labels and combine duplicates with those found
    # in the label map.  for labels which do not contain duplicates, this is an
    # no-op as union and intersect are idempotent when applied to themselves.
    for iwp_label in iwp_labels:
        label_key = get_iwp_augmented_label_key( iwp_label )

        # update the existing label with the current one according to the
        # merge type.
        #
        # NOTE: we don't check existence in our label map since it was
        #       created from the labels we're iterating through.  all
        #       labels are guaranteed to be present.
        #
        if merge_type == IWPLabelMergeType.UNION:
            label_map[label_key] = union_iwp_label( iwp_label,
                                                    label_map[label_key] )
        elif merge_type == IWPLabelMergeType.INTERSECTION:
            label_map[label_key] = intersect_iwp_label( iwp_label,
                                                        label_map[label_key] )

    # the normalized IWP labels are simply the label map's contents.
    normalized_iwp_labels = list( label_map.values() )

    return (normalized_iwp_labels, label_map)

def merge_iwp_labels( iwp_labels_a, iwp_labels_b, merge_type=IWPLabelMergeType.UNION ):
    """
    Merges two lists of IWP labels into a single, while appropriately handling duplicate
    labels.  Returns a new list containing updated copies of the original labels.
    Duplicates can be handled by 1) union, 2) intersection, or 3) error.  See union_iwp_label()
    and intersect_iwp_label() for the first two cases, respectively.

    Raises ValueError if an unknown merge type is supplied.

    Takes 3 arguments:

      iwp_labels_a - List of IWP labels to merge.
      iwp_labels_b - List of IWP labels to merge.
      merge_type   - Enumeration of type IWPLabelMergeType.

    Returns 1 value:

      merged_iwp_labels - List of merged IWP labels.  The order of the labels matches
                          the ordering of iwp_labels_a and iwp_labels_b.

    """

    if merge_type not in (IWPLabelMergeType.UNION,
                          IWPLabelMergeType.INTERSECTION,
                          IWPLabelMergeType.ERROR):
        raise ValueError( "Unknown merge type ({:s}) supplied!  Must be one of {:s}, {:s} or {:s}.".format(
            merge_type,
            IWPLabelMergeType.UNION,
            IWPLabelMergeType.INTERSECTION,
            IWPLabelMergeType.ERROR ) )

    # flatten both A and B's labels to simplify the merge logic below.  common
    # cases are that labels come from a canonical source (and are already
    # flattened) or come from a tool (and need to be flattened).
    #
    # NOTE: this makes copies of each, so we can modify them as needed without
    #       affecting the caller's versions.
    #
    (iwp_labels_a, label_map_a) = _normalize_iwp_labels( iwp_labels_a,
                                                         IWPLabelMergeType.UNION )
    (iwp_labels_b, label_map_b) = _normalize_iwp_labels( iwp_labels_b,
                                                         IWPLabelMergeType.UNION )

    # walk through each of B's labels and add them into A's map.  additions
    # are handled according to the merge method requested.
    for iwp_label_b in iwp_labels_b:
        label_key_b = (get_iwp_label_key( iwp_label_b ), iwp_label_b["id"])

        # add this B label into A's map.  handle an updates and additions
        # as appropriate.
        if label_key_b in label_map_a:

            # this label already exists in A.  update it unless duplicates are
            # catastrophic.
            if merge_type == IWPLabelMergeType.UNION:
                label_map_a[label_key_b] = union_iwp_label( iwp_label_b,
                                                            label_map_a[label_key_b] )
            elif merge_type == IWPLabelMergeType.INTERSECTION:
                label_map_a[label_key_b] = intersect_iwp_label( iwp_label_b,
                                                                label_map_a[label_key_b] )
            else:
                raise ValueError( "Duplicate labels encountered for (T={:d}, Z={:d}, id={:s}).".format(
                    label_map_a[label_key_b]["time_step_index"],
                    label_map_a[label_key_b]["z_index"],
                    label_map_a[label_key_b]["id"] ) )
        else:
            # add this label since it is not in the A list.
            label_map_a[label_key_b] = iwp_label_b

    # sort A's label map to get the merged labels.  all of B's labels were added
    # above so we simply need to reorder things so that they're in (time, z, id)
    # order.
    #
    # NOTE: this is a stable sort and preserves the original ordering of A's
    #       labels.
    #
    merged_iwp_labels = sorted( label_map_a.values(),
                                key=lambda label: (label["time_step_index"], label["z_index"], label["id"]) )

    return merged_iwp_labels

def normalize_iwp_label_coordinates( iwp_labels, width, height, in_place_flag=False ):
    """
    Normalizes IWP labels' bounding boxes so they are in the range of [0, 1] instead
    of an arbitrary width and height.

    See also scale_iwp_label_coordinates().

    Takes 4 arguments:

      iwp_labels    - List of IWP labels whose bounding boxes will be normalized.
      width         - Numeric width to normalize the bounding boxes' X coordinates.
      height        - Numeric height to normalize the bounding boxes' Y coordinates.
      in_place_flag - Optional flag specifying in place update or an update to a
                      copy of the labels.  If omitted, defaults to False and a new
                      list of IWP labels is returned.

    Returns 1 value:

      iwp_labels - List of normalized IWP labels.

    """

    if not in_place_flag:
        iwp_labels = copy.deepcopy( iwp_labels )

    for iwp_label in iwp_labels:
        iwp_label["bbox"]["x1"] = iwp_label["bbox"]["x1"] / width
        iwp_label["bbox"]["x2"] = iwp_label["bbox"]["x2"] / width
        iwp_label["bbox"]["y1"] = iwp_label["bbox"]["y1"] / height
        iwp_label["bbox"]["y2"] = iwp_label["bbox"]["y2"] / height

    return iwp_labels

def scale_iwp_label_coordinates( iwp_labels, width, height, in_place_flag=False ):
    """
    Scales IWP labels' bounding boxes by an arbitrary width and height.

    See also normalize_iwp_label_coordinates().

    Takes 4 arguments:

      iwp_labels    - List of IWP labels whose bounding boxes will be scaled.
      width         - Numeric width to scale the bounding boxes' X coordinates.
      height        - Numeric height to scale the bounding boxes' Y coordinates.
      in_place_flag - Optional flag specifying in place update or an update to a
                      copy of the labels.  If omitted, defaults to False and a new
                      list of IWP labels is returned.

    Returns 1 value:

      iwp_labels - List of scaled IWP labels.

    """

    if not in_place_flag:
        iwp_labels = copy.deepcopy( iwp_labels )

    for iwp_label in iwp_labels:
        iwp_label["bbox"]["x1"] = iwp_label["bbox"]["x1"] * width
        iwp_label["bbox"]["x2"] = iwp_label["bbox"]["x2"] * width
        iwp_label["bbox"]["y1"] = iwp_label["bbox"]["y1"] * height
        iwp_label["bbox"]["y2"] = iwp_label["bbox"]["y2"] * height

    return iwp_labels

def flipud_iwp_label_coordinates( iwp_labels, height, in_place_flag=False ):
    """
    Flips a list of IWP labels' Y coordinate about a midpoint to change their origin
    (from top to bottom, and vice versa).  This is useful for working with labels
    coming from tools that do not explicitly specify their coordinate system (e.g.
    Scalabel.ai) so they're usable in tools that do (e.g. ParaView).

    Takes 3 arguments:

      iwp_labels    - List of IWP labels whose bounding boxes will be scaled.
      height        - Numeric height to scale the bounding boxes' Y coordinates.
      in_place_flag - Optional flag specifying in place update or an update to a
                      copy of the labels.  If omitted, defaults to False and a new
                      list of IWP labels is returned.

    Returns 1 value:

      iwp_labels - List of IWP labels with flipped Y coordinates.

    """

    if not in_place_flag:
        iwp_labels = copy.deepcopy( iwp_labels )

    for iwp_label in iwp_labels:
        new_y2 = height - iwp_label["bbox"]["y1"]
        new_y1 = height - iwp_label["bbox"]["y2"]

        iwp_label["bbox"]["y1"] = new_y1
        iwp_label["bbox"]["y2"] = new_y2

    return iwp_labels

def filter_iwp_labels( iwp_labels, time_range=[], z_range=[], identifiers=[] ):
    """
    Filters a list of IWP labels by time range, XY slice range, or by identifier
    name.  IWP labels that match the specified criteria are returned.  Time and
    slice ranges are inclusive.

    Takes 4 arguments:

      iwp_labels  - List of IWP labels to filter.
      time_range  - Optional sequence of start and stop time indices to filter
                    by.  May be specified as an empty sequence to keep labels
                    from any time.  If omitted, defaults to an empty sequence.
      z_range     - Optional sequence of start and stop Z indices to filter by.
                    May be specified as an empty sequence to keep labels with
                    any Z index.  If omitted, defaults to an empty sequence.
      identifiers - Optional list of identifier names to filter by.  May be
                    specified as an empty sequence to keep labels with any
                    identifier.  If omitted, defaults to an empty sequence.

    Returns 1 value:

      filtered_iwp_labels - List of IWP labels that satisfy the criteria provided
                            by time_range, z_range, and identifiers.

    """

    filtered_iwp_labels = []

    for iwp_label in iwp_labels:
        # run the gauntlet of the filters.  only keep the labels that match
        # all of the provided constraints.
        if ((len( time_range ) == 2) and
            not (time_range[0] <= iwp_label["time_step_index"] <= time_range[1])):
            continue
        elif ((len( z_range ) == 2) and
            not (z_range[0] <= iwp_label["z_index"] <= z_range[1])):
            continue
        elif ((len( identifiers ) > 0) and
              iwp_label["id"] not in identifiers):
            continue

        filtered_iwp_labels.append( iwp_label )

    return filtered_iwp_labels

def convert_iwp_bboxes_to_corners( iwp_labels, z_coordinates=None, two_d_flag=False ):
    """
    Converts IWP labels' bounding boxes (upper-left and lower-right corners) to a
    flattened NumPy array containing the four corners.  Each corner is returned
    as a flattened array such that each corner's coordinates are interleaved
    as 12 values like so:

        (x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

    Optionally converts Z indices into Z coordinates if a coordinate axes is provided.

    This routine provides compatibility with systems that require explicit geometry,
    such as ParaView.

    Takes 3 arguments:

      iwp_labels    - List of IWP labels to convert.
      z_coordinates - Optional NumPy array-like containing the Z coordinates.  When
                      specified, each label's Z index is converted to a Z coordinate.
                      If omitted, defaults to None and skips coordinate conversion.
      two_d_flag    - Optional flag specifying whether the output array should be
                      2D or 3D.  If True, bboxes is 2D, otherwise 3D.  If omitted,
                      defaults to False.

    Returns 1 value:

      bboxes - NumPy array, shaped (len( iwp_labels ), N), containing the flattened
               corners.  N is 12 when 3D outputs are requested (two_d_flag == False),
               or 8 when 2D requested outputs are requested (two_d_flag == True).

    """

    # each label generates four (x, y, z) corners: 1) top left, 2) top right,
    # 3) bottom right, and 4) bottom left.
    bboxes = np.empty( (len( iwp_labels ), 12) )

    for label_index, iwp_label in enumerate( iwp_labels ):
        # corner #1 - top left.
        bboxes[label_index, 0] = iwp_label["bbox"]["x1"]
        bboxes[label_index, 1] = iwp_label["bbox"]["y1"]
        bboxes[label_index, 2] = iwp_label["z_index"]

        # corner #2 - top right.
        bboxes[label_index, 3] = iwp_label["bbox"]["x2"]
        bboxes[label_index, 4] = iwp_label["bbox"]["y1"]
        bboxes[label_index, 5] = iwp_label["z_index"]

        # corner #3 - bottom right.
        bboxes[label_index, 6] = iwp_label["bbox"]["x2"]
        bboxes[label_index, 7] = iwp_label["bbox"]["y2"]
        bboxes[label_index, 8] = iwp_label["z_index"]

        # corner #4 - bottom left.
        bboxes[label_index, 9]  = iwp_label["bbox"]["x1"]
        bboxes[label_index, 10] = iwp_label["bbox"]["y2"]
        bboxes[label_index, 11] = iwp_label["z_index"]

    # trim off the z coordinate if requested.
    if two_d_flag:
        bboxes = bboxes[:, 0:2]
    # otherwise, map from z indices to z coordinates when a lookup table is provided.
    elif z_coordinates is not None:
        bboxes[:, 2::3] = z_coordinates[bboxes[:, 2::3].astype( np.int32 )]

    return bboxes
