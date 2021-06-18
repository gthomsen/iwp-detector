import collections
import copy
import enum
import json

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

    # sort the labels by (time step, z_index, id) to make it easier to review.
    sorted_iwp = copy.copy( iwp_labels )
    sorted_iwp.sort( key=lambda label: (label["time_step_index"], label["z_index"], label["id"]) )

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
