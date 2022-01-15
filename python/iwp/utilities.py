import enum
import numpy as np
import xarray

import iwp.data_loader

# enumeration of color system types.  not all color systems are equal and both
# Matplotlib and PIL do their own things.  some portions (e.g. named colors and
# hex codes) overlap, though the distinct portions are large enough to require
# knowing where they come from so they can be properly validated.
#
#   matplotlib  - color specifications adhere to the "Matplotlib Specifying Colors"
#                 documentation found here:
#
#                    https://matplotlib.org/stable/tutorials/colors/colors.html
#
#   pil         - color specifications adhere to PIL.ImageColor's implementation.
#                 see that class' documentation for details.
@enum.unique
class ColorSystemType( enum.Enum ):
    MATPLOTLIB = 0
    PIL        = 1

def indices_to_regions( indices, is_sorted_flag=True ):
    """
    Creates a compact representation of the contiguous regions contained within a
    list of indices.  This is useful for succintly describing a sequence instead
    of enumerating all of its contents.

    For example, the following 10 inputs can be described as four distinct regions:

      Input:  [1, 2, 3, 5, 6, 8, 10, 11, 12, 13]
      Output: [(1, 3), (5, 6), (8, 8), (10, 13)]

    NOTE: This only properly handles inputs with repeated indices when the
          inputs are sorted.

    Takes 2 arguments:

      indices        - List or Array of integral indices to identify regions in.
      is_sorted_flag - Optional flag specifying whether indices are already sorted.
                       If specified as True, sorting the indices is skipped since the
                       caller took care of it.  If omitted, defaults to True.

    Returns 1 value:

      regions - List of tuples specifying a compact representation of the regions
                present in indices.  Each tuple contains the first and last index
                of a region of consecutive indices.  Tuples may describe only a
                single number (i.e. first and last are equal) for singleton regions.
                regions is an empty list when indices is.

    """

    # handle the case were we don't have any regions.
    if len( indices ) == 0:
        return []

    # ensure that we have have monotonically
    if not is_sorted_flag:
        indices = np.sort( indices )

    # take the difference between adjacent indices so we can find discontinuities.
    index_differences = np.diff( indices )

    # regions are separated by differences larger than 1.  book end the locations
    # in the original array with the first and last index so we have fence posts
    # of every region.  fence posts land on the first element of each region, except
    # the last which is one beyond the number of entries.
    #
    # for the following 10 inputs:
    #
    #  [1, 2, 3, 5, 6, 8, 10, 11, 12, 13]
    #
    # we compute the following 5 fence post indices:
    #
    #  [0, 3, 5, 6, 10]
    #
    # which represent the 4 regions:
    #
    #  [(1, 3), (5, 6), (8, 8), (10, 13)]
    #
    # NOTE: because of the difference above, we start at index -1 instead of 0 and
    #       adjust below.
    #
    region_indices = np.concatenate( (np.array( [-1] ),
                                      np.where( index_differences > 1 )[0],
                                      np.array( [len( indices ) - 1] )) )
    region_indices += 1

    # build a list of region indices.  for N + 1 fenceposts, we have N regions that
    # we get the (start, end) indices for.
    regions = []
    for region_number in range( 1, len( region_indices ) ):

        # we start on the previous region's fence post and end one
        # before this region's fence post.
        regions.append( [indices[region_indices[region_number - 1]],
                         indices[region_indices[region_number] - 1]] )

    return regions

def parse_range( range_string ):
    """
    Parse a range object from a string of the form:

      <start>:<stop>[:<step>]

    No validation is performed on <start>, <stop>, <step>.

    Takes 1 argument:

      range_string -

    Returns 1 value:

      range_object - range() object.

    """

    components = list( map( int, range_string.split( ":" ) ) )

    if len( components ) == 1:
        components = [components[0], components[0]]

    # adjust the end so it is inclusive rather than following Python's exclusive
    # semantics.  this makes things less surprising to end users.
    components[1] += 1

    try:
        range_object = range( *components )
    except:
        range_object = None

    return range_object

def validate_variables_and_ranges( dataset, variable_names, time_step_indices, xy_slice_indices ):
    """
    Validates a set of variable names, time step indices, and XY slice indices are
    compatible with a particular dataset.  Access and indexing by the supplied
    parameters must be compatible for validation to successful.

    Raises ValueError if the supplied parameters are incompatible with the dataset
    provided.

    Takes 4 arguments:

      dataset           - xarray.Dataset or xarray.DataArray to validate the supplied
                          parameters against.
      variable_names    - List of variable names to validate are present in dataset.
      time_step_indices - List of time step indices to validate are present in dataset.
      xy_slice_indices  - List of XY slice indices to validate are present in dataset.

    Returns nothing.

    """

    if isinstance( dataset, (xarray.Dataset, xarray.DataArray) ):
        truth_time_step_indices = dataset.coords["Cycle"]
        truth_xy_slice_indices  = dataset.coords["z"]
        truth_variable_names    = dataset.data_vars
    elif isinstance( dataset, iwp.data_loader.IWPDataset ):
        truth_time_step_indices = dataset.time_step_indices()
        truth_xy_slice_indices  = dataset.xy_slice_indices()
        truth_variable_names    = dataset.variables()
    else:
        raise ValueError( "Unknown type of dataset supplied ({:s})!".format(
            type( dataset ) ) )

    # verify that each of the time step indices provided is the data value of
    # at least one cycle.
    for time_step_index in time_step_indices:
        if time_step_index not in truth_time_step_indices:
            raise ValueError( "Time step index {:d} is not present in the dataset.".format(
                time_step_index ) )

    # verify that the XY slice indices map to a valid Z slice.
    for xy_slice_index in xy_slice_indices:
        if xy_slice_index > len( truth_xy_slice_indices ):
            raise ValueError( "XY slice index {:d} is not present in the dataset.".format(
                xy_slice_index ) )
        elif xy_slice_index < 0:
            raise ValueError( "XY slice indices cannot be negative ({:d}).".format(
                xy_slice_index ) )

    # verify that each variable requested is a grid variable.  we explicitly
    # check for membership in .data_vars rather than .variables to avoid false
    # positives when coordinate names are provided.
    for variable_name in variable_names:
        if variable_name not in truth_variable_names:
            raise ValueError( "'{:s}' is not present in the dataset.".format(
                variable_name ) )

    return

def get_xarray_subset( dataset, variable_name, time_step_indices, xy_slice_indices ):
    """
    Returns a subset of the provided dataset.  The source dataset is restricted to the
    variable and indices supplied.  When contiguous time step and XY slice indices are
    supplied, the resulting DataArray is a view of the original.

    Takes 4 arguments:

      dataset           - xarray.Dataset or xarray.DataArray object to return a view
                          from.
      variable_name     - Variable to restrict the view to.
      time_step_indices - Time step values to restrict the view to.
      xy_slice_indices  - XY slice indices to restrict the view to.

    Returns 1 value:

      data_array - xarray.DataArray object representing a subset of dataset.

    """

    # create a new DataArray from the indices provided.
    #
    # NOTE: we select time steps by value and XY slices by array index.  XY
    #       slices are generated contiguously, so array index maps to the slice
    #       index for the vast majority (all?) of use cases.  netCDF4 datasets
    #       may be constructed from multiple files selected via file name
    #       globbing, meaning time steps may not increase monotonically, meaning
    #       we should select time steps by value.
    #
    data_array = (dataset[variable_name]
                  .sel( Cycle=time_step_indices )
                  .isel( z=xy_slice_indices ))

    return data_array

def lookup_module_function( module_reference, function_name ):
    """
    Acquires a function reference from a module given an function name.

    Takes 2 arguments:

      module_reference - Module whose interface is searched for function_name.
      function_name    - String specifying the function whose handle is sought.

    Returns 1 value:

      function_reference - Reference to the requested function.  None if function_name
                           is not part of module_reference's interface or if there was
                           an error acquiring the reference.
    """

    # get a reference to the function requested.  return None if it isn't part
    # of the module's interface.
    function_reference = getattr( module_reference,
                                  function_name,
                                  None )

    return function_reference

def _normalize_iwp_color_like( color_like ):
    """
    Normalizes an IWP string color specification into a RGB(A) tuple.  Accepts
    colon-delimited strings specifying either integrals values in the range of [0, 255]
    or floating point values in the range of [0, 1] and returns a tuple of integer
    RGB(A) values in the range of [0, 255].

    Raises ValueError when the supplied color specification does not represent an RGB
    or RGBA value.

    Takes 1 value:

      color_like - Colon-delimited string of numeric values, in the range of [0, 255],
                   representing a color.  Must contain either three or four components
                   to specify a color via RGB or RGBA, respectively.

    Returns 1 value:

      normalized_color_like - Tuple, with either three or four integer components (RGB
                              and RGBA, respectively), representing color_like.

    """

    # check to see if we got a RGB(A) triplet.
    try:
        color_components = tuple( map( lambda x: float( x ),
                                       color_like.split( ":" ) ) )
    except:
        raise ValueError( "'{}' is not a colon delimited list of numeric values.".format(
            color_like ) )

    # ensure that we are either RGB or RGBA.
    if (len( color_components ) < 3) or (len( color_components ) > 4):
        raise ValueError( "'{:s}' does not look like a RGB(A) tuple.  Expected 3 "
                          "or 4 components, but received {:d}.".format(
                              color_like,
                              len( color_components ) ) )

    # ensure each of the values are at least in [0, 255].
    if any( map( lambda x: (x < 0) or (x > 255), color_components ) ):
        raise ValueError( "'{:s}' does not look like a valid RGB(A) tuple.  "
                          "One or more of the components are outside of [0, 255].".format(
                              color_like ) )

    # translate floating point values to uint8 by scaling from [0, 1] to
    # [0, 255].
    if all( map( lambda x: (x >= 0.) and (x <= 1.0), color_components ) ):
        normalized_color_like = tuple( map( lambda x: int( x * 255.0 ), color_components ) )
    else:
        # warn about any floating point values in [0, 255] while helping the
        # caller get integral values in the range.
        if any( map( lambda x: x != float( int( x ) ), color_components ) ):
            import warnings
            warnings.warn( "'{:s}' is a non-integral, floating point RGB(A) tuple in [0, 255].  "
                           "Fixing for PIL-compatibility.".format(
                               color_like ) )

        # map everything back to integers for PIL compatibility.
        normalized_color_like = tuple( map( lambda x: int( x ), color_components ) )

    return normalized_color_like

def normalize_color_like( color_like, validator_type ):
    """
    Normalizes a color-like string specification into its RGB(A) values.  Colors
    are validated and converted according to first the requested system
    (e.g. Matplotlib or PIL) and then against the IWP system as a fall back.
    This allows colors to be specified in a myriad of ways (e.g. "red", "#ff0000",
    "r") according to the larger system, but also specified in a command-line friendly,
    colon-separated list of values for IWP.

    Raises ValueError when the supplied color specification does not represent an
    RGBA(A) value either in the requested color system or IWP.

    Takes 2 arguments:

      color_like     - matplotlib color spec, PIL color spec, or colon-delimited
      validator_type - Enumeration of type iwp.utilities.ColorSystemType that specifies
                       which color system to validate color_like against.

    Returns 1 value:

      normalized_color_like - Tuple, with either three or four components (RGB and
                              RGBA, respectively), representing color_like.  Components
                              will be in the range of [0, 255] if validator_type is
                              PIL, otherwise in the range of [0, 1].

    """

    # ensure we have a known color system.
    if ((validator_type != ColorSystemType.MATPLOTLIB) and
        (validator_type != ColorSystemType.PIL)):
        raise ValueError( "Unknown color system specified ({}).".format(
            validator_type ) )

    # are we validating against Matplotlib's colors?
    if validator_type == ColorSystemType.MATPLOTLIB:
        import matplotlib.colors

        # see if the color is a string based color (e.g. "#ff0000", "r", "red",
        # "xkcd:red", etc - see 'Matplotlib Specifying Colors" for more
        # details).
        if matplotlib.colors.is_color_like( color_like ):
            normalized_color_like = matplotlib.colors.to_rgba( color_like )
        else:
            # fall back and see if we got an IWP color specification.
            try:
                normalized_color_like = _normalize_iwp_color_like( color_like )

                # map back to [0, 1] to keep Matplotlib happy.
                normalized_color_like = tuple( map( lambda x: x / 255.0,
                                                    normalized_color_like ) )
            except ValueError as e:
                raise ValueError( "'{}' is neither a Matplotlib nor IWP color specification ({:s}).".format(
                    color_like,
                    str( e ) ) )

        return normalized_color_like

    # or are we validating against PIL's colors?
    if validator_type == ColorSystemType.PIL:
        import PIL.ImageColor

        try:
            # see if the color is a string-based color (e.g. "red", "#ff0000",
            # "rgb( 127, 127, 127 )", "hsv( 128, 90%, 40%)", etc - see
            # PIL.ImageColor for more details).
            normalized_color_like = PIL.ImageColor.getrgb( color_like )
        except:
            # fall back and see if we got an IWP color specification.
            try:
                normalized_color_like = _normalize_iwp_color_like( color_like )
            except ValueError as e:
                raise ValueError( "'{}' is neither a PIL nor IWP color specification ({:s}).".format(
                    color_like,
                    str( e ) ) )

        # we've got something that is PIL-compatible.  return it to the caller.
        return normalized_color_like
