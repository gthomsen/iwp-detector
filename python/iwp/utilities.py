import xarray

import iwp.data_loader

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
