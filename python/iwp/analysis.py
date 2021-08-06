import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

# collection of routines to aide in analyzing IWP data.

def show_xy_slice( ax_h, slice_data, variable_name, grid_extents=None, color_map=cm.bwr, rotate_flag=False ):
    """

    Renders an XY slice via Matplotlib's imshow() and decorates it so it is easily
    interpreted.  A colorbar is added and the title is set to the supplied variable
    name, while the XY slice's extents can be set if supplied.  The image rendered
    follows IWP conventions and has the origin in the lower left.

    Takes 6 arguments:

      ax_h          - Axes handle to supply to imshow().
      slice_data    - 2D NumPy array containing the XY slice data, shaped (Y, X).  If
                      rotate_flag == True, this is shaped (X, Y).
      variable_name - String to use as the plot's title describing slice_data.
      grid_extents  - Optional tuple containing the extents for the X and Y dimensions,
                      with each entry containing a tuple specifying the (min, max) for
                      said dimension.  May be specified as either a pair containing
                      (X, Y) extents or  a triple containing (X, Y, Z) extents.  If
                      omitted, the plot axes are labeled with slice_data's dimensions.
      color_map     - Optional Matplotlib colormap to render slice_data with.  May
                      be specified as anything that imshow()'s cmap argument accepts.
                      If omitted, defaults to matplotlib.cm.bwr.
      rotate_flag   - Optional flag specifying whether slice_data should be rotated
                      before calling imshow().  If omitted, defaults to False.

    Returns 1 value:

      image_h - Image handle returned from imshow().

    """

    # unpack the extents if we were supplied them.
    if grid_extents is not None:
        # handle the case where we're working with 3D volumes and supply the
        # same extents data structure.
        if len( grid_extents ) == 3:
            (grid_x, grid_y, _) = grid_extents
        else:
            (grid_x, grid_y) = grid_extents

        if rotate_flag:
            grid_x, grid_y = grid_y, grid_x

    if rotate_flag:
        slice_data = slice_data.T

    # plot the slice.
    if grid_extents is not None:
        slice_h = ax_h.imshow( slice_data,
                               extent=[grid_x[0], grid_x[-1],
                                       grid_y[0], grid_y[-1]],
                               cmap=color_map,
                               origin="lower" )
    else:
        slice_h = ax_h.imshow( slice_data,
                               cmap=color_map,
                               origin="lower" )

    # add a colorbar.
    divider = make_axes_locatable( ax_h )
    cax_h   = divider.append_axes( "right", size="5%", pad=0.05 )
    plt.colorbar( slice_h, cax=cax_h )

    ax_h.set_title( variable_name )

    return slice_h

def find_nearest( data, value ):
    """
    Finds the nearest value in an array and returns it and its location.

    Takes 2 arguments:

      data  - NumPy Array-like to search for value.
      value - Target value to search for in data.

    Returns 2 values:

      nearest_value - Closest value to value found in data.
      nearest_index - Index of nearest_value in data.

    """

    # coerce the data into an actual array.
    data = np.asarray( data )

    value_index = (np.abs( data - value )).argmin()

    return data[value_index], value_index
