import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

# collection of routines to aide in analyzing IWP data.

def show_xy_slice( ax_h, slice_data, variable_name, grid_extents=None, color_map=cm.bwr, quantization_table=None, rotate_flag=False, colorbar_flag=True ):
    """

    Renders an XY slice via Matplotlib's imshow() and decorates it so it is easily
    interpreted.  A colorbar is added and the title is set to the supplied variable
    name, while the XY slice's extents can be set if supplied.  The image rendered
    follows IWP conventions and has the origin in the lower left.

    Takes 8 arguments:

      ax_h               - Axes handle to supply to imshow().
      slice_data         - 2D NumPy array containing the XY slice data, shaped (Y, X).
                           If rotate_flag == True, this is shaped (X, Y).
      variable_name      - String to use as the plot's title describing slice_data.
      grid_extents       - Optional tuple containing the extents for the X and Y
                           dimensions, with each entry containing a tuple specifying
                           the (min, max) for said dimension.  May be specified as
                           either a pair containing (X, Y) extents or a triple
                           containing (X, Y, Z) extents.  If omitted, the plot axes
                           are labeled with slice_data's dimensions.
      color_map          - Optional Matplotlib colormap to render slice_data with.
                           May be specified as anything that imshow()'s cmap argument
                           accepts. If omitted, defaults to matplotlib.cm.bwr.
      quantization_table - Optional quantization table to apply to slice_data.  Must
                           be compatible with NumPy's digitize() function.  If omitted,
                           defaults to None and a linear quantization between slice_data's
                           minimum and maximum is used.
      rotate_flag        - Optional flag specifying whether slice_data should be rotated
                           before calling imshow().  If omitted, defaults to False.
      colorbar_flag      - Optional flag specifying whether a colorbar should be added
                           to the supplied axes or not.  If omitted, defaults to True.

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

    # provide a mapping of the slice data to [0, 1].  use our quantization
    # table if we have one, otherwise fall back to linear between the minimum
    # and maximum.
    if quantization_table is None:
        normalizer = colors.Normalize( vmin=slice_data.min(),
                                       vmax=slice_data.max(),
                                       clip=True )
    else:
        normalizer = colors.BoundaryNorm( boundaries=quantization_table,
                                          ncolors=quantization_table.shape[0] )

    # plot the slice.
    if grid_extents is not None:
        slice_h = ax_h.imshow( slice_data,
                               extent=[grid_x[0], grid_x[-1],
                                       grid_y[0], grid_y[-1]],
                               cmap=color_map,
                               norm=normalizer,
                               origin="lower" )
    else:
        slice_h = ax_h.imshow( slice_data,
                               cmap=color_map,
                               norm=normalizer,
                               origin="lower" )

    # add a colorbar.
    if colorbar_flag:
        divider = make_axes_locatable( ax_h )
        cax_h   = divider.append_axes( "right", size="5%", pad=0.05 )
        plt.colorbar( slice_h, cax=cax_h )

    ax_h.set_title( variable_name,
                    fontweight="bold" )

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

def variable_name_to_title( variable_name ):
    """
    Translates a variable name into a title suitable for inclusion in Matplotlib
    title strings.  Variable names are assumed to be lowercased as found in IWP
    datasets and titles may include LaTeX markers for mathematical typesetting.
    Unknown variable names are returned as is.

    Takes 1 argument:

      variable_name - Variable name to translate.

    Returns 1 value:

      variable_title - Translated variable title.

    """

    if variable_name == "divh":
        return "Horizontal Divergence"
    elif variable_name == "p":
        return "Density"
    elif variable_name == "pprime":
        return "Density$'$"
    elif variable_name == "u":
        return "$velocity_x$"
    elif variable_name == "uprime":
        return "$velocity_x'$"
    elif variable_name == "v":
        return "$velocity_y$"
    elif variable_name == "vprime":
        return "$velocity_y'$"
    elif variable_name == "w":
        return "$velocity_z$"
    elif variable_name == "wprime":
        return "$velocity_z'$"
    elif variable_name == "vortx":
        return "$vorticity_x$"
    elif variable_name == "vorty":
        return "$vorticity_y$"
    elif variable_name == "vortz":
        return "$vorticity_z$"
    elif variable_name.startswith( "morlet" ):
        # Morlet wavelets have an angle preference which is encoded as either
        # "morlet+-angle", "morlet+angle", or "morlet-angle".  handle the
        # plus/minus case as special and let the positive/negative, single
        # angles fall through like normal text.

        variable_title = "2D CWT with Morlet"

        # decorate the base title depending on the format of the rest of the
        # variable.
        pm_index = variable_name.find( "+-" )
        if pm_index != -1:
            #
            # NOTE: we have to filter this as the non-default split parameter
            #       will *not* filter out empty strings...
            #
            pieces = list( filter( lambda piece: len( piece ) > 0,
                                   variable_name[pm_index+2:].split( "-" ) ) )

            # the first piece is the angle.  add the remaining pieces the
            # way we found them.
            if len( pieces ) == 1:
                suffix = ""
            else:
                suffix = " ({:s})".format( "-".join( pieces[1:] ) )

            # add "+-N degrees" in LaTeX and then append the remaining
            # components as a suffix.
            variable_title = "{:s}$\pm{:s}\circ${:s}".format(
                variable_title,
                pieces[0],
                suffix )
        elif len( variable_name ) > len( "morlet" ):
            # add the remaining text as a parenthetical.
            variable_title = "{:s} ({:s})".format(
                variable_title,
                variable_name[len( "morlet" ):] )

        return variable_title
    elif variable_name.startswith( "arc" ):
        return "2D CWT with Arc"
    elif variable_name.startswith( "halo" ):
        return "2D CWT with Halo"

    # we don't have a special title for this variable.  use what we have.
    return variable_name
