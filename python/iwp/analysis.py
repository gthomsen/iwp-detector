import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import iwp.labels

# collection of routines to aide in analyzing IWP data.

class FixedScientificFormatter( matplotlib.ticker.ScalarFormatter ):
    """
    Matplotlib tick formatter class for use with dynamic range floating point values
    where the caller knows precisely how their scientific representation should look.

    This class fills the gaps that the matplotlib.ticker.ScalarFormatter class has
    when working with near zero where the parent class dynamically switches the tick
    label format from scientific notation to floating point notation (with variable
    numbers of digits of precision) when the ticks are located "close" to zero.
    Instead, this class fixes the label format at object creation time based on
    caller parameters so as to provide consistent tick labels.

    """

    def __init__( self, order, preferred_format, offset_flag=True, math_text_flag=True ):
        """
        Construct a FixedScientificFormatter object.  Specifies overrides for tick's
        order of magnitudes and formatting string, which are normally dynamically
        computed from the ticks as needed.

        Takes 4 arguments:

          order            - Integer specifying the order of the exponent to use.
          preferred_format - Format string using the Python2 string formatting
                             syntax (e.g. "%1.1f").  Tick values are rendered to labels
                             with this.
          offset_flag      - Optional flag specifying whether tick labels should
                             use offset notation or not.  If specified as True, tick
                             labels are offset relative to the order parameter,
                             otherwise they are scaled by the order.  If omitted,
                             defaults to True.
          math_text_flag   - Optional flag specifying whether tick labels should
                             be rendered with LaTeX formatting.  If omitted, defaults to
                             True.

        Returns 1 value:

          self - Newly constructed FixedScientificFormatter object.

        """

        # initialize our parent's state.
        super().__init__( useOffset=offset_flag, useMathText=math_text_flag )

        # hang on to the formatting labels for later.
        self._order_of_magnitude = order
        self._preferred_format   = preferred_format

    def _set_format( self ):
        """
        Set the format string for the tick labels.  This overrides the
        matplotlib.ticker.ScalarFormatter implementation to force the format string
        specified at construction tiem.

        NOTE: This method ignores the current tick values.

        Takes no arguments.

        Returns no values.

        """

        # use our saved format string.
        self.format = self._preferred_format

        # decorate with LaTeX if we're using mathematical notation.
        if self._useMathText:
             self.format = "$\mathdefault{{{:s}}}$".format( self.format )

    def _set_order_of_magnitude( self ):
        """
        Sets the exponent for formatting ticks with scientific notation.  This overrides
        the matplotlib.ticker.ScalarFormatter implementation to force the order of
        magnitude regardless of the tick locations.

        NOTE: This method ignores the current tick values.

        Takes no arguments.

        Returns no values.

        """

        # use our saved order of magnitude.
        self.orderOfMagnitude = self._order_of_magnitude

def iwp_labels_to_rectangles( iwp_labels, grid_extents, label_color=None, line_width=2 ):
    """
    Converts IWP label bounding boxes to Matplotlib patches.Rectangles and
    coordinates suitable for addition to an existing axes.

    Takes 4 arguments:

      iwp_labels   - List of IWP labels to overlay on a Matplotlib axes.
      grid_extents - Data coordinates of the XY slice.  Sequence of two sequences, each
                     with a pair of elements specifying the (min, max) for the X and Y
                     axes (i.e. (min_x, max_x), (min_y, max_y)).  If omitted, defaults
                     to None and the XY slice axes are labeled in indices.
      label_colors - Optional Matplotlib-compatible label color.  May be a color string
                     (by English name, by Matplotlib code, by hex, etc) or a color tuple
                     (RGB or RGBA).  If omitted, defaults to a high contrast color.
      line_width   - Optional line width, in pixels, for the labels.  If omitted
                     defaults to 2.

    Returns 2 values:

      rectangles  - List of matplotlib.patches.Rectangles corresponding to the
                    iwp_labels.
      coordinates - List of coordinate lists corresponding to the iwp_labels.  Each
                    entry is a list of four coordinates specifying [anchor_x,
                    anchor_y, offset_x, offset_y].

    """

    # rename our grid extents so the code below is easier to read.
    #
    # NOTE: we don't just unpack in case we're called with a 3D grid extent.
    #
    (grid_x, grid_y) = (grid_extents[0], grid_extents[1])

    # if the caller doesn't have a preference, render labels as opaque magenta.
    # this has a high likelihood of having high contrast relative to the
    # underlying color map.
    if label_color is None:
        label_color = (1.0, 0.0, 1.0)

    rectangles  = []
    coordinates = []

    # list of IWP labels whose anchor points reside outside of the data
    # coordinates.  we track all of the labels that are invisible since it
    # likely indicates a coordinate system mismatch (label incorrectly stored in
    # data coordinates rather than normalized, resulting in positions in data
    # coordinates^2).  we flag these to the user to take corrective action
    # rather than silently deleting them.
    labels_out_of_bounds = []

    for iwp_label in iwp_labels:
        # convert the bounding box (top left, lower right) into (anchor, offset)
        # and transform from normalized coordinates into the grid coordinates
        # provided.
        current_coordinates = [iwp_label["bbox"]["x1"] * grid_x[-1],
                               iwp_label["bbox"]["y1"] * grid_y[-1],
                               (iwp_label["bbox"]["x2"] - iwp_label["bbox"]["x1"]) * grid_x[-1],
                               (iwp_label["bbox"]["y2"] - iwp_label["bbox"]["y1"]) * grid_y[-1]]

        # keep track of labels that are beyond the coordinate system boundaries.
        # these won't be displayed under normal circumstances, so we warn the
        # user something is awry so they can debug their coordinate systems.
        if (current_coordinates[0] > grid_x[-1] or
            current_coordinates[1] > grid_y[-1]):
            labels_out_of_bounds.append( iwp_label )

        current_rectangle = patches.Rectangle( current_coordinates[:2],
                                               current_coordinates[2],
                                               current_coordinates[3],
                                               linewidth=line_width,
                                               edgecolor=label_color,
                                               facecolor="none" )

        rectangles.append( current_rectangle )
        coordinates.append( current_coordinates )

    # let the user know if something went wrong.
    if len( labels_out_of_bounds ) > 0:
        import warnings

        warnings.warn( "{:d} label{:s} out of bounds!".format(
            len( labels_out_of_bounds ),
            "" if len( labels_out_of_bounds ) == 1 else "s" ) )

    return rectangles, coordinates

def show_xy_slice( ax_h, slice_data, variable_name, grid_extents=None, color_map=cm.bwr, quantization_table=None, iwp_labels=[], label_color=None, rotate_flag=False, colorbar_flag=True, colorbar_formatter=None ):
    """

    Renders an XY slice via Matplotlib's imshow() and decorates it so it is easily
    interpreted.  A colorbar is added and the title is set to the supplied variable
    name, while the XY slice's extents can be set if supplied.  The image rendered
    follows IWP conventions and has the origin in the lower left.

    Takes 12 arguments:

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
      iwp_labels         - Optional list of IWP labels to overlay.  If omitted, defaults
                           to an empty list and nothing is overlaid.
      label_color        - Optional Matplotlib-compatible label color.  May be a color
                           string (by English name, by Matplotlib code, by hex, etc)
                           or a color tuple (RGB or RGBA).  If omitted, defaults to a
                           high contrast color.
      rotate_flag        - Optional flag specifying whether slice_data should be rotated
                           before calling imshow().  If omitted, defaults to False.
      colorbar_flag      - Optional flag specifying whether a colorbar should be added
                           to the supplied axes or not.  If omitted, defaults to True.
      colorbar_formatter - Optional matplotlib.ticker.Formatter-derived tick formatter,
                           or a string constructor for the matplotlib.ticker.StrMethodFormatter
                           class.  Controls the formatting of the colorbar's tick
                           labels.  Has no effect when colorbar_flag is False.  If
                           omitted, defaults to None and uses the default colorbar
                           tick label formatting.

    Returns 1 value:

      image_h - Image handle returned from imshow().

    """

    # default to an index coordinate system if we weren't provided a data
    # coordinate system.
    if grid_extents is None:
        #
        # NOTE: mind the transition from matrix (Y, X) coordinates to cartesian
        #       coordinates (X, Y).
        #
        grid_extents = ((0, slice_data.shape[1]),
                        (0, slice_data.shape[0]))

    # if the caller doesn't have a preference, render labels as opaque magenta.
    # this has a high likelihood of having high contrast relative to the
    # underlying color map.
    if label_color is None:
        label_color = (1.0, 0.0, 1.0)

    # handle the case where we're working with 3D volumes and supply the
    # same extents data structure.
    if len( grid_extents ) == 3:
        (grid_x, grid_y, _) = grid_extents
    else:
        (grid_x, grid_y) = grid_extents

    # rotate the world by 90 degrees if requested.
    if rotate_flag:
        slice_data     = slice_data.T
        grid_x, grid_y = grid_y, grid_x

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
    slice_h = ax_h.imshow( slice_data,
                           extent=[grid_x[0], grid_x[-1],
                                   grid_y[0], grid_y[-1]],
                           cmap=color_map,
                           norm=normalizer,
                           origin="lower" )

    # convert our label bounding boxes (top left, bottom right) to
    # (anchor, offset)'s.
    label_rectangles, label_coordinates = iwp_labels_to_rectangles( iwp_labels,
                                                                    grid_extents,
                                                                    label_color=label_color )

    # compute the size of a single pixel in data coordinate space.  we use these
    # below to adjust the labels' text names so they're readable is most (all?)
    # cases.
    coordinate_epsilon = (((1 / slice_data.shape[1]) *
                           (grid_x[-1] - grid_x[0]) +
                           grid_x[0]),
                          ((1 / slice_data.shape[0]) *
                           (grid_y[-1] - grid_y[0]) +
                           grid_y[0]))

    for label_index in range( len( iwp_labels ) ):
        # add this label's outline to the axes.
        ax_h.add_patch( label_rectangles[label_index] )

        # overlay the label name (1st six characters) so that it is slightly
        # above, and to the right, the top of each label's upper left corner.
        # take care such that the name is always visible even if the label is at
        # the top or right of the image (the name is moved inside the label in
        # those cases).
        #
        # NOTE: the name is positioned relative to the data coordinates whose
        #       origin is in the bottom left.
        #
        # NOTE: the hardcoded scale factors below were empirically-derived and
        #       should be replaced with something a) correct and b) explainable.
        #

        # horizontally position the name 2 pixels inside the left edge of the
        # label, unless we're at the slice right hand side, in which case
        # position it 32 pixels inside the edge (should be enough space for a
        # label with all capital letters).
        label_name_x_coordinate = min( (label_coordinates[label_index][0] +
                                        2 * coordinate_epsilon[0]),
                                       grid_x[-1] - 32 * coordinate_epsilon[0] )

        # vertically position the name 2 pixels above the top edge of the label,
        # unless we're at the top of the slice, in which case position it 8
        # pixels below the top.
        #
        # NOTE: this placement is problematic for short labels at the edge.
        #       luckily, this is typically where the sponge layer is so we
        #       don't have to worry too much about it.
        #
        label_name_y_coordinate = min( (label_coordinates[label_index][1] +
                                        label_coordinates[label_index][3] +
                                        2 * coordinate_epsilon[1]),
                                       grid_y[-1] - 8 * coordinate_epsilon[1] )

        ax_h.text( label_name_x_coordinate,
                   label_name_y_coordinate,
                   iwp.labels.get_iwp_label_name( iwp_labels[label_index],
                                                  shortened_flag=True ),
                   color=label_color )

    # add a colorbar.
    if colorbar_flag:
        #
        # NOTE: this approach of adding a colorbar axes via a divider limits
        #       what we can do with respect to positioning and size.  as a
        #       result, some colorbar_formatters (e.g. those that use a
        #       ScalarFormatter to get scientific notation scales/orders of
        #       magnitude at the top of the colorbar) may result in missing
        #       labels depending on the parent figure's layout.
        #
        #       it appears that any modification to the colorbar axes position
        #       may be discarded at a later date due to
        #       the parent figure's layout being updaated for rendering
        #       (particularly constrained layouts - the axes position is
        #       seemingly reset when a call to show() or a canvas draw()).
        #
        #       it does not appear there are any great solutions to creating
        #       colorbars that have a smaller size, so that decorations
        #       above/below them don't fall outside of the rendered figure's
        #       boundaries.
        #
        divider = make_axes_locatable( ax_h )
        cax_h   = divider.append_axes( "right", size="5%", pad=0.05 )

        plt.colorbar( slice_h,
                      cax=cax_h,
                      format=colorbar_formatter )

        # align the colorbar y axes labels along the left side of it.  this
        # allows labels located above the colorbar to be flush along the left
        # side (so they overhang the right side) rather than being flush along
        # the right side (so they overhang the left side and possibly conflict
        # with a title)
        cax_h.yaxis.set_offset_position( "left" )

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

def variable_name_to_title( variable_name, latex_flag=True ):
    """
    Translates a variable name into a title suitable for inclusion in Matplotlib
    title strings.  Variable names are assumed to be lowercased as found in IWP
    datasets and titles may include LaTeX markers for mathematical typesetting.
    Unknown variable names are returned as is.

    Takes 2 arguments:

      variable_name - Variable name to translate.
      latex_flag    - Optional flag specifying whether LaTeX-encodings should be
                      used in the translation.  If specified as False, translations
                      will not use LaTeX.  If omitted, defaults to True.

    Returns 1 value:

      variable_title - Translated variable title.

    """

    if variable_name == "divh":
        return "Horizontal Divergence"
    elif variable_name == "p":
        return "Density"
    elif variable_name == "pprime":
        if latex_flag:
            return "Density$'$"
        else:
            return "Density'"
    elif variable_name == "u":
        if latex_flag:
            return "$velocity_x$"
        else:
            return "Velocity - X"
    elif variable_name == "uprime":
        if latex_flag:
            return "$velocity_x'$"
        else:
            return "Acceleration - X"
    elif variable_name == "v":
        if latex_flag:
            return "$velocity_y$"
        else:
            return "Velocity - Y"
    elif variable_name == "vprime":
        if latex_flag:
            return "$velocity_y'$"
        else:
            return "Acceleration - Y"
    elif variable_name == "w":
        if latex_flag:
            return "$velocity_z$"
        else:
            return "Velocity - Z"
    elif variable_name == "wprime":
        if latex_flag:
            return "$velocity_z'$"
        else:
            return "Acceleration - Z"
    elif variable_name == "vortx":
        if latex_flag:
            return "$vorticity_x$"
        else:
            return "Vorticity - X"
    elif variable_name == "vorty":
        if latex_flag:
            return "$vorticity_y$"
        else:
            return "Vorticity - Y"
    elif variable_name == "vortz":
        if latex_flag:
            return "$vorticity_z$"
        else:
            return "Vorticity - Z"
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

            if latex_flag:
                # add "+-N degrees" in LaTeX and then append the remaining
                # components as a suffix.
                variable_title = "{:s}$\pm{:s}\circ${:s}".format(
                    variable_title,
                    pieces[0],
                    suffix )
            else:
                variable_title = "{:s} +-{:s} degrees{:s}".format(
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
