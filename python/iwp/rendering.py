import io
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from PIL import Image, ImageDraw

import iwp.analysis
import iwp.labels
import iwp.statistics
import iwp.utilities

def array_to_pixels( array, quantization_table, color_map, scaler=1 ):
    """
    Quantizes a NumPy array of data, applies a color map, and converts the result to
    uint8.  The intermediate quantized data may optionally be scaled to better use
    the 8-bit range.

    Takes 4 arguments:

      array              - NumPy array of data to convert to pixels.  The data type
                           must be compatible with NumPy's digitize() function.
      quantization_table - Quantization table to apply to array.  Must be compatible
                           with NumPy's digitize() function.
      color_map          - Matplotlib color map to apply.
      scaler             - Optional scalar floating point to scale the quantized data
                           before color_map is applied.  If omitted, defaults to 1.0
                           so that the quantized data are used as is.

    Returns 1 value:

      data_pixels - NumPy array of colored, quantized, uint8 pixels corresponding to array.
                    Retains the same shape as array when color_map is an identity function,
                    otherwise adds a new inner dimension of length 4 to represent the
                    RGBA channels.

    """

    # quantize the data.  this generates int64's by default.
    data_slice = np.digitize( array,
                              quantization_table ).astype( np.float32 )

    # scale the data so it uses more of the pixels' available range.
    if scaler > 1:
        data_slice = data_slice * np.float32( scaler )

    # map into [0, 1] to apply the colormap, then back to [0, 255] before
    # casting to uint8.
    data_pixels = np.uint8( color_map( data_slice / 255.0 ) * 255.0 )

    return data_pixels

def array_to_image_PIL( array, quantization_table, color_map, iwp_labels=[], label_color=None, indexing_type="ij", title_text="", **kwargs ):
    """
    Converts a NumPy array to a PIL Image, and optionally burns in a title to
    the top of the image.  The supplied array is quantized and colorized prior
    to conversion to a PIL Image.

    Raises ValueError if the requested indexing method is unknown.

    Takes 8 arguments:

      array              - NumPy array of data to convert to pixels.  The data type
                           must be compatible with NumPy's digitize() function.
      quantization_table - Quantization table to apply to array.  Must be compatible
                           with NumPy's digitize() function.
      color_map          - Matplotlib color map to apply.
      iwp_labels         - Optional list of IWP labels to overlay.  If omitted, defaults
                           to an empty list and nothing is overlaid.
      label_color        - Optional PIL-compatible label color.  May be a color string
                           (by name, by hex, etc) or a color tuple (RGB or RGBA).
                           Color tuples may be normalized color values (in [0, 1]),
                           even though they're not natively supported by PIL.  If
                           omitted, defaults to a high contrast color.
      indexing_type      - Optional string specifying array's indexing method.  Must
                           be either "xy" (origin in top left) or "ij" (origin in
                           bottom left).  See numpy.meshgrid() for a detailed
                           description of indexing types.  If omitted, defaults to
                           "ij" to match IWP visualization conventions.
      title_text         - Optional title string to burn into the generated image.
                           If omitted, the image is created without alteration.
      kwargs             - Optional keyword arguments dictionary.  Accepted for
                           compatibility with array_to_image()'s calling convention.

    Returns 1 value:

      image - PIL Image created from array's data.

    """

    # if the caller doesn't have a preference, render labels as opaque magenta.
    # this has a high likelihood of having high contrast relative to the
    # underlying color map.
    if label_color is None:
        label_color = (255, 0, 255, 255)

    # map our data array to 8-bit integers with a colormap applied.
    #
    # XXX: compute the scaler based on the size of the quantization table.
    #      number_table_bits = ceil( log2( len( quantization_table ) )
    #      scaler            = 2.0**(8 - number_table_bits)
    #
    pixels = array_to_pixels( array,
                              quantization_table,
                              color_map )

    # render the image into a 4-byte per pixel image.  ensure that the origin
    # is placed correctly.
    if indexing_type == "xy":
        image = Image.fromarray( pixels, "RGBA" )
    elif indexing_type == "ij":
        image = Image.fromarray( np.flipud( pixels ), "RGBA" )
    else:
        raise ValueError( "Unknown indexing type requested ('{:s}').  "
                          "Must be either 'xy' or 'ij'.".format(
            indexing_type ) )

    # titles and labels share the same machinery for overlaying themselves onto
    # the pixels we just rendered.
    if (len( title_text ) > 0 or len( iwp_labels ) > 0):
        draw = ImageDraw.Draw( image )

        # burn in a title if requested.
        if len( title_text ) > 0:
            draw.text( (5, 5),
                       title_text,
                       fill=(255, 255, 255, 175) )

        if len( iwp_labels ) > 0:
            # PIL does not support normalized colors.  attempt scale a
            # normalized color (each component in [0, 1]) into an 8-bit unsigned
            # integer (each component in [0, 255]).
            if all( map( lambda x: type( x ) == float and (0.0 <= x <= 1.0), label_color ) ):
                label_color = tuple( map( lambda x: int( 255 * x ), label_color ) )

            # flip our labels above the horizontal line if the output image does
            # not have the same coordinate system as the IWP labels.
            if indexing_type == "xy":
                iwp_labels = iwp.labels.flipud_iwp_label_coordinates( iwp_labels,
                                                                      array.shape[0],
                                                                      in_place_flag=False )

            # map the IWP labels from normalized coordinates to pixel
            # coordinates.
            #
            # NOTE: we make a copy of the labels to avoid altering our caller's
            #       state.  we *could* combine this with the xy case's in place
            #       but it is a) not a common use case and b) the number of
            #       labels per image are small so we make a duplicate copy.
            #
            #       the amount of time it took to write this comment will far
            #       outweigh the time saved by minimizing the memory footprint.
            #
            iwp_labels = iwp.labels.scale_iwp_label_coordinates( iwp_labels,
                                                                 array.shape[1],
                                                                 array.shape[0],
                                                                 in_place_flag=False )

            for iwp_label in iwp_labels:
                # overlay the label outline.
                draw.rectangle( ((iwp_label["bbox"]["x1"], iwp_label["bbox"]["y1"]),
                                 (iwp_label["bbox"]["x2"], iwp_label["bbox"]["y2"])),
                                outline=label_color )

                # overlay the label name (1st six characters) so that it is
                # slightly above the top of each label's upper left corner.
                # take care such that the name is always visible even if the
                # label is at the top of the image (the name is moved inside the
                # label in that case).
                #
                # NOTE: the hardcoded 12 below accounts for the default PIL font
                #       size on my system (8-10 points?) and leaves 3 pixels
                #       between the bottom of the label name and the label's top
                #       edge.  this is incredibly brittle, but it's not worth
                #       the energy to do this correctly right now.
                #
                draw.text( (iwp_label["bbox"]["x1"],
                            max( iwp_label["bbox"]["y1"] - 12, 2 )),
                           "{:s}".format( iwp_label["id"][:6] ),
                           fill=label_color )

    return image

def array_to_image_imshow( array, quantization_table, color_map, title_text="", show_axes_labels_flag=True, iwp_labels=[], label_color=None, figure_size=None, colorbar_flag=True, grid_extents=None, indexing_type="ij", **kwargs ):
    """
    Takes a NumPy array and creates a Matplotlib figure decorated with title,
    axes labels, and a colorbar.  The array specified is quantized and colorized
    prior to rendering with imshow().  The resulting figure is converted into a
    PIL image.

    Takes 12 arguments:

      array                 - NumPy array of data to convert to pixels.  The data type
                              must be compatible with NumPy's digitize() function.
      quantization_table    - Quantization table to apply to array.  Must be compatible
                              with NumPy's digitize() function.
      color_map             - Matplotlib color map to apply.
      title_text            - Optional title string to burn into the generated image.
                              If omitted, the image is created without alteration.
      show_axes_labels_flag - Optional flag specifying whether axes labels should be
                              rendered.  If omitted, defaults to True.

                              NOTE: Axes ticks and tick labels are always rendered,
                                    this specifies whether the units label is rendered.

      iwp_labels            - Optional list of IWP labels to overlay.  If omitted, defaults
                              to an empty list and nothing is overlaid.
      label_color           - Optional Matplotlib-compatible label color.  May be a
                              color string (by English name, by Matplotlib code, by hex,
                              etc) or a color tuple (RGB or RGBA).  If omitted, defaults
                              to a high contrast color.
      figure_size           - Optional sequence specifying the rendered figure's height
                              and width.  If specified, must be two values in inches.
                              If omitted, defaults to (10, 8).
      colorbar_flag         - Optional flag specifying whether a colorbar should be
                              rendered to the right of the XY slice.  If omitted,
                              defaults to True.
      grid_extents          - Optional sequence specifying the data coordinates of the
                              XY slice.  Sequence of two sequences, each with a pair
                              of elements specifying the (min, max) for the X and Y
                              axes (i.e. (min_x, max_x), (min_y, max_y)).  If omitted,
                              defaults to None and the XY slice axes are labeled in
                              indices.
      indexing_type         - Optional string specifying array's indexing method.  Must
                              be either "xy" (origin in top left) or "ij" (origin in
                              bottom left).  See numpy.meshgrid() for a detailed
                              description of indexing types.  If omitted, defaults to
                              "ij" to match IWP visualization conventions.
      kwargs                - Optional keyword arguments dictionary.  Accepted for
                              compatibility with array_to_image()'s calling convention.

    Returns 1 value:

      image - PIL Image created from array's data.

    """

    # pick a largish default plot size.  if it is too big, callers can specify
    # something themselves.
    if figure_size is None:
        figure_size = (10, 8)

    try:
        # switch the Agg backend so we can render to an offscreen canvas and
        # export its contents into a PIL Image.  take care to track the previous
        # so we can restore it afterwards.  we run into problems when calling
        # this method interactively and the backend doesn't match if we don't.
        previous_backend = plt.get_backend()
        plt.switch_backend( "Agg" )

        # create a single axes subplot.
        #
        # NOTE: we use a constrained layout so that we get consistent axes
        #       layouts when rendering colorbars.  without this we have the
        #       issue where a colorbar tick label at the top of the bar may
        #       cause the XY slice and colorbar axes to shift slightly, so
        #       that a sequence of XY slices jitters around as the value
        #       ranges change slice-to-slice.
        #
        fig_h, ax_h = plt.subplots( 1, 1,
                                    figsize=figure_size,
                                    constrained_layout=True )

        #
        # NOTE: show_xy_slice() already sets the coordinate system to "ij", so
        #       we need to trick it and flip the input matrix (via an indexing
        #       view) if "xy" coordinates are requested.
        #
        if indexing_type == "xy":
            array = np.flipud( array )

        # render the figure.
        image_h = iwp.analysis.show_xy_slice( ax_h,
                                              array,
                                              title_text,
                                              color_map=color_map,
                                              quantization_table=quantization_table,
                                              grid_extents=grid_extents,
                                              iwp_labels=iwp_labels,
                                              label_color=label_color,
                                              colorbar_flag=colorbar_flag )

        # attempt to label our axes correctly.  grid extents specify we have
        # data coordinates, so we're either in meters or dimensionless units
        # (normalized by the tow body diameter).  otherwise, we're in indices.
        #
        # NOTE: we don't have a way to indicate that our coordinate system is in
        #       meters (straight from the simulation) or dimensionless (modified
        #       for analysis), so we attempt to guess based on the domain size.
        #       for the largest simulations the individual domain extents are at
        #       least 10D, so guess based on that.
        #
        if show_axes_labels_flag:
            if grid_extents is not None:
                if ((np.diff( grid_extents[0] ) > 10 or
                     np.diff( grid_extents[1] ) > 10)):
                    ax_h.set_xlabel( "x/D",
                                     fontweight="bold" )
                    ax_h.set_ylabel( "y/D",
                                     fontweight="bold" )
                else:
                    ax_h.set_xlabel( "x (m)",
                                     fontweight="bold" )
                    ax_h.set_ylabel( "y (m)",
                                     fontweight="bold" )
            else:
                ax_h.set_xlabel( "x index",
                                 fontweight="bold" )
                ax_h.set_ylabel( "y index",
                                 fontweight="bold" )

        fig_h.canvas.draw()
    finally:
        # ensure that we switch back to the previous backend no matter what.
        # it would be poor form to suddenly break image plotting in an
        # interactive session because something went awry before we could
        # switch back.
        #
        # NOTE: if an exception was raised above, this will run before leaving
        #       this method.
        #
        plt.switch_backend( previous_backend )

    # convert the figure's rendering into a PIL image.
    image = Image.frombytes( "RGB",
                             fig_h.canvas.get_width_height(),
                             fig_h.canvas.tostring_rgb() )

    return image

def array_to_image( array, quantization_table, color_map, **kwargs ):
    """
    Converts a NumPy array to a PIL Image, either as a raw matrix of pixels or as
    a decorated Matplotlib figure.  The supplied array is quantized and colorized
    prior to conversion to a PIL Image.

    Takes 4 arguments:

      array              - NumPy array of data to convert to pixels.  The data type
                           must be compatible with NumPy's digitize() function.
      quantization_table - Quantization table to apply to array.  Must be compatible
                           with NumPy's digitize() function.
      color_map          - Matplotlib color map to apply.
      kwargs             - Optional keyword arguments dictionary.  See array_to_image_imshow()
                           and array_to_image_PIL() for details on arguments not
                           described below.  Arguments specifically handled:

                             render_figure_flag - Optional flag specifying whether
                                                  a Matplotlib figure should be created.
                                                  If omitted, defaults to False and a
                                                  PIL image is created.

    Returns 1 value:

      image - PIL Image created from array's data.

    """

    # call the appropriate specialization to create the image.
    if kwargs.get( "render_figure_flag", False ):
        return array_to_image_imshow( array,
                                      quantization_table,
                                      color_map,
                                      **kwargs )
    else:
        return array_to_image_PIL( array,
                                   quantization_table,
                                   color_map,
                                   **kwargs)

def color_map_to_image( color_map, color_limits, orientation="vertical", figsize=None ):
    """
    Converts a colormap to a colorbar and renders it into a PIL Image.

    Takes 4 arguments:

      color_map    - Matplotlib color map to render into a colorbar.
      color_limits - Tuple of minimum and maximum values to set the extremes of the
                     rendered colorbar.
      orientation  - Optional string specifying the orientation of the colorbar.
                     Must be either "horizontal" or "vertical".  If omitted,
                     defaults to "vertical" for a vertically oriented colorbar.
      figsize      - Optional tuple specifying the size, width and height in inches,
                     of the colorbar.  If omitted, a default size is chosen.

                     NOTE: No sanity checks are performed against the orientation
                           and the color_limits to ensure that the rendered colorbar
                           will fit within the requested size.

    Returns 1 value:

      image_buffer - PIL Image of the rendered colorbar.

    """

    # we use Matplotlib to render a colorbar from our colormap and then
    # serialize it into a buffer that PIL can turn into an exportable image.

    # pick a reasonable default figure size based on the orientation.
    if figsize is None:
        if orientation == "vertical":
            figsize = (1.5, 5)
        elif orientation == "horizontal":
            figsize = (5, 1.5)
        else:
            raise ValueError( "Unknown colorbar orientation ({:s}).  "
                              "Cannot default figsize.".format(
                                  orientation ) )

    # build a figure of the appropriate size.  the single axes will contain
    # the colorbar.
    fig_h, ax_h = plt.subplots( 1, 1, figsize=figsize )

    # create a colorbar spanning our color limits using the supplied colormap.
    normalizer = mpl.colors.Normalize( vmin=color_limits[0],
                                       vmax=color_limits[1] )
    fig_h.colorbar( mpl.cm.ScalarMappable( norm=normalizer,
                                           cmap=color_map ),
                    cax=ax_h,
                    orientation=orientation )

    # attempt to remove excess space in the figure.
    fig_h.tight_layout()

    # serialize the figure to a buffer.
    #
    # NOTE: we use PNG so that we have a lossless round trip from Matplotlib
    #       figure to PIL image.
    #
    colorbar_buffer = io.BytesIO()
    fig_h.savefig( colorbar_buffer, format="png" )
    plt.close( fig=fig_h )
    colorbar_buffer.seek( 0 )

    # create a PIL image from the raw bytes.
    colorbar_image = Image.open( colorbar_buffer )

    return colorbar_image

def da_write_single_xy_slice_image( da, output_path, quantization_table, color_map, verbose_flag=False, **kwargs ):
    """
    Functor for creating an on-disk image for a single XY slice of data.  Quantizes
    the XY slice and applies a color map before writing.  The file format used is
    determined by the file path specified (e.g. "foo.png" is written as PNG).

    Takes 6 arguments:

      da                 - xarray DataArray to create XY slice images from.  Each of
                           the XY slices, for all time steps and variables, are written
                           to disk beneath output_root.
      output_path        - On-disk path where the XY slice image is written to.  May
                           be either an absolute or relative path.
      quantization_table - Quantization table to apply to array.  Must be compatible
                           with NumPy's digitize() function.
      color_map          - Matplotlib color map to apply.
      verbose_flag       - Optional boolean specifying whether execution should be
                           verbose.  If specified as True, diagnostic messages are
                           printed to standard output during execution.  If omitted,
                           defaults to False.
      **kwargs           - Optional keyword arguments dictionary.  See array_to_image()
                           for details on the arguments supported.

    Returns 1 value:

       da - xarray DataArray that was supplied by the caller.

    """

    # get the data coordinate extents for the XY slice.
    grid_extents = (da.coords["x"].values[[0, -1]],
                    da.coords["y"].values[[0, -1]])

    image = array_to_image( da.values.astype( np.float32 ),
                            quantization_table,
                            color_map,
                            grid_extents=grid_extents,
                            **kwargs )

    if verbose_flag:
        print( "Writing {:s}".format( output_path ) )

    # save the image to disk.
    image.save( output_path )

    return da

def da_write_xy_slice_images( da, output_root, experiment_name, xy_slice_indices, data_limits, color_map, quantization_table_builder, title_flag=True, verbose_flag=False, **kwargs ):
    """
    Functor for creating on-disk images for each XY slice of data present in an xarray
    DataArray object.  Iterates across all time steps, variables, and XY slices within
    the DataArray to quantize and apply a color map before writing to disk as a PNG.

    Individual image paths are of the form:

        <output_root>/<variable>/<experiment>-<variable>-z=<z_index>-Nt=<time_index>.png

    Takes 9 arguments:

      da                         - xarray.DataArray to create XY slice images from.  Each of
                                   the XY slices, for all time steps and variables, are written
                                   to disk beneath output_root.
      output_root                - On-disk path specifying where XY slice images are written.
      experiment_name            - Name of the experiment associated with the XY slices.  Used
                                   to construct the path for each XY slice's on-disk image.
      xy_slice_indices           - Sequence of XY slice indices corresponding to the slices
                                   contained in da.  These are used to lookup the Z index
                                   within the original data volume from the information
                                   present to the functor during execution.
      data_limits                - Tuple providing (minimum, maximum, standard deviation) of
                                   da to support global normalization.  If specified as None,
                                   data_limits are computed for each XY slice to support local,
                                   per-slice normalization.
      color_map                  - Matplotlib color map to apply to the underlying data.
      quantization_table_builder - Function that generates a quantization table when supplied
                                   four arguments: number of quantization levels, data minimum,
                                   data maximum, and data standard deviation.  If data_limits
                                   is specified, then this is called once per functor execution,
                                   otherwise it is called once per XY slice with local data limits.
      title_flag                 - Optional boolean specifying whether images should have
                                   slice metadata burned into the image or not, allowing
                                   for visual identification of the slice when its on-disk
                                   path is unknown.  If omitted, defaults to True.
      verbose_flag               - Optional boolean specifying whether execution should
                                   be verbose.  If specified as True, diagnostic messages
                                   are printed to standard output during execution.  If
                                   omitted, defaults to False.

    Returns 1 value:

      da - xarray DataArray that was supplied by the caller.

    """

    # ignore the initial setup call.  this is required since xarray makes a
    # single call before mapping the functor across the real dataset.
    if sum( da.shape ) == 0:
        return da

    # pull this time step's value.
    #
    # NOTE: this is not a time step index, but rather the actual underlying
    #       value (i.e. Nt or buoyancy time).  correcting the conflation
    #       between indices and values will be dealt with at a later date.
    #
    time_step_value = da.Cycle.values[0]

    # create 8-bit quantization tables since we're generating images.
    number_table_entries = 256

    # build a quantization table if statistics were provided, otherwise we
    # compute them slice-by-slice (to highlight local features) and quantize
    # each separately.
    if data_limits is not None:
        quantization_table = quantization_table_builder( number_table_entries,
                                                         *data_limits )

    # walk through slices in this data array and create an image for each.
    #
    # NOTE: we may be operating on a subset of a larger volume so the Z indices
    #       are relative to the XY slice indices supplied.
    #
    for z_index in range( len( da.z ) ):
        output_path = "{:s}/{:s}/{:s}-{:s}-z={:03d}-Nt={:03d}.png".format(
            output_root,
            da.name,
            experiment_name,
            da.name,
            xy_slice_indices[z_index],
            time_step_value )

        # create a title that matches the type of image being rendered.
        title_text = ""
        if title_flag:
            if kwargs.get( "render_figure_flag", False ):
                # figures have a proper title and can accommodate longer, more
                # descriptive text.
                title_text = "{:s}\nNt={:03d}, z={:.2f} ({:03d})".format(
                    iwp.analysis.variable_name_to_title( da.name ),
                    time_step_value,
                    da.z[z_index].values,
                    xy_slice_indices[z_index] )
            else:
                # rendered arrays have limited pixel space (one per element).
                # build a title to burn into the slice so it is recognizable
                # without additional metadata.
                title_text = "{:s} - Nt={:03d}, z={:.2f} ({:03d})".format(
                    da.name,
                    time_step_value,
                    da.z[z_index].values,
                    xy_slice_indices[z_index] )

        # compute local statistics on this slice if they're being normalized
        # independently rather than across an entire dataset.
        if data_limits is None:
            local_data_limits  = iwp.statistics.compute_statistics( da[0, z_index, :] )
            quantization_table = quantization_table_builder( number_table_entries,
                                                             *local_data_limits )

        # image this slice.
        da_write_single_xy_slice_image( da[0, z_index, :],
                                        output_path,
                                        quantization_table,
                                        color_map,
                                        title_text=title_text,
                                        verbose_flag=verbose_flag,
                                        **kwargs )

    return da

def ds_write_xy_slice_images( ds, output_root, experiment_name, variable_names, time_step_indices, xy_slice_indices, data_limits, color_map, quantization_table_builder, work_chunk_size=30, render_figure_flag=False, title_flag=True, verbose_flag=False ):
    """
    Creates on-disk images for each of the XY slices in a subset of the supplied dataset.
    A subset of time, XY slices, and variables are normalized, quantized, and colorized
    before rendering to images and written to disk.

    See da_write_xy_slice_images() for details on where images are written.

    Normalization is done via supplied statistics (typically pre-computed, global
    statistics) or with computed on demand for per-XY slice statistics.  Quantization
    is performed with a caller supplied table, as is colorization.

    Metadata information may be "burned in" to the images created for easy diagnostics
    when additional supporting information (e.g. file path) is not available.

    Image creation may be done in parallel and is distributed into chunks to avoid
    exhausting system resources.

    Takes 13 arguments:

      ds                         - xarray.Dataset or xarray.DataArray to create XY slice images
                                   from.
      output_root                - On-disk path specifying where XY slice images are written.
      experiment_name            - Name of the experiment associated with the XY slices.  Used
                                   to construct the path for each XY slice's on-disk image.
      variable_names             - Sequence of variable names to generate images for.
      time_step_indices          - Sequence of time step indices to generate images for.
      xy_slice_indices           - Sequence of XY slice indices to generate images for.
      data_limits                - Tuple providing (minimum, maximum, standard deviation) of
                                   da to support global normalization.  If specified as None,
                                   data_limits are computed for each XY slice to support local,
                                   per-slice normalization.
      color_map                  - Matplotlib color map to apply to the underlying data.
      quantization_table_builder - Function that generates a quantization table when supplied
                                   four arguments: number of quantization levels, data minimum,
                                   data maximum, and data standard deviation.  If data_limits
                                   is specified, then this is called once per functor execution,
                                   otherwise it is called once per XY slice with local data limits.
      work_chunk_size            - Optional positive integer specifying the chunk size of
                                   work to distribute when rendering images.  This is
                                   a crude means to balance compute workload against memory
                                   footprint.  If omitted, defaults to 30.

                                   NOTE: This interface will likely change in the future.

      render_figure_flag         - Optional boolean specifying whether images created should
                                   be rendered as a Matplotlib figure or as a direct
                                   translation of each XY slice.  If omitted, defaults to
                                   False.
      title_flag                 - Optional boolean specifying whether images should have
                                   slice metadata used as a title.  For direct translations
                                   of each XY slice (render_figure_flag == False), this
                                   title text is burned into the image for easy visual
                                   identification.  For figure renderings of each XY slice
                                   (render_figure_flag == True), a figure title is constructed
                                   instead.  If omitted, defaults to True.
      verbose_flag               - Optional boolean specifying whether execution should be
                                   verbose.  If specified as True, diagnostic messages are
                                   printed to standard output during execution.  If omitted,
                                   defaults to False.

    Returns nothing.

    """

    # parameters for generating images passed to the slice writing routines.
    image_parameters = {}

    # size up the amount of work we have in the time dimension.
    number_time_steps = len( time_step_indices )

    if verbose_flag:
        print( "Rendering XY slices for '{:s}'.".format(
            experiment_name ) )

    # propagate the request to create figures down the stack.
    if render_figure_flag:
        image_parameters["render_figure_flag"] = True

    # iterate through each of the variables.  this is our outer loop so that
    # data for a single variable is completely rendered before moving to the
    # next.
    for variable_name in variable_names:

        # ensure that the directory structure this variable will write to exists.
        os.makedirs( "{:s}/{:s}".format( output_root, variable_name ),
                     exist_ok=True )

        if verbose_flag:
            print( "    {:s}".format( variable_name ) )

        # acquire this variable's statistics if they were provided.  if they
        # were not provided, they'll be computed on the fly when rendering the
        # images.
        if data_limits is not None:
            variable_statistics = data_limits[variable_name]
        else:
            variable_statistics = None

        # iterate through the time steps in chunks.  we map from array indices
        # to time step indices below when we subset the dataset.
        #
        # NOTE: we break up the work into (untuned) chunks to avoid exhausting
        #       the system's memory and triggering an OOM event.  if we were
        #       smarter about the underlying distributed computation system
        #       we would simply map the function across the xarray.Dataaset.
        #
        for chunk_start_index in range( 0, number_time_steps, work_chunk_size ):
            chunk_end_index = min( chunk_start_index + work_chunk_size,
                                   number_time_steps )

            current_time_step_indices = time_step_indices[slice( chunk_start_index,
                                                                 chunk_end_index )]

            if verbose_flag:
                print( "        [{:d}:{:d}]".format(
                    time_step_indices[chunk_start_index],
                    time_step_indices[chunk_end_index - 1] ) )

            # get a DataArray for the time steps and XY slices of interest for
            # this variable, and generate images for this variable.
            da = iwp.utilities.get_xarray_subset( ds,
                                                  variable_name,
                                                  current_time_step_indices,
                                                  xy_slice_indices )

            da.map_blocks( da_write_xy_slice_images,
                           (output_root,
                            experiment_name,
                            xy_slice_indices,
                            variable_statistics,
                            color_map,
                            quantization_table_builder,
                            title_flag,
                            verbose_flag),
                           image_parameters ).compute()

    return
