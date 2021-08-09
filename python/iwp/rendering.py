import io
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from PIL import Image, ImageDraw

import iwp.statistics

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

def array_to_image( array, quantization_table, color_map, indexing_type="ij", title_text="" ):
    """
    Converts a NumPy array to a PIL Image, and optionally burns in a title to
    the top of the image.  The supplied array is quantized and colorized prior
    to conversion to a PIL Image.

    Raises ValueError if the requested indexing method is unknown.

    Takes 5 arguments:

      array              - NumPy array of data to convert to pixels.  The data type
                           must be compatible with NumPy's digitize() function.
      quantization_table - Quantization table to apply to array.  Must be compatible
                           with NumPy's digitize() function.
      color_map          - Matplotlib color map to apply.
      indexing_type      - Optional string specifying array's indexing method.  Must
                           be either "xy" (origin in top left) or "ij" (origin in
                           bottom left).  See numpy.meshgrid() for a detailed
                           description of indexing types.  If omitted, defaults to
                           "ij" to match IWP visualization conventions.
      title_text         - Optional title string to burn into the generated image.
                           If omitted, the image is created without alteration.

    Returns 1 value:

      image - PIL Image created from array's data.

    """

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

    # burn in a title if requested.
    if len( title_text ) > 0:
        draw = ImageDraw.Draw( image )
        draw.text( (5, 5),
                   title_text,
                   fill=(255, 255, 255, 175) )

    return image

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

def da_write_single_xy_slice_image( da, output_path, quantization_table, color_map, title_text="", verbose_flag=False ):
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
      title_text         - Optional title string to burn into the generated image.
                           If omitted, the image is created without alteration.
      verbose_flag       - Optional boolean specifying whether execution should be
                           verbose.  If specified as True, diagnostic messages are
                           printed to standard output during execution.  If omitted,
                           defaults to False.

    Returns 1 value:

       da - xarray DataArray that was supplied by the caller.

    """

    image = array_to_image( da.values.astype( np.float32 ),
                            quantization_table,
                            color_map,
                            title_text=title_text )

    if verbose_flag:
        print( "Writing {:s}".format( output_path ) )

    # save the image to disk.
    image.save( output_path )

    return da

def da_write_xy_slice_images( da, output_root, experiment_name, xy_slice_indices, data_limits, color_map, quantization_table_builder, title_flag=True, verbose_flag=False ):
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

        # build a title to burn into the slice so it is recognizable without
        # additional metadata.
        title_text = ""
        if title_flag:
            title_text = "Nt={:03d}, z={:.2f} ({:03d}), {:s}".format(
                time_step_value,
                da.z[z_index].values,
                xy_slice_indices[z_index],
                da.name )

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
                                        title_text,
                                        verbose_flag )

    return da

def ds_write_xy_slice_images( ds, output_root, experiment_name, variable_names, time_step_indices, xy_slice_indices, data_limits, color_map, quantization_table_builder, work_chunk_size=30, title_flag=True, verbose_flag=False ):
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

    Takes 12 arguments:

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

      title_flag                 - Optional boolean specifying whether images should have
                                   slice metadata burned into the image or not, allowing
                                   for visual identification of the slice when its on-disk
                                   path is unknown.  If omitted, defaults to True.
      verbose_flag               - Optional boolean specifying whether execution should be
                                   verbose.  If specified as True, diagnostic messages are
                                   printed to standard output during execution.  If omitted,
                                   defaults to False.

    Returns nothing.

    """

    # size up the amount of work we have in the time dimension.
    number_time_steps = len( time_step_indices )

    if verbose_flag:
        print( "Rendering XY slices for '{:s}'.".format(
            experiment_name ) )

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
                            verbose_flag) ).compute()

    return
