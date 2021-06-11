import numpy as np
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

def array_to_image( array, quantization_table, color_map, title_text="" ):
    """
    Converts a NumPy array to a PIL Image, and optionally burns in a title to
    the top of the image.  The supplied array is quantized and colorized prior
    to conversion to a PIL Image.

    Takes 4 arguments:

      array              - NumPy array of data to convert to pixels.  The data type
                           must be compatible with NumPy's digitize() function.
      quantization_table - Quantization table to apply to array.  Must be compatible
                           with NumPy's digitize() function.
      color_map          - Matplotlib color map to apply.
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

    # render the image into a 4-byte per pixel image.
    image = Image.fromarray( pixels, "RGBA" )

    # burn in a title if requested.
    if len( title_text ) > 0:
        draw = ImageDraw.Draw( image )
        draw.text( (5, 5),
                   title_text,
                   fill=(255, 255, 255, 175) )

    return image

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
                            title_text )

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

    time_step_index = da.Cycle.values[0] - 1

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
            time_step_index )

        # build a title to burn into the slice so it is recognizable without
        # additional metadata.
        title_text = ""
        if title_flag:
            title_text = "Nt={:03d}, z={:03d}, {:s}".format( time_step_index,
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
