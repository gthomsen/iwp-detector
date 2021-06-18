import io

#
# NOTE: this imports the python-pptx module, not ourselves...
#
import pptx

import iwp.labels
import iwp.rendering

def _add_xy_slice_shape_group( slide, xy_slice_position, xy_slice_image, colorbar, variable_name, normalized_iwp_labels, y_axis_label_flag=False ):
    """
    Adds an XY slice image, its colorbar, and axes label decorations to an existing
    slide. Optionally overlays IWP labels onto the XY slice image.  All generated
    shapes are added to a new shape group for easy manipulation.

    NOTE: This function updates the slide object provided.

    NOTE: The X- and Y-axes' tick are currently not rendered.  Space is left for them
          for when this is implemented, but no axes ticks (or tick labels) are
          generated.

    Takes 7 arguments:

      slide                 - pptx.slide.Slide object to add XY slices to.  This is
                              modified during execution.
      xy_slice_position     - Tuple specifying the left, top, width, and height of the
                              XY slices to add to slide.  Must be an object derived
                              from pptx.util.Length.
      xy_slice_image        - XY slice image to add to the slide.  Specified either
                              as a path or a buffer of serialized bytes.
      colorbar              - Colorbar image to add to the slide.  Specified either
                              as a path or a buffer of serialized bytes.
      variable_name         - Name of the variable associated with xy_slice_image.
      normalized_iwp_labels - List of IWP labels whose bounding boxes are normalized
                              into the range of [0, 1].  Each IWP label is rendered
                              as an unfilled box on the XY slice image.
      y_axis_label_flag     - Optional flag specifying whether the Y-axis label
                              should be generated.  Specify as True to the first, and
                              False to subsequent XY slices, if multiple are to be
                              added to a single slide to conserve horizontal space.
                              If omitted, defaults to False.

    Returns 1 value:

      xy_slice_group - The created pptx.shapes.group.GroupShape containing all of
                       the shapes associated with the XY slice image.

    """

    #
    # NOTE: all of the hard coded offsets and positions were copied or
    #       derived from manual layouts.
    #

    # name our coordinates so we can use them to derive other positions and
    # sizes.
    (xy_slice_left,
     xy_slice_top,
     xy_slice_width,
     xy_slice_height) = xy_slice_position

    xy_slice_group = slide.shapes.add_group_shape()

    # add the XY slice with a black border.
    xy_slice_picture = xy_slice_group.shapes.add_picture( xy_slice_image,
                                                          xy_slice_left,
                                                          xy_slice_top )

    xy_slice_picture.width  = xy_slice_width
    xy_slice_picture.height = xy_slice_height
    xy_slice_picture.name   = "XY Slice - {:s}".format(
        variable_name )

    xy_slice_border           = xy_slice_picture.line
    xy_slice_border.color.rgb = pptx.dml.color.RGBColor( 0x00, 0x00, 0x00 )
    xy_slice_border.width     = pptx.util.Pt( 1 )

    # title the slice using the variable name.
    xy_slice_title_left   = xy_slice_left + pptx.util.Inches( .69 )
    xy_slice_title_top    = pptx.util.Inches( 1.31 )
    xy_slice_title_width  = pptx.util.Inches( 0.86 )
    xy_slice_title_height = pptx.util.Inches( 0.4 )

    xy_slice_title = xy_slice_group.shapes.add_textbox( xy_slice_title_left,
                                                        xy_slice_title_top,
                                                        xy_slice_title_width,
                                                        xy_slice_title_height )

    # bold, 18pt, and centered.
    p           = xy_slice_title.text_frame.paragraphs[0]
    p.text      = variable_name
    p.font.bold = True
    p.font.size = pptx.util.Pt( 18 )
    p.alignment = pptx.enum.text.PP_ALIGN.CENTER

    # X-axis label.
    xy_slice_xaxis_left   = xy_slice_left + pptx.util.Inches( 0.69 )
    xy_slice_xaxis_top    = pptx.util.Inches( 5.23 )
    xy_slice_xaxis_width  = pptx.util.Inches( 0.86 )
    xy_slice_xaxis_height = pptx.util.Inches( 0.4 )

    xy_slice_xaxis = xy_slice_group.shapes.add_textbox( xy_slice_xaxis_left,
                                                        xy_slice_xaxis_top,
                                                        xy_slice_xaxis_width,
                                                        xy_slice_xaxis_height )
    xy_slice_xaxis.rotation = 0

    # bold, 18pt, and centered.
    p           = xy_slice_xaxis.text_frame.paragraphs[0]
    p.text      = "X/D"
    p.font.bold = True
    p.font.size = pptx.util.Pt( 18 )
    p.alignment = pptx.enum.text.PP_ALIGN.CENTER

    # Y-axis label.
    if y_axis_label_flag:
        #
        # NOTE: we don't position this relative to anything as it is assumed
        #       there will be at most one Y-axis label.
        #
        xy_slice_yaxis_left   = pptx.util.Inches( .04 )
        xy_slice_yaxis_top    = pptx.util.Inches( 3.21 )
        xy_slice_yaxis_width  = pptx.util.Inches( 0.86 )
        xy_slice_yaxis_height = pptx.util.Inches( 0.4 )

        xy_slice_yaxis = xy_slice_group.shapes.add_textbox( xy_slice_yaxis_left,
                                                            xy_slice_yaxis_top,
                                                            xy_slice_yaxis_width,
                                                            xy_slice_yaxis_height )
        xy_slice_yaxis.rotation = 270

        # bold, 18pt, and centered.
        p           = xy_slice_yaxis.text_frame.paragraphs[0]
        p.text      = "Y/D"
        p.font.bold = True
        p.font.size = pptx.util.Pt( 18 )
        p.alignment = pptx.enum.text.PP_ALIGN.CENTER

    # add the colorbar at slightly to the right of the XY slice.
    colorbar_left = xy_slice_left + pptx.util.Inches( 2.26 )
    colorbar_top  = pptx.util.Inches( 2.23 )

    colorbar_picture = xy_slice_group.shapes.add_picture( colorbar,
                                                          colorbar_left,
                                                          colorbar_top )
    colorbar_picture.width  = pptx.util.Inches( 0.47 )
    colorbar_picture.height = pptx.util.Inches( 2.37 )
    colorbar_picture.name = "XY Slice Colorbar - {:s}".format(
        variable_name )

    # generate bounding boxes for each of the labels supplied.
    for normalized_iwp_label in normalized_iwp_labels:

        # convert the normalized label's four corners into (left, top, width,
        # height) so we can create a box representing the label.
        normalized_label_left   = normalized_iwp_label["bbox"]["x1"]
        normalized_label_top    = normalized_iwp_label["bbox"]["y1"]
        normalized_label_width  = (normalized_iwp_label["bbox"]["x2"] -
                                   normalized_iwp_label["bbox"]["x1"])
        normalized_label_height = (normalized_iwp_label["bbox"]["y2"] -
                                   normalized_iwp_label["bbox"]["y1"])

        # convert the normalized label's coordinates into the XY slice
        # picture's coordinates and position it.
        label_box_left = xy_slice_left + int( xy_slice_picture.width *
                                              normalized_label_left )
        label_box_top  = xy_slice_top  + int( xy_slice_picture.height *
                                              normalized_label_top )

        # properly size the label by scaling it by the image dimensions.
        label_box_width  = int( xy_slice_picture.width  * normalized_label_width )
        label_box_height = int( xy_slice_picture.height * normalized_label_height )

        # white border for an unfilled rectangle.  1 pt line thickness.
        label_box = xy_slice_group.shapes.add_shape( pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
                                                     label_box_left,
                                                     label_box_top,
                                                     label_box_width,
                                                     label_box_height )

        label_box.fill.background()
        label_box.line.color.rgb = pptx.dml.color.RGBColor( 0xFF, 0xFF, 0xFF )
        label_box.line.width     = pptx.util.Pt( 1 )

    return xy_slice_group

def create_data_review_presentation( iwp_dataset, experiment_name, variable_names, time_xy_slice_pairs, data_limits, color_map, quantization_table_builder, iwp_labels=[] ):
    """
    Creates a Powerpoint presentation containing data review slides for a set of
    XY slices.  One slide per XY slice is generated, with up to three variables of
    the slice rendered side-by-side for comparison and review.  IWP labels may be
    overlaid on each XY slice rendered, and slices are normalized, quantized, and
    colored according to the caller's specification.

    Raises ValueError if too few or too many variables are specified.  This prevents
    the generated XY slices from being scaled down so much to be of no use.

    Takes 8 arguments:

      iwp_dataset                - iwp.data_loader.IWPDataset object containing the XY
                                   slices to generate review slides for.
      experiment_name            - String specifying the experiment that generated the slice.
      variable_names             - List of one, two, or three variable names to generate
                                   review images for.
      time_xy_slice_pairs        - List of tuples, (time index, XY slice index), specifying
                                   the XY slices to generate slides for.
      data_limits                - Dictionary of tuples, keys are variable names, keys are
                                   (minimum, maximum, standard deviation), specifying the
                                   global statistics to normalize each XY slice with.  If
                                   specified as None, statistics are computed for each XY
                                   slice independently.
      color_map                  - Matplotlib color map to apply to the underlying data.
      quantization_table_builder - Function that generates a quantization table when supplied
                                   four arguments: number of quantization levels, data minimum,
                                   data maximum, and data standard deviation.
      iwp_labels                 - Optional sequence of IWP labels to overlay on the
                                   generated data.

    Returns 1 value:

      presentation - pptx.Presentation object containing the slides generated.

    """

    # bail if we have too few or too many variables to review.  we have limited
    # space on each slide and cannot fit more than three XY slices before
    # running out of room.
    if not (0 < len( variable_names ) < 4):
        raise ValueError( "Must have between 1 and 3 variables to generate "
                          "review data for, received {:d}.".format(
                              len( variable_names ) ) )

    presentation = pptx.Presentation()

    # set a widescreen ratio (16:9) for this presentation's slides.
    presentation.slide_width  = pptx.util.Inches( 10 )
    presentation.slide_height = pptx.util.Inches( 5.625 )

    # start with a title-only slide.
    blank_slide_layout = presentation.slide_layouts[5]

    # specify the locations of each of the XY slices we're reviewing.  these
    # change as a function of slice count so that we use the most of the
    # available space and produce aesthetically pleasing results.
    if len( variable_names ) == 1:
        # single image centered in the middle of the slide.
        xy_slice_positions = [(pptx.util.Inches( 3.64 ),
                               pptx.util.Inches( 1.75 ),
                               pptx.util.Inches( 2.22 ),
                               pptx.util.Inches( 3.33 ))]
    elif len( variable_names ) == 2:
        # two images centered about the middle of the slide.
        xy_slice_positions = [(pptx.util.Inches( 1.75 ),
                               pptx.util.Inches( 1.75 ),
                               pptx.util.Inches( 2.22 ),
                               pptx.util.Inches( 3.33 )),
                              (pptx.util.Inches( 6.22 ),
                               pptx.util.Inches( 1.75 ),
                               pptx.util.Inches( 2.22 ),
                               pptx.util.Inches( 3.33 ))]
    elif len( variable_names ) == 3:
        # three images equally distributed across the slide.
        xy_slice_positions = [(pptx.util.Inches( 0.98 ),
                               pptx.util.Inches( 1.75 ),
                               pptx.util.Inches( 2.22 ),
                               pptx.util.Inches( 3.33 )),
                              (pptx.util.Inches( 3.99 ),
                               pptx.util.Inches( 1.75 ),
                               pptx.util.Inches( 2.22 ),
                               pptx.util.Inches( 3.33 )),
                              (pptx.util.Inches( 6.99 ),
                               pptx.util.Inches( 1.75 ),
                               pptx.util.Inches( 2.22 ),
                               pptx.util.Inches( 3.33 ))]

    # normalize the labels so their bounding boxes are [0, 1] rather than in
    # units of pixels.  this is needed so we can properly position them as
    # we're working in inches/points/EMUs/etc rather than raw pixels.
    #
    # NOTE: this assumes the supplied labels match the XY slice...
    #
    normalized_iwp_labels = iwp.labels.normalize_iwp_label_coordinates( iwp_labels,
                                                                        iwp_dataset.shape[0],
                                                                        iwp_dataset.shape[1] )

    # build a mapping from (time, xy slice) to labels so it is easy to identify
    # the relevant ones when building data review slides.
    normalized_iwp_labels_map = {}
    for normalized_iwp_label in normalized_iwp_labels:
        label_key = iwp.labels.get_iwp_label_key( normalized_iwp_label )

        # add this label to its (time, slice) list.
        if label_key not in normalized_iwp_labels_map:
            normalized_iwp_labels_map[label_key] = [normalized_iwp_label]
        else:
            normalized_iwp_labels_map[label_key].append( normalized_iwp_label )

    # iterate through each of the requested XY slices and make a slide for it.
    for time_index, xy_slice_index in time_xy_slice_pairs:
        current_slide = presentation.slides.add_slide( blank_slide_layout )

        # set the title.
        slide_title      = current_slide.placeholders[0]
        slide_title.text = "{:s}: Z={:03d}, Nt={:03d}".format(
            experiment_name,
            xy_slice_index,
            time_index )

        # pull the data for this XY slice.
        xy_slice_array = iwp_dataset.get_xy_slice( time_index,
                                                   xy_slice_index )

        # construct a label key so we can lookup the labels associated with
        # this XY slice.
        label_key = (time_index, xy_slice_index)

        # iterate through each of the variables and add a group containing
        # the rendered data with titles and axes labels.  size and positions
        # are provided allowing for variable count-specific layouts (e.g.
        # centered, big images for a single variable vs smaller, multi-column
        # layouts for multiple variables).
        for variable_index, variable_name in enumerate( variable_names ):

            # get this variable's statistics so we can quantize it properly.
            if data_limits is not None:
                # pull our variable's statistics out of the global statistics.
                variable_statistics = (data_limits[variable_name]["minimum"],
                                       data_limits[variable_name]["maximum"],
                                       data_limits[variable_name]["standard_deviation"])
            else:
                # compute our variable's local statistics from the current XY
                # slice.
                variable_statistics = (xy_slice_array[variable_index, :].min(),
                                       xy_slice_array[variable_index, :].max(),
                                       xy_slice_array[variable_index, :].std())

            quantization_table = quantization_table_builder( 256,
                                                             *variable_statistics )

            # render this XY slice to an image.  serialize it to PNG format into
            # an in-memory buffer.
            xy_slice_image        = iwp.rendering.array_to_image( xy_slice_array[variable_index, :],
                                                                  quantization_table,
                                                                  color_map )
            xy_slice_image_buffer = io.BytesIO()
            xy_slice_image.save( xy_slice_image_buffer, format="png" )

            # render this XY slice's colorbar to an image.  serialize it to PNG
            # format into an in-memory buffer.
            xy_slice_colorbar_image  = iwp.rendering.color_map_to_image( color_map,
                                                                         (quantization_table[0],
                                                                          quantization_table[-1]),
                                                                         orientation="vertical",
                                                                         figsize=(2, 5) )
            xy_slice_colorbar_buffer = io.BytesIO()
            xy_slice_colorbar_image.save( xy_slice_colorbar_buffer, format="png" )

            # add this XY slice to the slide in a group.  only generate the
            # y-axis labeling on the first image so we efficiently use our
            # horizontal space and avoid clutter.
            _add_xy_slice_shape_group( current_slide,
                                       xy_slice_positions[variable_index],
                                       xy_slice_image_buffer,
                                       xy_slice_colorbar_buffer,
                                       variable_name,
                                       normalized_iwp_labels_map.get( label_key, [] ),
                                       y_axis_label_flag=(variable_index == 0))

    return presentation
