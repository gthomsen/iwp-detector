import numpy as np
import paraview.simple as pv
import warnings

import vtk
import vtk.numpy_interface.dataset_adapter as dsa

# utility routines for working within a ParaView instance.  these are assumed to
# be run from within 1) a ParaView Python prompt, 2) a pvbatch instance, 3) a
# pvpython instance, or 4) a Jupyter ParaView kernel.
#
# NOTE: take care to ensure that the version of ParaView, Python, and VTK (and
#       their associated environments) are consistent as there are many dark
#       pits of despair that one may fall into when they are not.
#
# all routines added to this module should take care to capture variables
# returned from called method.  the Jupyter ParaView kernel prints each
# uncaptured variable to the local messages console which is incredibly
# annoying.

# thoughts on measuring Line-like objects:
#
#   I don't understand enough about ParaView's proxy architecture to correctly
#   identify particular object classes, nor do I have a comprehensive list of
#   Line-like sources.  given that, we consider any object that has the correct
#   attributes (i.e. "Point1" and "Point2") to satisfy the Line-like criterion
#   and work with that.
#
# thoughts on coordinate systems:
#
#   IWP convention is a left handed system (north, east, up) that is rotated 90
#   degrees clockwise, such that the positive X axis is "north" and the negative
#   Y axis is "east".  this means that azimuth is measured relative in degrees
#   clockwise to the X axis, and elevation is degrees up relative to the XY plane.
#
#   this matches the typical presentation of IWP data in literature where XY
#   planes are viewed from above, with the origin being located either in the
#   lower left, or left-center, and the tow body track moving from left to right
#   along the X axis.  positive Z is up such that XY slices with larger Z values
#   are above those with lower, and the top of the simulation domain is
#   positioned at the maximum Z value.

def _locate_data_in_multiblock( paraview_source ):
    """
    Identifies the first block in a multi-block ParaView source that contains data.

    Raises ValueError if the supplied source is not multi-block.

    Raises RuntimeError if a block with data cannot be located.

    Takes 1 argument:

      paraview_source - Multi-block ParaView object.

    Returns 1 value:

      block_index - Index, in the range of [0, number_blocks), to the first block
                    that contains data.

    """

    multiblock_flag = (paraview_source.GetDataInformation().GetDataSetType() ==
                       vtk.VTK_MULTIBLOCK_DATA_SET)

    # make sure we're working with a multi-block dataset.
    if not multiblock_flag:
        raise ValueError( "Supplied ParaView object ({}) is not a multi-block dataset.".format(
            paraview_source ) )

    composite_data_information = paraview_source.GetDataInformation().GetCompositeDataInformation()
    number_blocks              = composite_data_information.GetNumberOfChildren()

    # walk through each of the blocks and stop at the first one containing data.
    for block_index in range( number_blocks ):
        data_information = composite_data_information.GetDataInformation( block_index )

        # blocks that do not contain data (e.g. grid only) will not have a
        # DataInformation structure.
        if data_information is None:
            continue
        else:
            break

    # handle the case where we can't find a suitable block.
    if block_index == number_blocks:
        raise RuntimeError( "Failed to find a suitable block!" )

    return block_index

def extract_block( filter_name, multiblock_source, output_type, data_information, block_index ):
    """
    Extracts a block from a multi-block source and adds it into the pipeline.
    Preserves dataset extents when extracting outputs that are structured.  The
    extracted block is not displayed.

    Derived from:

      https://www.paraview.org/Wiki/ParaView/Python/Extracting_Multiple_Blocks

    Below is code that extracts all blocks from a multi-block source:

      multiblock_source = FindSource( "SomeMultiblockSource" )

      # extract the time series out of the multi-block dataset.
      composite_data_information = multiblock_source.GetDataInformation().GetCompositeDataInformation()
      number_blocks              = composite_data_information.GetNumberOfChildren()

      for block_index in range( number_blocks ):
          block_data_information = composite_data_information.GetDataInformation( block_index )

          # skip over blocks that don't have any data.
          if block_data_information is None:
              continue

          extracted_block = extract_block( composite_data_information.GetName( block_index ),
                                           multiblock_source,
                                           block_data_information.GetDataClassName(),
                                           block_data_information,
                                           block_index )

          _ = Show( extracted_block )

    Takes 5 arguments:

      filter_name       - Name of the programmable filter to create.  May be specified
                          as an empty string whereby ParaView will generate a name for
                          the filter.
      multiblock_source - Multi-block ParaView object to extract data from.
      output_type       - String specifying the extracted block's dataset type.  Must
                          be a valid string for a ProgrammableFilter's .OutputDataSetType
                          property.  If specified as one of 'vtkStructuredGrid',
                          'vtkRectilinearGrid', 'vtkImageData', or 'vtkUniformGrid' the
                          extracted block's extents will match those found in the source
                          object.
      data_information  - vtkDataInformation proxy for the block to extract.
      block_index       - Non-negative integer index specifying the block in multiblock_source
                          to extract.

    Returns 1 value:

      programmable_filter - paraview.simple.ProgrammableFilter created.

    """

    if filter_name:
        programmable_filter = pv.ProgrammableFilter( multiblock_source,
                                                     registrationName=filter_name )
    else:
        programmable_filter = pv.ProgrammableFilter( multiblock_source )

    # specify the output type.
    programmable_filter.OutputDataSetType = output_type

    # create a script that inserts the requested block into the output filter's
    # dataset.
    #
    # NOTE: no validation on the block index is done here...
    #
    programmable_filter.Script = """
input = self.GetInputDataObject( 0, 0 )
self.GetOutputDataObject( 0 ).ShallowCopy( input.GetBlock( {:d}) )
""".format( block_index )

    # copy over the extent information when working with a structured grid.
    # without this, the resulting data would have extents based on the size of
    # each dimension rather than the values associated with each.
    if (output_type == "vtkImageData" or
        output_type == "vtkStructuredGrid" or
        output_type == "vtkRectilinearGrid"):
        programmable_filter.RequestInformationScript = """
import paraview.util
paraview.util.SetOutputWholeExtent( self, {} )
""".format( list( data_information.GetExtent() ) )

    return programmable_filter

def get_variable_point_arrays( source_name, variable_names, shape=None, block_index=-1 ):
    """
    Extracts one or more variables PointArray data from a named source.  Data from
    multi-block datasets can be extracted by specifying a block of interest.

    Raises exceptions if the requested source does not exist, when the PointArray
    data cannot be reshaped or if the source object does not have the requested block.

    Raises ValueError if the named source does not exist or if it does not contain
    the requested variables.

    Raises RuntimeError if a specific block is not requested and a viable one cannot
    be found in the dataset.

    Takes 4 arguments:

      source_name    - Name of the ParaView source to extract data from.
      variable_names - List of variable names to extract PointArray data from
                       source_name.
      shape          - Optional shape tuple to reshape the extracted PointArray data
                       by.  The product of the shape values must be equal to the length
                       of the underlying data, otherwise an exception is raised.  One
                       of the dimensions may be specified as -1 so that it is computed
                       based on the underlying data's size, with respect to the remaining
                       shape values. If omitted, each variable's data are returned
                       as a 1D NumPy array.
      block_index    - Optional integer specifying which block to extract data from
                       when source_name specifies a multi-block dataset.  This is
                       ignored when operating on a single block dataset.  If omitted,
                       defaults to the first block containing data.

    Returns 1 value:

      point_arrays - List of VTK wrapped arrays, one per variable, matching the order
                     specified by variable_names.

    """

    # ensure that a ParaView source exists by this name.
    paraview_source = pv.FindSource( source_name )

    if paraview_source is None:
        raise ValueError( "'{:s}' does not exist!".format(
            source_name ) )

    # pull all of the data locally, and wrap it so we can get NumPy-compatible
    # Arrays.
    #
    # NOTE: this gathers all of the data to a single system.  be careful with
    #       large sources...
    #
    source_vtk      = pv.servermanager.Fetch( paraview_source )
    source_wrapper  = dsa.WrapDataObject( source_vtk )

    # determine how we find the arrays of interest.
    multiblock_flag = (paraview_source.GetDataInformation().GetDataSetType() ==
                       vtk.VTK_MULTIBLOCK_DATA_SET)

    # find a suitable block if one wasn't requested.
    #
    # NOTE: this raises RuntimeError if we can't find a suitable block.
    #
    if multiblock_flag and (block_index == -1):
        block_index = _locate_data_in_multiblock( paraview_source )

    arrays = []

    # iterate through each of the requested variables and add their point
    # data to the caller's list.
    for variable_name in variable_names:

        # get the point data, while respecting the multi-block organization.
        if multiblock_flag:
            array = source_wrapper.PointData[variable_name].Arrays[block_index]
        else:
            array = source_wrapper.PointData[variable_name]

        # verify that each of the variables requested exist in this source.
        if isinstance( array, dsa.VTKNoneArray ):
            raise ValueError( "'{:s}' does not exist in '{:s}' block index {:d}!".format(
                variable_name,
                source_name,
                block_index ) )

        # reshape the data if requested.
        #
        # XXX: we should see if the supplied shape is compatible earlier on
        #      rather than catch fire here.
        #
        if shape is not None:
            array = array.reshape( shape )

        arrays.append( array )

    return arrays

def load_xdmf_dataset( source_name, xdmf_path, variables_of_interest, render_variable=None, show_flag=True, xdmf_v2_flag=True ):
    """
    Loads a XDMF dataset using a ParaView XDMF data source and makes it visible,
    rendering with one of the variables loaded.

    NOTE: This resets the camera view to center on the XDMF dataset.  The active
          camera not modified if the dataset isn't requested to be visible (see
          show_flag below).

    Raises ValueError if the variables requested weren't in the underlying dataset, if
    variables weren't requested, or if the requested render variable doesn't exist.
    Any created ParaView objects are destroyed before raising an exception so as to
    avoid cluttering the data pipeline with partially configured objects.

    Takes 6 arguments:

      source_name           - Name of the XDMF dataset to load.  This is the name of
                              the loaded source object.
      xdmf_path             - Path to the XDMF file to load.
      variables_of_interest - Non-empty list of variable names to load.  Each variable
                              specified must exist in the dataset described by xdmf_path.
      render_variable       - String specifying the variable name to render the loaded
                              dataset with.  Must be in variables_of_interest.
      show_flag             - Optional flag specifying whether the source object should
                              be visible when the method returns.  If specified as False
                              the source object is hidden.  If omitted, defaults to True.
      xdmf_v2_flag          - Optional flag specifying whether the ParaView XDMF v2 reader
                              should be used, or if a v3 reader is needed.  If omitted,
                              defaults to True which is a reasonable default for IWP
                              datasets.

    Returns 1 value:

      xdmf_source - ParaView source object representing the dataset described by xdmf_path.

    """

    # ensure that we have data to load from this dataset.
    if len( variables_of_interest ) == 0:
        raise ValueError( "Must load at least one variable from '{:s}'.  None were specified.".format(
            xdmf_path ) )

    # show the first variable we loaded instead of something unhelpful like
    # block index.
    if render_variable is None:
        render_variable = variables_of_interest[0]

    # verify the render variable is part of variables of interest.
    if render_variable not in variables_of_interest:
        raise ValueError( "Must render the XDMF dataset with one of the loaded variables.  "
                          "'{:s}' is not one of {:s}.".format(
            render_variable,
            ", ".join( map( lambda name: "'" + name + "'", variables_of_interest ) ) ) )

    # load a dataset from its XDMF description using the v2 XDMFReader source.
    # only load the variables of interest.
    #
    # NOTE: we could alternatively use the Xdmf3ReaderS or Xdmf3ReaderT sources,
    #       though they don't properly expose data through the NumPy.
    #
    if xdmf_v2_flag:
        xdmf_source = pv.XDMFReader( registrationName=source_name,
                                     FileNames=[xdmf_path] )
    else:
        raise NotImplementedError( "We don't currently support non-XDMF2 loading.  Sorry!" )

    # remove the simulation domain from the blocks loaded.  this avoids
    # the "(partial)" suffix for grid variables as the domain doesn't
    # have any.
    #
    # NOTE: querying .GridStatus, removing "simulation_domain", and setting it
    #       doesn't work, so we set it to all available timesteps.
    #
    xdmf_source.GridStatus = list( map( lambda x: "{:.0f}".format( x ),
                                        xdmf_source.TimestepValues ) )

    # only expose some of the grid variables.  make sure that each of them
    # exist first.
    for variable_name in variables_of_interest:
        if variable_name not in xdmf_source.PointArrayStatus:
            # we don't have a variable of interest.  cleanup behind ourselves so
            # we don't leave a dangling, partially configured source.
            pv.Delete( xdmf_source )
            del xdmf_source

            raise ValueError( "'{:s}' is not a variable available in '{:s}'!  "
                              "Variables available are {:s} .".format(
                                  variable_name,
                                  source_name,
                                  ", ".join( map( lambda name: "'" + name + "'",
                                                  variables_of_interest ) ) ) )
    xdmf_source.PointArrayStatus = variables_of_interest

    render_view = pv.GetActiveViewOrCreate( "RenderView" )

    # make this source visible.
    #
    # NOTE: we do this regardless of show_flag so that we can set render
    #       parameters when it is made visible.
    #
    xdmf_source_display = pv.Show( xdmf_source, render_view )

    # display the dataset as a surface colored by the render variable.  we want
    # the user to see their data immediately rather than a wireframe outline
    # colored by block index.
    _ = xdmf_source_display.SetRepresentationType( "Surface" )
    _ = pv.ColorBy( xdmf_source_display, ["POINTS", render_variable] )

    # configure the render view if the data should remain visible.
    if show_flag:
        # turn on the color limits axis.
        _ = xdmf_source_display.SetScalarBarVisibility( render_view, True )

        # zoom the camera out far enough to see the entire dataset.
        #
        # set the view so we're looking down on the simulation domain from
        # above, with positive Y going up and positive X going right (toward the
        # tow body).  we only set the camera's orientation and let ParaView
        # figure out a default based on the loaded data's extents.
        #
        # NOTE: this is likely zoomed out too far.
        #
        render_view.CameraViewUp = [0.0, 1.0, 0.0]
        render_view.ResetCamera()

        # render everything in the view.
        _ = pv.Render()
    else:
        # the caller doesn't want this visible, so toggle it off.
        _ = pv.Hide( xdmf_source )

    # enable the animation controls so we can interact with multiple time steps.
    animation_scene = pv.GetAnimationScene()

    _ = animation_scene.UpdateAnimationUsingDataTimeSteps()

    return xdmf_source

def is_line_like( source_name, object_flag=False ):
    """
    Predicate to determine if a ParaView source is Line-like and has the attributes
    "Point1" and "Point2".

    Raises ValueError if a source by the supplied name does not exist, or if it is
    not Line-like.  Returns nothing when it is Line-like.

    Takes 2 arguments:

      source_name - Name of the ParaView object to test.
      object_flag - Optional flag specifying whether source_name is actually a ParaView
                    source object instead of a name.  If specified as True, source
                    lookup by name is skipped.  If omitted, defaults to False.

    Returns nothing.

    """

    # look up the object by name if we need to.
    if not object_flag:
        line_like = pv.FindSource( source_name )
    else:
        line_like = source_name

    # ensure this source exists.
    if line_like is None:
        raise ValueError( "'{:s}' does not exist!".format(
            source_name ) )

    # ensure that it has two points to draw a line through.
    #
    # NOTE: we force the source "name" to a string to ensure we can format
    #       exception properly.  otherwise we have a type mismatch when
    #       object_flag is True.
    #
    if getattr( line_like, "Point1", None ) is None:
        raise ValueError( "'{:s}' is not Line-like and is missing the 'Point1' attribute!".format(
            str( source_name ) ) )
    elif getattr( line_like, "Point2", None ) is None:
        raise ValueError( "'{:s}' is not Line-like and is missing the 'Point2' attribute!".format(
            str( source_name ) ) )

    return

def compute_polar_coordinates( source_name, object_flag=False ):
    """
    Computes the polar coordinates of a Line-like ParaView source.  Coordinates
    are computed relative to the source's origin, rather than ParaView's origin,
    so that the source's characteristics can be measured.

    This is a convenience wrapper around compute_azimuth(),
    compute_polar_angle(), and compute_magnitude().

    Raises ValueError if the requested source is not Line-like.  See is_line_like()
    for details.

    Takes 2 arguments:

      source_name - Name of the ParaView object to test.
      object_flag - Optional flag specifying whether source_name is actually a ParaView
                    source object instead of a name.  If specified as True, source
                    lookup by name is skipped.  If omitted, defaults to False.

    Returns 3 values:

      azimuth   - Azimuth of source_name's object in degrees, clockwise relative to
                  the line X=0.  See compute_azimuth() for details on the value
                  returned when magnitude is 0 or the elevation is +-90 degrees.
      elevation - Elevation of source_name's object in degrees, up relative to XY
                  plane.  See compute_elevation() for details on the value
                  returned when magnitude is 0.
      magnitude - Magnitude of the source_name's object.

    """

    # look up the object by name if we need to.
    if not object_flag:
        # find the object so we don't have to do this once per coordinate.
        source_name = pv.FindSource( source_name )
        object_flag = True

    return (compute_azimuth( source_name, object_flag=object_flag ),
            compute_elevation( source_name, object_flag=object_flag ),
            compute_magnitude( source_name, object_flag=object_flag ))

def compute_azimuth( source_name, object_flag=False ):
    """
    Computes the azimuth of a Line-like ParaView source.  Azimuth is measured in
    degrees clockwise from the X=0 line (in a rotated NEU system).

    Takes 2 arguments:

      source_name - Name of the ParaView object to test.
      object_flag - Optional flag specifying whether source_name is actually a ParaView
                    source object instead of a name.  If specified as True, source
                    lookup by name is skipped.  If omitted, defaults to False.

    Returns 1 value:

      azimuth - Degrees, clockwise, relative to the line X=0.  The returned value
                for Line-like objects with 0 magnitude or elevations of +-90
                matches NumPy's arctan2() method.

    """

    # get our source.
    if not object_flag:
        line_like = pv.FindSource( source_name )
    else:
        line_like = source_name

    #
    # NOTE: this only returns when we have a Line-like object, otherwise
    #       we raise ValueError.
    #
    is_line_like( line_like, object_flag=True )

    point1 = np.array( line_like.Point1 )
    point2 = np.array( line_like.Point2 )

    return (np.arctan2( (point2[0] - point1[0]),
                        (point2[1] - point1[1]) ) / np.pi * 180) - 90

def compute_elevation( source_name, object_flag=False ):
    """
    Computes the elevation of a Line-like ParaView source.  Elevation is measured in
    degrees up from the XY plane (in a rotated NEU system).

    Takes 2 arguments:

      source_name - Name of the ParaView object to test.
      object_flag - Optional flag specifying whether source_name is actually a ParaView
                    source object instead of a name.  If specified as True, source
                    lookup by name is skipped.  If omitted, defaults to False.

    Returns 1 value:

      elevation - Degrees, up, from the XY plane.  The returned value for Line-like
                  objects with 0 magnitude matches NumPy's arctan2() method.

    """

    # get our source.
    if not object_flag:
        line_like = pv.FindSource( source_name )
    else:
        line_like = source_name

    #
    # NOTE: this only returns when we have a Line-like object, otherwise
    #       we raise ValueError.
    #
    is_line_like( line_like, object_flag=True )

    point1 = np.array( line_like.Point1 )
    point2 = np.array( line_like.Point2 )

    return (np.arctan2( np.sqrt( (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 ),
                        (point2[2] - point1[2]) ) / np.pi * 180)

def compute_magnitude( source_name, object_flag=False ):
    """
    Computes the magnitude of a Line-like ParaView source.

    Takes 2 arguments:

      source_name - Name of the ParaView object to test.
      object_flag - Optional flag specifying whether source_name is actually a ParaView
                    source object instead of a name.  If specified as True, source
                    lookup by name is skipped.  If omitted, defaults to False.

    Returns 1 value:

      magnitude - Non-negative length of source_name.

    """

    # get our source.
    if not object_flag:
        line_like = pv.FindSource( source_name )
    else:
        line_like = source_name

    #
    # NOTE: this only returns when we have a Line-like object, otherwise
    #       we raise ValueError.
    #
    is_line_like( line_like, object_flag=True )

    point1 = np.array( line_like.Point1 )
    point2 = np.array( line_like.Point2 )

    return np.sqrt( (point2[0] - point1[0])**2 +
                    (point2[1] - point1[1])**2 +
                    (point2[2] - point1[2])**2 )

def delete( source_names, object_flag=False ):
    """
    Deletes one or more ParaView objects by name or reference.  Releases resources for
    both the ParaView object and the underlying Python object.

    Takes 2 arguments:

      source_names - List of ParaView source names to delete.
      object_flag  - Optional flag specifying whether source_names contains names
                     (False) or object references (True).  If omitted, defaults to False.

    Returns nothing.

    """

    render_view = pv.GetActiveViewOrCreate( "RenderView" )

    for source_name in source_names:
        if object_flag:
            paraview_object = source_name
        else:
            paraview_object = pv.FindSource( source_name )

            if paraview_object is None:
                warnings.warn( "Cannot find an object named '{:s}.  Skipping.'".format(
                    source_name ) )

        _ = pv.Hide( paraview_object )
        pv.Delete( paraview_object )
        del paraview_object

    # force a visual update to remove any objects we just deleted.
    pv.Render()

    return

def hide( source_names, object_flag=False ):
    """
    Hides one or more ParaView objects by name or reference.

    Takes 2 arguments:

      source_names - List of ParaView source names to hide.
      object_flag  - Optional flag specifying whether source_names contains names
                     (False) or object references (True).  If omitted, defaults to False.

    Returns nothing.

    """

    for source_name in source_names:
        if object_flag:
            paraview_object = source_name
        else:
            paraview_object = pv.FindSource( source_name )

            if paraview_object is None:
                warnings.warn( "Cannot find an object named '{:s}.  Skipping.'".format(
                    source_name ) )

        _ = pv.Hide( paraview_object )

    # force a visual update to remove any objects we just hid.
    pv.Render()

    return

def show( source_names, object_flag=False ):
    """
    Shows one or more ParaView objects by name or reference.

    Takes 2 arguments:

      source_names - List of ParaView source names to show.
      object_flag  - Optional flag specifying whether source_names contains names
                     (False) or object references (True).  If omitted, defaults to False.

    Returns nothing.

    """

    for source_name in source_names:
        if object_flag:
            paraview_object = source_name
        else:
            paraview_object = pv.FindSource( source_name )

            if paraview_object is None:
                warnings.warn( "Cannot find an object named '{:s}.  Skipping.'".format(
                    source_name ) )

        _ = pv.Show( paraview_object )

    # force a visual update to show any objects we just exposed.
    pv.Render()

    return

def get_structured_grid_coordinates( source_name, block_index=-1 ):
    """
    Returns the grid coordinates for a structured grid source.  Only

    Raises ValueError if the supplied source is not a structured grid.

    Raises RuntimeError if supplied a multi-block source and the requested block
    is invalid.

    Takes 2 arguments:

      source_name - Name of the ParaView source to extract coordinates from.  Must
                    be one of "vtkImageClass", "vtkRectilinearGrid", or
                   "vtkStructuredGrid" otherwise ValueError is raised.
      block_index - Optional index specifying which block, from a multi-block source,
                    to extract coordinates from.  If omitted, it defaults to the
                    first block that contains data.  This parameter is ignored when
                    source_name is not a multi-block object.

                      NOTE: This ignores non-data blocks which may contain grids.
                            If those are of interest, specify the appropriate block
                            index.

    Returns 1 value:

      coordinates - Tuple with 3 elements, each a NumPy array containing the coordinates
                    for the X-, Y-, and Z-axes, respectively.

    """

    # find the source object and get its underlying VTK object.
    paraview_source = pv.FindSource( source_name )
    vtk_source      = pv.servermanager.Fetch( paraview_source )

    multiblock_flag = (paraview_source.GetDataInformation().GetDataSetType() ==
                       vtk.VTK_MULTIBLOCK_DATA_SET)

    # hide(ish) the fact that the original object was multi-block.
    if multiblock_flag:
        # find a suitable block if one wasn't requested.
        #
        # NOTE: this raises RuntimeError if we can't find a suitable block.
        #
        if block_index == -1:
            block_index = _locate_data_in_multiblock( paraview_source )

        vtk_source = vtk_source.GetBlock( block_index )

    # we now have *the* VTK object.
    vtk_class_name = vtk_source.GetClassName()

    # ensure that we're working with an object that has 1D coordinate.
    structured_grid_names = ["vtkImageData",
                             "vtkRectilinearGrid",
                             "vtkStructuredGrid"]
    if vtk_class_name not in structured_grid_names:
        raise ValueError( "'{:s}' is not a structured grid!  Must be one of {:s}.".format(
            vtk_class_name,
            ", ".join( map( lambda x: "'" + x + "'",
                            structured_grid_names ) ) ) )

    # get the VTKDoubleArray objects for each of the coordinate axes.
    vtk_x_coordinates = vtk_source.GetXCoordinates()
    vtk_y_coordinates = vtk_source.GetYCoordinates()
    vtk_z_coordinates = vtk_source.GetZCoordinates()

    grid_shape = (vtk_x_coordinates.GetSize(),
                  vtk_y_coordinates.GetSize(),
                  vtk_z_coordinates.GetSize())

    # pre-allocate our coordinate arrays.
    x_coordinates = np.empty( (grid_shape[0], ) )
    y_coordinates = np.empty( (grid_shape[1], ) )
    z_coordinates = np.empty( (grid_shape[2], ) )

    # pull each of the arrays' entries one by one.
    #
    # NOTE: there has got to be a better interface here, though until then, be
    #       mindful of the performance characteristics of this method.
    #
    for x_index in range( grid_shape[0] ):
        x_coordinates[x_index] = vtk_x_coordinates.GetTuple1( x_index )
    for y_index in range( grid_shape[1] ):
        y_coordinates[y_index] = vtk_y_coordinates.GetTuple1( y_index )
    for z_index in range( grid_shape[2] ):
        z_coordinates[z_index] = vtk_z_coordinates.GetTuple1( z_index )

    return (x_coordinates, y_coordinates, z_coordinates)

def create_xy_labels( iwp_labels, z_coordinates, color=None ):
    """
    Creates 2D labels in ParaView from a list of IWP labels.  Each 2D label created
    is a box outline in a particular XY slice so as to highlight an IWP's features.
    Created labels do not have a concept of time and are simply positioned within the
    visualized domain.

    Each label has a structured name so as to be easily identified, and adhere to the
    following convention:

       XY Label - <label id> (z/D=<z coord>)

    Where <z coord> has 2 digits after the decimal point.

    If a ParaView label with the same name already exists, its position and extent
    are updated.

    NOTE: This method will create arbitrary numbers of label objects though callers
          should be mindful that large numbers of labels will slow ParaView during
          creation and clutter its pipeline browser.  Note that this is exacerbated
          even further if Python tracing is enabled.

    Takes 3 arguments:

      iwp_labels    - List of IWP labels to create XY labels for.
      z_coordinates - NumPy array of Z coordinates for the labels.  Used to translate
                      IWP labels' z_index into the coordinate system their bounding
                      boxes are already in.
      color         - Optional sequence, with three elements, specifying the RGB
                      values of the label.  Each element must be in the range [0, 1].
                      If omitted, defaults to magenta (1, 0, 1).

    Returns 2 values:

      paraview_labels - List of ParaView objects created or updated, one per label, in
                        the same order as iwp_labels.
      label_names     - List of label names created or updated, one per label, in the
                        same order as iwp_labels.

    """

    # default the label color if not supplied one.  we use magenta since it
    # is high contrast to most of the common colormaps.
    if color is None:
        color = [1.0, 0.0, 1.0]

    # get a rendering context so we can display the labels below.
    render_view = pv.GetActiveViewOrCreate( "RenderView" )

    # we return both objects and names of the labels created.
    paraview_labels = []
    label_names     = []

    for iwp_label in iwp_labels:
        # map from indices to coordinates.  we assume the rest of the label is
        # already in the appropriate coordinate system.
        z_coordinate = z_coordinates[iwp_label["z_index"]]

        label_name = "XY Label - {:s} (z/D={:.2f})".format(
            iwp_label["id"],
            z_coordinate )

        # create a new label if this one doesn't already exist.  otherwise we're
        # updating the existing.
        paraview_label = pv.FindSource( label_name )
        if paraview_label is None:
            paraview_label = pv.Box( registrationName=label_name )

        # convert our (top-left, bottom-right) labels into (center, extent)
        # boxes.
        #
        # NOTE: we're creating planar labels without any vertical extent.
        #
        paraview_label.Center = [(iwp_label["bbox"]["x2"] - iwp_label["bbox"]["x1"]) / 2 + iwp_label["bbox"]["x1"],
                                 (iwp_label["bbox"]["y2"] - iwp_label["bbox"]["y1"]) / 2 + iwp_label["bbox"]["y1"],
                                 z_coordinate]

        paraview_label.XLength = iwp_label["bbox"]["x2"] - iwp_label["bbox"]["x1"]
        paraview_label.YLength = iwp_label["bbox"]["y2"] - iwp_label["bbox"]["y1"]
        paraview_label.ZLength = 0

        paraview_label_display = pv.Show( paraview_label, render_view, "OutlineRepresentation" )

        # set the label's color.
        paraview_label_display.AmbientColor = color
        paraview_label_display.DiffuseColor = color

        # make these visible by thickening the border.
        paraview_label_display.LineWidth = 3.0

        # XXX: create a text annotation with the label identifier so they can be
        #      visually recognized.

        paraview_labels.append( paraview_label )
        label_names.append( label_name )

    # make each of the labels visible.
    pv.Render()

    return paraview_labels, label_names
