import paraview.simple as pv

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
    XXX: specify the actual exceptions.

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
    if multiblock_flag and (block_index == -1):
        composite_data_information = paraview_source.GetDataInformation().GetCompositeDataInformation()
        number_blocks              = composite_data_information.GetNumberOfChildren()

        # walk through each of the blocks and stop at the first one containing
        # data.
        for block_index in range( number_blocks ):
            data_information = composite_data_information.GetDataInformation()

            # blocks that do not contain data (e.g. grid only) will not have a
            # DataInformation structure.
            if data_information is None:
                continue
            else:
                break

        # handle the case where we can't find a suitable block.
        if block_index == number_blocks:
            raise RuntimeError( "Failed to find a suitable block!" )

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
