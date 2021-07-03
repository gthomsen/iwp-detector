import copy
import numpy as np
import os

import iwp.data_loader
import iwp.utilities

class XDMFGenerator( object ):
    """
    XDMF generator for IWP datasets.  Generators are validated against the metadata
    within a dataset to ensure the XDMF documents generated are correct as most
    tools consuming them are not robust to malformed/slightly erroneous documents.

    Creates a 3D rectilinear mesh so the underlying grid variables may be represented
    by structured data types.  Additionally, a collection of temporal grids are
    created, one per time step, each containing one or more grid variables.

    Each time step's grid geometry is identical by construction with XInclude
    references.  Currently each time step's identifier is the time step index in
    the underlying dataset, rather than the simulation time.

    NOTE: The generated XDMF document does *NOT* validate against the "official"
          XDMF DTD as it uses XML XIncludes to reference the computational grid
          and its topology within each time step.  While this is malformed with
          respect to the v2 specification, the documents generated are compatible
          with both the v2 and v3 XDMF readers in ParaView (tested with v5.7.x
          and v5.9.y).

    """

    #
    # NOTE: the 3D rectilinear mesh generated must have the following
    #       characteristics otherwise ParaView's v2 and v3 XDMF readers (built
    #       on VTK's) will fail to read them.  at best, ParaView will provide
    #       an error message giving something to search for in the code (rare),
    #       while at worst, ParaaView will crash without a message (common).
    #
    #         1. the simulation domain's topology type must be "3DRectMesh".
    #            this causes the XDMF reader to expect structured, compact
    #            coordinates to define the domain's geometry.
    #
    #         2. the simulation domain's geometry must be "VXVYVZ".  this
    #            causes the XDMF reader to expect vectors for each of the
    #            coordinate axes.  the implicit outer product of each of the
    #            axes (X, Y, and Z) specify the 3D mesh used.  no other
    #            specification of the geometry can be specified per
    #            the ParaView developer Utkarsh Ayachit:
    #
    #              https://vtk.org/Bug/view.php?id=9582#c17728
    #
    #            while this ensures there is only a single way to specify
    #            a rectilinear mesh, it is by no means well documented...
    #
    _xdmf_fragment = """<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
    <Domain>

        <!--
             Define a 3D rectilinear mesh (shaped {dimension_x:d}x{dimension_y:d}x{dimension_z:d}) and specify each of the
             axes in three, 1D vectors.

             NOTE: The (X, Y, Z) grid is specified in C-major order with the fastest
                   dimension on the right, causing the dimensions specification to be
                   reversed as (Z, Y, X).
        -->

        <Grid Name="simulation_domain" GridType="Uniform">
            <Topology TopologyType="3DRectMesh" Dimensions="{dimension_z:d} {dimension_y:d} {dimension_x:d}"/>

            <Geometry GeometryType="VXVYVZ">
                <DataItem Name="X" Dimensions="{dimension_x:d}" NumberType="{number_type_x:s}" Precision="{size_bytes_x:d}" Format="HDF">
                    {coordinates_path}:/x
                </DataItem>
                <DataItem Name="Y" Dimensions="{dimension_y:d}" NumberType="{number_type_y:s}" Precision="{size_bytes_y:d}" Format="HDF">
                    {coordinates_path}:/y
                </DataItem>
                <DataItem Name="Z" Dimensions="{dimension_z:d}" NumberType="{number_type_z:s}" Precision="{size_bytes_z:d}" Format="HDF">
                    {coordinates_path}:/z
                </DataItem>
            </Geometry>

        </Grid>

        <Grid Name="timesteps" CollectionType="Temporal" GridType="Collection">
{time_step_fragments:s}
        </Grid>
    </Domain>
</Xdmf>
"""
    _time_step_fragment = """
            <Grid Name="{time_step:d}" GridType="Uniform">
                <Time Value="{time_step:d}" />

                <!-- Reference the Topology and Geometry nodes via XInclude. -->
                <xi:include xpointer="element(/1/1/1/1)"/>
                <xi:include xpointer="element(/1/1/1/2)"/>
{variable_fragments:s}
            </Grid>
"""

    _variable_fragment = """
                <Attribute Name="{variable_name:s}" AttributeType="Scalar" Center="Node">
                    <DataItem NumberType="Float" Precision="{variable_byte_size:d}" Dimensions="{dimension_z:d} {dimension_y:d} {dimension_x:d}" Format="HDF">
                    {file_path:s}:/{variable_name:s}
                    </DataItem>
                </Attribute>"""

    def __init__( self, dataset_path_template, time_step_indices, variable_names, override_map={} ):
        """
        Initializes an XDMFGenerator object from a multi-file netCDF4 dataset.  Metadata
        for the specified time steps and grid variables are retrieved and validated
        against the dataset so that XDMF documents may be generated.

        The netCDF4 dataset described by this generator must exist when the object is
        initialized.

        XDMF documents referencing multiple multi-file netCDF4 datasets may be
        generated by specifying additional variables, and their associated dataset
        paths, via an override.  This supports datasets that were generated in
        stages rather than all at once (e.g. simulation creating one dataset and
        post-processing for analysis creating additional datasets).

        Raises ValueError if any of the datasets cannot be opened or are not compatible
        with the supplied arguments.

        Takes 4 arguments:

          dataset_path_template - Path template to the multi-file netCDF4 dataset.  Paths
                                  to individual time step files are instantiated by substituting
                                  each time step index into the template.
          time_step_indices     - Sequence of time step indices in the dataset.
          variable_names        - List of variable names to expose in generated XDMF
                                  documents.  Variables not in the override map must exist
                                  in each of the individual time step files instantiated
                                  from dataset_path_template.
          override_map          - Optional dictionary mapping variable names to alternative
                                  dataset path templates.  Each variable in the map must
                                  be specified in variable_names and, additionally, must
                                  exist in the dataset associated with it.  If omitted,
                                  defaults to an empty dictionary.

        """

        # template to the default dataset used for all variables and grid
        # coordinates.  only variables in the not in the override map come from
        # here.
        self._dataset_path_template = dataset_path_template

        # sequence of time step indices used to instantiate dataset path
        # templates into dataset paths.
        self._time_step_indices = time_step_indices

        # list of variable names, both default and overriden, that XDMF will be
        # generated for.
        self._variable_names = variable_names

        # map from variable name to numpy.dtype strings (e.g. 'float32').  has
        # one entry per variable in _variable_names.
        self._dtype_map = {}

        # 4-tuple specifying the shape of the grid variables, ordered as
        # (time, z, y, x), as stored within the netCDF4 datasets.
        self._grid_shape = ()

        # create a map from dataset path templates to a list of overriden
        # variables.  we use this so we only validate the override datasets
        # once, rather than once per overriden variable.
        override_variable_map = {}
        for override_variable_name, override_path_template in override_map.items():
            override_variable_map.setdefault( override_path_template,
                                              [] ).append( override_variable_name )

        # verify that the overriden variables are a subset of the variables
        # requested.
        for override_variable_name in override_map.keys():
            if override_variable_name not in variable_names:
                raise ValueError( "Unknown variable specified as an override ('{:s}').".format(
                    override_variable_name ) )

        # build a map from variable names to their dataset.  start with the
        # override map and default the remaining variables to the default path
        # template.
        self._dataset_path_map = copy.copy( override_map )
        for variable_name in self._variable_names:
            if variable_name not in self._dataset_path_map:
                self._dataset_path_map[variable_name] = self._dataset_path_template

        # open the dataset to get the underlying grid shape and variable data
        # types.
        try:
            # get each of the paths to this data set's individual time steps.
            dataset_paths = []
            for time_step_index in time_step_indices:
                dataset_paths.append( dataset_path_template.format( time_step_index ) )

            # create a list of variables that aren't overriden.  these need to
            # be validated in the default dataset.  we do not validate overriden
            # variables because they're not guaranteed to exist in this dataset.
            #
            # NOTE: we flatten our variables into a set to remove any duplicates
            #       so validation isn't made more complex.
            #
            dataset_variable_names = list( set( variable_names ) )

            for override_variable_name in override_map.keys():
                variable_index = dataset_variable_names.index( override_variable_name )
                del dataset_variable_names[variable_index]

            # open the dataset and get the characteristics needed to generate our
            # XDMF.
            (master_dataset,
             self._grid_shape,
             self._dtype_map) = self._open_and_validate_dataset( dataset_paths,
                                                                 dataset_variable_names,
                                                                 time_step_indices )

            # we no longer need the underlying dataset.  all XDMF generation is done
            # from the metadata we've acquired.
            del master_dataset

        except ValueError as e:
            raise ValueError( "Failed to open the dataset specified by '{:s}' - {:s}".format(
                dataset_path_template,
                str( e ) ) )

        # verify the override datasets have the same grid shape and variable
        # data types.  walk through each of the override datasets and validate
        # that they're compatible with the master dataset for each of the
        # variables they override.
        for override_path_template in override_variable_map.keys():
            try:
                # get each of the paths to this dataset's individual time steps.
                override_dataset_paths = []
                for time_step_index in time_step_indices:
                    override_dataset_paths.append( override_path_template.format( time_step_index ) )

                # validate this dataset against the expected dataset
                # characteristics.  get an updated data type map that includes
                # this dataset's overriden variables.
                (override_dataset,
                 _,
                 self._dtype_map) = self._open_and_validate_dataset( override_dataset_paths,
                                                                     override_variable_map[override_path_template],
                                                                     time_step_indices,
                                                                     target_grid_shape=self._grid_shape,
                                                                     target_dtype_map=self._dtype_map )

                # free the resources associated with this dataset.  we don't
                # actually use it for anything other than validation.
                del override_dataset

            except ValueError as e:
                raise ValueError( "Failed to open the override dataset for "
                                  "'{:s}' - {:s}".format(
                                      override_variable_name,
                                      str( e ) ) )

        return

    @staticmethod
    def _open_and_validate_dataset( dataset_paths, variable_names, time_step_indices, target_grid_shape=None, target_dtype_map={} ):
        """
        Static class method that opens a netCDF4 dataset from one or more paths.  The
        contents of the dataset are validated against the callers expectations and
        key characteristics are returned post-validation.

        Raises ValueError() if the underlying dataset does not exist or is inconsistent
        with the provided arguments.

        Takes 5 arguments:

          dataset_paths     - List of path names comprising the netCDF4 dataset.  The
                              order matches that supplied in time_step_indices.
          variable_names    - List of variable names to validate exist in each of
                              the dataset paths in dataset_paths.
          time_step_indices - List of time step indices to validate exist in the
                              dataset.  The i-th time step index must exist in the
                              i-th dataset path.
          target_grid_shape - Optional 4-tuple specifying the expected grid variables'
                              shape, with (time, x, y, z) as the ordering.  All grid
                              variables must have this shape to pass validation.  If
                              omitted, defaults to the shape of the first entry in
                              variable_names.
          target_dtype_map  - Optional dictionary mapping variable names to numpy.dtype
                              strings (e.g. 'float32').  Each variable in variable_names
                              must have the expected data type.  If omitted, or if it
                              does not include one of variable_names' variables, no data
                              type validation is performed.

        Returns 3 values:

          dataset       - xarray.Dataset opened from dataset_paths.
          grid_shape    - 4-tuple specifying the grid variables' shape with (time, x, y, z)
                          as the ordering.
          dtype_map - Dictionary mapping variable names to numpy.dtype strings (e.g.
                          'float32').  Provides a mapping for each of the variables in
                          variable_names.

        """

        # validate that we got one dataset path per time step.
        if len( dataset_paths ) != len( time_step_indices ):
            raise ValueError( "Received {:d} dataset paths, but only {:d} time step ind{:s}.".format(
                len( dataset_paths ),
                len( time_step_indices ),
                "ex" if len( time_step_indices ) == 1 else "ices" ) )

        # update a local copy of the target data types.
        dtype_map = copy.copy( target_dtype_map )

        # verify each of the paths exists on disk.  this simplifies debugging as
        # xarray is not great a giving a sensible error message when a dataset
        # path is non-existent.
        for dataset_path in dataset_paths:
            if not os.path.isfile( dataset_path ):
                raise ValueError( "Dataset member does not exist ({:s})!".format(
                    dataset_path ) )

        # create a dataset from the paths provided.
        #
        # NOTE: we use xarray to open the multi-file dataset so we have the most
        #       robustness against malformed netCDF4 files.  as of 2021/07/01
        #       IWP datasets do not have an aggregation dimension (i.e. a time
        #       dimension with unlimited size).
        #
        dataset = iwp.data_loader.open_xarray_dataset( dataset_paths )

        # validate that the variables and time steps of interest are present in
        # this data set.
        #
        # NOTE: we don't care about the XY slice indices here, so we request the
        #       first.
        #
        try:
            iwp.utilities.validate_variables_and_ranges( dataset,
                                                         variable_names,
                                                         time_step_indices,
                                                         [0] )
        except Exception as e:
            raise ValueError( "Failed to validate the dataset - {:s}".format(
                str( e ) ) )

        # ensure that we have a 3D grid named "x", "y", and "z".
        if not ("x" in dataset.dims and
                "y" in dataset.dims and
                "z" in dataset.dims):
            raise ValueError( "Dataset does not have an 3D grid named (x, y, z)." )

        # validate that each of the variables have the correct shape and data
        # type.  walk through and validate against supplied characteristics or
        # default to the first encountered.
        for variable_name in variable_names:

            variable = dataset.variables[variable_name]

            variable_shape      = variable.shape
            variable_dimensions = variable.dims
            variable_dtype      = variable.dtype.name

            # get the desired data type if it already exists.  otherwise we'll
            # default to this variable's data type later.
            target_dtype = dtype_map.get( variable_name, None )

            # validate the grid shape.  we must both have the proper shape as
            # well as the expected dimensions.
            if target_grid_shape is None:
                target_grid_shape = variable_shape
            else:
                if variable_shape != target_grid_shape:
                    raise ValueError( "Mismatched grid variable shapes! '{:s}' has "
                                      "shape {:s} though expected {:s}.".format(
                                          variable_name,
                                          "x".join( variable_shape ),
                                          "x".join( target_grid_shape ) ) )

                if not ("x" in variable_dimensions and
                        "y" in variable_dimensions and
                        "z" in variable_dimensions):
                    raise ValueError( "{:s} is not a 3D grid variable ({:s}).".format(
                        variable_name,
                        ", ".join( variable_dimensions ) ) )

            # validate the variable's data type.
            if target_dtype is None:
                # this variable sets the target.
                target_dtype = variable_dtype

                dtype_map[variable_name] = variable_dtype
            elif variable_dtype != target_dtype:
                raise ValueError( "'{:s}' is '{:s}' but expected '{:s}'.".format(
                    variable_name,
                    variable_dtype,
                    target_dtype ) )

        # treat the grid coordinates as grid variables and add into the
        # data type map.  use the corresponding grid variable if it exists,
        # otherwise fall back to the coordinate indices.
        #
        # NOTE: this won't overwrite a valid grid variable's dtype since we
        #       ensure these are dataset dimensions.  the corresponding
        #       variables (if they exist) should be one dimensional and would
        #       have failed validation for not being 3D above.
        #
        # NOTE: we only check that "x" is in the map rather than each
        #       individually since we ensure that they all exist above.
        #       this is equivalent to "have we updated the map or not".
        #
        if "x" not in dtype_map:
            if "x" in dataset.variables:
                dtype_map["x"] = dataset.variables["x"].dtype.name
            else:
                dtype_map["x"] = dataset.coords["x"].dtype.name
            if "y" in dataset.variables:
                dtype_map["y"] = dataset.variables["y"].dtype.name
            else:
                dtype_map["y"] = dataset.coords["y"].dtype.name
            if "z" in dataset.variables:
                dtype_map["z"] = dataset.variables["z"].dtype.name
            else:
                dtype_map["z"] = dataset.coords["z"].dtype.name

        return dataset, target_grid_shape, dtype_map

    def generate( self, time_step_indices=[], variable_names=[] ):
        """
        Generates an XDMF document describing the underlying netCDF4 datasets.  May
        generate a description of a subset of the underlying datasets if requested.

        Takes 2 arguments:

          time_step_indices - Optional sequence of time step indices to output in
                              the generated XDMF.  Must be a subset of the indices
                              specified during object creation.  If omitted, defaults
                              to the indices specified during object creation.
          variable_names    - Optional list of variable names to output in the
                              the generated XDMF.  Must be a subset of the names
                              specified during object creation.  If omitted, defaults
                              to the variables specified during object creation.

        Returns 1 value:

          xdmf_document - String containing the serialized XDMF document.

        """

        def numpy_dtype_to_xdmf_number_type( dtype ):
            """
            Converts a NumPy dtype to an XDMF NumberType.

            Takes 1 argument:

              dtype - NumPy dtype to convert.

            Returns 1 value:

              number_type - String indicating the XDMF NumberType corresponding to
                            dtype.  Will be "UNKNOWN" if dtype cannot be converted.

            """

            if dtype.kind == "f":
                return "Float"
            elif dtype.kind == "i":
                return "Int"
            elif dtype.kind == "u":
                return "Uint"

            return "UNKNOWN"

        # default to the metadata read during initialization if the caller isn't
        # overriding it.
        if len( time_step_indices ) == 0:
            time_step_indices = self._time_step_indices
        if len( variable_names ) == 0:
            variable_names = self._variable_names

        # ensure that we're generating data for things we've already validated
        # during initialization.
        for time_step_index in time_step_indices:
            if time_step_index not in self._time_step_indices:
                raise ValueError( "Invalid time step index requested ({:d}).  Must be one of {}.".format(
                    time_step_index,
                    self._time_step_indices ) )

        for variable_name in variable_names:
            if variable_name not in self._variable_names:
                raise ValueError( "Invalid variable requested ('{:s}').  Must be one of {:s}.".format(
                    variable_name,
                    ", ".join( map( lambda x: "'" + x + "'", self._variable_names ) ) ) )

        # decompose the grid into individual dimensions.  we skip the leading
        # time dimension.
        dimension_x = self._grid_shape[3]
        dimension_y = self._grid_shape[2]
        dimension_z = self._grid_shape[1]

        dtype_x = np.dtype( self._dtype_map["x"] )
        dtype_y = np.dtype( self._dtype_map["y"] )
        dtype_z = np.dtype( self._dtype_map["z"] )

        # get the size of each of the coordinate axes.  these *should* all be
        # the same, though we attempt to be robust against strange datasets...
        size_bytes_x = dtype_x.itemsize
        size_bytes_y = dtype_y.itemsize
        size_bytes_z = dtype_z.itemsize

        # get the data type of each of the coordinate axes.  again, these
        # *should* all be the same...
        number_type_x = numpy_dtype_to_xdmf_number_type( dtype_x )
        number_type_y = numpy_dtype_to_xdmf_number_type( dtype_y )
        number_type_z = numpy_dtype_to_xdmf_number_type( dtype_z )

        # build each time step's XML, one at a time.
        time_step_fragments = ""
        for time_step_index in time_step_indices:

            # assemble each of the variable's descriptions, one at a time.
            variable_fragments = ""
            for variable_name in variable_names:

                # lookup this variable's characteristics.
                dataset_path   = self._dataset_path_map[variable_name].format( time_step_index )
                variable_dtype = np.dtype( self._dtype_map[variable_name] )

                # instantiate the variable fragment.
                variable_fragments += XDMFGenerator._variable_fragment.format(
                    file_path=dataset_path,
                    variable_name=variable_name,
                    variable_byte_size=variable_dtype.itemsize,
                    dimension_x=dimension_x,
                    dimension_y=dimension_y,
                    dimension_z=dimension_z )

            # instantiate the time step fragment.
            time_step_fragments += XDMFGenerator._time_step_fragment.format(
                time_step=time_step_index,
                variable_fragments=variable_fragments )

        # instantiate the document.  fill in the grid characteristics and
        # assemble the time steps.
        xdmf_document = XDMFGenerator._xdmf_fragment.format(
            coordinates_path=self._dataset_path_template.format( time_step_indices[0] ),
            number_type_x=number_type_x,
            number_type_y=number_type_y,
            number_type_z=number_type_z,
            size_bytes_x=size_bytes_x,
            size_bytes_y=size_bytes_y,
            size_bytes_z=size_bytes_z,
            dimension_x=dimension_x,
            dimension_y=dimension_y,
            dimension_z=dimension_z,
            time_step_fragments=time_step_fragments )

        return xdmf_document
