#!/usr/bin/env python3

# Tests for the IWPDataloader class.

# TODO:
#
#   * Create TestRealIWPDataset that runs tests on real IWP data.
#   * Reorder tests into some logical order
#   * Verify that the fixture test names match the methods they go with
#   * Document the different shapes
#       - xy_slice ~ (variables, x, y)
#       - parameters[grid_size] ~ (x, y, z)
#       - variables ~ (z, y, x)
#   * permutation is a misnomer - it's really a subset.
#
# Tests:
#   1. Verify that get_xy_slices() works with indices in the original data set
#      instead of the indices in the subset.  XXX: define whether the interface
#      is in the original indices or relative to the subset.

import itertools
import netCDF4 as nc
import numpy as np
import os
import pytest
import random
import stat

import iwp.data_loader

class TestSyntheticIWPDataset:
    """
    Testing harness for synthetically generated IWP datasets.  Verifies the
    entirety of the IWPDataset interface by generating netCDF4 files with known
    patterns that are easily tested.  This is intended to obviate the need for
    large simulation outputs during testing.

    XXX
    """

    def get_unique_permutation( original_sequence, subset_length=0 ):
        """
        Returns a non-identity permutation for the supplied sequence.  The
        length of the permutation may be specified.

        Takes 2 arguments:

          original_sequence - Iterable to generate permutations for.  Must have
                              a length of at least two otherwise a ValueError
                              exception will be raised.
          subset_length     - Optional integral length of the permutations to
                              generate.  If omitted, defaults to a small cycle
                              length.

        Returns 1 value:

          permutation - Tuple of indices representing a non-identity permutation
                        for original_sequence.

        """

        # default the subset length to something reasonable if it wasn't
        # specified.
        if subset_length == 0:
            #
            # NOTE: large values will result in a exponentially increased memory
            #       footprint.
            #
            subset_length = min( 5, len( original_sequence ) )

        # ensure that we don't ask for a permutation bigger than the supplied
        # list can support, otherwise no permutations are generated below.
        subset_length = min( subset_length, len( original_sequence ) )

        all_permutations = list( itertools.permutations( original_sequence,
                                                         subset_length ) )

        # randomly pick one of the non-identity permutations.
        #
        # NOTE: this will raise ValueError if the input sequence has a single
        #       element.
        #
        permutation_index = random.randint( 1, len( all_permutations ) - 1 )

        return all_permutations[permutation_index]

    def change_netcdf_dimensions( netcdf_path, target_dimension_name, new_dimension_size ):
        """
        Changes the size of a single dimension in a netCDF4 file.  Every variable using said
        dimension is adjusted.

        Changing a dimension requires creating a new file, creating an adjusted dimension,
        and then copying over variables from the original with potentially new shapes.
        Variables that are truncated (smaller dimensions than before) will lose some of
        their original data, while variables that are enlarged (larger dimensions than
        before) will be expanded with the default fill value.  Once the copy is complete
        the original file is removed and the new one is moved into its place.

        Takes 3 arguments:

          netcdf_path           - Path to a netCDF4 file whose dimensions are to be changed.
                                  The original file is overwritten only if the adjustment
                                  is successful.
          target_dimension_name - Name of the dimension to change.  ValueError if raised
                                  if target_dimension_name is not a valid dimension in
                                  netcdf_path.
          new_dimension_size    - Integer specifying the new size of the adjusted dimension.
                                  ValueError is raised if new_dimension_size is the same
                                  as the dimension's original size.

        Returns 1 value:

          copied_flag - Boolean indicating whether the dimensionality change was
                        successful.  If True, then netcdf_path was updated, otherwise
                        it remains unchanged.

        """

        # create the temporary netCDF4 in the same directory as the original.
        temp_netcdf_path = "{:s}.temp".format( netcdf_path )

        copied_flag = False

        # make a copy of the original dataset with new dimensions as requested.
        # copy over the original data and use the fill value if dimensions are
        # expanded, or truncate the original data if the dimensions were
        # reduced.  netCDF4 does not allow alteration of dataset dimensions, so
        # we manually create a new dataset and copy over everything we want.
        with nc.Dataset( temp_netcdf_path, "w" ) as ds_temp, nc.Dataset( netcdf_path, "r" ) as ds_source:

            # make sure that we're actually changing a dimension in the data set.
            if target_dimension_name not in ds_source.dimensions:
                raise ValueError( "Invalid dimension specified ({:s}).".format(
                    target_dimension_name ) )
            if new_dimension_size == ds_source.dimensions[target_dimension_name].size:
                raise ValueError( "Requested to change {:s}'s size, but supplied the same value ({:d}).".format(
                    target_dimension_name,
                    new_dimension_size ) )

            # keep track of the delta between the original and new dimension size
            # so we can copy affected variables' data correctly.
            size_delta = (new_dimension_size -
                          ds_source.dimensions[target_dimension_name].size)

            # manually copy each of the dimensions.
            for dimension_name in ds_source.dimensions:
                if dimension_name == target_dimension_name:
                    # use the new dimension.
                    ds_temp.createDimension( dimension_name, new_dimension_size )
                else:
                    # use the original dimension.
                    ds_temp.createDimension( dimension_name,
                                             ds_source.dimensions[dimension_name].size )

            # copy the variables.
            for variable_name in ds_source.variables:
                variable = ds_source.variables[variable_name]

                # compute the slices representing the data to be copied from and
                # to under the dimension change.
                new_dimension_slices = []
                for dimension_name in variable.dimensions:
                    if dimension_name == target_dimension_name:
                        new_dimension_slices.append( slice( new_dimension_size ) )
                    else:
                        new_dimension_slices.append( slice( ds_temp.dimensions[dimension_name].size ) )

                old_dimension_slices = tuple( slice( dim ) for dim in variable.shape )

                # convert index slices into proper indices.
                old_indices = np.s_[old_dimension_slices]
                new_indices = np.s_[new_dimension_slices]

                # create the new variable using the dimension names.  these
                # correspond to the adjusted dimension sizes in the new variable
                # and the original dimension sizes in the old.
                values_temp    = ds_temp.createVariable( variable.name,
                                                         variable.dtype,
                                                         dimensions=variable.dimensions )

                # copy the data, taking care to respect the potentially new
                # dimensions.
                if size_delta < 0:
                    # the data volume shrunk in size, so only copy the volume of
                    # interest.
                    values_temp[new_indices] = ds_source.variables[variable.name][new_indices]
                else:
                    # the data volume either remained the same size, or grew, so
                    # copy the original volume.  if it grew, the new data will
                    # take on the fill value.
                    values_temp[old_indices] = ds_source.variables[variable.name][old_indices]

            copied_flag = True

        # move the new file into place, but only if we were successful in
        # creating the temporary file.
        if copied_flag:
            os.remove( netcdf_path )
            os.rename( temp_netcdf_path, netcdf_path )
        else:
            os.remove( temp_netcdf_path )

        return copied_flag

    def remove_netcdf_variables( netcdf_path, variable_names ):
        """
        Removes one or more variables from a netCDF4 file.

        Removing variables is done by creating a new file and copying over all the variables
        except those explicitly excluded.  Once the copy is performed, the original file is
        removed and the new file is moved into its place.

        Takes 2 arguments:

          netcdf_path    - Path to a netCDF4 file whose variables are to be removed.  The
                           original file is overwritten only if the variable removal was
                           successful.
          variable_names - Sequence of variable names to remove from netcdf_path.  Only
                           variables not in variable_names will remain in netcdf_path
                           after the method returns.

        Returns 1 value:

          copied_flag - Boolean indicating whether the variable removals were successful.
                        If True, then netcdf_path was updated, otherwise it remains
                        unchanged.

        """

        # create the temporary netCDF4 in the same directory as the original.
        temp_netcdf_path = "{:s}.temp".format( netcdf_path )

        copied_flag = False

        # make a copy of the original dataset without the variables of interest.
        # netCDF4 does not allow removal of a variable from a dataset, so we
        # manually create a new dataset and copy over everything we want.
        with nc.Dataset( temp_netcdf_path, "w" ) as ds_temp, nc.Dataset( netcdf_path, "r" ) as ds_source:

            # manually copy each of the dimensions.
            for dimension_name in ds_source.dimensions:
                ds_temp.createDimension( dimension_name,
                                         ds_source.dimensions[dimension_name].size )

            # copy the variables, if they're not in the blacklist.
            for variable_name in ds_source.variables:
                if variable_name in variable_names:
                    continue

                variable = ds_source.variables[variable_name]

                values_temp    = ds_temp.createVariable( variable.name,
                                                         variable.dtype,
                                                         dimensions=variable.dimensions )
                values_temp[:] = ds_source.variables[variable.name][:]

            copied_flag = True

        # move the new file into place, but only if we were successful in
        # creating the temporary file.
        if copied_flag:
            os.remove( netcdf_path )
            os.rename( temp_netcdf_path, netcdf_path )
        else:
            os.remove( temp_netcdf_path )

        return copied_flag

    def validate_all_slices_values( dataset, time_indices, xy_slice_indices ):
        """
        Walks through all of the XY slices in the dataset and verifies that they are
        returned in the expected order and contain the expected values.  AssertionError
        is raised if any of the XY slices retrieved from the dataset don't match the
        expected values.

        Takes 3 arguments:

          dataset          - The IWPDataset whose XY slices are under test.
          time_indices     - Sequence of time step indices that match those specified
                             when dataset was created.
          xy_slice_indices - Sequence of XY slice indices that match those specified
                             when dataset was created.

        Returns nothing.

        """

        # walk through each XY slice in the order requested and verify that we
        # see the correct slice, through both the direct and indirect
        # interfaces.
        slice_index = 0
        for time_index in time_indices:
            for xy_slice_index in xy_slice_indices:

                # get the candidate XY slices via the two public interfaces.
                # these should be equivalent according to the Dataset's API.
                candidate_indirect = dataset[slice_index]
                candidate_direct   = dataset.get_xy_slice( time_index,
                                                           xy_slice_index )

                #
                # NOTE: we're only interested in testing the IWPDataset's
                #       interface, though computing the expected XY slice's
                #       values directly makes debugging this *significantly*
                #       easier.
                #
                candidate_computed = TestSyntheticIWPDataset.compute_xy_slice_values(
                    time_index,
                    xy_slice_index,
                    candidate_indirect.shape[1:],
                    list( range( candidate_indirect.shape[0] ) ) )

                # ensure that the indirect and direct accessors produce the same
                # values.
                assert np.allclose( candidate_indirect, candidate_direct ), \
                    ("Indirect and direct indexing yield different XY slices "
                     "(slice index {:d} != (time step index {:d}, XY slice "
                     "index {:d}) is different.".format(
                        slice_index,
                        time_index,
                        xy_slice_index ))

                # ensure that the indirect accessor and the testing interface
                # produce the same values.  transitively, the computed and the
                # direct interface will be the same.
                assert np.allclose( candidate_indirect, candidate_computed ), \
                    ("Indirect and computed indexing yield different XY slices "
                     "(slice index {:d} != ((time step index {:d}, XY slice "
                     "index {:d}) is different.".format(
                        slice_index,
                        time_index,
                        xy_slice_index ))

                # verify that each of the indirect values have their expected
                # values.
                assert TestSyntheticIWPDataset.verify_xy_slice_values(
                    candidate_indirect,
                    time_index,
                    xy_slice_index ) == True, \
                    ("Indirect XY slice index {:d} (time step index {:d}, XY "
                     "slice index {:d}) is different.".format(
                        slice_index,
                        time_index,
                        xy_slice_index ))

                slice_index += 1

    def validate_all_slices_values_backwards( dataset, time_indices, xy_slice_indices ):
        """
        Walks through all of the XY slices in the dataset in reverse order and verifies
        that they are returned in the expected order and contain the expected values.
        AssertionError is raised if any of the XY slices retrieved from the dataset
        don't match the expected values.

        Takes 3 arguments:

          dataset          - The IWPDataset whose XY slices are under test.
          time_indices     - Sequence of time step indices that match those specified
                             when dataset was created.
          xy_slice_indices - Sequence of XY slice indices that match those specified
                             when dataset was created.

        Returns nothing.

        """

        # walk through each XY slice in the order requested and verify that we
        # see the correct slice, through both the direct and indirect
        # interfaces.
        slice_index = -1
        for time_index_index, time_index in enumerate( reversed( time_indices ), 1 ):
            for xy_slice_index_index, xy_slice_index in enumerate( reversed( xy_slice_indices ), 1 ):

                negative_time_index_index     = -time_index_index
                negative_xy_slice_index_index = -xy_slice_index_index

                candidate_indirect = dataset[slice_index]
                candidate_direct   = dataset.get_xy_slice( negative_time_index_index,
                                                           negative_xy_slice_index_index  )
                candidate_computed = TestSyntheticIWPDataset.compute_xy_slice_values(
                    time_index,
                    xy_slice_index,
                    candidate_indirect.shape[1:],
                    list( range( candidate_indirect.shape[0] ) ) )

                assert np.allclose( candidate_indirect, candidate_direct ), \
                    ("Indirect and direct indexing yield different XY slices "
                     "(slice index {:d} != (time step index {:d}, XY slice index {:d}) "
                     "is different.".format(
                         slice_index,
                         time_index,
                         xy_slice_index ))

                assert np.allclose( candidate_indirect, candidate_computed ), \
                    ("Indirect and computed indexing yield different XY slices "
                     "(slice index {:d} != ((time step index {:d}, XY slice index {:d}) "
                     "is different.".format(
                         slice_index,
                         time_index,
                         xy_slice_index ))

                assert TestSyntheticIWPDataset.verify_xy_slice_values(
                    candidate_indirect,
                    time_index,
                    xy_slice_index ) == True, \
                    "Indirect XY slice index {:d} (time step index {:d}, XY slice index {:d}) is different.".format(
                        slice_index,
                        time_index,
                        xy_slice_index )

                assert TestSyntheticIWPDataset.verify_xy_slice_values(
                    candidate_direct,
                    time_index,
                    xy_slice_index ) == True, \
                    "Direct XY slice for time step index {:d}, XY slice index {:d} is different.".format(
                        time_index,
                        xy_slice_index )

                slice_index -= 1

    def compute_xy_slice_values( time_step_index, slice_index, xy_slice_shape, variable_indices ):
        """
        Compute the expected values for a particular time step's XY slice.  For a given
        XY slice specified by (time step, slice index, variables) the values computed
        are:

           ((time step * 10000) +
            (slice index * 100) +
            (variable index)    +
            ((slice x index * slice y index) / (slice size)))

        NOTE: The equation above only computes unique values for each XY slice if there
              are fewer than 100 time steps or 100 slices.  If either condition is true,
              then the outputs for distinct (time step, slice index) pairs are duplicated.

        Takes 4 arguments:

          time_step_index  - Time step index to compute values for.
          slice_index      - XY slice index to compute values for.
          xy_slice_shape   - List of XY slice sizes, shaped (X, Y), to compute values for.
          variable_indices - List of variable indices to compute values for.

        Returns 1 value:

          values - float32 array, shaped (number_variables, x, y), containing
                   the expected values of the XY slice and variables requested.

        """

        # NOTE: we need "ij" indexing to get the correct shape, otherwise X and
        #       Y are transposed.
        V, X, Y = np.meshgrid( variable_indices,
                               np.arange( xy_slice_shape[0] ),
                               np.arange( xy_slice_shape[1] ),
                               indexing="ij" )

        # compute the expected values for each variable in the slice based on
        # the 1) time step, 2) slice index, and 3) variable index.
        values    = np.empty( V.shape, dtype=np.float32 )
        values[:] = ((time_step_index * 10000) +
                     (slice_index * 100) +
                     V + ((X * Y) / (xy_slice_shape[0] * xy_slice_shape[1])))

        return values

    def compute_grid_variable_values( time_step_index, grid_shape, variable_index ):
        """
        Computes the expected values for a grid variable for a specific time step.  For
        a given variable, each slice's values are computed by
        TestSyntheticIWPDataset.compute_xy_slice_values().

        Takes 3 arguments:

          time_step_index - Time step index to compute values for.
          grid_shape      - Sequence of 3 sizes defining the grid variable's shape.

                            NOTE: grid_shape's sizes are specified as (z, y, z) to
                                  match the convention specified by the IWPDataset's
                                  underlying netCDF4 variables.

          variable_index  - Index of the variable to compute the values for.  This
                            is relative to the list of variables specified when the
                            IWPDataset was created.

        Returns 1 value:

          values - float32 array, shaped (z, y, x), containing the expected values
                   for the specified time step and variable.
        """

        values = np.empty( grid_shape, dtype=np.float32 )

        # walk through each of the slices in the requested time step and compute
        # the expected values for the variables of interest.
        #
        # NOTE: it would be more efficient to compute all of these in one go,
        #       though we need to verify individual slices so it is easier to
        #       structure the construction (once) inefficiently to be more
        #       flexible during validation (many).
        #
        # NOTE: The dimension order is assumed to be (z, y, x) to match the
        #       underlying netCDF interface.
        #
        for z_index in range( grid_shape[0] ):
            values[z_index, :, :] = TestSyntheticIWPDataset.compute_xy_slice_values(
                time_step_index,
                z_index,
                grid_shape[1:],
                [variable_index] )

        return values

    # XXX: move me
    def create_netcdf_file( netcdf_path, time_step_index, grid_dimensions, variables, initialize_variables_flag=False ):
        """
        Creates a netCDF4 file for a single time step containing one or more grid variables.
        Variables are optionally initialized to a test pattern or are left uninitialized so
        the default fill value is used instead.

        Takes 5 arguments:

          netcdf_path               - Path to the netCDF4 file to create.  The contents
          time_step_index           - Time step index to compute values for.
          grid_dimensions           - Size of the underlying grid, shaped (x, y, z).
          variables                 - Sequence of variable names or a sequence of dictionaries
                                      specifying the variable names or variable names and data types
                                      for the netCDF4's contents.  Variables specified by name only
                                      default to a float32 data type.  Variables specified by dictionary
                                      must contain both a "string" and "dtype" key specifying the
                                      variable name and data type, respectively.
          initialize_variables_flag - Optional boolean specifying whether grid variables are
                                      initialized to the test pattern.  If omitted, defaults to
                                      False.

        Returns nothing.

        """

        # convert a sequence of variable names into a sequence of dictionaries
        # with names and float32 data types.
        if all( map( lambda x: type( x ) == str, variables ) ):
            variable_names = variables
            variables      = []

            for variable_name in variable_names:
                variables.append( {"name":  variable_name,
                                   "dtype": np.float32} )

        # build a dataset using the context manager interface.  this ensures
        # that the file is properly closed so it may be overwritten when opened
        # for writing a second time.
        with nc.Dataset( netcdf_path,
                         "w",
                         format="NETCDF4",
                         clobber=True ) as ds:

            # build the grid variables' dimension name tuple from those
            # specified.
            #
            # NOTE: this forces all grid variables to have the same, correct
            #       dimensions.
            #
            # NOTE: we reverse the order because dimensions are listed in
            #       row-major order.
            #
            variable_dimensions = tuple( map( lambda x: x["name"],
                                              grid_dimensions ) )[::-1]

            # create the netCDF grid dimensions and variables associated
            # with them.  each axes' coordinates are simply [0, size) in
            # steps of 1.
            for grid_dimension in grid_dimensions:
                dimension = ds.createDimension( grid_dimension["name"],
                                                grid_dimension["size"] )

                dimension_variable = ds.createVariable( grid_dimension["name"],
                                                        grid_dimension["dtype"],
                                                        dimensions=(grid_dimension["name"],) )

                dimension_variable[:] = np.arange( grid_dimension["size"] )

            # create each of the grid variables.
            for variable_index, variable in enumerate( variables ):
                v = ds.createVariable( variable["name"],
                                       variable["dtype"],
                                       dimensions=variable_dimensions )

                #
                # NOTE: the value set here depends on the order of the variables
                #       provided.  multiple calls with different orderings will
                #       result in different initialization values.
                #
                if initialize_variables_flag:
                    v[:] = TestSyntheticIWPDataset.compute_grid_variable_values(
                        time_step_index,
                        v.shape,
                        variable_index )

    @pytest.fixture( scope="session" )
    def create_netcdf_files( self, request, tmpdir_factory ):
        """
        netCDF4 file creation fixture.  Creates one or more netCDF4 files according
        to the parameters specified in the provided request structure.  Returns the
        parameters used during creation, the path pattern for each of the created files,
        and the paths to each of the individual time step files.

        Takes 2 arguments:

          request        - 4-tuple containing the fixture's parameters.  Contains the
                           following parameters:

                             'number_time_steps': Integral number of consecutive time steps
                                                  to create.
                             'grid_size':         Shape of the 3D grid to use during creation.
                             'variable_names':    Sequence of grid variable names.
                             'test_key':          String specifying the purpose of the family
                                                  of netCDF4 files.

          tmpdir_factory - pytest fixture for creating temporary paths.

        Returns 3 values:

           parameters     - Dictionary of parameters used to create the netCDF4 files.
                            Contains at least the following fields:

                             'grid_size':         Shape of the 3D grid to use during creation.
                             'number_time_steps': Integral number of consecutive time steps
                                                  to create.
                             'test_key':          String specifying the purpose of the family
                                                  of netCDF4 files.
                             'variable_names':    Sequence of grid variable names.

           output_pattern - Path pattern for the netCDF4 files created.  The path to a
                            specific time step file can be constructed by formatting
                            output_pattern with an integral time step index.
           output_paths   - List of paths to the netCDF4 files created, one per time step.

        """

        # unpack and map the request parameters into a dictionary for easy
        # access by the callers of this fixture.
        (number_time_steps,
         grid_size,
         variable_names,
         test_key) = request.param

        parameters = {
            "number_time_steps": number_time_steps,
            "grid_size":         grid_size,
            "variable_names":    variable_names,
            "test_key":          test_key,
        }

        # create netCDF4 files with a self-descriptive path for post-mortem
        # analysis.
        output_name = "iwp-{:s}-{:s}-{:s}".format(
            test_key,
            ",".join( map( lambda variable: str( variable ), variable_names ) ),
            "x".join( map( lambda size: str( size ), grid_size ) ) )

        # XXX: replace the temporary path creation with something that cleans up
        #      after itself.
        temp_path = tmpdir_factory.mktemp( "TestSyntheticIWPDataset" )

        # file pattern to each of the time step files in the dataset.  must be
        # instantiated with an integral time step to get a file path.
        output_pattern = "{}/{:s}.{{:d}}.nc".format(
            temp_path,
            output_name )

        output_paths = []

        # ensure that we have a 3D grid so we can build a dictionary of
        # dimensions and data types.
        if len( grid_size ) != 3:
            raise ValueError( "Expected 3D grid variables, but got a "
                              "{:d}-length list of grid dimensions.".format(
                                  len( grid_size ) ) )

        grid_dimensions = [{"name": "x", "size": grid_size[0], "dtype": np.float32},
                           {"name": "y", "size": grid_size[1], "dtype": np.float32},
                           {"name": "z", "size": grid_size[2], "dtype": np.float32}]

        # create a netCDF4 file for each time step.
        for time_step_index in range( number_time_steps ):
            output_path = output_pattern.format( time_step_index )

            output_paths.append( output_path )

            TestSyntheticIWPDataset.create_netcdf_file( output_path,
                                                        time_step_index,
                                                        grid_dimensions,
                                                        variable_names,
                                                        initialize_variables_flag=True )

        return (parameters, output_pattern, output_paths)

    def verify_xy_slice_values( xy_slice, time_step_index, z_index ):
        """
        Verifies the values of a particular XY slice match the expected test values a
        (time, slice) location within the dataset.  Expected values are generated using
        TestSyntheticIWPDataset.compute_xy_slice_values().

        Takes 3 arguments:

          xy_slice        - float32 array of slice values, shaped (number_variables, x, y),
                            to compare against the expected values.
          time_step_index - Time step index associated with xy_slice.
          z_index         - Slice index associated with xy_slice.

        Returns 1 value:

          result - Boolean indicating whether the specified slice was equivalent to its
                   expected values.

        """

        # compute the expected values.
        expected_values = TestSyntheticIWPDataset.compute_xy_slice_values(
            time_step_index,
            z_index,
            xy_slice.shape[1:],
            list( range( xy_slice.shape[0] ) ) )

        # check the supplied slice against the expected values with a modest
        # relative tolerance.
        return np.allclose( expected_values, xy_slice, rtol=1e-3 )

    def test_nonexistent_dataset( self, tmp_path ):
        """
        Verifies that non-existent netCDF4 files cannot be used to create an IWPDataset.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          tmp_path - Temporary directory to specify a non-existent path to a netCDF4 time step
                     file.  This path will not be written to..

        Returns nothing.

        """

        nonexistent_path = "{:s}/nonexistent.nc".format( str( tmp_path ) )

        # verify that a dataset can't be created when some of the time steps'
        # files are inaccessible.
        with pytest.raises( FileNotFoundError ):
            nonexistent_dataset = iwp.data_loader.IWPDataset( nonexistent_path, range( 1 ) )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [( 1, (8,  8,  1), ["u", "v", "w"], "unreadable"),
                               (10, (8, 16, 32), ["u", "v"],      "unreadable")],
                              indirect=True )
    def test_invalid_dataset_permissions( self, create_netcdf_files ):
        """
        Verifies that a IWPDataset cannot be created from netCDF4 files unless they
        have accessible permissions.

        Raises AssertionError if the dataset creation does not fail as expected.

        NOTE: Permissions on the created netCDF4 files are modified during the the though
              are restored after the test completes.  This is required due to how the
              tests are constructed and cached between runs.  Without restoration,
              additional tests may incorrectly fail.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        # keep track of the dataset's files' permissions so we can restore them
        # after the test.
        all_permissions = []
        for file_index in range( parameters["number_time_steps"] ):
            current_permissions = stat.S_IMODE( os.lstat( netcdf_paths[file_index] ).st_mode )

            all_permissions.append( current_permissions )

        # remove permissions for a number of files.
        NUMBER_FILES = 3

        # remove read permission from a handful of the dataset's underlying
        # files.
        for file_index in random.sample( range( parameters["number_time_steps"] ),
                                         min( parameters["number_time_steps"],
                                              NUMBER_FILES ) ):
            current_permissions = stat.S_IMODE( os.lstat( netcdf_paths[file_index] ).st_mode )
            os.chmod( netcdf_paths[file_index],
                      current_permissions & ~(stat.S_IRUSR |
                                              stat.S_IRGRP |
                                              stat.S_IROTH) )

        # verify that a dataset can't be created when some of the time steps'
        # files are inaccessible.
        with pytest.raises( PermissionError ):
            inaccessible_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                               range( parameters["number_time_steps"] ) )

        # restore the permissions to every file in the dataset.
        for file_index in range( parameters["number_time_steps"] ):
            os.chmod( netcdf_paths[file_index], all_permissions[file_index] )

        # get the permissions on the directory containing the dataset's first
        # time step file.
        dataset_dirname     = os.path.dirname( netcdf_paths[0] )
        current_permissions = stat.S_IMODE( os.lstat( dataset_dirname ).st_mode )

        os.chmod( dataset_dirname,
                  current_permissions & ~(stat.S_IRUSR | stat.S_IXUSR |
                                          stat.S_IRGRP | stat.S_IXGRP |
                                          stat.S_IROTH | stat.S_IXOTH) )

        # verify that a dataset can't be created when one of the time steps'
        # containing directory is inaccessible.
        with pytest.raises( PermissionError ):
            inaccessible_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                               range( parameters["number_time_steps"] ) )

        # restore the permissions to the first time step's containing directory.
        os.chmod( dataset_dirname, current_permissions )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(2, (8, 16, 32), ["u", "v", "w"], "incorrect_dimensions")],
                              indirect=True )
    def test_incorrect_dimensions( self, create_netcdf_files ):
        """
        Verifies that netCDF4 files containing variables with incorrect grid dimensions
        cannot be used to create an IWPDataset.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        # we change the shape of the grid variables in the last time step so
        # that it is inconsistent with the previous time steps.  ensure we have
        # a least two time steps to force the inconsistency.
        if len( netcdf_paths ) < 2:
            raise ValueError( "Must have at least two time steps to validate "
                              "incorrect dimension shapes.".format(
                                  len( netcdf_paths ) ) )

        # set a 2D grid.
        grid_size       = parameters["grid_size"]
        grid_dimensions = [{"name": "x", "size": grid_size[0], "dtype": np.float32},
                           {"name": "y", "size": grid_size[1], "dtype": np.float32}]

        # change the last file's grid variable's to 2D.
        TestSyntheticIWPDataset.create_netcdf_file(
            netcdf_paths[-1],
            parameters["number_time_steps"] - 1,
            grid_dimensions,
            parameters["variable_names"] )

        # verify that a dataset can't be created when there are inconsistent
        # grid variables.
        with pytest.raises( ValueError ):
            incorrect_dimensions_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                       range( parameters["number_time_steps"] ) )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(3, (8, 16, 32), ["u", "v", "w"], "missing_variable")],
                              indirect=True )
    def test_missing_variables( self, create_netcdf_files ):
        """
        Verifies that an IWPDataset cannot be created with variables that don't
        exist in the underlying netCDF4 files.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        # list of variables to expose from the IWP dataset.
        #
        # NOTE: this should contain the variable list in decorator above along
        #       with an additional variable that doesn't exist.
        #
        dataset_variables = ["u", "v", "w", "nonexistent"]

        # verify that a dataset can't be created when a nonexistent variable is
        # requested.
        with pytest.raises( ValueError ):
            missing_variable_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                   range( parameters["number_time_steps"] ),
                                                                   variables=dataset_variables )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(3, (8, 16, 32), ["u", "v", "w"], "inconsistent_variables")],
                              indirect=True )
    def test_inconsistent_variables( self, create_netcdf_files ):
        """
        Verifies that an IWPDataset cannot be created when time step files have inconsistent
        grid variables between them.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        number_time_steps = parameters["number_time_steps"]
        number_variables  = len( parameters["variable_names"] )

        # iterate through each of the files and remove one of the variables from
        # each.  for simplicity, we remove the ith variable from the ith file,
        # wrapping around when we have more files than variables.
        for time_step_index in range( number_time_steps ):
            TestSyntheticIWPDataset.remove_netcdf_variables(
                netcdf_paths[time_step_index],
                parameters["variable_names"][time_step_index % number_variables] )

        # verify that a dataset can't be created when the variables aren't
        # consistent between individual time step files.
        with pytest.raises( ValueError ):
            inconsistent_variables_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                         range( number_time_steps ) )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(3, (8, 16, 32), ["u", "v", "w"], "missing_grid_variables")],
                              indirect=True )
    def test_missing_grid_variables( self, create_netcdf_files ):
        """
        Verifies that an IWPDataset cannot be created when time step files do not have
        grid variables.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        number_time_steps = parameters["number_time_steps"]

        # iterate through each of the files and remove each of the grid
        # variables.
        #
        # NOTE: we update each file so we don't inadvertently succeed this test
        #       when the dataset loading code identifies inconsistent variables
        #       between time steps.
        #
        for time_step_index in range( number_time_steps ):
            TestSyntheticIWPDataset.remove_netcdf_variables(
                netcdf_paths[time_step_index],
                parameters["variable_names"] )

        # verify that a dataset can't be created when there aren't any grid
        # variables.
        with pytest.raises( ValueError ):
            missing_grid_variables_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                         range( number_time_steps ) )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(3, (8, 16, 32), ["u", "v", "w"], "integral_grid_variables")],
                              indirect=True )
    def test_integral_grid_variables( self, create_netcdf_files ):
        """
        Verifies that an IWPdataset cannot be created with non-floating point grid variables.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        # we change a grid variable's data type in the last time step so that it
        # is inconsistent with the previous time steps.  ensure we have a least
        # two time steps to force the inconsistency.
        if len( netcdf_paths ) == 1:
            raise ValueError( "Must have at least two time steps to validate "
                              "integral grid variable data types.".format(
                                  len( netcdf_paths ) ) )

        # prepare to build a single netCDF4 file through the low-level
        # interface.  explicitly create the grid dimensions and a list of grid
        # variables with their data types.
        grid_size       = parameters["grid_size"]
        grid_dimensions = [{"name": "x", "size": grid_size[0], "dtype": np.float32},
                           {"name": "y", "size": grid_size[1], "dtype": np.float32},
                           {"name": "z", "size": grid_size[1], "dtype": np.float32}]

        variables = [{"name": "u", "dtype": np.float32},
                     {"name": "v", "dtype": np.float32},
                     {"name": "w", "dtype": np.int32}]

        # overwrite the last time step with a grid variable that has an integral
        # data type.
        TestSyntheticIWPDataset.create_netcdf_file(
            netcdf_paths[-1],
            parameters["number_time_steps"] - 1,
            grid_dimensions,
            variables )

        # verify that a dataset can't be created when there are integral grid
        # variables in a later time step.
        with pytest.raises( ValueError ):
            missing_grid_variables_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                         range( parameters["number_time_steps"] ) )

        # overwrite the first time step with the same integral grid variable so
        # the IWPDataset's default variable list includes it.
        TestSyntheticIWPDataset.create_netcdf_file(
            netcdf_paths[0],
            0,
            grid_dimensions,
            variables )

        with pytest.raises( ValueError ):
            missing_grid_variables_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                         range( parameters["number_time_steps"] ) )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(3, (8, 16, 32), ["u", "v", "w"], "inconsistent_variable_dimensions")],
                              indirect=True )
    def test_inconsistent_variable_dimensions( self, create_netcdf_files ):
        """
        Verifies that netCDF4 files containing variables with inconsistent dimensions
        cannot be used to create an IWPDataset.  This tests both inconsistent dimension
        names (e.g. "dim1" vs "x") as well as incompatible dimension shapes (e.g. "x"
        with length N and N+1).

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        number_time_steps = parameters["number_time_steps"]
        grid_size         = parameters["grid_size"]

        # we change the shape of a variable's dimension in the last time step so
        # that it is inconsistent with the previous time steps.  ensure we have
        # a least two time steps to force the inconsistency.
        if len( netcdf_paths ) == 1:
            raise ValueError( "Must have at least two time steps to validate "
                              "inconsistent variable dimension handling." )

        # we change the X dimension's length to 1 to validate handling of
        # inconsistent dimension lengths.  ensure that we're actually changing
        # the length so we don't inadvertently succeed creating a dataset.
        if grid_size[0] == 1:
            raise ValueError( "Cannot test inconsistent grid variable sizes "
                              "with a singleton X dimension ({:s}).".format(
                                  netcdf_pattern ) )

        # adjust the X dimension to be a singleton.
        target_dimension_name = "x"
        target_dimension_size = 1

        # change one time step's variable dimensions to be inconsistent with the
        # remaining time steps.  this sets the last time step's X dimension to
        # be singleton.
        TestSyntheticIWPDataset.change_netcdf_dimensions(
            netcdf_paths[-1],
            target_dimension_name,
            target_dimension_size )

        # verify that a dataset can't be created when the variables' sizes
        # are not consistent between individual time step files.
        with pytest.raises( ValueError ):
            inconsistent_variables_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                         range( number_time_steps ) )

        # verify that grid variables must be both dimension name and dimension
        # shape compatible.  this changes the dimension's name but retains the
        # correct shape.
        grid_size       = parameters["grid_size"]
        grid_dimensions = [{"name": "x_different", "size": grid_size[0], "dtype": np.float32},
                           {"name": "y",           "size": grid_size[1], "dtype": np.float32},
                           {"name": "z",           "size": grid_size[2], "dtype": np.float32}]

        # overwrite the last time step with new dimension names on the grid
        # variables.
        TestSyntheticIWPDataset.create_netcdf_file(
            netcdf_paths[-1],
            parameters["number_time_steps"] - 1,
            grid_dimensions,
            parameters["variable_names"] )

        # verify that a dataset can't be created when the variables' dimension
        # names are not consistent between individual time step files.
        with pytest.raises( ValueError ):
            inconsistent_variables_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                         range( number_time_steps ) )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [( 2, (4,  4,  2), ["u", "v", "w"], "normal"),
                               (10, (8, 16, 32), ["u", "v"],      "normal")],
                              indirect=True )
    def test_data_values( self, create_netcdf_files ):
        """
        Verifies that the IWPDataset interface faithfully represents the underlying netCDF4
        data.  Test patterns are written to netCDF4 files and are verified through
        the IWPDataset's data accessors (i.e. __getitem__() and get_xy_slice()).

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        time_indices     = range( parameters["number_time_steps"] )
        xy_slice_indices = range( parameters["grid_size"][2] )

        # create a normal dataset (i.e. no permutation of indices, default
        # variables) and verify its contents.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              time_indices,
                                              xy_slice_indices=xy_slice_indices )

        TestSyntheticIWPDataset.validate_all_slices_values(
            dataset,
            time_indices,
            xy_slice_indices )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [( 2, (3,  5,  4), ["u", "v", "w"], "normal"),
                               ( 5, (7, 11, 17), ["u"],            "normal"),
                               (10, (8, 16, 32), ["u", "v"],      "normal")],
                              indirect=True )
    def test_permuted_slice_indices( self, create_netcdf_files ):
        """
        Verifies that the IWPDataset's interface for permuted time and slice indices
        works as expected.  Datasets with known test patterns are created and then accessed
        with one or both permutations and the values returned validated against the expected
        test patterns.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        # length of the permuted subset of data.  the value is arbitrary, but
        # shouldn't be trivially low.
        PERMUTATION_CYCLE_LENGTH = 5

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        # we compute permuted subsets of time and slice indices within the
        # underlying datasets.  ensure that we have at least two time steps and
        # two XY slices to verify things.
        #
        # NOTE: we don't force the sizes to be no shorter than the permutation
        #       cycle length as that mismatch is handled when we get our
        #       permutation.
        #
        if parameters["number_time_steps"] == 1:
            raise ValueError( "Must have at least two time steps to validate time step "
                              "permutations." )
        if parameters["grid_size"][2] == 1:
            raise ValueError( "Must have at least two XY slices to validate slice permutations." )

        # build our indices for both time steps and XY slices, for normal and
        # permuted.
        time_indices              = range( parameters["number_time_steps"] )
        time_indices_permuted     = TestSyntheticIWPDataset.get_unique_permutation(
            time_indices,
            PERMUTATION_CYCLE_LENGTH )
        xy_slice_indices          = range( parameters["grid_size"][2] )
        xy_slice_indices_permuted = TestSyntheticIWPDataset.get_unique_permutation(
            range( parameters["grid_size"][2] ),
            PERMUTATION_CYCLE_LENGTH )

        # verify that XY slices can be permuted via the dataset.
        normal_time_permuted_xy_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                      time_indices,
                                                                      xy_slice_indices=xy_slice_indices_permuted )

        TestSyntheticIWPDataset.validate_all_slices_values(
            normal_time_permuted_xy_dataset,
            time_indices,
            xy_slice_indices_permuted )

        # verify that time steps can be permuted via the dataset.
        permuted_time_normal_xy_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                      time_indices_permuted,
                                                                      xy_slice_indices=xy_slice_indices )

        TestSyntheticIWPDataset.validate_all_slices_values(
            permuted_time_normal_xy_dataset,
            time_indices_permuted,
            xy_slice_indices )

        # verify that both time steps and XY slices can be permuted via the data
        # set.
        permuted_time_permuted_xy_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                        time_indices_permuted,
                                                                        xy_slice_indices=xy_slice_indices_permuted )

        TestSyntheticIWPDataset.validate_all_slices_values(
            permuted_time_permuted_xy_dataset,
            time_indices_permuted,
            xy_slice_indices_permuted )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [( 2, (3,  5,  4), ["u", "v", "w"], "normal"),
                               ( 5, (7, 11, 17), ["u", "v", "w"], "normal"),
                               (10, (8, 16, 32), ["u", "v"],      "normal")],
                              indirect=True )
    def test_negative_slice_indices( self, create_netcdf_files ):
        """
        Verifies that the IWPDataset's slice accessors properly handle negative slice indices.
        Datasets with known test patterns are created and then accessed in reverse order with
        time and/or slice indices permuted.  The values returned are validated against the
        expected test patterns.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        # length of the permuted subset of data.  the value is arbitrary, but
        # shouldn't be trivially low.
        PERMUTATION_CYCLE_LENGTH = 5

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        # we compute permuted subsets of time and slice indices within the
        # underlying datasets.  ensure that we have at least two time steps and
        # two XY slices to verify things.
        #
        # NOTE: we don't force the sizes to be no shorter than the permutation
        #       cycle length as that mismatch is handled when we get our
        #       permutation.
        #
        if parameters["number_time_steps"] == 1:
            raise ValueError( "Must have at least two time steps to validate time step "
                              "permutations." )
        if parameters["grid_size"][2] == 1:
            raise ValueError( "Must have at least two XY slices to validate slice permutations." )

        # build our indices for both time steps and XY slices, for normal and
        # permuted.
        time_indices              = range( parameters["number_time_steps"] )
        time_indices_permuted     = TestSyntheticIWPDataset.get_unique_permutation(
            time_indices,
            PERMUTATION_CYCLE_LENGTH )
        xy_slice_indices          = range( parameters["grid_size"][2] )
        xy_slice_indices_permuted = TestSyntheticIWPDataset.get_unique_permutation(
            range( parameters["grid_size"][2] ),
            PERMUTATION_CYCLE_LENGTH )

        # start easy and verify that no permutations work.
        normal_time_normal_xy_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                    time_indices,
                                                                    xy_slice_indices=xy_slice_indices )

        TestSyntheticIWPDataset.validate_all_slices_values_backwards(
            normal_time_normal_xy_dataset,
            time_indices,
            xy_slice_indices )

        # verify that permuting just the time indices works.
        normal_time_permuted_xy_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                      time_indices,
                                                                      xy_slice_indices=xy_slice_indices_permuted )

        TestSyntheticIWPDataset.validate_all_slices_values_backwards(
            normal_time_permuted_xy_dataset,
            time_indices,
            xy_slice_indices_permuted )

        # verify that permuting just the slice indices works.
        permuted_time_normal_xy_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                      time_indices_permuted,
                                                                      xy_slice_indices=xy_slice_indices )

        TestSyntheticIWPDataset.validate_all_slices_values_backwards(
            permuted_time_normal_xy_dataset,
            time_indices_permuted,
            xy_slice_indices )

        # verify that permuting the time and slice indices, together, works.
        permuted_time_permuted_xy_dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                        time_indices_permuted,
                                                                        xy_slice_indices=xy_slice_indices_permuted )

        TestSyntheticIWPDataset.validate_all_slices_values_backwards(
            permuted_time_permuted_xy_dataset,
            time_indices_permuted,
            xy_slice_indices_permuted )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(10, (8, 16, 32), ["u", "v"], "invalid_dataset_indices")],
                              indirect=True )
    def test_invalid_dataset_indices( self, create_netcdf_files ):
        """
        Verifies that creating an IWPDataset while requesting non-existent XY slices fails.

        NOTE: We don't verify that requesting non-existent time step indices as that is
              tested elsewhere with non-existent time step files.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        # verify that we cannot create a dataset when requesting more data than
        # present in the underlying files.
        with pytest.raises( ValueError ):
            # request a single XY slice beyond the last one available.
            dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                  range( parameters["number_time_steps"] ),
                                                  xy_slice_indices=[parameters["grid_size"][2] + 1] )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(12, (8, 16, 10), ["u", "v"], "invalid_access_indices"),
                               (20, (8, 16, 20), ["u", "v"], "invalid_access_indices")],
                              indirect=True )
    def test_invalid_access_indices( self, create_netcdf_files ):
        """
        Verifies that accessing an IWPDataset's underlying data with invalid indices fails.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        # singleton dimensions to pick out of the test datasets.  these must be
        # non-zero so we 1) avoid getting lucky because of an easy edge case and
        # 2) so we can ensure there is an invalid index before each of these.
        SINGLETON_INDEX_TIME = 3
        SINGLETON_INDEX_XY   = 5

        number_time_steps = parameters["number_time_steps"]
        number_xy_slices  = parameters["grid_size"][2]

        # make sure that the underlying datasets are large enough to accommodate
        # the indices used to test singleton subsets.
        assert number_time_steps > SINGLETON_INDEX_TIME, \
            "{:s} needs at least {:d} time steps to test invalid indices, but only has {:d}.".format(
                netcdf_pattern,
                SINGLETON_INDEX_TIME + 1,
                number_time_steps )

        assert number_xy_slices > SINGLETON_INDEX_XY, \
            "{:s} needs at least {:d} XY slices to test invalid indices, but only has {:d}.".format(
                netcdf_pattern,
                SINGLETON_INDEX_XY + 1,
                number_xy_slices )

        # ensure the indices are non-zero so we can attempt to access the index
        # before them.
        assert SINGLETON_INDEX_TIME > 0, \
            "Singleton access verification needs a non-zero time step index."

        assert SINGLETON_INDEX_XY > 0, \
            "Singleton access verification needs a non-zero XY slice index."

        # create datasets for each of the combinations of time steps and XY
        # slices so as to ensure that we test datasets with singleton axes
        # (e.g. one XY slice for a sequence of time steps or a sequence of
        # XY slices for a single time step).
        dataset_all_time_all_xy    = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                 range( parameters["number_time_steps"] ) )
        dataset_all_time_single_xy = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                 range( parameters["number_time_steps"] ),
                                                                 xy_slice_indices=[SINGLETON_INDEX_XY] )
        dataset_single_time_all_xy = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                                 [SINGLETON_INDEX_TIME] )

        datasets = [dataset_all_time_all_xy,
                    dataset_all_time_single_xy,
                    dataset_single_time_all_xy]

        # test accessing beyond the logical bounds of the underlying dataset.
        for dataset in datasets:

            # get the shape of this data set.
            number_slices     = len( dataset )
            number_time_steps = dataset.number_time_steps()
            number_xy_slices  = dataset.number_xy_slices()

            # verify that indexing beyond the length of the dataset does not
            # work.  all datasets should fail the same.
            with pytest.raises( IndexError ):
                xy_slice = dataset[number_slices]

            # verify that indexing beyond the dataset's volume using the direct
            # method does not work.
            #
            #       1. all time, all xy - shouldn't wrap into the 2nd XY slice
            #       2. all time, one xy - exceeds the dataset length, same as
            #                             above
            #       3. one time, all xy - shouldn't wrap into a later XY slice
            #
            with pytest.raises( IndexError ):
                xy_slice = dataset.get_xy_slice( number_time_steps,
                                                 0 )

            #
            #       1. all time, all xy - shouldn't wrap into the 2nd time slice
            #       2. all time, one xy - shouldn't wrap into a later time slice
            #       3. one time, all xy - exceeds the dataset length, same as
            #                             above
            #
            with pytest.raises( IndexError ):
                xy_slice = dataset.get_xy_slice( 0,
                                                 number_xy_slices )

        datasets = [dataset_all_time_single_xy,
                    dataset_single_time_all_xy]

        # verify accessing "valid" indices for full datasets does not work for
        # datasets with singleton dimensions.

        # get the shape of this data set.
        number_time_steps = dataset_all_time_single_xy.number_time_steps()

        # verify that accessing an XY slice index other than the requested
        # singleton fails.
        with pytest.raises( IndexError ):
            xy_slice = dataset_all_time_single_xy.get_xy_slice( 0,
                                                                SINGLETON_INDEX_XY - 1 )
        with pytest.raises( IndexError ):
            xy_slice = dataset_all_time_single_xy.get_xy_slice( number_time_steps - 1,
                                                                SINGLETON_INDEX_XY - 1 )

        # get the shape of this data set.
        number_xy_slices = dataset_single_time_all_xy.number_xy_slices()

        # verify that accessing a time step index other than the requested
        # singleton fails.
        with pytest.raises( IndexError ):
            xy_slice = dataset_single_time_all_xy.get_xy_slice( SINGLETON_INDEX_TIME - 1,
                                                                0 )
        with pytest.raises( IndexError ):
            xy_slice = dataset_single_time_all_xy.get_xy_slice( SINGLETON_INDEX_TIME - 1,
                                                                number_xy_slices - 1 )


    @pytest.mark.parametrize( "create_netcdf_files",
                              [(3,  ( 8, 16, 32),  ["u", "v"], "interface_getters"),
                               (10, (16, 16, 8),  ["u", "v"],  "interface_getters"),
                               (17, ( 4,  4, 97), ["u", "v"],  "interface_getters")],
                              indirect=True )
    def test_interface_getters( self, create_netcdf_files ):
        """
        Verifies that the IWPDataset's getters work as expected.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              range( parameters["number_time_steps"] ) )

        # get the parameters and derived values from them so we can validate
        # what the dataset exposes.
        variable_names    = parameters["variable_names"]
        number_time_steps = parameters["number_time_steps"]
        number_xy_slices  = parameters["grid_size"][2]

        number_variables  = len( variable_names )
        number_slices     = (number_time_steps * number_xy_slices)

        x_coordinates = np.arange( parameters["grid_size"][0] )
        y_coordinates = np.arange( parameters["grid_size"][1] )
        z_coordinates = np.arange( parameters["grid_size"][2] )

        time_step_indices = list( range( number_time_steps ) )
        xy_slice_indices  = list( range( number_xy_slices ) )

        # verify that the dataset reports the correct, total number of slices.
        assert len( dataset ) == number_slices, \
            "{:s} has an invalid number of slices.  Expected {:d} but got {:d}.".format(
                netcdf_pattern,
                number_slices,
                len( dataset ) )

        # verify that the dataset reports the correct number of time steps.
        assert dataset.number_time_steps() == number_time_steps, \
            "{:s} has an invalid number of time steps.  Expected {:d} but got {:d}.".format(
                netcdf_pattern,
                number_time_steps,
                dataset.number_time_steps() )

        assert dataset.time_step_indices() == time_step_indices, \
            "{:s} has an invalid set of time step indices.  Expected {} but got {}.".format(
                netcdf_pattern,
                time_step_indices,
                dataset.time_step_indices() )

        # verify that the dataset reports the correct number of XY slices.
        assert dataset.number_xy_slices() == number_xy_slices, \
            "{:s} has an invalid number of XY slices.  Expected {:d} but got {:d}.".format(
                netcdf_pattern,
                number_xy_slices,
                dataset.number_xy_slices() )

        assert dataset.xy_slice_indices() == xy_slice_indices, \
            "{:s} has an invalid set of XY slice indices.  Expected {} but got {}.".format(
                netcdf_pattern,
                xy_slice_indices,
                dataset.xy_slice_indices() )

        # verify that the dataset reports the correct number of variables.
        assert dataset.number_variables() == number_variables, \
            "{:s} has an invalid number of variables.  Expected {:d} but got {:d}.".format(
                netcdf_pattern,
                len( parameters["variable_names"] ),
                dataset.number_variables() )

        # verify that the dataset reports the correct X coordinates.
        assert len( dataset.x_coordinates() ) == parameters["grid_size"][0], \
            "{:s} has the wrong length X axis.  Expected {:d} entries but got {:d}.".format(
                netcdf_pattern,
                parameters["grid_size"][0],
                len( dataset.x_coordinates() ) )

        assert np.allclose( dataset.x_coordinates(),
                            np.arange( parameters["grid_size"][0] ) ), \
            "{:s} has an invalid X axis.  Some entries to not match [0, N).".format(
                netcdf_pattern )

        # verify that the dataset reports the correct Y coordinates.
        assert len( dataset.y_coordinates() ) == parameters["grid_size"][1], \
            "{:s} has the wrong length Y axis.  Expected {:d} entries but got {:d}.".format(
                netcdf_pattern,
                parameters["grid_size"][1],
                len( dataset.y_coordinates() ) )

        assert np.allclose( dataset.y_coordinates(),
                            np.arange( parameters["grid_size"][1] ) ), \
            "{:s} has an invalid Y axis.  Some entries to not match [0, N).".format(
                netcdf_pattern )

        # verify that the dataset reports the correct Z coordinates.
        assert len( dataset.z_coordinates() ) == parameters["grid_size"][2], \
            "{:s} has the wrong length Z axis.  Expected {:d} entries but got {:d}.".format(
                netcdf_pattern,
                parameters["grid_size"][2],
                len( dataset.y_coordinates() ) )

        assert np.allclose( dataset.z_coordinates(),
                            np.arange( parameters["grid_size"][2] ) ), \
            "{:s} has an invalid Z axis.  Some entries to not match [0, N).".format(
                netcdf_pattern )

        # verify that the dataset's variable names match what was provided
        # during creation.
        #
        # NOTE: order matters, so equality is the right test.
        #
        assert dataset.variables() == variable_names, \
            "{:s} has an invalid variable names.  Expected '{:s}' but got '{:s}'.".format(
                netcdf_pattern,
                ", ".join( variable_names ),
                ", ".join( dataset.number_variables() ) )

    @pytest.mark.parametrize( "create_netcdf_files",
                              [(3,  ( 8, 16, 32), ["u", "v"], "dataset_index_types"),
                               (16, ( 4,  4, 32), ["u", "v"],  "dataset_index_types")],
                              indirect=True )
    def test_dataset_index_types( self, create_netcdf_files ):
        """
        Verifies that creating IWPDatasets with different data types for time step and slice
        indices works as expected.  Creation is tested with both valid and invalid data
        types.

        Raises AssertionError if the dataset creation does not fail as expected.

        Takes 1 argument:

          create_netcdf_files - pytest fixture specifying the parameters, path pattern, and
                                individual time step file paths.

        Returns nothing.

        """

        parameters, netcdf_pattern, netcdf_paths = create_netcdf_files

        time_indices     = range( parameters["number_time_steps"] )
        xy_slice_indices = range( parameters["grid_size"][2] )

        # verify the XY slice indices can be specified as a range.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              time_indices,
                                              xy_slice_indices=xy_slice_indices )

        # verify the XY slice indices can be specified as a list.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              time_indices,
                                              xy_slice_indices=list( xy_slice_indices ) )

        # verify the XY slice indices can be specified as a tuple.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              time_indices,
                                              xy_slice_indices=tuple( xy_slice_indices ) )

        # verify the XY slice indices can be specified as a scalar integer.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              time_indices,
                                              xy_slice_indices=parameters["grid_size"][2] - 1 )

        # verify that other strange XY slice indices raise an exception.
        with pytest.raises( ValueError ):
            dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                  time_indices,
                                                  xy_slice_indices=list( map( lambda x: float( x ),
                                                                              xy_slice_indices ) ) )

        with pytest.raises( ValueError ):
            dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                  time_indices,
                                                  xy_slice_indices=0.0 )

        with pytest.raises( ValueError ):
            dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                  time_indices,
                                                  xy_slice_indices="abc" )

        # verify the time indices can be specified as a range.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              time_indices )

        # verify the time indices can be specified as a list.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              list( time_indices ) )

        # verify the time indices can be specified as a tuple.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              tuple( time_indices ) )

        # verify the time indices can be specified as a scalar integer.
        dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                              parameters["number_time_steps"] - 1 )

        # verify that other strange time indices raise an exception.
        with pytest.raises( ValueError ):
            dataset = iwp.data_loader.IWPDataset( netcdf_pattern,
                                                  list( map( lambda x: float( x ),
                                                             time_indices ) ) )

        with pytest.raises( ValueError ):
            dataset = iwp.data_loader.IWPDataset( netcdf_pattern, 0.0 )

        with pytest.raises( ValueError ):
            dataset = iwp.data_loader.IWPDataset( netcdf_pattern, "abc" )


if __name__ == "__main__":
    pytest.main()
