#!/usr/bin/env python3

import collections.abc
import copy
import netCDF4
import numpy as np
import torch
import torch.utils.data.dataset
import xarray as xr

# XXX: add interface to get the time and slice indices from the data object?
#      this changes how the tests should be structured as there is far fewer
#      things to pass around.

#    XXX: assumes each netCDF file corresponds to a single time step.  broken up to avoid overly large files.


class IWPDataset( torch.utils.data.dataset.Dataset ):
    """
    Representation of an internal wave packet (IWP) dataset from a collection of netCDF4 files
    containing stacks of XY slices of CFD simulation outputs, one time step per file.  Provides
    a pytorch Dataset interface so that it may be accessible during training and evaluation.
    Basic accessor methods are provided so the dataset is also useful for exploratory analysis.


    XXX: description of what is required.

    """

    def __init__( self, dataset_path_pattern, time_indices, variables=[], xy_slice_indices=[] ):
        """
        Creates an IWPDataset object from a subset of the netCDF4 files specified.  The dataset
        may be a subset of time and/or XY slices and specific grid variables may be requested,
        allowing for an iterator over a precise subset of data in the underlying files.  The
        grid variables' dimensions must be "x", "y", and "z".  All grid variables must have the
        same dimensions (name and shape) throughout all time steps.

        NOTE: The underlying netCDF4 files are held open for reading for the duration of the
              IWPDataset object's life time.

        Raises ValueError exceptions if the specified parameters are incompatible with the underlying
        datasets.  Access to the netCDF4 files may result in other exceptions being raised (e.g.
        opening a non-existent file, or accessing a file without the required permissions, etc)



        Takes 4 arguments:

          dataset_path_pattern - Path pattern of the dataset, with a format specifier to instantiate
                                 into a path for each time step.  Each time step's dataset path is
                                 created by formatting dataset_path_pattern with the corresponding
                                 index in time_indices.
          time_indices         - Sequence of time step indices to create the dataset from.  Scalar
                                 integers may also be specified to indicate a specific time step.
          variables            - Optional list of variable names in the netCDF files to extract XY
                                 slices when creating the dataset.  If omitted, defaults to all of
                                 the grid variables in the netCDF4 dataset's first time step file
                                 (i.e. all 3D, floating point variables in the file associated
                                 with time_indices[0])).

                                 NOTE: The order of the concatenated XY slices matches the order of
                                       variables.

          xy_slice_indices     - Optional sequence of XY slice indices to create the dataset from.  If
                                 omitted, defaults to all XY slices present.  Scalar integers may
                                 also be specified to indicate a specific XY slice.

        Returns 1 value:

          self - IWPDataset object representing the time/volume subset for the variables requested.

        """

        super().__init__()

        # grids must be three dimensional and described with names "x", "y", and
        # "z".  this choice is slightly inflexible, but simplifies the interface
        # and resulting logic to implement.
        #
        # NOTE: dimensions are specified reverse order since underlying data are
        #       in row--major order.
        #
        target_dimensions = ("z", "y", "x")

        # ensure that our time step indices are properly typed.
        if isinstance( time_indices, int ):
            time_indices = [time_indices]
        elif not isinstance( time_indices, collections.abc.Sequence ):
            raise ValueError( "time_indices must be a scalar, sequence, or "
                              "range of time step indices (got {:s}).".format(
                                  str( type( time_indices ) ) ) )
        for index_index, time_index in enumerate( time_indices ):
            if not isinstance( time_index, int ):
                raise ValueError( "time_indices[{:d}] is a {:s} instead of an int.".format(
                    index_index,
                    str( type( time_index ) ) ) )

        # ensure that our xy slice indices are properly typed.
        if isinstance( xy_slice_indices, int ):
            xy_slice_indices = [xy_slice_indices]
        elif not isinstance( xy_slice_indices, collections.abc.Sequence ):
            raise ValueError( "xy_slice_indices must be a scalar, sequence, or "
                              "range of time step indices (got {:s}).".format(
                                  str( type( xy_slice_indices ) ) ) )
        for index_index, xy_slice_index in enumerate( xy_slice_indices ):
            if not isinstance( xy_slice_index, int ):
                raise ValueError( "xy_slice_indices[{:d}] is a {:s} instead of an int.".format(
                    index_index,
                    str( type( xy_slice_index ) ) ) )

        # get a handle to each of the netCDF files.
        #
        # NOTE: this will raise an exception if the file can't be opened.
        #
        self._netcdf_files = []
        for index_index, time_index in enumerate( time_indices ):
            current_netcdf_path = dataset_path_pattern.format( time_index )

            # attempt to open the netCDF file for reading.
            self._netcdf_files.append( netCDF4.Dataset( current_netcdf_path, "r" ) )

            # verify that each of the variables requested is in this file.
            for variable in variables:
                if not variable in self._netcdf_files[index_index].variables:
                    raise ValueError( "'{:s}' is not a variable in {:s}!".format(
                        variable,
                        current_netcdf_path ) )

        # get the available variables if the caller did not specify any.
        if len( variables ) == 0:
            # build a list of variables that have grid dimensions and are of a floating
            # point data type.
            variables = []
            for variable in self._netcdf_files[0].variables.keys():
                if ((self._netcdf_files[0][variable].dimensions != target_dimensions) or
                    (self._netcdf_files[0][variable].dtype.kind != "f")):
                    continue

                variables.append( variable )

            if len( variables ) == 0:
                raise ValueError( "Could not find any grid variables in {:s}!".format(
                    self._netcdf_files[index_index].filepath() ) )

        # ensure that the variables specified are 3D with (X, Y, Z) as their
        # dimensionality.
        for variable in variables:
            # check that the dimensions are named properly in each time step.
            for time_index in range( len( time_indices ) ):

                # make sure that this time step has the variable of interest.
                if variable not in self._netcdf_files[time_index].variables:
                    raise ValueError( "{:s} is not present in time step #{:d} (index {:d}).".format(
                        variable,
                        time_indices[time_index],
                        time_index ) )

                # make sure this variable has the proper symbolic dimensions.
                if self._netcdf_files[time_index][variable].dimensions != target_dimensions:
                    raise ValueError( "{:s} is not a 3D grid variable in {:s} "
                                      "(({:s}) vs ({:s})).".format(
                                          variable,
                                          self._netcdf_files[time_index].filepath(),
                                          ", ".join( self._netcdf_files[time_index][variable].dimensions ),
                                          ", ".join( target_dimensions ) ) )

                # make sure this variable is a floating point data type.
                if not np.issubdtype( self._netcdf_files[time_index][variable].dtype,
                                      np.floating ):
                    raise ValueError( "{:s} is {:s}, though expected a floating "
                                      "point data type (e.g. np.float32).".format(
                                          variable,
                                          str( self._netcdf_files[time_index][variable].dtype ) ) )

        # ensure that the variable dimensions match across all of the files.
        for time_index in range( 1, len( time_indices ) ):
            # make sure this time step's dimensions' sizes match the first.  it
            # is not sufficient to check that the dimension names match between
            # files when different time steps may have different dimension
            # shapes.
            for dimension in target_dimensions:
                if (self._netcdf_files[0].dimensions[dimension].size !=
                    self._netcdf_files[time_index].dimensions[dimension].size):
                    raise ValueError( "Dimension '{:s}' has differing shapes "
                                      "(({:d},) vs ({:d},)) in {:s} and {:s}.".format(
                                          dimension,
                                          self._netcdf_files[0].dimensions[dimension].size,
                                          self._netcdf_files[time_index].dimensions[dimension].size,
                                          self._netcdf_files[0].filepath(),
                                          self._netcdf_files[time_index].filepath() ) )

        # default to all available slices if the caller did not request any.
        if len( xy_slice_indices ) == 0:
            xy_slice_indices = range( 0, self._netcdf_files[0][variables[0]].shape[0] )

        # ensure that the XY slice indices within the available data.
        for index_index, xy_slice_index in enumerate( xy_slice_indices ):
            if (xy_slice_index < 0) or (xy_slice_index >= self._netcdf_files[0][variables[0]].shape[0]):
                raise ValueError( "XY index #{:d} ({:d}) is out of bounds for the grid ([0, {:d}].)".format(
                    index_index,
                    xy_slice_index,
                    self._netcdf_files[0][variables[0]].shape[0] ) )

        # keep track of the variables we were provided, or discovered.
        self._variables = variables

        # ensure that our indices are lists in case range()'s were provided.
        self._time_indices     = list( time_indices )
        self._xy_slice_indices = list( xy_slice_indices )

        # keep track of the size of our data volume and time span.
        self._number_xy_slices  = len( self._xy_slice_indices )
        self._number_time_steps = len( self._time_indices )
        self._number_surfaces   = len( self._variables )

        # compute the shape of each item we load.  this is a convenience for
        # debugging.
        self._datum_shape = (self._number_surfaces,
                             *self._netcdf_files[0][self._variables[0]].shape[1:])

        # get the underlying dimensions (x, y, z) from the netCDF files.
        self.shape = (self._netcdf_files[0].dimensions["x"].size,
                      self._netcdf_files[0].dimensions["y"].size,
                      self._number_xy_slices)

    def __del__( self ):
        """
        Destroys an IWPDataset object.

        Takes no arguments.

        Returns nothing.

        """

        # close each of the netCDF files we've opened.  take care to not access
        # the attribute if it doesn't exist, which happens if an exception was
        # thrown before the Dataset was initialized.
        if hasattr( self, "_netcdf_files" ):
            for netcdf_file in self._netcdf_files:
                netcdf_file.close()

    def __len__( self ):
        """
        Returns the number of XY slices, across all time steps, in the dataset.

        Takes no arguments.

        Returns 1 value:

          length - The number of XY slices available in the dataset.

        """

        return self._number_xy_slices * self._number_time_steps

    def __getitem__( self, index ):
        """
        Gets a specific XY slice from the underlying dataset.  XY slices are ordered
        in slice-order such that all slices for time step i occur before time step i+1.
        The index specified maps to the following time and slice indices:

           time_index     = index // number_xy_slices
           xy_slice_index = index  % number_xy_slices

        Both time_index and xy_slice_index are indices into the time and XY slice
        ranges provided when the dataset was constructed.  This distinction matters
        when the indices represent a permutation of the underlying data.

        NOTE: This does not support sliced access.

        Takes 1 argument:

          index - Integral index specifiying an XY slice.

        Returns 1 value:

          item - Array, shaped (number_variables, Y, X), containing the XY slice at
                 index.

        """

        # handle negative indexing properly.  take care to handle extremely
        # large negative values and wrap them back into an acceptable range.
        if index < 0:
            index = index % len( self )

        # break the slice index into constitute indices.
        time_index     = index // self._number_xy_slices
        xy_slice_index = index %  self._number_xy_slices

        # give a more helpful exception when we've indexed out of bounds.
        #
        # NOTE: we only check the time index as the XY slice index is guaranteed
        #       to be bounded by its construction.
        #
        if time_index >= self._number_time_steps:
            raise IndexError( "Invalid index ({:d}).  Must be in [0, {:d}].".format(
                index,
                self._number_time_steps * self._number_xy_slices ) )

        # apply the dataset's permutations the constitute indices so we walk
        # through the permutations rather than the native ordering.
        return self.get_xy_slice( self._time_indices[time_index],
                                  self._xy_slice_indices[xy_slice_index] )

    def number_time_steps( self ):
        """
        Returns the number of time steps in the dataset.

        Takes no arguments.

        Returns 1 value:

          number_time_steps - Integral number of time steps in the data set.

        """

        return copy.copy( self._number_time_steps )

    def number_xy_slices( self ):
        """
        Returns the number of XY slices in each time step.

        Takes no arguments.

        Returns 1 value:

          number_xy_slices - Integral number of XY slices within each time step.

        """

        return copy.copy( self._number_xy_slices )

    def number_variables( self ):
        """
        Returns the number of variables multiplexed within each XY slice.

        Takes no arguments.

        Returns 1 value:

          number_variables - Integral number of variables within each XY slice.

        """

        return copy.copy( self._number_surfaces )

    def variables( self ):
        """
        Returns a list of variable names corresponding to the variables multiplexed
        within each XY slice.

        Takes no arguments.

        Returns 1 value:

          variable_names - List of strings naming the variables within each XY slice.

        """

        return copy.copy( self._variables )

    def x_coordinates( self ):
        """
        Returns a NumPy array of the grid's X coordinates.

        Takes no arguments.

        Returns 1 value:

          x_coordinates - NumPy array containing the grid's X coordinates.

        """

        # make a copy of the X coordinates for the caller.
        #
        # NOTE: we explicitly return an numpy.ndarray rather than a masked array
        #       since all of the grid coordinates must be present.
        #
        return np.copy( self._netcdf_files[0].variables["x"][:].data )

    def y_coordinates( self ):
        """
        Returns a NumPy array of the grid's Y coordinates.

        Takes no arguments.

        Returns 1 value:

          y_coordinates - NumPy array containing the grid's Y coordinates.

        """

        # make a copy of the Y coordinates for the caller.
        #
        # NOTE: we explicitly return an numpy.ndarray rather than a masked array
        #       since all of the grid coordinates must be present.
        #
        return np.copy( self._netcdf_files[0].variables["y"][:].data )

    def z_coordinates( self ):
        """
        Returns a NumPy array of the grid's Z coordinates.

        Takes no arguments.

        Returns 1 value:

          z_coordinates - NumPy array containing the grid's Z coordinates.

        """

        # make a copy of the Z coordinates for the caller.
        #
        # NOTE: we explicitly return an numpy.ndarray rather than a masked array
        #       since all of the grid coordinates must be present.
        #
        return np.copy( self._netcdf_files[0].variables["z"][:].data )

    def time_step_indices( self ):
        """
        Returns a list of time step indices in the dataset.

        Takes no arguments.

        Returns 1 value:

          time_step_indices - List of time step indices.

        """

        return copy.copy( self._time_indices )

    def xy_slice_indices( self ):
        """
        Returns a list of XY slice indices in the dataset.

        Takes no arguments.

        Returns 1 value:

          xy_slice_indices - List of XY slice indices.

        """

        return copy.copy( self._xy_slice_indices )

    def get_xy_slice( self, time_index, xy_slice_index ):
        """
        Extracts an XY slice from the underlying dataset specified by the time and XY slice
        indices provided.

        Raises an IndexError exception if either of the specified indices are out of bounds.

        Takes 2 arguments:

          time_index     - Integral index into the dataset's time steps.  Must be in the
                           range [0, number_time_steps).
          xy_slice_index - Integral index into the dataset's XY slice.  Must be in the
                           range [0, number_xy_slices).

        Returns 1 value:

          xy_slice - Array, shaped (number_variables, Y, X), containing the XY slice at
                     index.

        """

        # handle negative indexing such that it behaves the same way as with
        # __getitem__().
        #
        # NOTE: care is taken to handle extremely large negative values and wrap
        #       them back into an acceptable range, which is different than
        #       Python's default behavior.  this seems more desirable than
        #       unnecessarily throwing an IndexError exception, though could
        #       be accommodated if this breaks something downstream.
        #
        if time_index < 0:
            time_index = self._time_indices[time_index % self._number_time_steps]

        if xy_slice_index < 0:
            xy_slice_index = self._xy_slice_indices[xy_slice_index % self._number_xy_slices]

        # ensure the requested indices, both time and XY, makes sense for the
        # collection of files we have open.  this is required to support opening
        # datasets with non-contiguous subsets in time and/or Z.
        if time_index not in self._time_indices:
            raise IndexError( "Requested time step index {:d}, though have {} time steps available.".format(
                time_index,
                self._time_indices ) )
        elif xy_slice_index not in self._xy_slice_indices:
            raise IndexError( "Requested XY slice index {:d}, though have {} XY slice indices available.".format(
                xy_slice_index,
                self._xy_slice_indices ) )

        # find the position of the target time step so we can access its
        # corresponding netCDF4 file.
        time_index_index = self._time_indices.index( time_index )

        # ensure the requested XY slice is within the available grid.
        if xy_slice_index >= self._netcdf_files[time_index_index].dimensions["z"].size:
            raise IndexError( "Requested XY slice (Z index #{:d}, though only {:d} slices present.".format(
                xy_slice_index,
                self._netcdf_files[time_index_index].dimensions["z"].size ) )

        # create an empty buffer big enough for each of the variables' XY slice
        # stacked together.
        xy_slice = np.empty( self._datum_shape, dtype=np.float32 )

        # walk through each variable and copy its data into the buffer.
        for variable_index, variable_name in enumerate( self._variables ):
            xy_slice[variable_index, :] = self._netcdf_files[time_index_index][variable_name][xy_slice_index, :]

        return xy_slice

def open_xarray_dataset( dataset_path_pattern ):
    """
    Opens an IWP dataset as an xarray.Dataset from one or more netCDF4 files.

    Takes 1 argument:

      dataset_path_pattern - Path to the dataset to open.  May include '*' for simple
                             globbing, or specified as a list of paths to open.

    Returns 1 value;

      dataset - xarray.Dataset associated with dataset_path_pattern.

    """

    def add_timestep_as_coord( ds ):
        """
        Adds time-based coordinate so it may be properly concatenated when
        accessed as a multi-file data set.  Care is detected to identify
        old-style files with "Cycle" attributes as well as new-style files
        with "time_step" attributes.

        Takes argument:

          ds - xarray.Dataset to modify.

        Returns 1 value:

          ds - Modified xarray.Dataset with the time-based coordinate added.

        """

        if "Cycle" in ds.attrs:
            time_step_attribute = "Cycle"
        else:
            time_step_attribute = "time_step"

        ds.coords[time_step_attribute] = ds.attrs[time_step_attribute]

        return ds

    # flatten a list of a single entry into a string.
    #
    # NOTE: this is required to be compatible with xarray's open_mfdataset()
    #       interface.  version 0.18.0 does not allow a list of patterns, but
    #       only a list of paths.  this provides a convenience for the case
    #       where a single pattern is provided, but the caller did not attempt
    #       to detect whether it was a pattern or a path.
    #
    if (type( dataset_path_pattern ) == list) and (len( dataset_path_pattern ) == 1):
        dataset_path_pattern = dataset_path_pattern[0]

    # each timestep is generated independently of the others, so we specify
    # nested concatenation across timesteps.  we preprocess each dataset as its
    # read to promote the timestep attribute to a coordinate so it may be
    # concatenated.
    #
    # NOTE: we cannot detect if we have an old-style or new-style file until
    #       we open the dataset and have access to its attributes.  iterate
    #       through the time-based attributes ordered by frequency we'll
    #       encounter them.  while opening each file in parallel, for each
    #       style, can be costly we'll almost always open the file once as
    #       new-style files will be generated for several years, while old-style
    #       files have not been created for several years (and will eventually
    #       disappear).
    #
    for time_step_attribute in ["time_step", "Cycle"]:
        try:
            ds = xr.open_mfdataset( dataset_path_pattern,
                                    parallel=True,
                                    combine="nested",
                                    concat_dim=[time_step_attribute],
                                    preprocess=add_timestep_as_coord )
            break
        except:
            # we don't do anything when we can't open a particular style of
            # field data.  below we raise an exception should this dataset
            # not match any of the known styles.
            pass
    else:
        raise RuntimeError( "Could not open '{:s}' as any known format of field data.".format(
            dataset_path_pattern ) )

    return ds
