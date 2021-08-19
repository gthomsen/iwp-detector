import math
import netCDF4 as nc

import iwp.wavelet

# this module provides a framework for generating data transforms in
# systematic, reproducible manner that can also be used in a variety of
# execution contexts, be they serial or parallel.  care is taken so that
# transforms can be worked with in both an interactive development context
# or in a production-like batch processing context.
#
# the framework implemented is centered around tranformer function factories
# that are (primarily) acquired through a lookup function that translates a
# transformer specification string into the transform function.  said transform
# function encodes its parameterization and internal data flow so that it
# can be called with the target data without requiring any additional
# parameterization.
#
# the above roughly maps to the following execution flow:
#
#   1. select and parameterize the transform:
#
#     >>> transformer_spec = "symmetric_morlet_max_modulus:alpha=50:scales=2,4,8,16,32"
#
#   2. acquire the transformer:
#
#     >>> transformer, _, _ = lookup_transform( transform_spec )
#
#   3. apply the transformer to data:
#
#    >>> transformed_data = transformer( array_data )
#
# the underlying transformer factories are part of the public interface to
# allow more direct acquisition of the transformer (skipping the need of
# encoding parameters into a transformer specification).
#
# the following priorities drove the design of this module:
#
#   * expose transforms that are reproducible so that computations are
#     consistent and resistant to human error.
#
#   * support development in both interactive sessions (i.e. Jupyter) as
#     well as batch processing (i.e. command line scripts).
#
#   * deal with the dependency requirements for IWP datasets and the
#     development environment.  this requires working around limitations
#     in Jupyter (i.e. not allowing interactive multiprocess pools)
#     and in the netCDF4/HDF5 stack (i.e. not playing well with concurrent
#     accesses and threading).
#
# adding new transforms to the module can be done via the following:
#
#    1. add a new constant - this gives the transformer specification a
#       name to work with.
#
#    2. add the transform generator - this is the meat of the task.  be
#       sure to update the generated transform's docstring to be descriptive.
#
#    3. add the transform specification's parameter parser - this validates
#       the parameters required by the generator and makes the module
#       robust to external inputs (e.g. command line arguments).
#
#    4. add the transform to the _transform_map - this wires the transform
#       into the public interface via lookup_transform().
#
# going forward, the following things should be done to make this more
# maintainable and usable to the end user:
#
#   * rework the internal structure so that transform types are
#     classes rather than implicitly coupled groups of functions.
#     this would make it easier for maintainers to make changes
#     without worrying if they changed everything as it can be
#     enforced via a class interface.
#
#   * figure out a way to make the interface validation code less
#     tedious without either pulling in a ton of dependencies or
#     adding a mountain of indirection.
#
# names of the transforms supported by this module.  see the associated
# functions in the _transform_map dictionary for details on what each transform
# is.
#
# NOTE: these need to stay in sync with lookup_transform()'s documentation.
#
SYMMETRIC_MORLET_MAX_MODULUS  = "symmetric_morlet_max_modulus"
SYMMETRIC_MORLET_SINGLE_SCALE = "symmetric_morlet_single_scale"

class IWPnetCDFTransformer( object ):
    """
    Class implementing transformation of a single IWP variable, in an on-disk
    netCDF4 file, into one or more output variables.  This provides support to
    post-process data on disk, either for workflow preparation or for development
    purposes.

    The following pattern describes this class' use case:

      1. Initialize an IWPnetCDFTransformer object describing the reference
         variable's relationship to the output variables.
      2. Invoke the IWPnetCDFTransformer object on one or more netCDF4 files to execute
         the transformations setup in #1.  This may be done in parallel.

    This class is structured such that IWPnetCDFTransformer objects may be
    serialized and distributed to parallel execution contexts
    (e.g. multiprocess.pool.Pool) so they are executed concurrently.  Care is taken
    during object initialization to partially defer setup so that
    serialization/deserialization is possible via the pickle module.

    """

    def __init__( self, input_name, output_names, transform_specs, verbose_flag=False ):
        """
        Creates an IWPnetCDFTransformer object that will apply one or more
        transformations from a reference variable in a netCDF4 file and write the
        results into named output variables.

        Transformations are validated during object initialization, though transformer
        creation and data processing are deferred until the object is called, allowing
        for serialization/deserialization of the object through Python's native pickling
        solution.  Consequently, calls may fail due to the underlying netCDF4 file being
        incompatible with the internal parameters (e.g.  the reference variable doesn't
        exist).

        Takes 4 arguments:

          input_name      - Name of the reference variable to apply transformations to.

                              NOTE: This must exist in the netCDF4 file supplied when
                                     transformations are applied.  Processing is aborted
                                     and an error returned if it does not.

          output_names    - List of variable names to write transformer outputs.  These
                            will either be created or overwritten depending on whether
                            they exist at the time when transformations are applied.  Must
                            have the same number of elements as transformer_specs.
          transform_specs - List of transformation specifications (see lookup_transform()
                            for details) to instantiate and apply to input_name during
                            transformation application.
          verbose_flag    - Optional boolean flag specifying whether execution should
                            be verbose.  If omitted, defaults to False.

        Returns 1 value:

          self - Newly created IWPnetCDFTransformer object.

        """

        # make sure we have one name per transform specification.
        if len( output_names ) != len( transform_specs ):
            raise ValueError( "Must have variable name per transform specification.  "
                              "Received {:d} name{:s} and {:d} transform{:s}.".format(
                                  len( output_names ),
                                  "" if len( output_names ) == 1 else "s",
                                  len( transform_specs ),
                                  "" if len( transform_specs ) == 1 else "s" ) )
        elif any( map( lambda x: len( x ) == 0, output_names ) ):
            raise ValueError( "Output names must be non-empty.  At least one name "
                              "is invalid ('{:s}').".format(
                                  output_names ) )

        # make sure we have valid transform specifications.  let the user know
        # about problems now, rather than later, when we're potentially in a
        # parallel region (and all of the workers have an issue).
        for transform_index, transform_spec in enumerate( transform_specs ):
            if not is_valid_transform_spec( transform_spec ):
                raise ValueError( "Transform #{:d} ('{:s}') is invalid.".format(
                    transform_index + 1,
                    transform_spec ) )

        self._input_name   = input_name
        self._output_names = output_names

        self._verbose_flag = verbose_flag

        #
        # NOTE: we store the transform specifications at object construction so
        #       that we meet Python's pickling requirements of no
        #       locally-captured functions which lookup_transforms() does.
        #       instead, we validate the requested transforms now so we can
        #       raise an exception in a (hopefully) non-parallel context and
        #       then acquire said transforms on demand in a (hopefully) parallel
        #       context.
        #
        self._transforms      = []
        self._transform_specs = transform_specs

    def __call__( self, netcdf_path ):
        """
        Opens a IWP netCDF4 file and applies transforms to the reference variable,
        storing the result in each of the output variables configured during
        initialization.  The netCDF4 file is opened for append so output variables
        can be created or overwritten.

        Takes 1 argument:

          netcdf_path - Path to an IWP netCDF4 file to process.

        Returns 2 values:

          status_code    - Integer status code capturing the result of the call.  0 if
                           the operation was successful, non-zero if an error occurred.
          status_message - String describing the result of the call.

        """

        # if this is first time we've been called, lookup the transforms we
        # need.
        if len( self._transforms ) == 0:
            self._setup_transforms()

        if self._verbose_flag:
            print( "Processing '{:s}': Transforming '{:s}' into {:s} with {:d} transform{:s}.".format(
                netcdf_path,
                self._input_name,
                ", ".join( map( lambda x: "'" + x + "'", self._output_names ) ),
                len( self._transforms ),
                "" if len( self._transforms ) == 1 else "s" ) )

        # open the file and apply each of the transforms in a serial manner.
        #
        # wrap the entire block with a try/except so that we can gracefully
        # handle whatever errors bubble up from the netCDF4 package and
        # return something sensible to the caller.
        try:

            # since we're either adding a new variable or overwriting an
            # existing, we need to open the file for append.  opening for
            # write will wipe out the existing file.
            #
            # NOTE: make sure we work with the file as a context manager so that
            #       the netCDF4 file is closed.  netCDF4/HDF5 seems prone to
            #       corrupt files and prevent future access without this.
            #
            with nc.Dataset( netcdf_path, mode="a" ) as input_nc:
                # make sure the reference variable exists.  we couldn't do this
                # at initialization time, so check now.
                if self._input_name not in input_nc.variables:
                    raise RuntimeError( "The reference variable '{:s}' does not "
                                        "exist in '{:s}'.".format(
                                            self._input_name,
                                            netcdf_path ) )

                reference_variable = input_nc.variables[self._input_name]

                # walk through each the transforms.
                for transform_index, transform_tuple in enumerate( self._transforms ):
                    (transform,
                     transform_name,
                     transform_parameters) = transform_tuple

                    variable_name = self._output_names[transform_index]

                    if variable_name not in input_nc.variables:
                        if self._verbose_flag:
                            print( "  Creating '{:s}'".format(
                                variable_name ) )

                        # create this variable as a template of the reference.
                        # this inherits the reference's data type, dimensions,
                        # chunk sizes, and filters (i.e. compression
                        # configuration).  without this, we're likely to get
                        # sub-optimal defaults.
                        variable = input_nc.createVariable( variable_name,
                                                            reference_variable.datatype,
                                                            reference_variable.dimensions,
                                                            chunksizes=reference_variable.chunking(),
                                                            **(reference_variable.filters()) )

                        # copy over any attributes that the reference has.
                        #
                        # NOTE: this may not be correct in all cases if this
                        #       transform changes the units of the data.  this
                        #       is *very* unlikely to affect anyone as its
                        #       metadata-only and would add a fair amount of
                        #       complexity to do right.  let's pretend this
                        #       isn't an issue and call it a day.
                        #
                        input_nc[variable_name].setncatts( input_nc[self._input_name].__dict__ )
                    elif self._verbose_flag:
                        print( "  '{:s}' already exists.".format(
                            variable_name ) )

                    # work each of the XY slices one at a time.
                    for z_index in range( reference_variable.shape[0] ):
                        if (z_index % 50) == 0 and self._verbose_flag:
                            print( "        Z={:d}".format( z_index ) )

                        input_nc[variable_name][z_index, :, :] = transform( reference_variable[z_index, :, :] )

        except Exception as e:
            # XXX: we may need something more descriptive here
            return (1, str( e ) )

        return (0, "Success")

    def _setup_transforms( self ):
        """
        Translates transformer specifications into the underlying transform functions.
        Each of the specifications validated at object construction time are
        looked up and stored internally for future use.

        Takes no arguments.

        Returns nothing.

        """

        #
        # NOTE: while this is normally fragile, though we can't get here unless
        #       the object was constructed with one or more valid transform
        #       specifications.
        #
        self._transforms = []
        for transform_spec in self._transform_specs:
            self._transforms.append( lookup_transform( transform_spec ) )

def get_symmetric_morlet_max_modulus_transform( transform_parameters ):
    """
    Gets a transform function to compute the modulus of a symmetric 2D continuous
    wavelet transform (CWT) using a directional Morlet wavelet function at multiple
    length scales.  Builds a function from the supplied transform parameters that
    operates on 2D NumPy arrays.

    The Morlet wavelet has a preferred orientation angle, alpha, though is not
    actually symmetric.  This transformer computes two CWTs at both alpha and -alpha
    and takes the maximum modulus across both.  The resulting output is clamped on
    the lower end to avoid numerical artifacts producing negative modulus values.

    See iwp.wavelet.cwt_2d() for details on the CWT.

    Takes 1 argument:

      transform_parameters - Dictionary with the following (key, value) pairs:

                               alpha       - Floating point angle specifying the
                                             preferred orientation of the Morlet CWT.
                                             The symmetric Morlet CWT is computed at
                                             both alpha and -alpha.
                               scales      - List of floating point values specifying
                                             the length scales to compute the Morlet CWT
                                             at.
                               scale_index - Optional integer specifying the scale in
                                             scales to compute the Morlet CWT at.  If
                                             omitted, defaults to 0 and selects the
                                             first length scale in scales.

    Returns 1 value:

      transform_function - Function to compute the transform.

    """

    # name the parameters we use in the transformer build process for clarity.
    preferred_angle = transform_parameters["alpha"]
    length_scales   = transform_parameters["scales"]

    def symmetric_morlet_max_modulus_transform( data, minimum_value=None ):
        """
        Computes the modulus of a symmetric 2D continuous wavelet transform (CWT) using
        a directional Morlet wavelet function, at multiple length scales, and returns its
        modulus.

        The Morlet wavelet has a preferred orientation angle, alpha, though is not
        actually symmetric.  This transform computes two CWTs at both alpha and -alpha
        and takes the maximum modulus across both.  The resulting output is clamped on
        the lower end to avoid numerical artifacts producing negative modulus values.

        This transform computes with the following parameters:

          alpha:  {{{alpha:.1f}, -{alpha:.1f}}} degrees
          scales: {length_scales}

        See iwp.wavelet.cwt_2d() for details on the CWT.

        Takes 2 arguments:

          data          - 2D NumPy array to compute the 2D CWT modulus for.
          minimum_value - Optional lower bound to clamp the modulus with.  If omitted,
                          defaults to a small value near zero.

        Returns 1 value:

          transformed_data - 2D NumPy array containing the maximum modulus across the
                             computed CWTs.

        """

        # pick a lower bound close to zero if the caller does not provide one.
        if minimum_value is None:
            minimum_value = 1e-7

        return iwp.wavelet.cwt_max_modulus( [iwp.wavelet.cwt_2d( data,
                                                                 length_scales,
                                                                 "morlet",
                                                                 alpha=preferred_angle ),
                                             iwp.wavelet.cwt_2d( data,
                                                                 length_scales,
                                                                 "morlet",
                                                                 alpha=(-1.0 * preferred_angle) )],
                                             minimum_value=minimum_value )

    # update the transform's docstring with its parameters so it is
    # self-descriptive.
    symmetric_morlet_max_modulus_transform.__doc__ = symmetric_morlet_max_modulus_transform.__doc__.format(
        alpha=preferred_angle,
        length_scales=length_scales )

    return symmetric_morlet_max_modulus_transform

def parse_symmetric_morlet_max_modulus_parameters( parameter_map ):
    """
    Parses parameters for the maximum modulus, symmetric Morlet wavelet generator.
    Takes a dictionary of (key, string value)'s and casts the values to the types
    expected by get_symmetric_morlet_max_modulus_transform().

    The following keys are expected in the supplied parameter map:

      alpha  - Morlet wavelet preferred angle, in degrees.  Must be a finite
               floating point value.
      scales - Comma separated list of length scales.  Must be positive floating
               point values.

    Raises ValueError if any of the required parameters are missing, they cannot be
    cast to the required data type, they have invalid values for their use, or if
    they're incompatible with each other.

    Takes 1 argument:

      parameter_map - Dictionary of parameters to parse for
                      get_symmetric_morlet_max_modulus_transform().

    Returns 1 value:

      parameter_map - Dictionary of parameters with type cast values.

    """

    # parse a floating point angle.
    try:
        parameter_map["alpha"] = float( parameter_map["alpha"] )
    except Exception as e:
        raise ValueError( "Failed to parse a floating point angle, \"alpha\", "
                          " from '{:s}' ({:s}).".format(
                              parameter_map["alpha"],
                              str( e ) ) )

    # ensure this is a finite value.
    if not math.isfinite( parameter_map["alpha"] ):
        raise ValueError( "Alpha is not finite ({:f}).".format(
            parameter_map["alpha"] ) )

    # parse a list of floating point values.
    try:
        parameter_map["scales"] = list( map( lambda scale: float( scale ),
                                             parameter_map["scales"].split( "," ) ) )
    except Exception as e:
        raise ValueError( "Failed to parse a list of floating point length "
                          "scales, \"scales\", from '{:s}' ({:s}).".format(
                              parameter_map["scales"],
                              str( e ) ) )

    # ensure all of the length scales are finite.
    if not all( map( lambda scale: math.isfinite( scale ) and scale > 0,
                     parameter_map["scales"] ) ):
        raise ValueError( "Length scales must be positive and finite ({:s}).".format(
            ", ".join( map( lambda scale: str( scale ),
                            parameter_map["scales"] ) ) ) )

    # make sure we have at least one length scale to work with.
    if len( parameter_map["scales"] ) == 0:
        raise ValueError( "Must have at least one length scale!" )

    return parameter_map

def get_symmetric_morlet_single_scale_transform( transform_parameters ):
    """
    Gets a transform function to compute the modulus of a symmetric 2D continuous
    wavelet transform (CWT) using a directional Morlet wavelet function at a single
    length scale.  Builds a function from the supplied transform parameters that
    operates on 2D NumPy arrays.

    The Morlet wavelet has a preferred orientation angle, alpha, though is not
    actually symmetric.  This transformer computes two CWTs at both alpha and -alpha
    and takes the maximum modulus across both.  The resulting output is clamped on
    the lower end to avoid numerical artifacts producing negative modulus values.

    See iwp.wavelet.cwt_2d() for details on the CWT.

    Takes 1 argument:

      transform_parameters - Dictionary with the following (key, value) pairs:

                               alpha       - Floating point angle specifying the
                                             preferred orientation of the Morlet CWT.
                                             The symmetric Morlet CWT is computed at
                                             both alpha and -alpha.
                               scales      - List of floating point values specifying
                                             the length scales to compute the Morlet CWT
                                             at.
                               scale_index - Optional integer specifying the scale in
                                             scales to compute the Morlet CWT at.  If
                                             omitted, defaults to 0 and selects the
                                             first length scale in scales.

    Returns 1 value:

      transform_function - Function to compute the transform.

    """

    # name the parameters we use in the transformer build process for clarity.
    preferred_angle = transform_parameters["alpha"]
    length_scales   = transform_parameters["scales"]
    scale_index     = transform_parameters.get( "scale_index", 0 )

    length_scale = length_scales[scale_index]

    def symmetric_morlet_single_scale_transform( data, minimum_value=None ):
        """
        Computes the modulus of a symmetric 2D continuous wavelet transform (CWT) using
        a directional Morlet wavelet function, at a single length scale, and returns its
        modulus.

        The Morlet wavelet has a preferred orientation angle, alpha, though is not
        actually symmetric.  This transform computes two CWTs at both alpha and -alpha
        and takes the maximum modulus across both.  The resulting output is clamped on
        the lower end to avoid numerical artifacts producing negative modulus values.

        This transform computes with the following parameters:

          alpha: {{{alpha:.1f}, -{alpha:.1f}}} degrees
          scale: {length_scale:.1f}

        See iwp.wavelet.cwt_2d() for details on the CWT.

        Takes 2 arguments:

          data          - 2D NumPy array to compute the 2D CWT modulus for.
          minimum_value - Optional lower bound to clamp the modulus with.  If omitted,
                          defaults to a small value near zero.

        Returns 1 value:

          transformed_data - 2D NumPy array containing the maximum modulus across the
                             computed CWTs.

        """

        # pick a lower bound close to zero if the caller does not provide one.
        if minimum_value is None:
            minimum_value = 1e-7

        return iwp.wavelet.cwt_max_modulus( [iwp.wavelet.cwt_2d( data,
                                                                 [length_scale],
                                                                 "morlet",
                                                                 alpha=preferred_angle ),
                                             iwp.wavelet.cwt_2d( data,
                                                                 [length_scale],
                                                                 "morlet",
                                                                 alpha=(-1.0 * preferred_angle) )],
                                             minimum_value=minimum_value )

    # update the transform's docstring with its parameters so it is
    # self-descriptive.
    symmetric_morlet_single_scale_transform.__doc__ = symmetric_morlet_single_scale_transform.__doc__.format(
        alpha=preferred_angle,
        length_scale=length_scale )

    return symmetric_morlet_single_scale_transform

def parse_symmetric_morlet_single_scale_parameters( parameter_map ):
    """
    Parses parameters for the single scale, symmetric Morlet wavelet generator.
    Takes a dictionary of (key, string value)'s and casts the values to the
    types expected by get_symmetric_morlet_single_scale_transform().

    The following keys are expected in the supplied parameter map:

      alpha       - Morlet wavelet preferred angle, in degrees.  Must be a finite
                    floating point value.
      scales      - Comma separated list of length scales.  Must be positive floating
                    point values.
      scale_index - Optional index specifying the length scale to use.  If supplied,
                    must be an non-negative, integral index smaller than the number
                    of length scales specified in scales.  If omitted, defaults to 0
                    and selects the first length scale.

    Raises ValueError if any of the required parameters are missing, they cannot be
    cast to the required data type, they have invalid values for their use, or if
    they're incompatible with each other.

    Takes 1 argument:

      parameter_map - Dictionary of parameters to parse for
                      get_symmetric_morlet_single_scale_transform().

    Returns 1 value:

      parameter_map - Dictionary of parameters with type cast values.

    """

    # the single scale symmetric Morlet transform uses the same parameters as
    # the maximum modulus version.  start with those parameters.
    parameter_map = parse_symmetric_morlet_max_modulus_parameters( parameter_map )

    # the scale index is optional and defaults to the first length scale.
    try:
        parameter_map["scale_index"] = int( parameter_map.get( "scale_index", 0 ) )
    except Exception as e:
        raise ValueError( "Failed to parse an integer scale index, \"scale_index\", "
                          "from '{:s}' ({:s}).".format(
                              parameter_map["scale_index"],
                              str( e ) ) )

    # make sure we're not indexing outside of the available length scales.
    if not( 0 <= parameter_map["scale_index"] < len( parameter_map["scales"] ) ):
        raise ValueError( "Invalid scale index provided ({:d}).  Must be in the "
                          "range of [0, {:d}).".format(
                          parameter_map["scale_index"],
                          len( parameter_map["scales"] ) ) )

    return parameter_map

def _parse_transform_spec( transform_spec ):
    """
    Parses a transform specification into its name and parameters dictionary.

    Raises ValueError if the specification is invalid, it represents an unknown
    transform, or if the encoded parameters do not match the transform's expected
    types.

    Takes 1 argument:

      transform_spec - Transform specification string.  See lookup_transform() for
                       details.

    Returns 2 values:

      transform_name       - Name of the specified transform.
      transform_parameters - Dictionary of parameters for the specified transform.
                             Dictionary values are cast to the types expected
                             the transform.

    """

    try:
        # break the "<name>:<parameters>" string.  make sure we don't break
        # the <parameters> into multiple components so it can contain colons
        # in the (key, value) pairs.
        (transform_name,
         transform_parameters_spec) = transform_spec.split( ":",
                                                            maxsplit=1 )
    except ValueError:
        raise ValueError( "Failed to get a transform name and parameters "
                          "specification from '{:s}'.".format(
                          transform_spec ) )

    # make sure this is a known transform.
    if transform_name not in _transform_map:
        raise ValueError( "Unknown transform '{:s}'!".format(
            transform_name ) )

    # get the associated parameter parser for this transform.
    _, parameter_parser = _transform_map[transform_name]

    try:
        # split the remaining <parameters> into (key, value) pairs.  each
        # (key, value) set is colon-delimited, and each set equal
        # sign-delimited.
        #
        # e.g. "parameter1=value1:parameter2=value2a,value2b,value2c"
        #
        transform_parameters = dict( map( lambda key_value: key_value.split( "=" ),
                                          transform_parameters_spec.split( ":" ) ) )

        # map individual parameters to their expected data types.
        transform_parameters = parameter_parser( transform_parameters )
    except ValueError as e:
        raise ValueError( "<parameters> -> (<key>, <value>) ({:s})".format(
            str( e ) ) )

    return (transform_name, transform_parameters)

def is_valid_transform_spec( transform_spec ):
    """
    Predicate for determining whether a transform specification is valid.
    Valid specifications can be mapped to transform functions via lookup_transform().

    Takes 1 argument:

      transform_spec - Transform specification string.  See lookup_transform() for
                       details.

    Returns 1 value:

      validity_flag - Boolean flag indicating whether transform_spec is valid or
                      not.

    """

    # if we can parse the specification, this is valid.
    try:
        transform_name, _ = _parse_transform_spec( transform_spec )
    except Exception as e:
        return False

    return True

def lookup_transform( transform_spec ):
    """
    Looks up a transform specification and returns the associated transform
    function.  The returned function takes a data object, applies its transform
    operations, and returns the result.

    The following table shows which transforms are supported and the transform
    generator that has their details:


                 Transform Name                                 Details
       ----------------------------------     ---------------------------------------------
       Symmetric Morlet - Maximum Modulus     Constant: SYMMETRIC_MORLET_MAX_MODULUS
                                              Name:     "symmetric_morlet_max_modulus"
                                              Function: get_symmetric_morlet_max_modulus_transform()

                                              Computes the 2D CWT using Morlet with an
                                              angle and its reflection (angle and -angle)
                                              for one or more length scales, and computes
                                              the maximum modulus across all length scales
                                              for each (x, y) point in the 2D CWT.

       Symmetric Morlet - Single Scale        Constant: SYMMETRIC_MORLET_SINGLE_SCALE
                                              Name:     "symmetric_morlet_single_scale"
                                              Function: get_symmetric_morlet_single_scale_transform()

                                              Computes the 2D CWT using Morlet with an
                                              angle and its reflection (angle and -angle)
                                              for one length scale and computes the modulus
                                              for each (x, y) point in the 2D CWT.

    Transform specifications have the following structure:

      <transform name>:<parameters>

    <parameters>'s is a colon-delimited list of (key, value) pairs separated by
    equal signs like so:

      <param1>:<value1>:<param2>:<value2a>,<value2b>...

    See the transform of interest's generating function for details on the parameters
    accepted and their requirements.  (key, value) pairs may come in any order.

    Raises ValueError if the transform specification is invalid or if there was an
    error acquiring the transform function.

    Takes 1 argument:

      transform_spec - Transform specification string as described above.

    Returns 3 values:

      transform            - The transform function.  See the associated transform
                             generator for its signature.
      transform_name       - Name of the transform as parsed from transform_spec.
      transform_parameters - Dictionary of type cast parameters and their values
                             parsed from transform_spec according to the
                             requested transform's requirements.

    """

    # see if this is a well formed specification.
    try:
        (transform_name,
         transform_parameters) = _parse_transform_spec( transform_spec )
    except ValueError as e:
        raise ValueError( "Failed to parse the transform spec '{:s}' ({:s}).".format(
            transform_spec,
            str( e ) ) )

    # route the parameters to the appropriate transform builder.
    try:
        transform = _transform_map[transform_name][0]( transform_parameters )
    except ValueError as e:
        raise ValueError( "Failed to get a transform function for '{:s}' ({:s}).".format(
            transform_name,
            str( e ) ) )

    return (transform, transform_name, transform_parameters)

# internal mapping from symbolic name to 1) the transform generator and 2) the
# generator's parameter parser.
_transform_map = {
    SYMMETRIC_MORLET_MAX_MODULUS:  (get_symmetric_morlet_max_modulus_transform,
                                    parse_symmetric_morlet_max_modulus_parameters),
    SYMMETRIC_MORLET_SINGLE_SCALE: (get_symmetric_morlet_single_scale_transform,
                                    parse_symmetric_morlet_single_scale_parameters)
}
