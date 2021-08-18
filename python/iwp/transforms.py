import math

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
