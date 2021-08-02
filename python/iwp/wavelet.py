import numpy as np

# Implementation of 2D continuous wavelet transform (CWT) with three wavelet
# filters.  See "2-D wavelet transforms: generalisation of the Hardy Space and
# application to experimental studies" by Dallard and Spedding (European Journal
# of Mechanics, B/Fluids, Volume 12, Number 1, 1993) for definition of Halo and
# Arc functions, as well as the 2D extension of the Morlet function.
#
# cwt_2d() is the core routine that performs a 2D CWT with the requested wavelet
# function.  Returns a 3D cube with one 2D, complex CWT per length scale.
#
# The framework to compute 2D CWTs is structured so that it can be easily ported
# to GPUs in the future.  The frequency plane construction is decoupled from the
# wavelet function computation in the CPU implementation, though may be fused
# (in several ways) to exploit hardware acceleration.

# scale factor from Dallard and Spedding 1993.  this is required to make the
# Morlet, Arc, and Halo wavelets "almost" satisfy the wavelet admissibility
# criterion (equation 17).  it should not need to be changed, though if it does
# it must be large enough to make the wavelet's response negligible when
# significantly far away from the Gaussian bump's peak.
K0_MAGNITUDE = 5.5

# named constants for each of the supported wavelets.
CWT_MORLET = "morlet"
CWT_HALO   = "halo"
CWT_ARC    = "arc"

def _cwt_2d_morlet( omega_x, omega_y, scale, alpha=0. ):
    """
    2D extension of the Morlet, directional wavelet.

    Uses a real-valued function, g(x), to select wave-vectors with both a
    preferred wavenumber and orientation.

       g(k) = scale * exp( -|(scale * k) - k0|^2 / 2 )

    Takes 4 arguments:

      omega_x - Wavenumber grid, 2D NumPy array, for the X dimension.
      omega_y - Wavenumber grid, 2D NumPy array, for the Y dimension.
      scale   - Scale factor, in wavenumbers, specifying the sensitive wavelength
                of the wavelet.
      alpha   - Orientation angle, in degrees clockwise from the X-axis,
                specifying the sensitive direction of the wavelet.

    Returns 1 value:

      morlet_values - 2D NumPy array containing the 2D Morlet wavelet function.

    """

    #
    # NOTE: the wavelet admissibility condition is not strictly satisfied but is
    #       only satisfied when the KO_MAGNITUDE is sufficiently large.
    #

    # compute the wavelet's preferred vector (direction and angle).
    k0_cos = K0_MAGNITUDE * np.cos( alpha * np.pi / 180. )
    k0_sin = K0_MAGNITUDE * np.sin( alpha * np.pi / 180. )

    # compute a scaled Gaussian with unit width at the origin specified by (k0,
    # alpha):
    #
    #   g(k) = exp( -|s*k - k0|^2 / 2 )
    #
    return scale * np.exp( -((scale * omega_x - k0_cos)**2 + (scale * omega_y - k0_sin)**2) / 2 )

def _cwt_2d_halo( omega_x, omega_y, scale ):
    """
    2D isotropic wavelet, inspired from the 2D Morlet wavelet.

    Uses a real-valued function, g(k), to select wave-vectors with a preferred
    wavenumber but is insensitive to orientation.  g(k) is defined as such:

        g(k) = exp( -(|k| - |k0|)^2 / 2 )

    NOTE: The above function is not a Hardy function and, thus, cannot retrieve
          phase of resonant signals.  See the 2D Arc transform if phase is
          important.

    Takes 3 arguments:

      omega_x - Wavenumber grid, 2D NumPy array, for the X dimension.
      omega_y - Wavenumber grid, 2D NumPy array, for the Y dimension.
      scale   - Scale factor, in wavenumbers, specifying the sensitive wavelength
                of the wavelet.

    Returns 1 value:

      halo_values - 2D NumPy array containing the 2D Halo wavelet function.

    """

    # NOTE: the wavelet admissibility condition is not strictly satisfied but is
    #       only satisfied when the KO_MAGNITUDE is sufficiently large.

    # compute a scaled, isotropic halo:
    #
    #   g(k) = scale * exp( -(|scale * k| - |k0|)^2 / 2 )
    #
    return scale * np.exp( -((scale * np.sqrt( omega_x**2 + omega_y**2 ) - K0_MAGNITUDE)**2 / 2 ) )

def _cwt_2d_arc( omega_x, omega_y, scale, rotate_flag=False ):
    """
    2D isotropic wavelet, inspired from the 2D Morlet wavelet, capable of
    recovering phase information from signals.

    Uses a complex Hardy function, g(k) to select wave-vectors with a preferred
    wavenumber but is insensitive to orientation.

    g(k) is defined as such:

        g(k) = exp( -(|k| - |k0|)^2 / 2 )

    With the complex plane partitioned into two half spaces, pi_1 and pi_2, such
    that:

        g(k) = g_hat(k)  for k in pi_1
        g(k) = 0         for k in pi_2

    The complex plane is partitioned such that pi_1 is {y<0 or y=0, x<0} and pi_2
    is the complement of pi_1.

    Takes 4 arguments:

      omega_x     - Wavenumber grid, 2D NumPy array, for the X dimension.
      omega_y     - Wavenumber grid, 2D NumPy array, for the Y dimension.
      scale       - Scale factor, in wavenumbers, specifying the sensitive wavelength
                    of the wavelet.
      rotate_flag - Optional flag specifying that the partitioning of pi_1 and pi_2
                    should be rotated 90 degrees, counter-clockwise.  If omitted,
                    defaults to False.

    Returns 1 value:

      arc_values - 2D NumPy array containing the 2D Arc wavelet function.

    """

    # start with the real-valued isotropic wavelet.
    halo_values = _cwt_2d_halo( omega_x, omega_y, scale )

    # adjust the filter so that it meets the wavelet admissibility condition.
    #
    # XXX: since we haven't needed to recover phase, this implementation has not
    #      been checked as carefully it deserves.  these are 90 rotations,
    #      counter-clockwise, of each other, though do not match the original
    #      MATLAB...
    #
    if rotate_flag == False:
        # zero Y's negative half-plane, as well as Y's negative DC.
        halo_values[(omega_y.shape[0] // 2):, :] = 0
        halo_values[0, :(omega_x.shape[1] // 2)] = 0
    else:
        # zero X's negative half-plane, as well as X's positive DC.
        halo_values[:, (omega_x.shape[1] // 2):] = 0
        halo_values[:(omega_y.shape[0] // 2), 0] = 0

    return halo_values

# name to kernel map for the  functions we support.
_wavelet_filters_map = {
    CWT_MORLET: _cwt_2d_morlet,
    CWT_HALO:   _cwt_2d_halo,
    CWT_ARC:    _cwt_2d_arc
}

def _create_frequency_plane( width, height, dtype=np.float64 ):
    """
    Creates a meshgrid for a normalized frequency plane spanning one wave period (two pi
    radians) in X and Y.  The plane created has the zero frequency components in the
    upper left quadrant, laid out like so:

       +----------+----------+
       |          |          |
       |  Px, Py  |  Nx, Py  |
       |          |          |
       +----------+----------+
       |          |          |
       |  Px, Ny  |  Nx, Ny  |
       |          |          |
       +----------+----------+

    For the positive X frequencies (Px), the negative X frequencies (Nx), positive
    Y frequencies (Py), and the negative Y frequencies (Ny), respectively.

    Takes 3 arguments:

      width  - Width of the plane.
      height - Height of the plane.
      dtype  - Optional NumPy data type to create the frequency plane with.
               If omitted, defaults to numpy.float64.

    Returns 2 values:

      omega_X - 2D NumPy array, shaped (height, width), containing grid values in the
                horizontal dimension.
      omega_Y - 2D NumPy array, shaped (height, width), containing grid values in the
                vertical dimension.

    """

    #
    # NOTE: we do not scale the frequency plane by the wavelet's length scale
    #       here so that we can more easily implement high performance GPU
    #       kernels.  having a single period frequency plane makes it easier to
    #       translate verbatim, or fuse into individual CWT kernels.
    #

    half_width  = (width - 1) // 2
    half_height = (height - 1) // 2

    omega_x = np.empty( (width,), dtype=dtype )
    omega_y = np.empty( (height,), dtype=dtype )

    # create each of the frequency plane's axes.  each spans one period, shifted
    # so positive wavenumbers precede negative.
    omega_x[0:(half_width+1)] = 2.0 * np.pi / width * np.arange( 0, half_width + 1 )
    omega_x[(half_width+1):]  = 2.0 * np.pi / width * np.arange( half_width - width + 1, 0 )

    omega_y[0:(half_height+1)] = 2.0 * np.pi / height * np.arange( 0, half_height + 1 )
    omega_y[(half_height+1):]  = 2.0 * np.pi / height * np.arange( half_height - height + 1, 0 )

    # build the frequency plane's grid.
    omega_X, omega_Y = np.meshgrid( omega_x, omega_y, indexing="xy" )

    return omega_X, omega_Y

def cwt_2d( data, scales, wavelet_name, **wavelet_parameters ):
    """
    Computes the 2D continuous wavelet transform (CWT) of a real-domained function.
    The CWT is computed at the length scales specified using the requested filter,
    with optional parameters to control the function's behavior (e.g. specifying the
    directionality of signal of interest).

    Currently supports the following filters:

      CWT_MORLET       2D extension of the Morlet wavelet.  See cwt_2d_morlet() for details.
      CWT_HALO         2D extension of the Halo wavelet.  See cwt_2d_halo() for details.
      CWT_ARC          2D extension of the Arc wavelet.  See cwt_2d_arc() for details.

    Raises ValueError if the wavelet type requested is unknown or the input data's
    data type isn't 32- or 64-bit floating point.

    Takes 4 arguments:

      data               - Data, shaped (height, width), to compute the 2D CWT of.  Must
                           be of type numpy.dtype( 'float32' ) or numpy.dtype( 'float64' ).
      scales             - List of length scales to compute the 2D CWT at.  Must
      wavelet_name       - Name of the wavelet to compute with.  Must be one of
                           CWT_MORLET, CWT_HALO, or CWT_ARC.
      wavelet_parameters - Optional dictionary providing parameters to the filter
                           kernels.  If omitted, the default parameters for each of
                           the kernels is used.

    Returns 1 value:

      cwt - Complex, 2D CWT of data, shaped (len( scales ), height, width), with the
            same data type as data (e.g. np.complex64 for np.float32 input).

    """

    # ensure we have an implementation of the requested wavelet.
    if wavelet_name not in _wavelet_filters_map:
        raise ValueError( "Unknown wavelet requested ('{:s}'), must be one of : {:s}.".format(
            wavelet_name,
            ", ".join( map( lambda name: "'" + name + "'", _wavelet_filters_map.keys() ) ) ) )

    # ensure that we got floating point data to work with.
    #
    # NOTE: we could be more friendly and accept integral data types, but it is
    #       not a) a case we deal with or b) something easily done correctly
    #       by choosing the proper data type to promote to.  as such, we simply
    #       punt to the caller to make the choice themselves.
    #
    if data.dtype == np.dtype( np.float32 ):
        cwt_dtype = np.complex64
    elif data.dtype == np.dtype( np.float64 ):
        cwt_dtype = np.complex128
    else:
        #
        # NOTE: we should be better about reporting a string value of the dtype,
        #      this is a weird corner of NumPy so we don't bother right now.
        #
        raise ValueError( "Unknown data type requested ({}).  Must be either "
                          "'np.float32' or 'np.float64'.".format(
                              data.dtype ) )

    # move the data into the Fourier domain so we can do element-wise
    # multiplications instead of expensive, 2D convolutions.
    data_spectra = np.fft.fft2( data )

    # create the base frequency plane.  each CWT scales this as necessary.
    omega_x, omega_y = _create_frequency_plane( data.shape[1], data.shape[0] )

    # pre-allocate space for our transformed data.
    cwt = np.empty( (len( scales ),
                     data.shape[0],
                     data.shape[1]),
                    dtype=cwt_dtype )

    # walk through each of the length scales, build the wavelet's Fourier
    # response, and filter the data.
    for scale_index, scale in enumerate( scales ):
        wavelet_filter = _wavelet_filters_map[wavelet_name]( omega_x,
                                                             omega_y,
                                                             scale,
                                                             **wavelet_parameters )

        # filter in the Fourier domain and invert back to the data domain.
        #
        # NOTE: this may involve a type cast because NumPy promotes inputs to
        #       its FFT routines to 64-bit.
        #
        cwt[scale_index, ...] = np.fft.ifft2( data_spectra * wavelet_filter )

    return cwt
