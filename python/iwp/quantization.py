import numpy as np

def build_two_sigma_quantization_table( number_entries, minimum, maximum, standard_deviation ):
    """
    Builds a quantization table, suitable for use with NumPy's digitize(), that emphasizes
    data within two sigma from the mean.  All data outside of the mean, plus or minus two sigma,
    are mapped to two entries, one for values below mean minus two sigma and one for values
    above mean plus two sigma, respectively.  This ensures that the majority of the table
    entries map onto the majority of the data values.

    The quantization table generated looks like so:

                   (number_entries - 1)                              1
            [                |                |                      ]
    -2*standard_deviation   mean      2*standard_deviation        maximum


    NOTE: This assumes the data are zero mean.

    Takes 4 arguments:

      number_entries     - Number of entries in quantization_table.
      minimum            - Minimum value of the data to quantize.
      maximum            - Maximum value of the data to quantize.
      standard_deviation - Standard deviation of the data to quantize.

    Returns 1 value:

      quantization_table - NumPy array, of type float32, containing the quantization table.

    """

    # attempt to handle non-normal data.  adjust the extrema so that +- one
    # standard deviation is within the data.
    #
    # NOTE: this is a kludge.
    #
    if (-2.0*standard_deviation < minimum):
        minimum = -2.1 * standard_deviation
    if (2.0*standard_deviation > maximum):
        maximum =  2.1 * standard_deviation

    quantization_table = np.empty( (number_entries,), dtype=np.float32 )

    # build the table so that it covers two standard deviations from the mean.
    # the final entry is set to the maximum so all of the data are covered by
    # the table when quantization is done such that x[i-1] < value <= x[i].
    quantization_table[:-1] = np.linspace( -2.0*standard_deviation,
                                            2.0*standard_deviation,
                                            number_entries - 1 )
    quantization_table[-1]  = maximum

    return quantization_table

def build_outliers_quantization_table( number_entries, minimum, maximum, standard_deviation ):
    """
    Builds a quantization table, suitable for use with NumPy's digitize(), that emphasize
    data more than one sigma from the mean.  All data within one sigma of the mean are
    mapped to a single entry in the quantization table ensuring that the majority of the
    table entries map onto the outliers in the data's distribution.

    The quantization table generated looks like so:

         (number_entries - 1)                        (number_entries - number_left_entries - 1)
   floor[--------------------]              1
                  2
       [                      |                           |                      ]
    minimum          -standard_deviation   mean   standard_deviation          maximum

    For tables with even number of entries, the levels are skewed towards the maximum with
    an extra entry.  The generated table is closed from above and open from below,
    which corresponds to digitize()'s right parameter being set to True.

    NOTE: This assumes the data are zero mean.

    Takes 4 arguments:

      number_entries     - Number of entries in quantization_table.
      minimum            - Minimum value of the data to quantize.
      maximum            - Maximum value of the data to quantize.
      standard_deviation - Standard deviation of the data to quantize.

    Returns 1 value:

      quantization_table - NumPy array, of type float32, containing the quantization table.

    """

    # compute the number of entries in each section, starting with the middle.
    number_middle_bins = 1
    number_left_bins   = (number_entries - number_middle_bins) // 2
    number_right_bins  = number_entries - (number_left_bins + number_middle_bins)

    # attempt to handle non-normal data.  adjust the extrema so that +- one
    # standard deviation is within the data.
    #
    # NOTE: this is a kludge.
    #
    if (-1.0*standard_deviation < minimum):
        minimum = -1.1 * standard_deviation
    if (standard_deviation > maximum):
        maximum =  1.1 * standard_deviation

    quantization_table = np.empty( (number_entries,), dtype=np.float32 )

    # build the table as three linearly (different) spaced regions.
    quantization_table[:number_left_bins] = np.linspace( minimum,
                                                         -1.0*standard_deviation,
                                                         number_left_bins,
                                                         endpoint=False )
    quantization_table[number_left_bins:(number_left_bins+number_middle_bins)] = np.linspace( -1.0*standard_deviation,
                                                                                              standard_deviation,
                                                                                              number_middle_bins,
                                                                                              endpoint=False )
    quantization_table[(number_left_bins+number_middle_bins):] = np.linspace( standard_deviation,
                                                                              maximum,
                                                                              number_right_bins,
                                                                              endpoint=True )


    return quantization_table
