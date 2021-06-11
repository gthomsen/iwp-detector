import json

# XXX: create a class to wrap statistics.  methods/attributes for each of the
#      the descriptive variables.  method to load and save stats?

def compute_statistics( dataset ):
    """
    Computes summary statistics of a dataset.  Generates the minimum, maximum, and standard
    deviation of the entire dataset provided.

    Takes 1 arguments:

      dataset - xarray Dataset or DataArray to compute statistics over.

    Returns 3 values:

      minimum - Minimum value of dataset.
      maximum - Maximum value of dataset.
      stddev  - Standard deviation of dataset.

    """

    # compute the minimum, maximum, and standard deviation.
    #
    # NOTE: this reduces over all grid variables as well as time steps.  caller
    #       is responsible for providing the correct dataset.
    #
    task_min = dataset.min()
    task_max = dataset.max()
    task_std = dataset.std()

    (variable_min,
     variable_max,
     variable_std) = (task_min.compute().values,
                      task_max.compute().values,
                      task_std.compute().values)

    return (variable_min, variable_max, variable_std)

def save_statistics( statistics_path, statistics, pretty_flag=True ):
    """
    Serializes a statistics dictionary to disk as JSON.

    Takes 3 arguments:

      statistics_path - Path to save the statistics to.
      statistics      - Dictionary of statistics to save.  Keys are variable
                        names and the values are dictionaries containing
                        the underlying statistics.
      pretty_flag     - Optional flag specifying whether the serialized JSON is
                        pretty-printed or not.  If omitted, defaults to True.

    Returns nothing.

    """

    with open( statistics_path, "w" ) as statistics_fp:
        #
        # NOTE: we sort the dictionary keys so that it is easier to compare
        #       different labels without custom tools.
        #
        json.dump( statistics, statistics_fp, indent=4, sort_keys=True )

    return

def load_statistics( statistics_path ):
    """
    Loads statistics from an on-disk JSON file.

    Takes 1 argument:

      statistics_path - Path to load statistics from.

    Returns 1 value:

      statistics - Dictionary of the statistics loaded.  Keys are variable names
                   and the values are dictionaries containing the underlying
                   statistics.

    """

    with open( statistics_path, "r" ) as statistics_fp:
        statistics = json.load( statistics_fp )

    return statistics
