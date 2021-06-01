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

    #
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
