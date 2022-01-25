# Development Environment

It is assumed that the code base is installed within a Linux environment that
has a CUDA-compatible GPU.  Non-ML workflows may run in MacOS or Windows
environments, though these have not been tested and are unsupported.  It is
recommended to use a [development container](#development-container-images) on
non-Linux systems to provide the base environment should that setup be
necessary.

# Environment Variables

Different workflows touch data with different characteristics and life cycles so
it makes sense to keep them in different locations on the system.  Though
nothing in the code base requires them, using environment variables to refer to
the top-level directories for each makes life significantly easier.

The following table provides a summary of the environment variables referenced
throughout the documentation, along with their access patterns, typical size,
and various notes.

| Environment Variable | Purpose | Access | Size | Notes |
| --- | --- | --- | --- | --- |
| `IWP_DATA_ROOT` | Flow datasets | Mostly read | Very large (TBs) | Simulation outputs as a collection of netCDF files.  See the [flow dataset documentation](data-organization.md#flow-datasets) for details. |
| `IWP_LABELS_ROOT` | IWP labels and Scalabel.ai playlists | Read/write | Small (MBs) | IWP labels, in various forms, serialized to JSON.  See the [IWP labels documentation](data-organization.md#iwp-labels) and [Scalabel playlists documentation](data-organization.dm#scalabel-playlists) for details. |
| `IWP_IMAGES_ROOT` | Flow data slices rendered as images | Read/write | Moderate (few GBs) | Subset of flow data rendered into images for use in external tools.  See the [rendered images documentation](data-organization.md#rendered-images) for details. |
| `IWP_ARTIFACTS_ROOT` | Generated artifacts during workflows | Read/write | Small to Moderate (MBs - GBs) | Collection of workflow outputs, including animations, grid variable statistics, etc. |
| `IWP_REPO_ROOT` | Code base | Read or read/write | Small (MBs) | Checkout of the IWP Detectors repository.  Read/write when actively developing code rather than labeling or analyzing data. |

The following can be copied and modified for use in environment configuration
and shell scripts:

```shell
export IWP_DATA_ROOT=/path/to/data/netcdf
export IWP_LABELS_ROOT=/path/to/labels
export IWP_IMAGES_ROOT=/path/to/data/images
export IWP_ARTIFACTS_ROOT=/path/to/artifacts
export IWP_REPO_ROOT=/path/to/
```

# Python Environment

Unless one is using [development containers](#development-container-images) for
development, a Python environment needs to be installed.  If a development
container is used, this section can be skipped.

 to the complexity of certain packages (e.g. CUDA and PyTorch) Python
dependencies are managed by Conda .  Install a version
of [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) that
provides a recent version of Python (Python 3.8 and above have been tested).
Either Anaconda or Miniconda can provide the base Python installation, choosing
one over the other only changes the initial on-disk footprint and how many
packages must be downloaded during creation of the `iwp` environment.

Once Conda is installed, create the `iwp` environment (change `iwp` if using a
shared installation with multiple developers):

```shell
# change the environment name 'iwp' as needed.
$ conda env create --file python/environment.yml --prefix ${CONDA_ENV}/envs/iwp
```

If IWP detection workflows will be a regular activity, one can
configure their shell environment to always use the `iwp` Conda environment by
adding the following into their startup files:

```shell
# change the environment name 'iwp' as needed.
$ source activate iwp
```

# `iwp` Package

Installing IWP detection Python package is as simple as:

```shell
$ cd ${IWP_REPO_ROOT}/python
$ pip install iwp
```

# Preparing Datasets

Flow datasets need to be prepared before use within the data exploration,
labeling, and analysis workflows.  Since flow simulations are often **VERY**
expensive to generate (upwards of millions of compute hours), preparation
involves making a copy of the simulation outputs so as to avoid inadvertent
modification (or, worse, removal).  At a high level, the following steps are
required for preparation:

1. Create a directory for each [flow dataset](data-organization.md#flow-datasets)
2. Copy and re-chunk [netCDF files](data-organization.md#netcdf-file-structure) of interest
3. Post-process each file as necessary

## Dataset Directory Creation

The IWP tools do not impose a naming convention on datasets so users can use
names (`<dataset>`) that make organizational sense.  Create one directory
per dataset:

```shell
$ mkdir ${IWP_DATA_ROOT}/<dataset>
```

## Copy Dataset netCDF Files

Not all of a flow dataset is necessary for IWP detection, so grid variables of
interest, from the time range of interest, should be copied from the source of
truth.  The `utilities/prepare-iwp-dataset.sh` script simplifies that process.

First, identify the files with the timesteps of interest.  The following command
uses all files in the source of truth dataset and assigns them to a temporary
variable `ORIGINAL_FILES_LIST`.  Adjust or replace the `find` command as
necessary.

```shell
$ ORIGINAL_FILES_LIST=`find /path/to/source/of/truth/dataset -name 'data.*.nc' | xargs`
```

Second, identify the variables of interest for IWP detection.  A subset of
variables may be copied from the source of truth to (greatly) reduce the
on-disk footprint.  The example below only copies the horizontal divergence
(`divh`) and pressure (`p`) variables.

How data are stored within individual netCDFs, the chunk size, can drastically
impact I/O rates and optimal I/O patterns for creating simulation outputs likely
do not match optimal patterns for data exploration and analysis.  Changing the
chunk size to horizontal grid slices during the copy can be done via the `-c`
option.

Depending on the workflows of interest, it may be useful to scale individual
netCDF file's dimensions to transform the data
into
[dimensionless coordinates](data-organization.md#scaled-vs-unscaled-coordinates)
that are more useful for analysis workflows.  Note that scaling will likely go
away and be properly absorbed into the analysis code that depends
on them.  Set the `SCALE_FACTOR` below as appropriate.

**NOTE:** Preparation can take a while depending on how large the flow dataset
is.

```shell
$ SCALE_FACTOR=1.0
$ ${IWP_REPO_ROOT}/utilities/prepare-iwp-dataset.sh \
    -c \
    -D ${SCALE_FACTOR} \
    -V divh,p \
    ${IWP_DATA_ROOT}/<dataset> \
    ${ORIGINAL_FILES_LIST}
```

## Post-process netCDF Data

Once a copy of the data has been made, post-processing can be applied to the
local dataset.  Existing grid variables can be modified and new grid variables
can be added to the dataset without affecting the source of truth dataset.

The following applies a 2D Morlet continuous wavelet transform (CWT) to the
horizontal divergence grid variable (`divh`).  See `iwp_postprocess_netcdf.py`'s
help for details on the transform parameters.

```shell
$ NUMBER_CORES=`grep MHz /proc/cpuinfo | wc -l`
$ POSTPROC_FILES_LIST=`find ${IWP_DATA_ROOT}/<dataset> -name '*.nc' | xargs | tr ' ' ','`
$ iwp_postprocess_netcdf.py \
    -n ${NUMBER_CORES} \
    divh \
    ${POSTPROC_FILES_LIST} \
    divh_morlet+-50:symmetric_morlet_max_modulus:alpha=55:scales=2,4,8,16,32
```

# Scalabel.ai Container Image

Pulling Scalabel.ai's labeling application container to the local system's
registry is required for labeling workflows.  Use the desired container
run-time to make a copy of the container image to the local registry:

```shell
# NOTE: change 'podman' to the appropriate container run-time.
$ podman pull docker.io/scalabel/www
```

# Paraview

ParaView is used to visualize flow datasets that have their contents described
via XDMF.  While manual visualization is useful, ParaView's utility really shines
when it is driven by Python code, either command line scripts (e.g. to
programatically generate animations) or via kernel backends for Jupyter Notebook
access.

XXX: Finish me

# Development Container Images

Container images can be used for consistent development environments across
developers and researchers.  See
the [containers documentation](containers.md#custom-iwp-containers) for a detailed
overview of how to build, deploy, and use containers for IWP detection.
