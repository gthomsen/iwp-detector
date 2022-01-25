# Internal Wave Packet Detection Tools

Collection of tools to identify and analyze turbulence-based internal wave
packets (IWP) in simulated flow data.  These are intended to be the foundation for
several research workflows that couple existing flow solvers to new analysis
techniques, automated by machine learning.

Currently the focus is on foundational components (data loading, coordinate
systems, labeling, and basic analysis) and does not include specialized
tools related to flows, turbulence, or internal waves analysis.  See
the [workflows documentation](documentation/workflows.md) for details on what is
supported.

# Quick Start

Installing the Python `iwp` module is a dependency for all workflows.  Conda is
strongly recommended for setting up the required dependencies as they span
multiple non-Python tools:

```shell
# change the environment name 'iwp' as needed.
$ conda env create --file python/environment.yml --prefix ${CONDA_ENV}/envs/iwp
```

Install the `iwp` package:

``` shell
$ source activate iwp
$ cd python
$ pip install iwp
```

See the [setup documentation](documentation/setup.md) for setting up additional
workflows.

# Setup

Full setup of a development environment that supports all of
the [workflows](documentation/workflows.md) (data exploration, labeling, and
analysis) is covered by the [setup instructions](documentation/setup.md).  See
below for an overview of the dependencies required.

## Software Requirements

The following packages are required:

- Python 3
- CUDA
- PyTorch
- Jupyter
- netCDF4
- Dask
- xarray
- python-pptx

Optional packages depending on the workflows of interest:

- Labeling:
    - Podman or Docker
    - Scalabel.ai
- Data exploration and analysis:
    - netCDF Operators (NCO)
    - ParaView
    - ParaView kernel for Jupyter
- Visualization and data review:
    - ffmpeg
- Development in containers
    - Podman or Docker

## Data

This repository does not contain a flow solver, nor provides any data suitable
for use - you must provide your own.  See
the [data organization documentation](documentation/data-organization.md) for
details on the organizational structure expected.

# Documentation

High-level documentation on various [workflows](documentation/workflows.md) is
complemented with command line references, organized
by [use cases](documentation/use-cases-reference.md).

Additionally, high-level documentation on [labeling](documentation/labeling.md)
and [containers](documentation/containers.md) also is available.
