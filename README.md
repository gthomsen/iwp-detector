# Internal Wave Packet Detection Tools

# Requirements

The following packages are required:

* Python 3
* CUDA
* PyTorch
* NetCDF4
* Dask
* xarray
* python-pptx

The following versions have been tested and are explicitly set in the
`requirements.txt` file referenced below:

* Python 3.8
* CUDA 11.1
* PyTorch 1.8.1
* NetCDF4 1.5.3

# Installation

Create a new Conda environment:

```shell
# change the environment name 'iwp' as needed.
$ conda env create --file python/environment.yml --prefix ${CONDA_ENV}/envs/iwp
```
