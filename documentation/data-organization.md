# Overview

There are four main types of data handled by the IWP tools:

1. Flow data in netCDF files
2. 2D slices of flow data rendered as images
3. IWP labels in JSON
4. Scalabel.ai playlists in JSON

Of the above list, only the flow data needs to be provided before using the IWP
tools.  The remaining data are generated during the course of data exploration,
labeling, and analysis.

The following environment variables are used refer to common paths throughout
the documentation:

- `IWP_DATA_ROOT`: Top-level directory containing flow datasets
- `IWP_IMAGES_ROOT`: Top-level directory containing flow data rendered as 2D
  slices
- `IWP_LABELS_ROOT`: Top-level directory containing IWP labels and Scalabel.ai
  playlists

See the [setup documentation](setup.md#environment-variables) for additional
details.

# Flow Datasets

Flow datasets are comprised of one or more [netCDF4 files](#netcdf-file-structure),
each containing data for a single timestep.

Datasets should be organized beneath a common location, `IWP_DATA_ROOT`, with
one dataset per subdirectory.  Each dataset's subdirectory should hold all of
the timesteps so they can be referred to via a file glob (e.g. `timestep.*.nc`).

Example directory structure with two data sets, each with different naming
conventions and number of timesteps.

```
${IWP_DATA_ROOT}/R5F04/timestep.001.nc
                       timestep.002.nc
                       ...
                       timestep.009.nc
                 R5F04-test/data.00001.nc
                            data.00002.nc
                            ...
                            data.00230.nc
```

## netCDF File Structure

3D field outputs are stored one timestep per file in netCDF
files.  Each timestep must have the same dimensionality and grid variables of
interest.

The following global attributes are required:

| Attribute Name | Data Type | Description |
| --- | --- | --- |
| `Cycle` | Integer | Sequence number of the timestep |
| `Nt` | Floating point | Buoyancy time for the timestep |

The following dimensions are required:

| Dimension Name | Description |
| --- | --- |
| `x` | Streamwise dimension |
| `y` | Transverse dimension |
| `z` | Vertical dimension |

The 3D grid defined by `x`, `y`, and `z` uses a right-handed Cartesian
coordinate system.  While not mandated in any of the data loading and
serialization code, the analysis tools assume a wake centered along the midpoint
of the Y dimension, running the length of the X dimension.  The wake center
relative to the bottom of the domain is simulation-dependent.

The following variables are required:

| Variable Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| `x` | `x` | Floating point | X coordinates (scaled or unscaled) |
| `y` | `z` | Floating point | Y coordinates (scaled or unscaled) |
| `z` | `z` | Floating point | Y coordinates (scaled or unscaled) |
| Varies | (`x`, `y`, `z`) | Floating point | Grid variable |

Multiple grid variables are supported, though the code does not expect any
particular variable name to be present.

### Scaled vs Unscaled Coordinates

Simulation outputs have unscaled, physical units (e.g. domain dimensions in
meters), while analysis outputs often work in scaled, dimensionless units (e.g.
domain dimensions scaled by a reference size).  Currently the code does not
scale on the fly and assumes scaled units, though the dataset preparation
scripts do not provide a flag indicating what the data represent.

# Rendered Images

To fit into Scalabel.ai's video labeling model, 3D flow data are exported to
disk as sequences of 2D slices rendered as images.  Renderings take a raw grid
variable, normalize it (either by dataset or local statistics), quantize it, and
then apply a colormap to produce an image that highlights internal wave packet
features.

Since 3D flow simulations may be large (a 3D domain sized 256x384x320 results in
120 MiB per variable per timestep) rendered slices are generated on demand
rather than offline prior to working with the data set.  Also, due to the
complex dynamics inherent in simulated flow data multiple renderings may
be required by the workflow (e.g. different normalization and quantization
schemes may be useful during labeling).

These workflow characteristics result in irregular subsets of the 3D flow data
residing on disk with the following organization structure:

```shell
${IWP_IMAGES_ROOT}/<experiment>/<variable>/
```

`<experiment>` is free-form text, though should be descriptive given that there
will likely be many of them for a given dataset.  It is recommended using a
structure like `<dataset>-<quantization_schema>-<statistics_type>` or
`<dataset>-<statistics_type>-<colormap>` to easily organize and identify
different renderings.  `<variable>` are variable names from
a [flow dataset's variables](#netcdf-file-structure).

Currently only XY slices are rendered and stored in
`${IWP_IMAGES_ROOT}/<experiment>/<variable>/`, and their file names have the
form:

```shell
<dataset>-<variable>-z=<z_index>-Nt=<timestep>.png
```

Both `<z_index>` and `<timestep>` are three digit, zero padded indices
specifying the source XY slice in the flow dataset.  While `<dataset>` is
not explicitly connected to the flow datasets it is recommended to use the same
naming convention to avoid confusion.

# IWP Labels

Labels are currently simple 2D, horizontal bounding boxes associated with a
particular XY slice in the dataset.  IWP labels are considered the source of
truth throughout the code base as they are exported to other formats
(e.g. [Scalabel.ai](#scalabel-playlists)) and loaded into third-party tools
(e.g. ParaView) as needed.

Collections of labels are stored on disk as files.  There is no overarching
organization between labels in different files, and all tools (as of January
2022) expect all labels to be present in a single file.  Tools exist to
merge files (i.e. `iwp_merge_labels.py`) that allows manual organization of
labels.

The code base does not assume a particular workflow and one must be established
to avoid confusion and errors when moving between tools.  It is recommended that
all labels be stored in a central location, e.g. beneath `IWP_LABELS_ROOT`.

## Serialized Format

Serialized labels are stored as JSON with the following high level structure:

```
- labels [ ]:
    - label001
    - label002
    ...
```

where each label in `labels` has the format:

```
- bbox:
    - x1: float
    - y1: float
    - x2: float
    - y2: float
- category: string
- id: string
- time_step_index: int
- z_index: int
```

Bounding boxes are stored in normalized coordinates (with a range of [0, 1]) so
they're independent of the original grid resolution and rendered image used to
label with.  Boxes are stored as two coordinates, (`x1`, `y1`) and (`x2`, `y2`),
that define the upper left and lower right corners. The labeling origin is in
the bottom left (IJ coordinates) rather than the top left (XY coordinates,
typical for images) to match the [flow data's](#netcdf-file-structure)
coordinate systems.

Label categories are strings, though the code currently only supports the `"iwp"`
category (as of January 2022).

Label identifiers are unique within the flow dataset so that labels on different
XY slices can be associated into 3D volumes.

The timestep and Z indices (`time_step_index` and `z_index`, respectively)
specify a 2D slice within the associated flow data that the label corresponds
to.  `time_step_index` corresponds to a netCDF file's `Cycle` attribute.

# Scalabel Playlists

Scalabel labels are used when interacting with Scalabel.ai, and are referred to
as playlists throughout the code base.  Playlists adhere to
Scalabel.ai's [exporting format](https://doc.scalabel.ai/format.html).

Playlists are short-lived and only have utility when creating or reviewing
labels.  Though there is no requirement on where they are stored, organizing
them next to the [source of truth IWP labels](#iwp-labels) (beneath
`IWP_LABELS_ROOT`) is recommended.
