# Overview

Command lines for a variety of use cases are provided below.  These are not
intended to be run end-to-end, but rather as needed for a given workflow.  See
the [workflows documentation](documentation/workflows.md) for details on supported
workflows.

**NOTE:** The command lines described below assume
that [setup](documentation/setup.md) has been completed and various environment
variables (e.g. `IWP_DATA_ROOT`) are present.

# Render XY Slices as Images

Create PNGs from the horizontal divergence grid variable (`divh`) using
a 2-sigma quantization table with per-slice normalization, and rendered
using Matplotlib's `bwr` colormap.  The first command fixes an XY slice
(`z_index=150`) and renders it for all available timesteps, while the second
fixes a timestep (`Nt=30`) and renders each of the XY slices available.

```shell
# creates ${IWP_IMAGES_ROOT}/R5F04-2sigma-local/divh/R5F04-divh-z=150-Nt=*.png.
$ iwp_create_labeling_data.py \
    -L \
    -q build_two_sigma_quantization_table \
    -c bwr \
    -z 150:150 \
    -v \
    "${IWP_DATA_ROOT}/timestep.*.nc" \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    R5F04 \
    divh

# creates ${IWP_IMAGES_ROOT}/R5F04-2sigma-local/divh/R5F04-divh-z=*-Nt=30.png.
$ iwp_create_labeling_data.py \
    -L \
    -q build_two_sigma_quantization_table \
    -c bwr \
    -t 30:30 \
    -v \
    "${IWP_DATA_ROOT}/timestep.*.nc" \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    R5F04 \
    divh
```

## Rendering XY Slices as Matplotlib Figures

Rendering uses PIL by default to map data to pixels with a minimum of
decorations.  Using Matplotlib figures allows labeling of axes and
data ranges for more detailed analysis.  Due to Matplotlib being a
big hammer, image rendering takes significantly longer (i.e. 10-15x).

**NOTE:** Do not use this for labeling data as the bounding box coordinates
will be incorrect!

```shell
# creates ${IWP_IMAGES_ROOT}/R5F04-2sigma-local-figures/divh/R5F04-divh-z=*-Nt=30.png.
$ iwp_create_labeling_data.py \
    -F \
    -L \
    -q build_two_sigma_quantization_table \
    -c bwr \
    -t 30:30 \
    -v \
    "${IWP_DATA_ROOT}/timestep.*.nc" \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local-figures \
    R5F04 \
    divh
```

## Rendering XY Slices in Parallel via Dask

Dask can be used to render images in parallel to greatly accelerate the
process.

Create PNGs for timesteps Nt=[10, 59] in parallel using one Dask process per
logical core on the system.  The Dask cluster is created, workers spun up,
and then shutdown after rendering is complete.

```shell
$ NUMBER_CORES=`grep MHz /proc/cpuinfo | wc -l`
$ iwp_create_labeling_data.py \
    -n ${NUMBER_CORES} \
    -L \
    -q build_two_sigma_quantization_table \
    -c bwr \
    -t 10:59 \
    -v \
    "${IWP_DATA_ROOT}/timestep.*.nc" \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    R5F04 \
    divh
```

Create PNGs for timesteps Nt=[10, 59] in parallel using an existing Dask
cluster at `dask-host:12345`.  Work is distributed across all available
Dask workers.

```shell
$ DASK_CLUSTER=dask-host:12345
$ iwp_create_labeling_data.py \
    -C ${DASK_CLUSTER} \
    -L \
    -q build_two_sigma_quantization_table \
    -c bwr \
    -t 10:59 \
    -v \
    "${IWP_DATA_ROOT}/timestep.*.nc" \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    R5F04 \
    divh
```

## Rendering XY Slices with Pre-computed Statistics

Pre-computed statistics can be used in lieu of locally-computed statistics
so that normalization produces comparable outputs across all XY slices.

Rendering via statistics generated offline via `iwp_compute_statistics.py`
and serialized to `${IWP_ARTIFACTS_ROOT}/statistics/R5F04-statistics.json`:

```shell
$ iwp_create_labeling_data.py \
    -S ${IWP_ARTIFACTS_ROOT}/statistics/R5F04-statistics.json \
    -q build_two_sigma_quantization_table \
    -c bwr \
    -t 30:50 \
    -v \
    "${IWP_DATA_ROOT}/timestep.*.nc" \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    R5F04 \
    divh
```

Rendering with statistics specified on the command line is possible
though requires specifying statistics for each variable rendered:

```shell
# normalize horizontal divergence with a range of [-1e-3, 1e-3] and
# a standard deviation of 3e-4.
$ iwp_create_labeling_data.py \
    -s -1e-3:1e-3:3e-4 \
    -q build_two_sigma_quantization_table \
    -c bwr \
    -t 30:50 \
    -v \
    "${IWP_DATA_ROOT}/timestep.*.nc" \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    R5F04 \
    divh
```

## Rendering XY Slices with Burned-in Labels

Rendering images with labels permanently overlaid (i.e. burned-in) can be
useful for analysis, review, or sharing with collaborators.  Labels can
be overlaid for both undecorated and figure-based images via the `-l` option.

```shell
# use bright green for labels.
$ LABEL_COLORSPEC=169:209:142

# NOTE: omit ",${LABEL_COLORSPEC}" from below to use the default
#       label color.
$ iwp_create_labeling_data.py \
    -L \
    -q build_two_sigma_quantization_table \
    -c bwr \
    -l ${IWP_LABELS_ROOT}/iwp-labels.json,${LABEL_COLORSPEC} \
    -t 30:30 \
    -v \
    "${IWP_DATA_ROOT}/plot_deflated.*.nc" \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    R5F04 \
    divh
```

# Labeling

## Starting the Scalabel.ai Container

Launch the Scalabel.ai labeling application using Docker:

```shell
$ ${IWP_REPO_ROOT}/scalabel/run-scalabel.sh \
    ${IWP_IMAGES_ROOT}/ \
    ${IWP_REPO_ROOT}/scalabel/
```

Alternative container run-times are supported via the `-l` option.
This launches the application using Podman:

```shell
$ ${IWP_REPO_ROOT}/scalabel/run-scalabel.sh \
    -l "podman run -it --rm" \
    ${IWP_IMAGES_ROOT}/ \
    ${IWP_REPO_ROOT}/scalabel/
```

### Alternative Ports

Multiple labeling applications can run on the same host (e.g. development,
per-user labeling, etc), though require using different ports.

Running Scalabel.ai on `localhost:8687`:

```shell
$ REPO_ROOT=/path/to/repo/working-copy
$ ${REPO_ROOT}/scalabel/run-scalabel.sh \
    -p localhost:8687 \
    ${IWP_IMAGES_ROOT} \
    ${REPO_ROOT}/scalabel
```

## Stopping the Scalabel.ai Container

Identify the container using the run-time and kill it:

```shell
# NOTE: change 'docker' to the appropriate container run-time.
$ docker ps | grep scalabel
e93c16a43230  docker.io/scalabel/www:latest  node app/dist/mai...  2 days ago  Up 2 days ago  127.0.0.1:8686->8686/tcp  great_moser

# NOTE: confirm this is the correct container on multi-user systems!
$ docker kill e93c16a43230
```

## Labeling in Scalabel.ai

### Create Labeling Playlist

Creating a playlist, named `${IWP_LABELS_ROOT}/scalabel-playlist.json` to label
timestep `Nt=30` using the pre-rendered images beneath
`${IWP_IMAGES_ROOT}/R5F04-2sigma-local/`.  XY slices are sequenced so that time
advances as they move up in the grid, allowing for labeling above the wake
center.  Assumes that the Scalabel.ai container runs at `http://localhost:8686/`
and maps `${IWP_IMAGES_ROOT}/R5F04-2sigma-local/` on the host to `/items/` in
the container.

```shell
$ scalabel_generate_playlist.py \
    -L xy_slices \
    ${IWP_LABELS_ROOT}/scalabel-playlist.json \
    R5F04 \
    divh \
    30:30 \
    1:300 \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    http://localhost:8686/items \
    3
```

#### Labeling Beneath the Wake

Scalabel.ai's frame interpolation works when time advances forward.  Labeling
features beneath the wake require time advancing backwards as said features
evolve downward (i.e. with decreasing `z_index`).

This creates a playlist where XY slices have a reversed order:

```shell
$ scalabel_generate_playlist.py \
    -L xy_slices \
    ${IWP_LABELS_ROOT}/scalabel-playlist.json \
    R5F04 \
    divh \
    30:30 \
    300:1:-1 \
    ${IWP_IMAGES_ROOT}/R5F04-2sigma-local \
    http://localhost:8686/items \
    3
```

### Creating a new Project

XXX: copy the playlist to the client

XXX: upload the playlist

### Exporting Labels

Assuming Scalabel.ai is running at `http://localhost:8686`, visit the project's
dashboard at `http://localhost:8686/dashboard?project_name=<project>` and click
"Download Labels".

Scalabel.ai's labels use pixel coordinates and need to be scaled by the rendered
XY slice size.  Assuming labeling was done on images that were 256x384
resolution, the following extracts labels from the export and convert them to
IWP format:

```shell
$ scalabel_extract_iwp_labels.py \
    ${IWP_LABELS_ROOT}/project-name-2022-01-24_14-36-48.json \
    ${IWP_LABELS_ROOT}/iwp-labels-2020-01-24_14-36-48.json \
    256 \
    384
```

# Reviewing

Data review is complex, as each tool has a myriad of ways to visualize and
present information present in the labeled flow data (which is 4D per grid
variable).

Below are a handful of common use cases for data review.

## PowerPoint Exports

Creating PowerPoint slides for three XY slices for timestep `Nt=30` at
`z_index={175, 180, 185}`.  Both horizontal divergence and wavelet modulus are
rendered using the Matplotlib `bwr` and `inferno` colormaps, respectively.
Labels are overlaid with default colors.

```shell
$ iwp_export_pptx.py \
      -c bwr,inferno \
      -l ${IWP_LABELS_ROOT}/iwp-labels.json \
      -q build_two_sigma_quantization_table,build_linear_quantization_table \
      ${IWP_DATA_ROOT}/R5F04/timestep.{:06}.nc \
      ${IWP_ARTIFACTS_ROOT}/R5F04-Nt=030-divh_morlet+-50.pptx \
      R5F04 \
      divh,morlet+-50 \
      30,175 \
      30,180 \
      30,185
```

## Movies

Animations comprised of sequences of XY slices can be created via `ffmpeg`.

Since parameterizing `ffmpeg` is not for the faint of heart, a wrapper
script `utilities/create-dataset-animation.sh` was written to hide its
complexity.

```shell
$ create-dataset-animation.sh \
    ${IWP_ARTIFACTS_ROOT}/R5F04-divh-2sigma-Nt=030.mp4 \
    ${IWP_IMAGES_ROOT}/R5F04-2sigmal-local \
    R5F04 \
    divh \
    1:322 \
    30
```

This wrapper does not generate XY slice renderings and uses existing data.  See
the [image rendering use cases](#render-xy-slices-as-images) for creating the
source images needed for an animation.

### Specify Path to `ffmpeg`

Not all instances of `ffmpeg` are equal and specifying a full path to the
version of interest is supported via the `-F` option:

```shell
# use the system's ffmpeg instead of Conda's.
$ create-dataset-animation.sh \
    -F /usr/bin/ffmpeg \
    ${IWP_ARTIFACTS_ROOT}/R5F04-divh-2sigma-Nt=030.mp4 \
    ${IWP_IMAGES_ROOT}/R5F04-2sigmal-local \
    R5F04 \
    divh \
    1:322 \
    30
```
