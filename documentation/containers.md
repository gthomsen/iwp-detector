# Scalabel.ai Labeling Application

Out of the box, the labeling container addresses all IWP labeling needs.
Aside for security vulnerability updates or the introduction of backwards
label interpolation, there is little reason to upgrade the container once
it is pulled to the local system.

Once the [container is pulled](setup.md#scalabelai-container-image),
[starting](use-cases-reference.md#starting-the-scalabelai-container)
and [stopping](use-cases-reference.md#stopping-the-scalabelai-container)
the container are the only interactions required, and these are infrequent
operations at best.

## References

Useful references when working with Scalabel.ai:
- [Scalabel.ai Home](https://www.scalabel.ai/)
- [Dockerhub page](https://hub.docker.com/r/scalabel/www)
- [Github](https://github.com/scalabel/scalabel)

# Custom IWP Containers

Due to the complexity of setting up dependencies, we chose to build containers
for development to ensure consistency amongst collaborators.

Currently these only provide a base Python environment via Conda (the
`iwp-development` image), though will ultimately include all of the dependencies
and tools required by the code base.

Podman is the default container run-time for building containers so as to
minimize privileges required.  Other run-times, such as Docker, are supported so
long as they provide the same command line interface as the `docker` binary.

## Running Development Containers

**NOTE:** This workflow is not fully fleshed out and has sharp edges.

It is intended that all workflows will eventually be supported in containerized
development environments (provided the host can provide the appropriate
hardware, that is).

Assuming a configured environment, running the `iwp-development` image can be
with the following:

```shell
$ podman run --rm -it \
    -v ${IWP_DATA_ROOT}:/data/flows \
    -v ${IWP_LABELS_ROOT}:/data/labels \
    -v ${IWP_ARTIFACTS_ROOT}:/data/artifacts \
    -v ${IWP_IMAGES_ROOT}:/data/images \
    -v ${IWP_REPO_ROOT}:/home/user/iwp
    iwp-development:latest
```

Things to note:
- Drop any volume mounts that are needed or don't exist on the host system.
- Accessing network services inside the container on the host can be done by
  adding `-p <HOST_PORT>:<CONTAINER_PORT>` to the command line above.
- There is no persistence between container execution outside of changes made
  to the volumes mapped into the container.  Be mindful of changes to the user
  environment inside the container as it will be lost when it is shutdown.

### Containers on MacOS with Apple Silicon

Running on Apple Silicon hardware is partially supported for non-GPU-based
components of the workflows.  This requires Docker Desktop to be installed.

For non-networked applications, no changes are needed to launch and interact
with the container.  Simply follow
the [instructions above](#running-development-containers).

Additional work is required when using network applications such as Jupyter,
which is roughly the following:

1. Launch the container with the appropriate ports exposed.  This is done by
   adding `-p <HOST_PORT>:<CONTAINER_PORT>` to the launch command.
2. Configure processes within the container to listen on the container's
   VM network interface instead of `localhost`.

The work above stems from the fact that the `iwp-development` image is `x86-64`
rather than ARM-based resulting in the container being run as a VM rather than
as a lightweight container.  This results in the following headaches:

- Jupyter does not bind to all devices (i.e. `0.0.0.0`) by default, and
  explicitly requesting to bind to `0.0.0.0` somehow does not map to all of the
  local devices.
- MacOS does not map ports listening on the container's `localhost`/`127.0.0.1`
  address, only to the VM's local interface

To work around this, identify the interfaces available and start Jupyter with
the VM's interface (typically using the network `172.17.0.x`):

```shell
# NOTE: use 'ifconfig | grep inet' instead of 'awk' when it is installed.
$ awk '/32 host/ { print f } {f=$2}' <<< "$(</proc/net/fib_trie)" | grep -v 127.0.0.1 | sort -u
172.17.0.2
$ jupyter notebook --ip 172.17.0.2
```

## Building Containers

To build the `iwp-development` image:

```shell
$ cd ${IWP_REPO_ROOT}/docker
$ make iwp-development
Copying the Python environment for iwp-development.
Copying the certificate authority bundle for iwp-development.
STEP 1/25: FROM ubuntu:hirsute
...
STEP 25/25: ENTRYPOINT ["/bin/bash", "-i"]
COMMIT iwp-development:latest
--> 75b7b297686
Successfully tagged localhost/iwp-development:latest
75b7b297686113a9e7f0431873bd09578a1d763fbe53b92a8c9360f0637dcd2a
```

**NOTE:** Due to the dependency size, the container images generated are large.
The base image with Anaconda, CUDA, and PyTorch weighs in somewhere north of
5GB.

Configuration options to the container image build system can be found by
building the `help` target:

```shell
$ cd ${IWP_REPO_ROOT}/docker
$ make help
```

### Debugging the Container Build Process

Debugging the build can be done by adding the `DEBUG=yes` environment variable
to the command line.  This prints out the commands run prior to execution,
which also provides a dry run trace of commands that would be executed when
combined with Make's `-n` option:

```shell
$ cd ${IWP_REPO_ROOT}/docker
$ make -n iwp-development DEBUG=yes
podman build  -t iwp-development:latest iwp-development/
```

### Building on MacOS with Apple Silicon

Building images on MacOS is supported, even with Apple Silicon.  Assuming Docker
Desktop has been installed, only one additional step is required to export the
system's certificate authority chain from the System Key Ring:

```shell
$ security \
    find-certificate \
    -a \
    -p /System/Library/Keychains/SystemRootCertificates.keychain \
    > host-certificate-chain.crt
```

With `host-certificate-chain.crt` the build command line then becomes:

```shell
$ cd ${IWP_REPO_ROOT}/docker
$ make \
    iwp-development \
    HOST_CA_BUNDLE=host-certificate-chain.crt \
    CONTAINER_RUNTIME=docker
```

**NOTE:** Building on Apple Silicon will take a significant amount of time as an
`x86-64` environment must be emulated rather than executed in a sandbox like a
native ARM build would.

**NOTE:** [Caching via a MITM proxy](#building-with-a-caching-mitm-proxy) is not
supported on Apple Silicon.


### Building with a Caching MITM Proxy

It is possible to setup Squid as a man-in-the-middle caching proxy to speed up
container builds...

XXX: create a certificate chain, setup Squid, specify HOST_CA_BUNDLE and
     PROXY_URL during build.

## Container Design Philosophy

Best practices for container design should be followed (e.g. move most rapidly
evolving layers at the end for caching, don't unnecessarily bloat layers with
artifacts, etc).

Images are built with a layered design so that successively larger development
environments are available allowing users to pick the smallest environment
needed for their workflows.  This allows fully featured ML and
OpenGL-accelerated ParaView containers to exist for those who need them, while
sitting on top of the same base Python container suitable for data processing
and labeling.
