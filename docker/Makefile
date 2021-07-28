# build with Podman by default.
#
# NOTE: in theory, any runtime should be able to build these images though
#       only Docker and Podman have been tested.
#
CONTAINER_RUNTIME = podman

# suffix to use when tagging image names.
TAG_SUFFIX = latest

# build without a proxy unless we're given one.
PROXY_URL =

# warn users that proxies with Docker are unnecessarily difficult.
ifneq ($(strip $(PROXY_URL)),)
ifeq ($(CONTAINER_RUNTIME),docker)
$(warning "PROXY_URL is not supported with Docker.  You must update the appropriate config.json to specify one.  This build *MAY* fail.")
endif
endif

# locations of certificate authority bundles by system.  only one of these
# will exist on any given system, which CA_BUNDLE is set to below.
POTENTIAL_CA_BUNDLES = /etc/ssl/certs/ca-certificates.crt \
                       /etc/ssl/certs/ca-bundle.crt

# make sure we have a CA bundle.  pick the first bundle that exists.  there
# should only be one...
#
# NOTE: the build is viable without a bundle, though limitations of Docker
#       prevent conditional execution when it isn't provided.  Podman, however,
#       is fine.
#
HOST_CA_BUNDLE = $(word 1, $(realpath $(POTENTIAL_CA_BUNDLES)))
ifeq ($(strip $(HOST_CA_BUNDLE)),)
$(error "Could not find a certificate authority bundle in '$(POTENTIAL_CA_BUNDLES)'!")
endif

# name of the staged CA bundle used inside of the build.  this must match the
# contents of the Dockerfiles that use it.
STAGED_CA_BUNDLE = ca-certificates.crt

# name of the Conda environment YAML.  this must match the contents of the
# Dockerfiles that use it.
STAGED_CONDA_ENVIRONMENT = environment.yml

# list of artifacts that must be staged to build an image.
STAGED_ARTIFACTS = $(STAGED_CONDA_ENVIRONMENT) \
                   $(STAGED_CA_BUNDLE)

# be quiet unless we're debugging.  executed commands are not echoed to
# output by default.
ifeq ($(DEBUG),yes)
QUIET =
else
QUIET = @
endif

# show the user help if they don't provide a build target.
help:
	@echo "Internal Wave Packet (IWP) Detector container image build system.  The following"
	@echo "images are available to build:"
	@echo
	@echo "   iwp-development:   Full development stack"
	@echo
	@echo "Built images are tagged and stored in the local registry."
	@echo
	@echo "The build process is influenced by the following variables that can be specified on"
	@echo "the command line."
	@echo
	@echo "   CONTAINER_RUNTIME: Container runtime to build the image with.  Docker (docker)"
	@echo "                      and Podman (podman) are supported."
	@echo "   DEBUG:             Enables debugging during the build.  Executed commands"
	@echo "                      are echoed to standard output prior to execution."
	@echo "   HOST_CA_BUNDLE:    Path to the certificate authority bundle to use during the"
	@echo "                      build, specified as a path on the host."
	@echo "   PROXY_URL:         URL to a proxy to build the image with.  This proxy is"
	@echo "                      not embedded into the image and does not influence users"
	@echo "                      running the image after creation."
	@echo
	@echo "                        NOTE: This cannot be used with Docker.  Proxies must be"
	@echo "                              set via the appropriate config.json."
	@echo
	@echo "   TAG_SUFFIX:        Suffix to build the image tag with.  The build creates"
	@echo "                      tags of the form <image>:<tag>."
	@echo
	@echo "The above variables are currently set like so:"
	@echo
	@echo "   CONTAINER_RUNTIME=$(CONTAINER_RUNTIME)"
	@echo "   DEBUG=$(DEBUG)"
	@echo "   HOST_CA_BUNDLE=$(HOST_CA_BUNDLE)"
	@echo "   PROXY_URL=$(PROXY_URL)"
	@echo "   TAG_SUFFIX=$(TAG_SUFFIX)"
	@echo

# build each image when everything is requested.
all: $(DEFAULT_BUILD_LIST)

clean:
	@echo "Cleaning the build tree."
	$(QUIET) rm -f $(patsubst %, iwp-development/%, $(STAGED_ARTIFACTS))

# each image lives in a directory with the same name.  ensure that we build
# the image regardless of directories' last modified time.
.PHONY: iwp-development

# create the development image from the contents of its eponymous directory.
iwp-development: $(patsubst %, iwp-development/%, $(STAGED_ARTIFACTS))
	$(QUIET) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) $(CONTAINER_RUNTIME) build -t $@:$(TAG_SUFFIX) $@/

# copy the Conda environment YAML into a location visible to the container image
# build.
iwp-%/$(STAGED_CONDA_ENVIRONMENT): ../python/$(STAGED_CONDA_ENVIRONMENT)
	@echo "Copying the Python environment for `dirname $@`."
	$(QUIET) cp -f $^ $@

# copy the certificate authority bundle into a location visible to the container
# image build.
iwp-%/$(STAGED_CA_BUNDLE): $(HOST_CA_BUNDLE)
	@echo "Copying the certificate authority bundle for `dirname $@`."
	$(QUIET) cp -f $^ $@
