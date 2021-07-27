# build against Ubuntu 21.04 LTS so we have a recent, stable foundation.
FROM ubuntu:hirsute

# create a generic user.  we don't expect to run in a privileged container
# so we create a separate, non-root entity to make that assumption explicit.
ARG USERNAME=user
ARG UID=1000
ARG GID=1000

# specify the installation and build locations.
ARG INSTALL_ROOT=/opt
ARG BUILD_ROOT=/tmp/build

# name of the Conda environment we create.
ARG IWP_CONDA_ENVIRONMENT=iwp

# certificate authority (CA) certificates to use when creating the image.
# overriding this is useful when internet access is behind a man-in-the-middle
# (MITM) proxy as typically found in caching or whitelist/blacklist setups.
#
# NOTE: we do not specify the sytem's certificate store (e.g. /etc/ssl/certs)
#       as any system update would wipe this out.
#
ARG CA_BUNDLE=${BUILD_ROOT}/ca-certificates.crt

# configure curl to use our CA certificates.
#
# NOTE: use the system's curl instead of looking it up in PATH where we may
#       inadvertently pick up Conda's curl instead.
#
ARG CURL_CA_BUNDLE=${CA_BUNDLE}
ARG CURL_CMD="/usr/bin/curl --cacert ${CURL_CA_BUNDLE}"

# location of our Conda install.  we need this to reference binaries beneath it.
ARG MINICONDA_ROOT=${INSTALL_ROOT}/miniconda3

# work out of our staging directory.
WORKDIR ${BUILD_ROOT}

# copy the CA chain into the working directory.
#
# NOTE: this becomes a permanent fixture in the final image.
#
COPY ca-certificates.crt ${CA_BUNDLE}

# add Miniconda's Python executable to every user's PATH.  users' environments
# will start inside of the Conda environment we create below.
RUN \
    echo "export PATH=${MINICONDA_ROOT}/bin:\${PATH}" >> /etc/skel/.bashrc && \
    echo "source activate ${IWP_CONDA_ENVIRONMENT}" >> /etc/skel/.bashrc

# add the user with appropriate UID/GID.  we create a user and then set their
# primary group in two steps in case the GID specified is created along with the
# user.
RUN \
    useradd -m -u ${UID} ${USERNAME} && \
    usermod -g ${GID} ${USERNAME}

# we don't assume we're building with privileges so ensure the build directory
# can be accessed by the user.
RUN \
    chown ${UID}:${GID} ${BUILD_ROOT} && \
    chmod 2755 ${BUILD_ROOT}

# create the installation root and ensure that it is owned by the user.  this
# allows modifications at run-time should they be needed.
RUN \
    mkdir -p ${INSTALL_ROOT} && \
    chown ${UID}:${GID} ${INSTALL_ROOT} && \
    chmod 2755 ${INSTALL_ROOT}

# update the OS and install core utilities.
RUN \
    apt update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# transition into our non-privileged user.
USER ${USERNAME}

# install Miniconda 4.10.3.
RUN \
    ${CURL_CMD} -O https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh && \
    echo "1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f  Miniconda3-py39_4.10.3-Linux-x86_64.sh" > Miniconda3-py39_4.10.3-Linux-x86_64.sh.sha256 && \
    sha256sum -c Miniconda3-py39_4.10.3-Linux-x86_64.sh.sha256 && \
    sh Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p ${MINICONDA_ROOT} && \
    rm -rf Miniconda3-py39_4.10.3-Linux-x86_64.sh*

# setup Conda with a system configuration.
COPY conda/.condarc ${MINICONDA_ROOT}/.condarc

# copy the environment file into the image for reference.
COPY environment.yml ${MINICONDA_ROOT}/${IWP_CONDA_ENVIRONMENT}-environment.yml

# install Mamba so we have a SAT solver that works.  otherwise the environment
# created below will have problems.
RUN \
    ${MINICONDA_ROOT}/bin/conda install -y -c conda-forge mamba && \
# create the IWP environment.
#
# NOTE: we spawn a shell to set an additional environment variable forcing
#       Conda's requests Python module to respect our certificate authority
#       bundle.  without this, the installation can fail if certificates have
#       been added to the local certificate authority (and specified via
#       CURL_CA_BUNDLE).
#
    sh -c "REQUESTS_CA_BUNDLE=${CURL_CA_BUNDLE} ${MINICONDA_ROOT}/bin/mamba env create --file ${MINICONDA_ROOT}/${IWP_CONDA_ENVIRONMENT}-environment.yml" && \
# cleanup behind ourselves so we don't have unused packages embedded into the
# layers.  these can be downloaded again if modifications are needed at
# run-time.
    ${MINICONDA_ROOT}/bin/conda clean -av

# default to an interactive shell in the user's home directory.
WORKDIR /home/${USERNAME}
ENTRYPOINT ["/bin/bash", "-i"]
