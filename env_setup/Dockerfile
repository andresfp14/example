###############################################################################
# Nvidia already has a docker image where they installed all their drivers 
###############################################################################
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

###############################################################################
# Small fix for the internal time of the container. Logging has issues otherwise
###############################################################################
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

###############################################################################
# Commonly needed system libs
###############################################################################
RUN apt-get update \
    && apt-get install -y \
        tzdata git wget cmake make zip curl unzip\
        pkg-config build-essential \
        libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev \ 
        libxmlsec1-dev libffi-dev liblzma-dev \
        autoconf libtool flex bison \
        libgl1-mesa-glx rsync libedit-dev htop \
    && apt-get clean

###############################################################################
# Installing pyenv
###############################################################################
ENV HOME="/root"
WORKDIR ${HOME}
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

###############################################################################
# Installing your global python version (in this case is miniconda)
###############################################################################
ENV PYTHON_VERSION=miniconda3-latest
WORKDIR /home
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}
RUN conda init

# Installing venv and jupyterlab in the base environment.
# in case you want to run the container as jupyter notebooks.

RUN pip install virtualenv jupyterlab
CMD export SHELL=/bin/bash && python -m jupyter lab --allow-root --ip='*' --NotebookApp.token='' --NotebookApp.password=''