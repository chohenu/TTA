# syntax=docker/dockerfile:1.4
# 최상단 주석은 작동을 위해 필요하므로 삭제하지 말 것.

ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG PYTHON_VERSION
ARG LINUX_DISTRO
ARG DISTRO_VERSION

# Visit https://hub.docker.com/r/nvidia/cuda/tags for all available images.
ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG DEPLOY_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-${LINUX_DISTRO}${DISTRO_VERSION}

########################################################################
FROM ${BUILD_IMAGE} AS build-base

LABEL maintainer=mi.ret@vuno.co
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
ARG CONDA_URL

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      git \
      libjpeg-turbo8-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fksSL -v -o /tmp/miniconda.sh -O ${CONDA_URL} && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda config --set ssl_verify no && \
    conda config --append channels conda-forge && \
    conda config --remove channels defaults && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

ENV PYTHONHTTPSVERIFY=0
RUN {   echo "[global]"; \
        echo "trusted-host=pypi.org files.pythonhosted.org"; \
    } > /opt/conda/pip.conf

########################################################################
FROM build-base AS build-pillow
ARG PILLOW_SIMD_VERSION=9.0.0.post1
RUN if [ -n "$(lscpu | grep avx2)" ]; then CC="cc -mavx2"; fi && \
    python -m pip wheel --no-deps --wheel-dir /tmp/dist \
        Pillow-SIMD==${PILLOW_SIMD_VERSION}

########################################################################
FROM build-base AS build-torch

ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG PYTORCH_HOST
ARG PYTORCH_INDEX_URL
RUN python -m pip wheel --no-deps \
            --wheel-dir /tmp/dist \
            --index-url ${PYTORCH_INDEX_URL} \
            --trusted-host ${PYTORCH_HOST} \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION}

########################################################################
FROM build-base AS build-pure

ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

########################################################################
FROM ${BUILD_IMAGE} AS deploy-builds

COPY --link --from=build-base   /opt/conda /opt/conda
COPY --link --from=build-torch  /tmp/dist  /tmp/dist
COPY --link deploy-requirements.txt        /tmp/requirements.txt

ENV PYTHONHTTPSVERIFY=0
RUN {   echo "[global]"; \
        echo "trusted-host=pypi.org files.pythonhosted.org"; \
    } > /opt/conda/pip.conf

ENV PATH=/opt/conda/bin:$PATH
ARG PIP_CACHE_DIR=/tmp/.cache/pip
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    python -m pip install --find-links /tmp/dist \
        -r /tmp/requirements.txt \
        /tmp/dist/*.whl

########################################################################
FROM ${DEPLOY_IMAGE} AS deploy

LABEL maintainer=mi.ret@vuno.co
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# Timezone set to UTC for consistency.
ENV TZ=UTC
ARG DEBIAN_FRONTEND=noninteractive

COPY --link apt-deploy-requirements.txt /opt/apt-requirements.txt
RUN apt-get update && sed 's/#.*//g; s/\r//g' /opt/apt-requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ARG PROJECT_ROOT=/opt/lct
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}
COPY --link --from=deploy-builds /opt/conda /opt/conda
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:/opt/conda/lib/libiomp5.so:$LD_PRELOAD
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

WORKDIR ${PROJECT_ROOT}
CMD ["/bin/bash"]

########################################################################
FROM ${BUILD_IMAGE} AS train-builds

COPY --link --from=build-base   /opt/conda /opt/conda
COPY --link --from=build-pillow /tmp/dist  /tmp/dist
COPY --link --from=build-torch  /tmp/dist  /tmp/dist
COPY --link --from=build-pure   /opt/zsh   /opt/zsh
COPY --link requirements.txt    /tmp/requirements.txt

ENV PYTHONHTTPSVERIFY=0
RUN {   echo "[global]"; \
        echo "trusted-host=pypi.org files.pythonhosted.org"; \
    } > /opt/conda/pip.conf

ENV PATH=/opt/conda/bin:$PATH
ARG PIP_CACHE_DIR=/tmp/.cache/pip
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    python -m pip install --find-links /tmp/dist \
        -r /tmp/requirements.txt \
        /tmp/dist/*.whl

########################################################################
FROM ${TRAIN_IMAGE} AS train

LABEL maintainer=mi.ret@vuno.co
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONHTTPSVERIFY=0
ARG DEB_OLD
ARG DEB_NEW
COPY --link apt-requirements.txt /opt/apt-requirements.txt
RUN if [ ${DEB_NEW} ]; then sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list; fi && \
    apt-get update && sed 's/#.*//g; s/\r//g' /opt/apt-requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ARG GID
ARG UID
ARG GRP
ARG USR
ARG PASSWD=ubuntu
# Create user with home directory and password-free sudo permissions.
# This may cause security issues. Use at your own risk.
RUN groupadd -g ${GID} ${GRP} && \
    useradd --shell /bin/zsh --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -aG sudo ${USR}

ARG PROJECT_ROOT=/opt/lct
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}
COPY --link --chown=${UID}:${GID} --from=train-builds /opt/conda /opt/conda
RUN conda config --set ssl_verify no && \
    conda config --append channels conda-forge && \
    conda config --remove channels defaults

RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:/opt/conda/lib/libiomp5.so:$LD_PRELOAD
# https://android.googlesource.com/platform/external/jemalloc_new/+/6e6a93170475c05ebddbaf3f0df6add65ba19f01/TUNING.md
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

USER ${USR}

ARG HOME=/home/${USR}
ARG PURE_PATH=$HOME/.zsh/pure
ARG ZSH_FILE=$HOME/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
COPY --link --chown=${UID}:${GID} --from=train-builds /opt/zsh ${HOME}/.zsh
RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        echo "prompt pure"; \
        echo "source ${ZSH_FILE}"; \
        echo "alias ll='ls -lh'"; \
        echo "alias wns='watch nvidia-smi'"; \
    } >> ${HOME}/.zshrc

# Enable mouse scrolling for tmux by uncommenting.
# iTerm2 users should change settings to use scrolling properly.
# RUN echo 'set -g mouse on' >> ${HOME}/.tmux.conf

WORKDIR ${PROJECT_ROOT}
CMD ["/bin/zsh"]
