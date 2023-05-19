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

# 'defaults' 채널은 라이선싱 이슈로 인해 제외함.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      git \
      libjpeg-turbo8-dev && \
    rm -rf /var/lib/apt/lists/*

# Miniconda는 오픈소스 라이선스로 기업에서도 사용 가능하다.
# `curl`의 `-k` flag와 conda SSL verification을 해제함으로써 방화벽 내에서도 설치.
RUN curl -fksSL -v -o /tmp/miniconda.sh -O ${CONDA_URL} && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda config --set ssl_verify no && \
    conda config --append channels conda-forge && \
    conda config --remove channels defaults && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

# 방화벽이 있어도 Python 설치가 가능하게 함.
ENV PYTHONHTTPSVERIFY=0
RUN {   echo "[global]"; \
        echo "trusted-host=pypi.org files.pythonhosted.org"; \
    } > /opt/conda/pip.conf

########################################################################
FROM build-base AS build-pillow
# Pillow-SIMD 라이브러리 설치를 통해 Pillow의 가속화 라이브러리 사용..
# Condition ensures that AVX2 instructions are built only if available.
ARG PILLOW_SIMD_VERSION=9.0.0.post1
RUN if [ -n "$(lscpu | grep avx2)" ]; then CC="cc -mavx2"; fi && \
    python -m pip wheel --no-deps --wheel-dir /tmp/dist \
        Pillow-SIMD==${PILLOW_SIMD_VERSION}

########################################################################
FROM build-base AS build-torch
# 실제로는 download이지만 일관성을 위해 build라고 부른다.

ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG PYTORCH_HOST
ARG PYTORCH_INDEX_URL
# PyTorch CUDA 11에 최적화된 wheel을 받기 위해 별도의 index URL 사용.
RUN python -m pip wheel --no-deps \
            --wheel-dir /tmp/dist \
            --index-url ${PYTORCH_INDEX_URL} \
            --trusted-host ${PYTORCH_HOST} \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION}

########################################################################
FROM build-base AS build-pure

# Z-shell 관련 패키지. 터미널 UI 및 UX 향상.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

########################################################################
FROM ${BUILD_IMAGE} AS deploy-builds

# 빌드가 끝난 결과물을 가지고 옴.
COPY --link --from=build-base   /opt/conda /opt/conda
COPY --link --from=build-torch  /tmp/dist  /tmp/dist
COPY --link deploy-requirements.txt        /tmp/requirements.txt

# 방화벽이 있어도 Python 설치가 가능하게 함.
ENV PYTHONHTTPSVERIFY=0
RUN {   echo "[global]"; \
        echo "trusted-host=pypi.org files.pythonhosted.org"; \
    } > /opt/conda/pip.conf

# 모든 Python 패키지는 pip으로 설치하는 것을 원칙으로 함.
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

# sed와 xargs를 사용해 requirements 파일로 apt 설치가 가능하게 함.
COPY --link apt-deploy-requirements.txt /opt/apt-requirements.txt
RUN apt-get update && sed 's/#.*//g; s/\r//g' /opt/apt-requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ARG PROJECT_ROOT=/opt/lct
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}
COPY --link --from=deploy-builds /opt/conda /opt/conda
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

# CPU 최적화 설정. Binary 파일 경로는 패키지 업데이트에 따라 변할 수 있으니 주의.
ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:/opt/conda/lib/libiomp5.so:$LD_PRELOAD
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

WORKDIR ${PROJECT_ROOT}
CMD ["/bin/bash"]

########################################################################
FROM ${BUILD_IMAGE} AS train-builds

# 빌드가 끝난 결과물을 가지고 옴.
COPY --link --from=build-base   /opt/conda /opt/conda
COPY --link --from=build-pillow /tmp/dist  /tmp/dist
COPY --link --from=build-torch  /tmp/dist  /tmp/dist
COPY --link --from=build-pure   /opt/zsh   /opt/zsh
COPY --link requirements.txt    /tmp/requirements.txt

# 방화벽이 있어도 Python 설치가 가능하게 함.
ENV PYTHONHTTPSVERIFY=0
RUN {   echo "[global]"; \
        echo "trusted-host=pypi.org files.pythonhosted.org"; \
    } > /opt/conda/pip.conf

# 모든 Python 패키지는 pip으로 설치하는 것을 원칙으로 함.
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

# tzdata를 통한 시간대 설정.
ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive

# 방화벽이 있어도 Python 설치가 가능하게 함.
ENV PYTHONHTTPSVERIFY=0
ARG DEB_OLD
ARG DEB_NEW
# apt를 통한 설치 과정을 sed와 xargs를 통해 requirements 파일을 사용 가능하게 함.
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
# Conda는 가상환경 매니저로만 사용하고 모든 패키지 설치는 pip으로 진행한다.
# 다만, 혹시 모르니 채널 설정은 COPY를 진행할 때 제거되므로 채널 순서를 지정해둔다.
RUN conda config --set ssl_verify no && \
    conda config --append channels conda-forge && \
    conda config --remove channels defaults

# Python 패키지를 link directory에 추가.
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

# Intel OpenMP 및 Jemalloc을 통한 CPU 성능 최적화.
ENV KMP_BLOCKTIME=0
# Ubuntu 18 이하에서는 libjemalloc1 파일의 위치를 사용할 것.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:/opt/conda/lib/libiomp5.so:$LD_PRELOAD
# https://android.googlesource.com/platform/external/jemalloc_new/+/6e6a93170475c05ebddbaf3f0df6add65ba19f01/TUNING.md
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

USER ${USR}

# 터미널 설정 변경.
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
