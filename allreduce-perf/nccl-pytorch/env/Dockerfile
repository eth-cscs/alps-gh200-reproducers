ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.03-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

RUN pip install -vvv \
  git+https://github.com/mlcommons/logging.git@2.1.0-rc1 \
  git+https://github.com/NVIDIA/mlperf-common.git \
  zarr \
  git+https://github.com/google/tensorstore.git
