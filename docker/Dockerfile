# Adapted from https://github.com/alpa-projects/jax-alpa/blob/main/build/Dockerfile

FROM gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010

WORKDIR /
RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-get update
RUN apt-get install libffi-dev
RUN git clone --branch v1.2.21 https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
RUN /pyenv/bin/pyenv install 3.7.2
RUN /pyenv/bin/pyenv install 3.8.0
RUN /pyenv/bin/pyenv install 3.9.0

# We pin numpy to the minimum permitted version to avoid compatibility issues.
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.7.2 && pip install numpy==1.19.5 setuptools wheel six auditwheel
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.8.0 && pip install numpy==1.19.5 setuptools wheel six auditwheel
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.9.0 && pip install numpy==1.19.5 setuptools wheel six auditwheel

# Change the CUDA version if it doesn't match the installed version in the base image
# which is 10.0
ARG JAX_CUDA_VERSION=11.1
COPY install_cuda.sh /install_cuda.sh
RUN chmod +x /install_cuda.sh
RUN /bin/bash -c 'if [[ ! "$CUDA_VERSION" =~ ^$JAX_CUDA_VERSION.*$ ]]; then \
  /install_cuda.sh $JAX_CUDA_VERSION; \
  fi'


WORKDIR /
COPY build_wheel_docker_entrypoint.sh /build_wheel_docker_entrypoint.sh
RUN chmod +x /build_wheel_docker_entrypoint.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_wheel_docker_entrypoint.sh"]
