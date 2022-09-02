FROM gcr.io/tensorflow-testing/nosla-cuda11.1-cudnn8-ubuntu18.04-manylinux2010-multipython

WORKDIR /
SHELL ["/bin/bash", "-c"]
RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-get update
RUN apt-get install -y python3-virtualenv
RUN virtualenv --python=python3.7 python3.7-env
RUN virtualenv --python=python3.8 python3.8-env
RUN virtualenv --python=python3.9 python3.9-env

# We pin numpy to the minimum permitted version to avoid compatibility issues.
RUN source python3.7-env/bin/activate && pip install --upgrade pip && pip install numpy==1.20 setuptools wheel six auditwheel
RUN source python3.8-env/bin/activate && pip install --upgrade pip && pip install numpy==1.20 setuptools wheel six auditwheel
RUN source python3.9-env/bin/activate && pip install --upgrade pip && pip install numpy==1.20 setuptools wheel six auditwheel

# Change the CUDA version if it doesn't match the installed version in the base image
# which is 10.0
ARG JAX_CUDA_VERSION=11.1
COPY scripts/install_cuda.sh /install_cuda.sh
RUN chmod +x /install_cuda.sh
RUN /bin/bash -c 'if [[ ! "$CUDA_VERSION" =~ ^$JAX_CUDA_VERSION.*$ ]]; then \
  /install_cuda.sh $JAX_CUDA_VERSION; \
  fi'


WORKDIR /
COPY scripts/build_jaxlib_docker_entrypoint.sh /build_jaxlib_docker_entrypoint.sh
RUN chmod +x /build_jaxlib_docker_entrypoint.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_jaxlib_docker_entrypoint.sh"]
