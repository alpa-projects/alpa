FROM gcr.io/tensorflow-testing/nosla-cuda11.1-cudnn8-ubuntu18.04-manylinux2010-multipython

WORKDIR /
SHELL ["/bin/bash", "-c"]
RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-get update
RUN apt-get install -y python3-virtualenv
RUN virtualenv --python=python3.8 python3.8-env

# We pin numpy to the minimum permitted version to avoid compatibility issues.
RUN source python3.8-env/bin/activate && pip install --upgrade pip \
  && pip install numpy==1.20 setuptools wheel six auditwheel \
  tqdm scipy numba pulp tensorstore prospector yapf coverage cmake  \
  pybind11 ray[default] matplotlib transformers uvicorn fastapi

# Install PyTorch dependencies
WORKDIR /
COPY scripts/install_torch.sh /install_torch.sh
RUN chmod +x /install_torch.sh
RUN source python3.8-env/bin/activate && /install_torch.sh

# We determine the CUDA version at `docker build ...` phase
ARG JAX_CUDA_VERSION=11.1
COPY ../scripts/install_cuda.sh /install_cuda.sh
RUN chmod +x /install_cuda.sh
RUN /bin/bash -c 'if [[ ! "$CUDA_VERSION" =~ ^$JAX_CUDA_VERSION.*$ ]]; then \
  /install_cuda.sh $JAX_CUDA_VERSION; \
  fi'

# Install cupy
RUN source python3.8-env/bin/activate && pip install cupy-cuda11x

# WORKDIR /
# COPY run_benchmark.sh /run_benchmark.sh
# RUN chmod +x /run_benchmark.sh

# WORKDIR /build
# ENV TEST_TMPDIR /build
# ENTRYPOINT ["/run_benchmark.sh"]
