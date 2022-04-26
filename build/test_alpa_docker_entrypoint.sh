#!/bin/bash
# Adapted from https://github.com/alpa-projects/jax-alpa/blob/main/build/build_wheel_docker_entrypoint.sh
set -xev
if [ ! -d "/alpa-dist" ]
then
  echo "/alpa-dist must be mounted to produce output"
  exit 1
fi

export PYENV_ROOT="/pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

PY_VERSION="$1"
echo "Python version $PY_VERSION"


usage() {
  echo "usage: ${0##*/} [3.7.2|3.8.0|3.9.0] [11.1|11.2|11.3]"
  exit 1
}

if [[ $# -lt 2 ]]
then
  usage
fi

# Builds and activates a specific Python version.
pyenv local "$PY_VERSION"

export JAX_CUDA_VERSION=$2
export CUPY_VERSION=${JAX_CUDA_VERSION//.}

git clone https://github.com/alpa-projects/alpa.git
# TODO(Hao): remove this
cd /build/alpa && git checkout hao-wheel-2

# install ILP solver
sudo apt install -y coinor-cbc
sudo apt install -y libsqlite3-dev
# install cupy
pip install cupy-cuda${JAX_CUDA_VERSION//.}
python -m cupyx.tools.install_library --library nccl --cuda $JAX_CUDA_VERSION
pip install coverage cmake pybind11 ray pysqlite3

pip install /alpa-dist/jaxlib-alpa/jaxlib-0.3.5-cp38-none-manylinux2010_x86_64.whl
pip install /alpa-dist/jax-alpa/jax-0.3.5.tar.gz

python setup.py install
ray start --head
coverage run -m unittest tests/test_*.py
coverage report -m
coverage html
mkdir -p /alpa-dist/coverage
coverage xml -o /alpa-dist/coverage/