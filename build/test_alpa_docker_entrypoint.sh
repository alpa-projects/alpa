#!/bin/bash
# Adapted from https://github.com/alpa-projects/jax-alpa/blob/main/build/build_wheel_docker_entrypoint.sh
set -xev
if [ ! -d "/alpa-dist" ]
then
  echo "/alpa-dist must be mounted to produce output"
  exit 1
fi

usage() {
  echo "usage: ${0##*/} [3.7|3.8|3.9] [11.1|11.2|11.3]"
  exit 1
}

if [[ $# -lt 2 ]]
then
  usage
fi

apt install -y coinor-cbc glpk-utils
export JAX_CUDA_VERSION=$2
export CUPY_VERSION=${JAX_CUDA_VERSION//.}

# Enter python env
source /python${PY_VERSION}-env/bin/activate


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
