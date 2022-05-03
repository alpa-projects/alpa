#!/bin/bash
# Adapted from https://github.com/alpa-projects/jax-alpa/blob/main/build/build_wheel_docker_entrypoint.sh
set -xev
if [ ! -d "/alpa-dist" ]
then
  echo "/alpa-dist must be mounted to produce output"
  exit 1
fi

usage() {
  echo "usage: ${0##*/} [3.7|3.8|3.9] [11.1|11.2|11.3] [alpa-branch]"
  exit 1
}

if [[ $# -lt 3 ]]
then
  usage
fi

ALPA_BRANCH="$3"
apt install -y coinor-cbc glpk-utils
export PY_VERSION=$1
export JAX_CUDA_VERSION=$2
export CUPY_VERSION=${JAX_CUDA_VERSION//.}

# Enter python env
source /python${PY_VERSION}-env/bin/activate

# switch to the merge commit
git clone https://github.com/alpa-projects/alpa.git
cd /build/alpa
git fetch origin
git checkout ${ALPA_BRANCH##*/}

# install cupy
pip install cupy-cuda${JAX_CUDA_VERSION//.}
python -m cupyx.tools.install_library --library nccl --cuda $JAX_CUDA_VERSION
pip install /alpa-dist/jaxlib-alpa/jaxlib-0.3.5-cp38-none-manylinux2010_x86_64.whl
pip install /alpa-dist/jax-alpa/jax-0.3.5.tar.gz

pip install -e .[dev]
ray start --head
cd tests
python run_all.py
