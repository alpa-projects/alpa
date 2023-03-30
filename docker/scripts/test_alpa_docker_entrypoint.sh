#!/bin/bash
set -xev
if [ ! -d "/alpa-dist" ]
then
  echo "/alpa-dist must be mounted to produce output"
  exit 1
fi

usage() {
  echo "usage: ${0##*/} [3.7|3.8|3.9] [alpa-branch]"
  exit 1
}

if [[ $# -lt 2 ]]
then
  usage
fi

export PY_VERSION=$1
ALPA_BRANCH="$2"

# Enter python env
source /python${PY_VERSION}-env/bin/activate
# switch to the merge commit
git clone https://github.com/alpa-projects/alpa.git
cd /build/alpa
git fetch origin +${ALPA_BRANCH}
git checkout -qf FETCH_HEAD

# install jaxlib and jax
pip install /alpa-dist/jaxlib-alpa-ci/jaxlib-0.3.22+cuda111.cudnn805-cp38-cp38-manylinux2014_x86_64.whl
pip install jax==0.3.22

# install cupy
pip install cupy-cuda11x
python -m cupyx.tools.install_library --library nccl --cuda 11.1
pip install -e .[dev]
ray start --head
cd tests
python run_all.py
