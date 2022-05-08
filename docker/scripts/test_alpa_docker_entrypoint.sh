#!/bin/bash
# Adapted from https://github.com/alpa-projects/jax-alpa/blob/main/build/build_wheel_docker_entrypoint.sh
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
pip install /alpa-dist/jaxlib-alpa-ci/jaxlib-0.3.5-cp38-none-manylinux2010_x86_64.whl
pip install /alpa-dist/jax-alpa/jax-0.3.5.tar.gz

pip install -e .[dev]
ray start --head
cd tests
python run_all.py
