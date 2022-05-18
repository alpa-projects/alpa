#!/bin/bash
set -xev
if [ ! -d "/dist" ]
then
  echo "/dist must be mounted to produce output"
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
pip install cmake

# switch to the merge commit
git clone https://github.com/alpa-projects/alpa.git
cd /build/alpa
git fetch origin +${ALPA_BRANCH}
git checkout -qf FETCH_HEAD

# install jaxlib and jax
export VERSION=$(bash GENVER --pep440)
python setup.py bdist_wheel --plat-name=manylinux2014_x86_64

if ! python -m auditwheel show dist/alpa-*.whl  | egrep 'platform tag: "(manylinux2010_x86_64|manylinux_2_12_x86_64)"' > /dev/null; then
  # Print output for debugging
  python -m auditwheel show dist/alpa-*.whl
  echo "jaxlib wheel is not manylinux2010 compliant"
  exit 1
fi

cp -r dist/*whl /dist/
