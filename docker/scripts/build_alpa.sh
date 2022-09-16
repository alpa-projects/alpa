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

if [ $PY_VERSION = "3.7" ]; then
  #alias python="/opt/python/cp37-cp37m/bin/python"
  ln -fs /opt/python/cp37-cp37m/bin/python /usr/bin/python3
  python3 -m ensurepip --upgrade
  python3 -m pip install cmake auditwheel pybind11
  ln -fs /opt/python/cp37-cp37m/bin/pybind11-config /usr/bin/pybind11-config
elif [ $PY_VERSION = "3.8" ]; then
  #alias python="/opt/python/cp38-cp38/bin/python"
  ln -fs /opt/python/cp38-cp38/bin/python /usr/bin/python3
  python3 -m ensurepip --upgrade
  python3 -m pip install cmake auditwheel pybind11
  ln -fs /opt/python/cp38-cp38/bin//pybind11-config /usr/bin/pybind11-config
elif [ $PY_VERSION = "3.9" ]; then
  #alias python="/opt/python/cp39-cp39/bin/python"
  ln -fs /opt/python/cp39-cp39/bin/python /usr/bin/python3
  python3 -m ensurepip --upgrade
  python3 -m pip install cmake auditwheel pybind11
  ln -fs /opt/python/cp39-cp39/bin/pybind11-config /usr/bin/pybind11-config
else
  echo "Unsupported Python version: $PY_VERSION"
  exit 1
fi

ALPA_BRANCH="$2"

# switch to the merge commit
git clone https://github.com/alpa-projects/alpa.git
cd alpa
git fetch origin +${ALPA_BRANCH}
git checkout -qf FETCH_HEAD

# install jaxlib and jax
python3 update_version.py --git-describe
python3 setup.py bdist_wheel sdist

#if ! python3 -m auditwheel show dist/alpa-*.whl  | egrep 'platform tag: "(manylinux2014_x86_64|manylinux_2_17_x86_64)"' > /dev/null; then
#  # Print output for debugging
#  python3 -m auditwheel show dist/alpa-*.whl
#  echo "jaxlib wheel is not manylinux2014 compliant"
#  exit 1
#fi

#rename 'linux' manylinux2014 dist/*.whl
cp -r dist/*whl /dist/
