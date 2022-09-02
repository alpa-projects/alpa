#!/bin/bash
# Adapted from https://github.com/alpa-projects/jax-alpa/blob/main/build/build_wheel_docker_entrypoint.sh
set -xev
if [ ! -d "/dist" ]
then
  echo "/dist must be mounted to produce output"
  exit 1
fi

export CC=/dt8/usr/bin/gcc
export GCC_HOST_COMPILER_PATH=/dt8/usr/bin/gcc
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

usage() {
  echo "usage: ${0##*/} [3.7|3.8|3.9] [cuda|nocuda] [11.1|11.2|11.3] [alpa branch name] [tensorflow-alpa branch name]"
  exit 1
}

if [[ $# -lt 3 ]]
then
  usage
fi

PY_VERSION="$1"
echo "Python version $PY_VERSION"

# switch tensorflow-alpa branch if necessary
git clone --recursive https://github.com/alpa-projects/alpa.git

# switch alpa branch
if [[ $# -eq 4 ]]
then
  ALPA_BRANCH="$4"
  echo "Switch to alpa branch ALPA_BRANCH"
  cd /build/alpa
  git fetch origin +${ALPA_BRANCH}
  git checkout -qf FETCH_HEAD
  git submodule update --recursive
fi

# switch tensorflow-alpa branch, this will overwrite the above
if [[ $# -eq 5 ]]
then
  TF_BRANCH="$5"
  echo "Switch to tensorflow-alpa branch $TF_BRANCH"
  cd /build/alpa/third_party/tensorflow-alpa
  git fetch origin +${TF_BRANCH}
  git checkout -qf FETCH_HEAD
fi

mkdir /build/tmp
mkdir /build/root
export TMPDIR=/build/tmp

# Builds and activates a specific Python version.
source /python${PY_VERSION}-env/bin/activate

# Workaround for https://github.com/bazelbuild/bazel/issues/9254
export BAZEL_LINKLIBS="-lstdc++"
export JAX_CUDA_VERSION=$3
export CUPY_VERSION=${JAX_CUDA_VERSION//.}

if [ $JAX_CUDA_VERSION = "11.0" ]; then
  export JAX_CUDNN_VERSION="805"
elif [ $JAX_CUDA_VERSION = "11.1" ]; then
  export JAX_CUDNN_VERSION="805"
elif [ $JAX_CUDA_VERSION = "11.2" ]; then
  export JAX_CUDNN_VERSION="810"
elif [ $JAX_CUDA_VERSION = "11.3" ]; then
  export JAX_CUDNN_VERSION="820"
elif [ $JAX_CUDA_VERSION = "11.4" ]; then
  export JAX_CUDNN_VERSION="822"
else
  echo "Unknown CUDNN version for CUDA version: $JAX_CUDA_VERSION"
  exit 1
fi


# install cupy
pip install cupy-cuda${JAX_CUDA_VERSION//.}
python -m cupyx.tools.install_library --library nccl --cuda $JAX_CUDA_VERSION

# start building
cd /build/alpa/build_jaxlib
case $2 in
  cuda)
    python build/build.py --enable_cuda --bazel_startup_options="--output_user_root=/build/root" --bazel_options=--override_repository=org_tensorflow=$(pwd)/../third_party/tensorflow-alpa 
    ;;
  nocuda)
    python build/build.py --enable_tpu --bazel_startup_options="--output_user_root=/build/root" --bazel_options=--override_repository=org_tensorflow=$(pwd)/../third_party/tensorflow-alpa
    ;;
  *)
    usage
esac

if ! python -m auditwheel show dist/jaxlib-*.whl | egrep 'platform tag: "(manylinux2014_x86_64|manylinux_2_17_x86_64)"' > /dev/null; then
  # Print output for debugging
  python -m auditwheel show dist/jaxlib-*.whl
  echo "jaxlib wheel is not manylinux2014 compliant"
  exit 1
fi
cp -r dist/* /dist
