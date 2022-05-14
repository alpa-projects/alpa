#!/bin/bash
# Adapted from https://github.com/alpa-projects/jax-alpa/blob/main/build/build_wheel_docker_entrypoint.sh
set -xev
if [ ! -d "/dist" ]
then
  echo "/dist must be mounted to produce output"
  exit 1
fi

export CC=/dt7/usr/bin/gcc
export GCC_HOST_COMPILER_PATH=/dt7/usr/bin/gcc
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

usage() {
  echo "usage: ${0##*/} [3.7|3.8|3.9] [cuda|nocuda] [11.0|11.1|11.2] [tensorflow-alpa branch name]"
  exit 1
}

if [[ $# -lt 4 ]]
then
  usage
fi

PY_VERSION="$1"
echo "Python version $PY_VERSION"
TF_BRANCH="$4"

git clone -b $TF_BRANCH https://github.com/alpa-projects/tensorflow-alpa.git /build/tensorflow-alpa
git clone https://github.com/alpa-projects/jax-alpa.git /build/jax
cd /build/jax/build

mkdir /build/tmp
mkdir /build/root
export TMPDIR=/build/tmp
export TF_PATH=/build/tensorflow-alpa

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
else
  echo "Unknown CUDNN version for CUDA version: $JAX_CUDA_VERSION"
  exit 1
fi


# install cupy
pip install cupy-cuda${JAX_CUDA_VERSION//.}
python -m cupyx.tools.install_library --library nccl --cuda $JAX_CUDA_VERSION

case $2 in
  cuda)
    python build.py --enable_cuda --bazel_startup_options="--output_user_root=/build/root" --tf_path=$TF_PATH
    ;;
  nocuda)
    python build.py --enable_tpu --bazel_startup_options="--output_user_root=/build/root" --tf_path=$TF_PATH
    ;;
  *)
    usage
esac

if ! python -m auditwheel show dist/jaxlib-*.whl | egrep 'platform tag: "(manylinux2010_x86_64|manylinux_2_12_x86_64)"' > /dev/null; then
  # Print output for debugging
  python -m auditwheel show dist/jaxlib-*.whl
  echo "jaxlib wheel is not manylinux2010 compliant"
  exit 1
fi
cp -r dist/* /dist
