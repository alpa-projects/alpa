# Adapted from https://github.com/alpa-projects/jax-alpa/blob/main/build/build_wheel_docker_entrypoint.sh
set -xev
if [ ! -d "/alpa-dist" ]
then
  echo "/alpa-dist must be mounted to produce output"
  exit 1
fi

apt-get install -y virtualenv


# Builds and activates a specific Python version.
virtualenv --python=python3.8 python3.8-env
source python3.8-env/bin/activate
pip install --upgrade pip && pip install numpy==1.19.5 setuptools wheel six auditwheel

export JAX_CUDA_VERSION=11.1
export CUPY_VERSION=${JAX_CUDA_VERSION//.}

git clone https://github.com/alpa-projects/alpa.git
# TODO(Hao): remove this
cd /build/alpa && git checkout hao-wheel-2

# install ILP solver
apt install -y coinor-cbc glpk-utils
#sudo apt install -y libsqlite3-dev
# install cupy
pip install cupy-cuda${JAX_CUDA_VERSION//.}
python -m cupyx.tools.install_library --library nccl --cuda $JAX_CUDA_VERSION
pip install coverage cmake pybind11 ray pysqlite3

pip install /alpa-dist/jaxlib-alpa/jaxlib-0.3.5-cp38-none-manylinux2010_x86_64.whl
pip install /alpa-dist/jax-alpa/jax-0.3.5.tar.gz

python setup.py install
ray start --head
cd /build/alpa/tests
python -m unittest