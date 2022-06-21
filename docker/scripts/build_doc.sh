#!/bin/bash

set -xev

if [ ! -d "/alpa-dist" ]
then
  echo "/alpa-dist must be mounted to produce output"
  exit 1
fi

source /python3.8-env/bin/activate
pip install /alpa-dist/jaxlib-alpa-ci/jaxlib-0.3.5+cuda111.cudnn805-cp38-none-manylinux2010_x86_64.whl
pip install jax==0.3.5

git clone https://github.com/alpa-projects/alpa.git
cd alpa
pip install cupy-cuda111
python -m cupyx.tools.install_library --library nccl --cuda 11.1
pip install -e .[doc]
cd /alpa/docs
make html
cp -r _build/html/* /alpa-dist/docs/
