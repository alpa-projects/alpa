#!/bin/bash

set -xev

if [ ! -d "/alpa-dist" ]
then
  echo "/alpa-dist must be mounted to produce output"
  exit 1
fi

pip install /alpa-dist/jaxlib-alpa/jaxlib-0.3.5-cp38-none-manylinux2010_x86_64.whl
pip install /alpa-dist/jax-alpa/jax-0.3.5.tar.gz

git clone https://github.com/alpa-projects/alpa.git
cd alpa
pip install -e .
cd /alpa/docs
make html

