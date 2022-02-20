Installing Alpa
===============

Requirements
------------
- CuDNN >= 8.1
- CUDA >= 11.1
- python >= 3.7

Install from Source
-------------------
Alpa depends on its own fork of jax and tensorflow.
To install alpa from source, we need to build these forks.

1.  Clone repos

  .. code:: bash
  
    git clone git@github.com:alpa-projects/alpa.git
    git clone git@github.com:alpa-projects/jax-alpa.git
    git clone git@github.com:alpa-projects/tensorflow-alpa.git

2. Install dependencies

  - CUDA Toolkit: cuda and cudnn
  - Python packages:

  .. code:: bash

    pip3 install cmake numpy scipy flax numba pybind11 ray[default]
    pip3 install cupy-cuda111   # use your own CUDA version

    # In case NCCL is not automatically installed during cupy installation, please install it manually
    python3 -m cupyx.tools.install_library --library nccl --cuda 11.1  # use your own CUDA version

  - ILP Solver:

  .. code:: bash

    sudo apt install coinor-cbc glpk-utils
    pip3 install pulp

3. Build and install jaxlib

  .. code:: bash
  
    cd jax-alpa
    export TF_PATH=~/tensorflow-alpa  # update this with your path
    python3 build/build.py --enable_cuda --dev_install --tf_path=$TF_PATH
    cd dist
    pip3 install -e .

4. Install jax

  .. code:: bash
  
    cd jax-alpa
    pip3 install -e .

5. Install Alpa

  .. code:: bash
  
    cd alpa
    pip3 install -e .

6. Build XLA pipeline marker custom call

  .. code:: bash
  
    cd alpa/pipeline_parallel/xla_custom_call_marker
    bash build.sh

.. note::

  All installations are in development mode, so you can modify python code and it will take effect immediately.
  To modify c++ code in tensorflow, you only need to run the command below from step 3 to recompile jaxlib::

    python3 build/build.py --enable_cuda --dev_install --tf_path=$TF_PATH

