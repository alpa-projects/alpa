Install Alpa
============

Requirements
------------
- CUDA >= 11.1
- CuDNN >= 8.1
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

  - CUDA Toolkit: `cuda <https://developer.nvidia.com/cuda-toolkit>`_, `cudnn <https://developer.nvidia.com/cudnn>`_, and `nccl <https://developer.nvidia.com/nccl>`_
  - Python packages:

  .. code:: bash

    pip3 install cmake tqdm numpy scipy numba pybind11 ray flax==0.4.1
    pip3 install cupy-cuda114  # use your own CUDA version. Here cuda-cuda114 means cuda 11.4.

  - ILP Solver:

  .. code:: bash

    sudo apt install coinor-cbc
    pip3 install pulp

  If you do not have sudo permission, please try install via `binary <https://projects.coin-or.org/Cbc#DownloadandInstall>`_ or `conda <https://anaconda.org/conda-forge/coincbc>`_.

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

    cd alpa/alpa/pipeline_parallel/xla_custom_call_marker
    bash build.sh

.. note::

  All installations are in development mode, so you can modify python code and it will take effect immediately.
  To modify c++ code in tensorflow, you only need to run the command below from step 3 to recompile jaxlib::

    python3 build/build.py --enable_cuda --dev_install --tf_path=$TF_PATH

Check Installation
------------------
You can check the installation by running the following test script.

.. code:: bash

  cd alpa
  ray start --head
  python3 tests/test_install.py

