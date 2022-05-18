Install Alpa
============

Requirements
------------
- CUDA >= 11.1
- CuDNN >= 8.1
- python >= 3.7

Install from Source
-------------------
1.  Clone repos

  .. code:: bash
  
    git clone --recursive git@github.com:alpa-projects/alpa.git

2. Install dependencies

  - CUDA Toolkit:
      Follow the official guides to install `cuda <https://developer.nvidia.com/cuda-toolkit>`_ and `cudnn <https://developer.nvidia.com/cudnn>`_.
  - Python packages:

      .. code:: bash
    
        pip3 install cmake tqdm pybind11 numba numpy pulp ray tensorstore flax==0.4.1 jax==0.3.5
        pip3 install cupy-cuda114  # use your own CUDA version. Here cuda-cuda114 means cuda 11.4.

  - NCCL:
      First, check whether your system already has NCCL installed.

      .. code:: bash

        python3 -c "from cupy.cuda import nccl"

      If it prints nothing, then nccl is already installed.
      Otherwise, follow the printed instructions to install nccl.

  - ILP Solver:
      If you have sudo permission, use

      .. code:: bash
    
        sudo apt install coinor-cbc

      Otherwise, please try to install via `binary <https://projects.coin-or.org/Cbc#DownloadandInstall>`_ or `conda <https://anaconda.org/conda-forge/coincbc>`_.

3. Build and install jaxlib

  .. code:: bash
  
    cd alpa/build_jaxlib
    python3 build/build.py --enable_cuda --dev_install --tf_path=$(pwd)/../third_party/tensorflow-alpa
    cd dist
    pip3 install -e .

4. Install Alpa

  .. code:: bash
  
    cd alpa
    pip3 install -e .

5. Build XLA pipeline marker custom call

  .. code:: bash

    cd alpa/alpa/pipeline_parallel/xla_custom_call_marker
    bash build.sh

.. note::

  All installations are in development mode, so you can modify python code and it will take effect immediately.
  To modify c++ code in tensorflow, you only need to run the command below from step 3 to recompile jaxlib::

    python3 build/build.py --enable_cuda --dev_install --tf_path=$(pwd)/../third_party/tensorflow-alpa

Check Installation
------------------
You can check the installation by running the following test script.

.. code:: bash

  cd alpa
  ray start --head
  python3 tests/test_install.py

