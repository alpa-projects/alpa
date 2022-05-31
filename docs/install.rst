Install Alpa
============

Requirements
------------
- CUDA >= 11.0
- CuDNN >= 8.0.5
- python >= 3.7

Install from Wheels
-------------------
Alpa provides wheels for the following CUDA (CuDNN) and Python versions:

- CUDA (CuDNN): 11.0 (8.0.5), 11.1 (8.0.5), 11.2 (8.1.0)
- Python: 3.7, 3.8, 3.9

  .. code:: bash

    # Install Alpa
    pip3 install alpa

    # Install Alpa-modified Jaxlib. Make sure CUDA and CuDNN versions match the wheel version following:
    # jaxlib==0.3.5+cuda{cuda_version}.cudnn{cudnn_version}
    pip3 install --trusted-host 169.229.48.123 --index-url http://169.229.48.123:8080/simple/ jaxlib==0.3.5+cuda110.cudnn805

If you need other CUDA or Python versions, please follow the section :ref:`Install from Source<install-from-source>`.

.. _install-from-source:

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
    pip3 install -e .[dev]  # Note that the suffix `[dev]` is required to build custom modules.


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


Troubleshooting
---------------

Using Alpa on Slurm
###################
Since Alpa relies on Ray to manage the cluster nodes, Alpa can run on a Slurm cluster as long as Ray can run on it.
We recommend to follow `this guide <https://docs.ray.io/en/latest/cluster/slurm.html>`__ to setup Ray on Slurm and make sure simple Ray examples
can run without any problem, then move to install and run Alpa on the same environment.
You might also find the discussoin under `issue#452 <https://github.com/alpa-projects/alpa/issues/452>`__ helpful.

Jaxlib and Jax Version Mismatch
###############################
If you see the following error:

.. code:: bash



Numpy versions
##############


You might also find similar issues addressed in the `Alpa issue <https://github.com/alpa-projects/alpa/issues?q=is%3Aissue+is%3Aclosed>`__ list.

If you still have troubles with installing Alpa, please join `Alpa Slack <https://forms.gle/YEZTCrtZD6EAVNBQ7>`__ and ask questions.