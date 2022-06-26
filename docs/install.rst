Install Alpa
============

This page provides instructions to install alpa from Python wheels or from source. The minimum supported python version is 3.7.

Prerequisites
-------------

Regardless of installing from wheels or from source, there are a few prerequisite packages:

1. CUDA toolkit:

  Follow the official guides to install `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ and `cuDNN <https://developer.nvidia.com/cudnn>`_.
  Alpa requires CUDA >= 11.1 and  cuDNN >= 8.0.5.

2. Install the ILP solver used by Alpa:

    If you have sudo permission, use

      .. code:: bash

        sudo apt install coinor-cbc

    Otherwise, please try to install via `binary <https://projects.coin-or.org/Cbc#DownloadandInstall>`_ or `conda <https://anaconda.org/conda-forge/coincbc>`_.

3. Update pip version and install cupy:

  .. code:: bash

    # Update pip
    pip3 install --upgrade pip

    # Use your own CUDA version. Here cuda-cuda114 means cuda 11.4
    pip3 install cupy-cuda114

  Then, check whether your system already has NCCL installed.

  .. code:: bash

    python3 -c "from cupy.cuda import nccl"

  If it prints nothing, then NCCL has already been installed.
  Otherwise, follow the printed instructions to install NCCL.

.. _install-from-wheels:

Methods
-------
Choose one of the methods below.

Method 1: Install from Python Wheels
####################################

Alpa provides wheels for the following CUDA (cuDNN) and Python versions:

- CUDA (cuDNN): 11.1 (8.0.5), 11.2 (8.1.0), 11.3 (8.2.0)
- Python: 3.7, 3.8, 3.9

If you need to use other CUDA, cuDNN, or Python versions, please follow the next section to :ref:`install from source<install-from-source>`.

1. To install from wheels, first install Alpa:

  .. code:: bash

    pip3 install alpa

2. Then install the Alpa-modified Jaxlib from our `self-hosted PyPI server <http://169.229.48.123:8080/simple/>`_,
   and make sure that the jaxlib version corresponds to the version of the existing CUDA and cuDNN installation you want to use.
   You can specify a particular CUDA and cuDNN version for jaxlib explicitly via:

  .. code:: bash

    pip3 install --trusted-host 169.229.48.123 -i http://169.229.48.123:8080/simple/ jaxlib==0.3.5+cuda{cuda_version}.cudnn{cudnn_version}

  For example, to install the wheel compatible with CUDA >= 11.1 and cuDNN >= 8.0.5, use the following command:

  .. code:: bash

    pip3 install --trusted-host 169.229.48.123 -i http://169.229.48.123:8080/simple/ jaxlib==0.3.5+cuda111.cudnn805

  You can see all available wheel versions we provided at our `PyPI index <http://169.229.48.123:8080/simple/jaxlib/>`_.

.. note::

  As of now, Alpa modified the original jaxlib at the version ``jaxlib==0.3.5``. Alpa regularly rebases the official jaxlib repository to catch up with the upstream.
  If you need features from newer versions of jaxlib, please open an issue at the `Alpa GitHub Issue Page <https://github.com/alpa-projects/alpa/issues>`_.


.. _install-from-source:

Method 2: Install from Source
#############################

1.  Clone repos

  .. code:: bash

    git clone --recursive git@github.com:alpa-projects/alpa.git

2. Install Alpa

  .. code:: bash

    cd alpa
    pip3 install -e ".[dev]"  # Note that the suffix `[dev]` is required to build custom modules.

3. Build and install jaxlib

  .. code:: bash

    cd alpa/build_jaxlib
    python3 build/build.py --enable_cuda --dev_install --tf_path=$(pwd)/../third_party/tensorflow-alpa
    cd dist

    pip3 install -e .


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

[Optional] PyTorch Frontend
-------------------------------------

While Alpa is mainly designed for Jax, Alpa also provides an experimental PyTorch frontend.
Alpa supports PyTorch models that meet the following requirements:

1. No input-dependent control flow
2. No weight sharing

To enable Alpa for PyTorch, install the following dependencies:

  .. code:: bash

    # Install nightly version of torch and torchdistx
    pip3 uninstall -y torch torchdistx
    pip install torch torchdistx --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu

    # Build functorch from source
    git clone https://github.com/pytorch/functorch
    cd functorch/
    python3 setup.py install

Please look at ``tests/torch_frontend/test_simple.py`` for usage examples.

Troubleshooting
---------------

Using Alpa on Slurm
###################
Since Alpa relies on Ray to manage the cluster nodes, Alpa can run on a Slurm cluster as long as Ray can run on it.
If you have trouble running Alpa on a Slurm cluster, we recommend to follow `this guide <https://docs.ray.io/en/latest/cluster/slurm.html>`__ to setup Ray on Slurm and make sure simple Ray examples
can run without any problem, then move forward to install and run Alpa in the same environment.

Common issues of running Alpa on Slurm include:

- The Slurm cluster has installed additional networking proxies, so XLA client connections time out. Example errors can be found in `this thread <https://github.com/alpa-projects/alpa/issues/452#issuecomment-1134260817>`_.
  The slurm cluster users might need to check and fix those proxies on their slurm cluster and make sure processes spawned by Alpa can see each other.

- When launching a Slurm job using ``SRUN``, the users do not request enough CPU threads or GPU resources for Ray to spawn many actors on Slurm.
  The users need to adjust the value for the argument ``--cpus-per-task`` passed to ``SRUN`` when launching Alpa. See `Slurm documentation <https://slurm.schedmd.com/srun.html>`_ for more information.

You might also find the discussion under `Issue #452 <https://github.com/alpa-projects/alpa/issues/452>`__ helpful.

Jaxlib, Jax, Flax Version Problems
##################################
Alpa is compatible with the following Jaxlib, Jax, and Flax versions:
- Jax==0.3.5
- Flax==0.4.1
- Alpa-modified Jaxlib distributed at `self-hosted PyPI <http://169.229.48.123:8080/simple/>`_ or compiled from source.

However, sometimes the users might have installed other versions of Jax-based neural network libraries, such as Flax or Optax in their environment, an incompatible version of
Jaxlib or Jax will be automatically installed by pip, and the following error might appear when importing alpa:

.. code:: bash

  >>> import alpa
    ......
    RuntimeError: jaxlib version 0.3.7 is newer than and incompatible with jax version 0.3.5. Please update your jax and/or jaxlib packages

Make sure your jax version is 0.3.5, Flax version is 0.4.1 by reinstalling them following:

.. code:: bash

  pip3 install jax==0.3.5
  pip3 install flax==0.4.1

Make sure you install **Alpa-modified Jaxlib** by either using :ref:`our prebuilt wheels<install-from-wheels>` or :ref:`Install from Source<install-from-source>`.

Numpy Version Problems
#######################
If you start with a clean Python virtual environment and have followed the procedures in this guide strictly, you should not see problems about Numpy versions.

However, sometimes due to the installation of other Python packages, another version of numpy might be silently installed before compiling jaxlib,
and you might see numpy version errors similar to the following one when launching Alpa after installing from source:

.. code:: bash

  >>> python3 tests/test_install.py
    ......
    RuntimeError: module compiled against API version 0xf but this version of numpy is 0xd
    ImportError: numpy.core._multiarray_umath failed to import
    ImportError: numpy.core.umath failed to import
    2022-05-20 21:57:35.710782: F external/org_tensorflow/tensorflow/compiler/xla/python/xla.cc:83] Check failed: tensorflow::RegisterNumpyBfloat16()
    Aborted (core dumped)

This is because you have used a higher version of numpy when compiling jaxlib, but later used a lower version of numpy to run Alpa.

To address the problem, please first downgrade the numpy in your Python environment to ``numpy==1.20`` via ``pip install numpy==1.20``,
then follow the procedures in :ref:`install from source<install-from-source>` to rebuild and reinstall jaxlib.
Optionally, you can switch back to use the higher version of numpy (``numpy>=1.20``) to run Alpa and your other applications, thanks to numpy's backward compatibility.

See `Issue#461 <https://github.com/alpa-projects/alpa/issues/461>`_ for more discussion.
