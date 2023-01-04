===============
Developer Guide
===============

Code Organization
=================

The code in alpa's repository is organized as follows:
  - `alpa <https://github.com/alpa-projects/alpa/tree/main/alpa>`__: the python source code of Alpa
  - `benchmark <https://github.com/alpa-projects/alpa/tree/main/benchmark>`__: benchmark scripts
  - `build_jaxlib <https://github.com/alpa-projects/alpa/tree/main/build_jaxlib>`__: build scripts for Alpa's version of jaxlib
  - `docs <https://github.com/alpa-projects/alpa/tree/main/docs>`__: documentation and tutorials
  - `examples <https://github.com/alpa-projects/alpa/tree/main/examples>`__: public examples
  - `playground <https://github.com/alpa-projects/alpa/tree/main/playground>`__: experimental scripts
  - `tests <https://github.com/alpa-projects/alpa/tree/main/tests>`__: unit tests
  - `third_party <https://github.com/alpa-projects/alpa/tree/main/third_party>`__: third party repos

In addition, Alpa maintains a tensorflow fork. This is because alpa modifies the XLA compiler, whose code
is hosted in the tensorflow repo.

- `tensorflow-alpa <https://github.com/alpa-projects/tensorflow-alpa>`__: The TensorFlow fork for Alpa.
  The c++ source code of Alpa mainly resides in ``tensorflow/compiler/xla/service/spmd``.


Contribute to Alpa
==================
Please submit a `pull request <https://github.com/alpa-projects/alpa/compare>`__ if you plan to contribute to Alpa.

Formatting and Linting
----------------------
We follow `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`__.

Install yapf and pylint via:

.. code-block:: bash

    pip install yapf==0.32.0 pylint==2.14.0

Use the following script to format the code and check linting errors:

.. code-block:: bash

    ./format.sh

Unit Testing
------------
Every New feature should come with a unit test. See this `README.md <https://github.com/alpa-projects/alpa/tree/main/tests/README.md>`_ on how to run tests locally.

Updating submodule tensorflow-alpa
----------------------------------
Alpa repo stores a commit hash of the submodule tensorflow-alpa, so git knows which version of tensorflow-alpa should be used.
However, commands like ``git pull`` do not update the submodule to the latest stored commit. You need to additionally use the commands below.

.. code-block:: bash

    git submodule update --init --recursive

Contributing to submodule tensorflow-alpa
-----------------------------------------
If you want to contribute code to tensorflow-alpa, you can follow the steps below

1. Contributors send a pull request to tensorflow-alpa.
2. Maintainers review the pull request and merge it to tensorflow-alpa.
3. Contributors send a pull request to alpa. The pull request should update the stored hash commit of the submodule and other modifications to alpa if necessary.
4. Maintainers review the pull request and merge it to alpa.


Building jaxlib from source
---------------------------

In case you need to modify and debug Alpa's jaxlib level of changes, you will need to build jaxlib from source, where Alpa added its auto sharding passes in SPMD.

We have a python script for you `build.py` that will help to download bazel and do the building process:

.. code-block:: bash

    cd alpa/build_jaxlib
    python3 build/build.py --enable_cuda --dev_install --bazel_options=--override_repository=org_tensorflow=$TF_PATH

TF_PATH here points to the tensorflow-alpa folder such as `alpa/third_party/tensorflow-alpa`

.. warning::

    Building Alpa with gcc 9.4.0 might lead to build issues, which happens to be a common default gcc verison on public cloud containers.

In this case you will need to update your gcc version to another one, such as gcc-10 to re-build by executing `build.py` above again with clean bazel cache.

.. code-block:: bash

    sudo apt install gcc-8 g++-8 gcc-9 g++-9 gcc-10 g++-10
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10
    # Clean bazel cache
    rm -rf ~/.cache/bazel