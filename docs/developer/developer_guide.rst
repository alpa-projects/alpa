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

Unittest
--------
Every New feature should come with a unittest.

`How to run unit test locally <https://github.com/alpa-projects/alpa/tree/main/tests/README.md>`_
