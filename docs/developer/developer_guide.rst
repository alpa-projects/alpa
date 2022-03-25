===============
Developer Guide
===============

Code Organization
=================
The code in this repository is organized as follows:
  - `alpa <https://github.com/alpa-projects/alpa/tree/main/alpa>`__: the python source code of Alpa
  - `benchmark <https://github.com/alpa-projects/alpa/tree/main/benchmark>`__: benchmark scripts
  - `docs <https://github.com/alpa-projects/alpa/tree/main/docs>`__: documentation and tutorials
  - `examples <https://github.com/alpa-projects/alpa/tree/main/examples>`__: public examples
  - `playground <https://github.com/alpa-projects/alpa/tree/main/playground>`__: experimental scripts
  - `tests <https://github.com/alpa-projects/alpa/tree/main/tests>`__: unit tests

There are two additional repositories:

- `tensorflow-alpa <https://github.com/alpa-projects/tensorflow-alpa>`__: The TensorFlow fork for Alpa.
  The c++ source code of Alpa mainly resides in ``tensorflow/compiler/xla/service/spmd``.
- `jax-alpa <https://github.com/alpa-projects/jax-alpa>`__: The JAX fork for Alpa, in which we do not change any
  functionality, but modify the building scripts to make building with tensorflow-alpa easier.



Contribute to Alpa
==================
Please submit a `pull request <https://github.com/alpa-projects/alpa/compare>`__ if you plan to contribute to Alpa.


Formatting and Linting
----------------------
We follow `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`__.

Install prospector and yapf via:

.. code-block:: bash

    pip install prospector yapf


Use yapf to automatically format the code:

.. code-block:: bash

    ./format.sh

Then use prospector to run linting for the folder ``alpa/``:

.. code-block:: bash

    prospector alpa/
