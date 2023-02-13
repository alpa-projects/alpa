=====================================
Code Structure of the Intra-op Solver
=====================================

The specific code of the intra-op solver (a.k.a auto-sharding) is scattered
in various files of the project.
This page contains some pointers to key components of the intra-op solver and
help you navigate the complicated code base.

.. note::

  All the links below are based on alpa v0.2.2


Key Pointers
============

- Main entrance:
   - python entrance (``run_auto_sharding_pass``): https://github.com/alpa-projects/alpa/blob/181de4f5577a72c9b30525ed3da09e5b2138cc2c/alpa/shard_parallel/auto_sharding.py#L172
   - c++ entrance: https://github.com/alpa-projects/tensorflow-alpa/blob/cd865615b9b518bc507fbdc71dc44c7cc76618ac/tensorflow/compiler/xla/service/spmd/auto_sharding.cc#L2124

- Where the possible sharding strategies are registred:
   - for matmul: https://github.com/alpa-projects/tensorflow-alpa/blob/cd865615b9b518bc507fbdc71dc44c7cc76618ac/tensorflow/compiler/xla/service/spmd/auto_sharding_dot_handler.cc#L327-L408
   - for elementwise operators: https://github.com/alpa-projects/tensorflow-alpa/blob/cd865615b9b518bc507fbdc71dc44c7cc76618ac/tensorflow/compiler/xla/service/spmd/auto_sharding.cc#L967-L1016

- Where the ILP solver is called:
   - c++ side: https://github.com/alpa-projects/tensorflow-alpa/blob/cd865615b9b518bc507fbdc71dc44c7cc76618ac/tensorflow/compiler/xla/service/spmd/auto_sharding.cc#L2259
   - python side: https://github.com/alpa-projects/alpa/blob/181de4f5577a72c9b30525ed3da09e5b2138cc2c/alpa/shard_parallel/auto_sharding.py#L588


How to Read and Learn the Code
==============================
.. _learn-intra-op-solver:

Run some simple examples
~~~~~~~~~~~~~~~~~~~~~~~~
You can run the unit tests under https://github.com/alpa-projects/alpa/tree/v0.2.2/tests/shard_parallel and set break points in the python entrance ``run_auto_sharding_pass``.
You can start from the most basic ones in ``test_basic.py``.

Inspect the sharding strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can print the HLO before and after the ``run_auto_sharding_pass``.


How to Debug
============
- Set global environment variable ``ALPA_DEBUG_PRINT_AS_STRATEGY=1``. This will print the choosen sharding strategy for each instruction and edge costs in a prettier way.
- Check batch dim analysis https://github.com/alpa-projects/tensorflow-alpa/blob/721260d122f096040762b2d226b37e8ab23f74b8/tensorflow/compiler/xla/service/spmd/auto_sharding_util.cc#L857
