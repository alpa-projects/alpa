"""
Differences between alpa.parallelize, jax.pmap and jax.pjit
===========================================================

The most common tool for parallelization or distributed computing in jax is
`pmap <https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap>`_.
With several lines of code change, we can use ``pmap`` for data parallel
training. However, we cannot use ``pmap`` for model parallel training,
which is required for training large models with billions of parameters.

On the contrary, ``alpa.parallelize`` supports both data parallelism and
model parallelism in an automatic way. ``alpa.parallelize`` analyzes the
jax computational graph and picks the best strategy.
If data parallelism is more suitable, ``alpa.parallelize`` achieves the same
performance as ``pmap`` but with less code change.
If model parallelism is more suitable, ``alpa.parallelize`` achieves better performance
and uses less memory than ``pmap``.

In this tutorial, we are going to compare ``alpa.parallelize`` and ``pmap`` on two
workloads. A more detailed comparison among ``alpa.parallelize``, ``pmap``, and ``xmap``
is also attached at the end of the article.
"""

################################################################################
# When data parallelism is prefered
# ---------------------------------

# TODO

################################################################################
# When model parallelism is prefered
# ----------------------------------

# TODO

################################################################################
# Comparing ``alpa.parallelize``, ``pmap``, ``xmap``, and ``pjit``
# ----------------------------------------------------------------
# Besides ``pmap``, jax also provides
# `xmap <https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html>`_ and
# `pjit <https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html>`_
# for more advanced parallelization.
# The table below compares the features of ``alpa.parallelize``, ``pmap``, ``xmap``
# and ``pjit``. In summary, ``alpa.parallelize`` supports more parallelism
# techniques in a more automatic way.
#
# ================  ================ ==================== ==================== =========
# Transformation    Data Parallelism Operator Parallelism Pipeline Parallelism Automated
# ================  ================ ==================== ==================== =========
# alpa.parallelize  yes              yes                  yes                  yes
# pmap              yes              no                   no                   no
# xmap              yes              yes                  no                   no
# pjit              yes              yes                  no                   no
# ================  ================ ==================== ==================== =========
#
# .. note::
#   Operator parallelism and pipeline parallelism are two forms of model parallelism.
#   Operator parallelism partitions the work in a single operator and assigns them
#   to different devices. Pipeline parallelism partitions the computational
#   graphs and assigns different operators to different devices.
