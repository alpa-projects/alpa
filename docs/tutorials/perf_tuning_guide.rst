Performance Tuning Guide
========================

This tutorial provides some tips for performance tuning and debugging.

Choosing Parallel Methods
-------------------------
Alpa relies on analyses of primitives tensor operators to perform auto-parallelization.
These analyses can be tricky for complicated computational graphs, especially thoes with many indexing/slicing/concatenating operators.
To make sure Alpa can perform auto-parallelization correctly, we can start with simple parallel methods and gradually move to more advanced ones.

1. Start with the basic ``DataParallel``

   Try a small configuration of your model and run it with ``alpa.parallelize(func, method=alpa.DataParallel())``. This is used to make sure Alpa's basic analyses work correctly.
   If you see warnings like "Detect unexpected behaviors in the auto-sharding pass.", this means some analyses fail on the model. You can submit an issue with a reproducible script to report the error.

2. Try ``Zero2Parallel``

   Next, try the ``Zero2Parallel`` method. You are expected to see the allocation memory size is lower if you use optimizers with element-wise states, such as Adam. Note that `nvidia-smi` does not correctly report the memory usage, you can use ``executable.get_total_allocation_size()`` as we did in the :ref:`quick start <Alpa Quickstart>`.

3. Try ``ShardParallel``

   Next, try the more general ``ShardParallel`` method with different logical mesh shapes.

4. Enable gradient accumulation.

   Next, enable gradient accumulation by

   1. replace ``jax.grad`` and ``jax.value_and_grad`` with ``alpa.grad`` and ``alpa.value_and_grad``, respectively.
   2. set a larger global batch size and increase ``num_micro_batches`` accordingly.

5. Try ``PipeshardParallel``

   Try to combine pipeline parallelism with shard parallelism. 

   1. Layer construction. You can use the automatic layer construction by using ``@automatic_layer_construction``.
      You can try a few choices of the ``layer_num`` argument and see the performance. The best choice of this value depends on the number of nodes in your cluster and the number of repetitive blocks in your model.
      You can also do layer construciton manually by using ``@manual_layer_construction`` and ``mark_pipeline_boundary``
   2. Number of micro batches. The ``num_micro_batches`` also affects the performance a lot. You can fix a large global batch size and try a few choices.

Reducing Runtime Overhead
-------------------------
Alpa uses a single-controler architecture. This architecture is easier to use and understandbut can potentially lead to  runtime overhead

The user script runs on a CPU driver and sends
commands to GPU workers.
