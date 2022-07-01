Performance Tuning Guide
========================

This tutorial provides some tips for performance tuning and debugging.

Choosing Parallel Methods
-------------------------
Alpa relies on analyses of primitives tensor operators to perform auto-parallelization.
These analyses can be tricky for complicated computational graphs, especially those with many indexing/slicing/concatenating operators.
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

   Try to combine pipeline parallelism and shard parallelism. 

   1. Layer construction. You can use the automatic layer construction by using ``layer_option=AutoLayerOption(layer_num=...)``.
      You can try a few choices of the ``layer_num`` argument and see the performance. The best choice of this value depends on the number of nodes in your cluster and the number of repetitive blocks in your model.
      You can also do layer construction manually by using ``layer_option="manual"`` and ``mark_pipeline_boundary``
   2. Number of micro batches. The ``num_micro_batches`` also affects the performance a lot. You can fix a large global batch size and try a few choices of ``num_micro_batches``.

Reducing Runtime Overhead
-------------------------
Alpa uses a single-controller architecture. In this architecture, the user script runs on a CPU driver and sends commands to GPU workers. Users can just think of the device cluster as a single big device.

This architecture is easier to use and understand but can potentially lead to significant runtime overhead. The runtime overhead includes:

- Send commands to launch the computation on workers
- Send data to workers
- Fetch data from workers

To reduce the overhead, we should avoid frequent synchronization, so we can overlap the computation with runtime scheduling.
Printing or accessing the value of a ``DistributedArray`` is a case of synchronization because we have to fetch the data from workers' GPUs to the driver's CPU.
However, accessing metadata such as `shape` and `dtype` does not need synchronization because the metadata is stored on the driver.
