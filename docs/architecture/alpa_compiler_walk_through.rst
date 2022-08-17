.. _Alpa Compiler Walk-Through:

==========================
Alpa Compiler Walk-Through
==========================

This document provides a walk-through of the compiler part of Alpa.

.. note::
  This document is based on the workflow as in `this commit <https://github.com/alpa-projects/alpa/tree/388594f>`__. While some specific details might not be the same as in the latest version, the general idea should be the same.


Starting from an arbitrary JAX function (i.e., computational graph) of a neural network training step, Alpa’s overall workflow includes the following steps:

1. **Layer construction:** Cluster different operators in the
   computational graph into a sequential list of pipeline layers.
2. **Stage construction:** Cluster the pipeline layers into pipeline
   stages and assign each stage a subset of devices for pipeline
   execution (i.e., inter-operator parallelism).
3. **Auto sharding:** Figure out how to shard each operator within each pipeline stage on its corresponding devices with SPMD parallelism (i.e., intra-operator parallelism).

Let’s start with the following code snippet:

.. code:: python

   class ManualPipelineMLPModel(nn.Module):
       hidden_dim: int

       @nn.compact
       def __call__(self, x):
           x = nn.Dense(features=self.hidden_dim * 4)(x)
           x = nn.relu(x)
           x = nn.Dense(features=self.hidden_dim)(x)
           x = nn.relu(x)
           # Use this boundary marker to separate the model into two stages.
           alpa.mark_pipeline_boundary()
           x = nn.Dense(features=self.hidden_dim * 4)(x)
           x = nn.relu(x)
           x = nn.Dense(features=self.hidden_dim)(x)
           x = nn.relu(x)
           return x

   @alpa.parallelize(method=alpa.PipeshardParallel(num_micro_batches=16,
                                                   layer_option="manual"))
   def manual_pipeline_train_step(state, batch):

       def loss_func(params):
           out = state.apply_fn(params, batch["x"])
           loss = jnp.mean((out - batch["y"])**2)
           return loss
       # Use `alpa.grad` here to slice the forward/backward stages and the
       # gradient update stage
       grads = alpa.grad(loss_func)(state.params)
       new_state = state.apply_gradients(grads=grads)
       return new_state

Compared to original JAX/Flax, this code snippet additionally calls ``alpa.mark_pipeline``, ``alpa.parallelize``, and ``alpa.grad``. Below, we will show how Alpa uses these functions and decorators to compile the original single device computational graph into a distributed version.

Layer Construction
==================

The first transformation we perform is in ``alpa.grad``
(`link <https://github.com/alpa-projects/alpa/blob/388594f00d1ee0fe4dc0d51c2d8567da13226fdf/alpa/api.py#L213>`__)
for layer construction. It is a thin wrapper of the original ``jax.grad`` in JAX,
which additionally performs the following tasks:

1. Process pipeline markers to form forward pipeline layers.
2. Call the original ``jax.grad``. We directly use JAX's autograd to map
   the forward layers to the backward layers.
3. Mark all the gradients with a special marker so that we can perform
   gradient accumulation for them.
4. Mark all the operators after the gradient computation as the
   gradient update phase.

We form the pipeline layers by inserting pipeline markers into the JAX
automatically or manually with user annotations.
``layer_option="manual"`` in the code example above indicates that we
are inserting the markers manually.

The definition of pipeline markers can be found in
`primitive_def.py <https://github.com/alpa-projects/alpa/blob/388594f00d1ee0fe4dc0d51c2d8567da13226fdf/alpa/pipeline_parallel/primitive_def.py>`__.
We define a new JAX primitive ``pipeline_p`` and an XLA custom call
``pipeline_marker``. All these markers behave exactly the same as an
identity function that returns all the input
arguments.

We distinguish between ``start`` and ``end`` markers. The ``start``
marker captures all the inputs to a pipeline layer, and the ``end`` marker captures the outputs. To preserve the forward/backward
stage mapping, we set the gradient of a ``start`` marker to be an ``end``
marker, and the gradient of an ``end`` to be a ``start``.

A complete pipeline layer has the following structure:

::

   marked_inputs = pipeline_marker[type="start"] layer_inputs
   ...
   layer_outputs = some_jax_operator marked_inputs
   ...
   marked_outputs = pipeline_marker[type="end"] layer_outputs

Note that all the inputs of the JAX operators within the pipeline layer
should take the marked inputs or the intermediate results within the
layer. All the outputs of the layer will be marked by the ``end``
marker.

In the manual case, we provide a simpler API that doesn’t require two
markers for a stage and the users do not need to specify the input and
output variables. Instead, the users only need to call
``alpa.mark_pipeline_boundary`` at the boundary of two pipeline layers.
The ``layer_level_jaxpr_transformation`` function
(`link <https://github.com/alpa-projects/alpa/blob/388594f00d1ee0fe4dc0d51c2d8567da13226fdf/alpa/pipeline_parallel/layer_construction.py#L424-L432>`__)
will transform it to the above form.

**Note:** Alpa can also perform rematerialization (i.e., gradient checkpointing) at these pipeline stage
boundaries. See these functions:
`link <https://github.com/alpa-projects/alpa/blob/388594f00d1ee0fe4dc0d51c2d8567da13226fdf/alpa/pipeline_parallel/layer_construction.py#L475-L547>`__.

Stage Construction
==================

The transformed function with layer markers is then transformed by
``@alpa.parallelize``. The most important option of
``@alpa.parallelize`` is ``method``, which specifies which type of
parallelism to use. Here we set it to ``alpa.PipeshardParallel``,
indicating that we are using both pipeline parallelism (inter-operator
parallelism) and SPMD-shard parallelism (intra-operator parallelism).

``@alpa.parallelize`` transforms the original function to a
``ParallelizedFunc``. ``ParallelizedFunc`` is a Python class that
behaves like the original function but with some additional methods.
``ParallelizedFunc`` flattens the input arguments, and will compile the
JAX function according to the ``method``. In our case, it eventually
calls ``compile_pipeshard_executable()``
`here <https://github.com/alpa-projects/alpa/blob/388594f00d1ee0fe4dc0d51c2d8567da13226fdf/alpa/pipeline_parallel/compile_executable.py#L42-L50>`__,
which transforms the input as follows:

1. ``compile_pipeshard_executable`` first traces the original function
   to JAXPR. Note that we trace the function with both full batch size
   and the smaller micro-batch size for gradient accumulation. Then we
   call into ``compile_pipeshard_executable_internal``.

2. ``split_compute_grad_and_apply_grad`` splits the ``apply_grad`` part
   from the rest of the function. There is a special transformation for
   the case where a single parameter ``x`` is used in multiple pipeline
   layers ``l1(x)``, ``l2(x)``, ... For example in language models' tied-embedding layer, the embedding matrix is used by both the first
   and the last stage. In this case, the backward pass of JAX will
   generate some equations that are not captured by pipeline markers to
   calculate the gradient to ``x``: ``grad_x = grad_l1_x + grad_l2_x``.
   We move these kinds of equations to the ``apply_grad`` part and let
   each layer perform gradient accumulation separately.

3. ``compute_grad_to_accumulate_grad`` transforms the original
   a ``compute_grad`` JAXPR that only computes gradient to
   an ``accumulate_grad`` JAXPR that performs gradient accumulation. More
   specifically, the structure of ``accumulate_grad`` is shown in the following pseudo-code:

   .. code:: python

      def accumulate_grad(compute_grad_inputs, accumulated_grad):
          grad = compute_grad(compute_grad_inputs)
        accumulated_grad += grad
          return accumulated_grad

   Note that the ``+=`` above is only correct when the gradients can be
   summed up. When the output is per input data (e.g., inference
   output), we use ``concat`` instead of ``+=``. The analysis of which
   operator to use is done in ``_get_full_batch_apply_grad`` by
   comparing full-batch and micro-batch codes.

4. ``slice_closed_jaxpr_by_full_pipeline_marks`` slices the
   ``accumulate_grad`` JAXPR into many pipeline layers.

5. ``mark_missing_vars_in_backward_computation_pipeline_marks``. When
   JAX derives the backward JAXPR, the backward layer will directly use
   the intermediate results of the forward layer instead of adding it
   to the backward layer’s start pipeline marker. This function fixes
   this issue. In addition, it removes all ``Literal`` in start markers
   and all ``DropVar`` in end markers.

6. ``cluster_layers_and_slice_mesh`` performs stage construction. it
   clusters different pipeline layers into pipeline stages, slice the
   compute cluster represented as a 2D device mesh into many submeshes,
   and assign each stage a submesh. Right now, a forward layer and its
   corresponding backward layer will always be on the same submesh. See
   the full automatic algorithm in `the Alpa paper <https://arxiv.org/abs/2201.12023>`__.

7. ``process_apply_gradient`` splits the single ``apply_grad`` JAXPR into
   #submeshes parts, each part processes the gradient updates and
   optimizer states related to the variables on a specific submesh.

8. ``create_donation_mapping`` and ``split_donate_invars``: Process
   donated invars for each pipeline stage, and also add donation variables for gradient accumulation.

Auto Sharding
=============

Then, in ``shard_each_stage`` we run the auto-sharding pass for each
pipeline stage. Because we include distributed compilation for
different stages to accelerate the compilation, the code is nested here.
Specifically, the following two functions are the two most important ones:

1. In ``generate_sharded_xla_computations_arguments``
   (`code <https://github.com/alpa-projects/alpa/blob/388594f00d1ee0fe4dc0d51c2d8567da13226fdf/alpa/pipeline_parallel/computation.py#L827>`__),
   we concat the JAXPRs of all stages on a submesh (which typically
   include forward/backward/update of a single stage) and compile it to
   an ``HLOModule``.
2. Then we call ``run_auto_sharding_pass``
   (`code <https://github.com/alpa-projects/alpa/blob/388594f00d1ee0fe4dc0d51c2d8567da13226fdf/alpa/shard_parallel/auto_sharding.py#L183>`__),
   which eventually calls ``RunAutoShardingPass`` we wrote in XLA
   (`code <https://github.com/alpa-projects/tensorflow-alpa/blob/445b4588a93c01a155053d6b77f4621b5f704a68/tensorflow/compiler/xla/service/spmd/alpa_compile.cc#L89-L90>`__).
   This XLA function:

   1. First run a subset of XLA passes before SPMD partitioner.
   2. Then we run the Alpa ``AutoSharding`` pass
      (`code <https://github.com/alpa-projects/tensorflow-alpa/blob/445b4588a93c01a155053d6b77f4621b5f704a68/tensorflow/compiler/xla/service/spmd/auto_sharding.cc>`__)
      that automatically annotate the graph with GSPMD annotations.
   3. Then run the ``SliceAutoShardedStages`` pass
      (`code <https://github.com/alpa-projects/tensorflow-alpa/blob/445b4588a93c01a155053d6b77f4621b5f704a68/tensorflow/compiler/xla/service/spmd/slice_auto_sharded_stages.cc>`__)
      that slices the concated stages back to individual stages, and
      return these stages back to Python.

The result of ``shard_each_stage`` will be a list of SPMD sharded
pipeline stages. Then the whole pipeline and sharding execution schedule
will be summarized and organized via a ``PipelineInstEmitter``
(`code <https://github.com/alpa-projects/alpa/blob/388594f00d1ee0fe4dc0d51c2d8567da13226fdf/alpa/pipeline_parallel/compile_executable.py#L221-L233>`__).
The result ``pipeshard_config`` will be sent to the runtime to be
executed.

.. note::
  To debug and visualize each step, you can debug via simply adding print instructions to the JAXPR in Python or the HLO in XLA.
