Design and Architecture
=======================

This document aims to describe the architecture of Alpa and explain several core concepts and compilation passes introduced by Alpa at a high level.
This document is organized as follows: 

- An overview of Alpa's architecture, including core terms and componenents introduced by Alpa;
- Walking-through example to explain the workflow of Alpa.


You are recommended to read the the following materials as well:

- `Alpa research paper`_.
- `12-min introductory talk`_ about Alpa.

.. _Alpa research paper: https://arxiv.org/pdf/2201.12023.pdf 
.. _12-min introductory talk: https://youtu.be/Jqz34CV-UqU

Overview
--------

The picture below shows a high-level diagram of Alpa's architecture.

.. image:: alpa-arch.png
  :width: 550px


Like many existing machine learning compilers, Alpa parallelizes the ML computation in two steps: a compilation step, followed by a runtime step. 
In the compilation step, Alpa takes a model description and a device cluster as input and performs a few compilation passes and optimizations to generate  
a model-parallel execution plan, which is suitable for the model and cluster; Alpa also generates executables based on the training code and parallel execution plan.
In the runtime step, Alpa orchestrates the parallel execution of these executables on the cluster devices.

Compilation
^^^^^^^^^^^

Before we start introducing the compilation architecture, we bring in two important terms introduced by Alpa.
Unlike many existing distributed ML training systems, Alpa views existing ML parallelization approaches into two orthogonal categories: 
*intra-operator* and *inter-operator* parallelisms. They are distinguished by the fact that if they involve partitioning operators along any tensor axis. 
Some examples falling into the two categories are listed below:

- **Intra-op parallelism**: data parallelism, Megatron-LM's tensor model parallelism, operator parallelism such as those in ToFu and FlexFlow, etc.
- **Inter-op parallelism**: device placement, pipeline parallelism and its variants.

For a deeper dive into what these two classes of parallelism entail and the rationale behind this new videw, please read the article 
about [our rationale](archicture/parallelism-and-rationale). 

This new view of ML parallelization techniques is the core part that drives Alpa's design: Alpa unifies existing ML parallelization methods following this 
view by realizing them in a two-level hierarchy shown in the diagram. At the upper level, Alpa designs a set of algorithms and compilation passes, which we call 
*inter-op pass* to generate parallel execution plan corresponding to all inter-op parallelisms; at the lower level, Alpa designs another set of algorithms and 
compilatoin passes, which we call *intra-op pass*, to generate the parallel execution plan mapping to all intra-op parallelisms.

Alpa can guarantee the plan generated at each individual level is locally optimal. 
Once the two-level plans are generated, Alpa runs a third pass *runtime orchestration pass*. In this pass, Alpa applies the plans on the input computational graph, 
performs some post-processing, and finally compile the original, single-node graph into parallel executables; It then sends the parallel executables to devices on the cluster.


The following concepts are necessary to understand what each pass is precisely doing during compilatoin.

- **Device cluster**: Alpa runs on a cluster of compute devices, managed by Ray_. For example, a cluster of 8 AWS p3.16xlarge nodes, with 8 GPUs on each node, form a 8x8 device cluster, illustrated 
  in the figure below. We also call this device cluster *the cluster mesh*.

- **Device mesh**: Alpa's inter-op compilation pass will slice the cluster mesh into multiple groups of devices; Each group might contain a number of devices with high communication
  bandwidth. We call each group a device mesh. The figure below shows how a cluster mesh is sliced into 4 device meshes.

- **Worker**: Each device mesh might consist of partial or full devices from a single node (such as device mesh 4) or from multiple nodes. Alpa uses a worker to manage multiple devices from
  one node; hence a device mesh might contain multiple workers, each mapping to a process that manage multiple devices on a node. For example, mesh 1 in Figure 2 contains all devices from the first node
  in the cluster mesh, mesh 2 contains ....

- **Stage**: Alpa slices the input computational graph into multiple, adjacent subgraphs. We call each subgraph a stage. 

- **Resharding**: TODO


Inter-op Pass
#############

Inter-op pass slices the computational graph into multiple stages and the cluster mesh into multiple smaller device meshes; it then assigns each stage to a mesh. 
Alpa generates the slicing and assignment scheme optimally using a dynamic programming algorithm to minimize the inter-op parallel execution latency.

Intra-op Pass
#############
Intra-op pass looks at each <stage, mesh> pair, and generates the optimal intra-op parallelism execution plan for this stage to run on its assigned mesh.


Runtime Orchestratoin Pass
##########################


.. _XLA: https://www.tensorflow.org/xla
.. _GSPMD: https://arxiv.org/pdf/2105.04663.pdf

These three compilation passes are implemented on top of XLA_ and GSPMD_, 
which additionally perform some other necessary compilation passes and optimizations to improve single-device execution performance.


Runtime
^^^^^^^
Alpa implements a runtime_ to orchestrate the inter-op parallel execution of different stages on these meshes.
For each stage, Alpa uses the GSPMD runtime to parallelize its execution on its assigned device mesh, following the intra-op parallelism execution plan generated by the intra-op pass.

.. _Ray: https://github.com/ray-project/ray
.. _MLP: tutorial/getting_started
.. _worker: https://github.com/alpa-projects/alpa/blob/main/alpa/device_mesh.py#L64
.. _runtime: https://github.com/alpa-projects/alpa/blob/main/alpa/pipeline_parallel/decentralized_distributed_runtime.py


Next, we will walk through the process of how the single-node code of an MLP, such as the MLP_ in the tutorial  is converted to a distributed version by Alpa.

Work-through Example: Distributing an MLP
-----------------------------------------

