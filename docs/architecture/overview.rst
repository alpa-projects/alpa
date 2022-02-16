Design and Architecture
=======================

This document aims to describe the architecture of Alpa and explain several core concepts and compilation passes introduced by Alpa at a high level.
This document is organized as follows: 

- An overview of Alpa's architecture, including core terms and componenents introduced by Alpa;
- Working-through example to explain the workflow of Alpa.


You are recommended to read the the following materials as well:

- `Alpa research paper`_.
- `12-min introductory talk`_ about Alpa.

.. _Alpa research paper: https://arxiv.org/pdf/2201.12023.pdf 
.. _12-min introductory talk: https://youtu.be/Jqz34CV-UqU

Overview
--------

The diagram below shows a high-level diagram of Alpa's architecture.

.. image:: alpa-arch.png
  :width: 550px


Like many existing machine learning compilers, Alpa parallelizes the ML computation in two phases: a compilation phase, followed by a runtime phase. 
In the compilation phase, Alpa takes a model description and a device cluster as input, and perform a few compilation passes and optimizations to generate  
model-parallel execution plans suitable for the model and cluster. Alpa also generate executables based on the training code and parallel execution plan.
In the runtime phase, Alpa orchestrates the parallel execution of these executables on the cluster devices.

Compilation
^^^^^^^^^^^

Before we start introducing the compilation architecture, we introduce two important terms introduced by Alpa.
Unlike many existing distributed ML training systems, Alpa views existing ML parallelization approaches into two orthogonal categories: 
*intra-operator* and *inter-operator* parallelisms. They are distinguished by the fact that if they involve partitioning operators along any tensor axis. 
Some examples falling into the two categories are listed below:

- Intra-op parallelism: data parallelism, Megatron-LM's tensor model parallelism, operator parallelism such as those in ToFu and FlexFlow, etc.
- Inter-op parallelism: device placement, pipeline parallelism and its variants.

For a deeper dive into what these two classes of parallelism entail and the rationale behind this new videw, please read the article 
about [our rationale](archicture/parallelism-and-rationale). 

This new view of ML parallelization techniques is the core part that drives Alpa's design: Alpa unifies existing ML parallelization methods following this 
view by realizing them in a two-level hierarchy shown in the diagram. At the upper level, Alpa designs a set of algorithms and compilation passes, which we call 
*inter-op pass* to generate parallel execution plan corresponding to all inter-op parallelisms; at the lower level, Alpa designs another set of algorithms and 
compilatoin passes, which we call *intra-op pass*, to generate the parallel execution plan mapping to all intra-op parallelisms.
Alpa can gurantee the plan generated at each individual level is locally optimal. 
Once the two-level plans are generated, Alpa runs a third pass *runtime orchestration pass*. In this pass, Alpa applies the plans on the input computational graph, 
performs some post-processing, and finally compile the original, single-node graph into parallel executables; It then sends the parallel executables to devices on the cluster.

.. _XLA: https://www.tensorflow.org/xla
.. _GSPMD: https://arxiv.org/pdf/2105.04663.pdf

These three compilation passes are implemented on top of XLA_ and GSPMD_, 
which additionally perform some other necessary compilation passes and optimizations to improve single-device execution performance.


Runtime
^^^^^^^

Alpa implements a new runtime to orchestrate the two-level parallelism over a cluster of compute devices. The following concepts are necessary to understand the 
Alpa runtime design:

- device cluster:
- device mesh:
- worker:
- stage:


Work-through Example: Distributing an MLP
-----------------------------------------

