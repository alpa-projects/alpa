'''
The @parallelize Decorator
================================================
The :code:`@parallelize` decorator calls the :code:`parallelize_callable` function in `api.py`, which caches the function to be parallelized to speed up recompilation the next time the same function with the same parameters is called. :code:`parallelize_callable` then calls a dedicated function for each parallel strategy. We will inspect the :code:`3d_parallel` strategy, implemented in :code:`three_d_parallel_callable` in `three_d_parallel.py`. 

'''

################################################################################
# Parameters
# -------------------------------------------
# The decorator function takes in as parameters:
#
# * :code:`fun`, the function to be parallelized;
# * :code:`in_tree`, the tree-like structure of the input object; 
# * :code:`out_tree_thunk`, the tree-like structure of the output object;
# * :code:`donated_invars`, an array flagging which variables can handle donation;
# * :code:`batch_invars`, an array flagging which variables are batched inputs;
# * :code:`devices`, the workers to run this parallel program over;
# * :code:`memory_budget_per_device`, the memory limit of each worker;
# * :code:`*avals`,  the shapes and data types of all input objects.

################################################################################
# Overview
# -------------------------------------------
# This tutorial will walk through each step of the :code:`@parallelize` function, from the input function :code:`fun` to the output XLA stages. As an overview, the input function is first converted to Jaxpr, then split into layers and clustered into stages, before finally being outputted as an XLA file. The diagram below illustrates the process (and the respective functions) that the input function will pass through.
#
# .. figure:: /gallery/tutorials/images/Diagram1.png
#    :alt: Stage Construction Flowchart
#
#    A flowchart of stage construction. From an input function, the Jaxpr representation is created, then sliced and sharded into different layers and stages, until an xla output is produced.


################################################################################
# Representing Jaxpr
# -------------------------------------------
# The input function is passed to :code:`trace_jaxpr_with_micro_batch` to return a ClosedJaxpr object representing the input function in Jaxpr. A ClosedJaxpr object stores the Jaxpr along with any global constants and literals. 
#
# .. figure:: /gallery/tutorials/images/Diagram2.png
#    :alt: Computational Diagram
#
#    Computational diagram of a neural network. Each minibatch of input is fed into the forward layer to generate the loss, which is accumulated across all minibatches to form the overall loss. Each minibatched loss is fed through the backward layer to acquire the gradient, which is also accumulated across all minibatches. Finally, the gradients are used to perform weight updates. 
#
# .. figure:: /gallery/tutorials/images/Diagram3.png
#    :scale: 60%
#    :alt: Recursive Computational Diagram
#
#    Recursive diagram of the same process.

################################################################################
# Splitting into compute_grad and apply_grad
# -------------------------------------------
# Next, the Jaxpr representation is split into the :code:`compute_grad` and the :code:`apply_grad` sections in :code:`split_compute_grad_and_apply_grad`. Each neural network training loop is structured as input → forward pass → output → backwards pass → gradient → gradient update. Or, in other words,
# 
# .. code-block::
#
#     for each minibatch of input: 
#         forward → val
#         backward → gradient
#         /* gradient accumulation: */
#         previous_val = val + previous_val
#         previous_grad = gradient + previous_grad
#     for each weight:
#         update weight with previous_grad
#
# We want to separate all steps before the gradient update with all steps that occur after, so that we can identify where to compute gradient accumulation and where to distribute the final gradient to each of the weights for weight updates. 
#
# The last :code:`compute_grad` layer is marked by the user through a named dummy identity layer. The :code:`split_compute_grad_and_apply_grad` function searches for this layer and splits the Jaxpr representation into two at this point. 


################################################################################
# Gradient accumulation
# -------------------------------------------
# The gradient accumulation computation is added after that with :code:`compute_grad_to_accumulate_grad`. Gradient accumulation sums up all the gradients computed with each minibatch input, and averages it out across all minibatches. The loss output is also accumulated and averaged out across all minibatches to be returned. 


################################################################################
# Layer Clustering
# -------------------------------------------
# The Jaxpr of the input function is then split into layers (which are either user-defined or automatically marked). Donation mappings are initialized by mapping the output variables of each donatable layer to the input variables of the next layer. 
#
# Next, layers are clustered together into pipeline stages with a dynamic programming algorithm in :code:`cluster_layers_and_slice_mesh`. The most cost-effective method of grouping layers in devices is computed and corresponding layers are clustered together. To minimize communication costs, the same layers that are clustered together must be clustered together in the :code:`apply_gradient` part of the function as well, since the same device would contain both the existing weights and the gradients needed without needing to communicate with other devices extensively. Thus, :code:`process_apply_gradient` is used to cluster the :code:`apply_gradient` layers into their corresponding pipeline stages. 


################################################################################
# Sharding Stages
# -------------------------------------------
# After the pipeline stages are constructed, a call to :code:`shard_each_stage` takes in all stages and variables, the available device meshes, and their corresponding communication costs and memory budgets to separate out different segments of the input matrix within each operator (intra-operator parallelism) and assign them to their respective device meshes.


################################################################################
# Generating Runtime Object
# -------------------------------------------
# Finally, all physical meshes are launched in parallel, and a distributed runtime object is created that runs the training loop passed in. 
