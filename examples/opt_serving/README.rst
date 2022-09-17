===========================
Serving OPT-175B using Alpa
===========================

This tutorial shows how to setup a serving system to serve the largest available pretrained language model `OPT-175B <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_.
You can also try a live demo at `Alpa-OPT Demo <https://opt.alpa.ai>`_.

Overview
========
As a serving system, Alpa offers the following unique advantages:

* **Designed for large models**: Cannot fit the model into a single GPU? Not a problem, Alpa is designed for training and serving big models like GPT-3.

* **Support commodity hardware**: With Alpa, you can serve OPT-175B using your in-house GPU cluster, without needing the latest generations of A100 80GB GPUs nor fancy InfiniBand connections -- no hardware constraints!

* **Flexible parallelism strategies**: Alpa will automatically figure out the appropriate model-parallel strategies based on your cluster setup and your model architecture.

In this example, we use Alpa to serve the open-source OPT model, supporting all sizes ranging from 125M to 175B. Specifically, Alpa provides:

* A distributed backend to perform efficient model-parallel inference for the large OPT models.

* A web frontend to collect and batch inference requests from users.

.. note::

  The pre-trained OPT model weights can be obtained from `Metaseq <https://github.com/facebookresearch/metaseq>`_, subject to their license.

.. note::

  You will need at least 350GB GPU memory on your entire cluster to serve the OPT-175B model.
  For example, you can use 4 x AWS p3.16xlarge instances, which provide 4 (instance) x 8 (GPU/instance) x 16 (GB/GPU) = 512 GB memory.

  You can also follow this guide to setup a serving system to serve smaller versions of OPT, such as OPT-66B, OPT-30B, etc.
  Pick an appropriate size from `OPT weight downloading page <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_ based on your available resources.

Demo
====
The code below shows how to use huggingface/transformers interface and Alpa distributed backend for large model inference.

.. code:: python

  from transformers import AutoTokenizer
  from opt_serving.model.wrapper import get_model

  # Load the tokenizer. We have to use the 30B version because
  # other versions have some issues. The 30B version works for all OPT models.
  tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
  tokenizer.add_bos_token = False

  # Load the model. Alpa automatically downloads the weights to the specificed path
  model = get_model(model_name="alpa/opt-2.7b", path="~/opt_weights/")

  # Generate
  prompt = "Paris is the capital city of"

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
  generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

  print(generated_string)


Requirements
============
1. Install Alpa following the `installation guide <https://alpa-projects.github.io/install.html>`_. You can either install by python wheel or build from source.

2. Install additional requirements for ``opt_serving``:

  .. code:: shell

    pip3 install transformers flask omegaconf

    # Install torch corresponding to your CUDA version, e.g., for CUDA 11.3:
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

3. Clone the ``alpa`` repo. If you install alpa by python wheel, please clone the alpa repo. If you install from source, you already did this step.

  .. code:: shell

    git clone git@github.com:alpa-projects/alpa.git

4. Install ``opt_serving`` package. Go to the examples folder and install the package.

  .. code:: shell

    cd alpa/examples
    pip3 install -e .


Convert Weights Format
======================

The weights of OPT 125M--66B models are publicly available. Huggingface hosts copies of these weights.
For OPT 125M--66B, you **do not need** to download or convert the weights manually. Alpa will automatically download the weights from huggingface to the given path if Alpa cannot find cached weights locally.

The weights of OPT-175B can be got from meta by filling a `request form <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_ .
You then need to manually convert the obtained weights into Alpa format.

Convert OPT-175B weights into Alpa formats
------------------------------------------
We provide detailed instructions below on how to convert the original OPT-175B weights into Alpa-compatible formats. You can skip this section if you only want to run smaller models.

  .. note::

    The procedures below for converting OPT-175B weights will take about 1 hour.

1. Download and verify the original weights
    First, download Metaseq's original OPT-175B weights in 992 shards, verify the `MD5 of each shard <https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/assets/opt175b_md5sum_shards.csv>`_ , and put the shards under a folder, say, ``PATH_TO_992_SHARDS/``.

2. Consolidate the weights from 992 shards into one single checkpoint
    Use the script `step_2_consolidate_992_shards_to_singleton.py <https://github.com/alpa-projects/alpa/tree/main/examples/opt_serving/scripts/step_2_consolidate_992_shards_to_singleton.py>`_ as:

  .. code:: shell

    python3 step_2_consolidate_992_shards_to_singleton.py --read-prefix [PATH_TO_992_SHARDS]/checkpoint_last --save-prefix [PATH_TO_SAVE_CHECKPOINT]

  The consolidated checkpoint will be saved at ``PATH_TO_SAVE_CHECKPOINT`` as specified in the command.

  .. note::

    The above script will require a peak memory (RAM) usage as large as twice of the model size.
    For example, if you are performing consolidation for the 175B model, it will approximately have a peak memory usage of 175B x 2 bytes x 2 = 700GB.
    Please make sure your RAM is sufficient to run the script without throwing an OOM exception.

  .. note::

    The above script will save the model weights as a single consolidated checkpoint at ``PATH_TO_SAVE_CHECKPOINT``, hence will require at least 350GB disk space available.

3. Convert the single checkpoint into Alpa-compatible formats
    Alpa ingests weights simply from numpy formats. Use the script `step_3_convert_to_numpy_weights.py <https://github.com/alpa-projects/alpa/tree/main/examples/opt_serving/scripts/step_3_convert_to_numpy_weights.py>`_ to convert the
    single checkpoint into numpy formats:

    .. code:: shell

      python3 step_3_convert_to_numpy_weights.py --ckpt-path PATH_TO_SAVE_CHECKPOINT --output-folder OUTPUT_PATH


    The weights will be saved at the folder ``OUTPUT_PATH`` as specified in the command.

  .. note::

    The above script also requires 350GB free disk space to write the numpy-formatted weights.

Converted weights for other models
----------------------------------
You do not need to download the weights manually for OPT 125M--66B. However, if you have trouble with the automatic downloading or huggingface. We also provide the converted weights for the following models.

  * `OPT-125M weights <https://drive.google.com/file/d/1Ps7DFD80wNO7u2t39YCYcBX-9XwypGzl/view?usp=sharing>`_
  * `OPT-2.7B weights <https://drive.google.com/file/d/1ayIaKRhxF9osZWgcFG-3vSkjcepSWdQd/view?usp=sharing>`_
  * `OPT-30B weights <https://drive.google.com/file/d/1_MBcgwTqHFboV0JkGWR03AOHusrxcHlu/view?usp=sharing>`_

Copy Weights to Multiple Nodes
------------------------------
If you want to run the model on multiple nodes, you can use one of the following methods to copy the weights to all nodes.

1. Put the weights under a shared network file system, so all nodes can access it.
2. Run the script first on a driver node. The driver node will download the weights to its local disk, but the script will fail later because worker nodes cannot access the weights.
   You can then manually copy all downloaded weights under ``path`` from the driver node to all worker nodes.

Run Generation in the Command Line
==================================

The code of this tutorial is under `examples/opt_serving <https://github.com/alpa-projects/alpa/tree/main/examples/opt_serving>`_.

- Run generation using the 125M model with PyTorch/HuggingFace backend on a single GPU:

  .. code:: shell

    python3 textgen.py --model facebook/opt-125m


- Run generation using the 125M model with JAX backend on a single GPU:

  .. code:: shell

    python3 textgen.py --model jax/opt-125m


- Run model-parallel generation using the 2.7B model with Alpa on multiple GPUs:

  .. code:: shell

    # Start ray on the node
    ray start --head

    python3 textgen.py --model alpa/opt-2.7b


- Run distributed generation using the 175B model with Alpa on a cluster of GPU nodes.
  Note you will need >350GB total GPU memory in the entire cluster to successfully run the inference.

  Before running the command below, start Ray on the cluster following `this guide <https://docs.ray.io/en/latest/cluster/cloud.html#manual-cluster>`_. You can check the cluster status by ``ray status``. You should be able to see all GPUs and all nodes in the output.

  .. code:: shell

    python3 textgen.py --model alpa/opt-175b

Launch a Web Server to Serve the OPT Models
===========================================

Launch the web server:

.. code:: shell

  # Serve the OPT-175B model at port 20001
  python3 interactive_hosted.py --model alpa/opt-175b --port 20001

Then open ``https://[IP-ADDRESS]:20001`` in your browser to try out the model!

Improving Generation Speed
==========================
Here are some tips for improving the generation speed.

1. Batching. Single sequence generation cannot fully utilize the GPU power.
   Applying batching can greatly boost the performace. See ``textgen.py`` for the usage.
2. Tune the ``encoder_chunk_sizes`` argument of ``get_model``.
   Alpa compiles multiple executables and uses these executables to encode a prompt chunk by chunk. This argument controls the possible chunk sizes. Depending on the length of your prompt, you can try different combinations. For example, if your prompt lengths are around 1000-1500, a good combination is ``[1, 256, 1024]``.
3. Tune parallelization strategy. If you are familiar with alpa, you can tune the ``method`` argument of ``alpa.parallelize`` and try different parallelization methods.

If you find the generation speed too slow and want to accelerate it, please join `Alpa slack <https://forms.gle/YEZTCrtZD6EAVNBQ7>`_ and tell us your use cases. We are acitvely working on improving the performance.

License
=======
The use of the OPT pretrained weights is subject to the `Model License <https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md>`_ by Metaseq.
