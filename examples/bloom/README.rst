===========================
Serving Bloom-176B using Alpa
===========================

This tutorial shows how to setup a serving system to serve the largest available pretrained language model `Bloom <https://huggingface.co/bigscience/bloom>`_.

This tutorial is best to be read in its rendered version on the `Alpa documentation page <https://alpa-projects.github.io/tutorials/opt_serving.html>`_.


Overview
========


Demo
====
Use huggingface/transformers interface and Alpa backend for distributed inference on a Ray cluster.

.. code:: python

  from transformers import AutoTokenizer
  from opt_serving.model.wrapper import get_model

  # Load the tokenizer. We have to use the 30B version because
  # other versions have some issues. The 30B version works for all OPT models.
  tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
  tokenizer.add_bos_token = False

  # Load the model
  model = get_model(model_name="bigscience/bloom",
                    device="cuda",
                    path="/home/ubuntu/bloom_weights/")

  # Generate
  prompt = "Paris is the capital city of"

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
  output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
  generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

  print(generated_string)


Requirements
============
1. Install Alpa following the `installation guide <https://alpa-projects.github.io/install.html>`_. You can either install by python wheel or build from source.

2. Install additional requirements for ``opt_serving``:

  .. code:: shell

    pip3 install transformers flask cython omegaconf

    # Install torch corresponding to your CUDA version, e.g., for CUDA 11.3:
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

3. Clone the ``alpa`` repo. If you install alpa by python wheel, please clone the alpa repo. If you install from source, you already did this step.

  .. code:: shell

    git clone git@github.com:alpa-projects/alpa.git

4. Install ``opt_serving`` package. Go to the examples folder and install the package.

  .. code:: shell

    cd alpa/examples
    pip3 install -e .


Get Alpa-compatible Bloom Weights
===============================
There are two ways to obtain Alpa-compatible OPT weights: converting the weights by yourself or downloading a copy of processed weights provided by the Alpa team.

.. _process-weights:

Convert weights into Alpa formats by yourself
---------------------------------------------
We provide detailed instructions below on how to convert the original OPT-175B weights into Alpa-compatible formats.
For processing other sizes of OPT (125M - 66B), you can skip Step 1 and start from :ref:`the latter part of Step 2<download-singleton>`.

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

.. _download-singleton:

  .. note::
    If you use Alpa to target smaller versions of OPT (125M, 350M, 1.3B, 2.7B, 6.7B, 13B, 30B), you can skip the above procedures
    and download the consolidated singleton checkpoint using the links below, then proceed to the next step.

      * `OPT-125M <https://huggingface.co/patrickvonplaten/opt_metaseq_125m/blob/main/model/restored.pt>`_
      * `OPT-350M <https://dl.fbaipublicfiles.com/opt/v1_20220502/350m/reshard.pt>`_
      * `OPT-1.3B <https://huggingface.co/patrickvonplaten/opt_metaseq_1300m/blob/main/model/restored.pt>`_
      * `OPT-2.7B <https://huggingface.co/patrickvonplaten/opt_metaseq_2700m/blob/main/model/restored.pt>`_
      * `OPT-6.7B <https://huggingface.co/patrickvonplaten/opt_metaseq_6700m/blob/main/model/restored.pt>`_
      * `OPT-13B <https://huggingface.co/patrickvonplaten/opt_metaseq_13000m/blob/main/model/restored.pt>`_
      * `OPT-30B <https://huggingface.co/patrickvonplaten/opt_metaseq_30000m/blob/main/model/restored.pt>`_


3. Convert the single checkpoint into Alpa-compatible formats
    Alpa ingests weights simply from numpy formats. Use the script `step_3_convert_to_numpy_weights.py <https://github.com/alpa-projects/alpa/tree/main/examples/opt_serving/scripts/step_3_convert_to_numpy_weights.py>`_ to convert the
    single checkpoint into numpy formats:

    .. code:: shell

      python3 step_3_convert_to_numpy_weights.py --ckpt_path PATH_TO_SAVE_CHECKPOINT --output-folder OUTPUT_PATH


    The weights will be saved at the folder ``OUTPUT_PATH`` as specified in the command.

  .. note::

    The above script also requires 350GB free disk space to write the numpy-formatted weights.


Download Alpa-compatible weights
--------------------------------
Alternatively, we provide links to download the preprocessed 125M, 2.7B, 30B model weights below.

 * `OPT-125M weights <https://drive.google.com/file/d/1Ps7DFD80wNO7u2t39YCYcBX-9XwypGzl/view?usp=sharing>`_
 * `OPT-2.7B weights <https://drive.google.com/file/d/1ayIaKRhxF9osZWgcFG-3vSkjcepSWdQd/view?usp=sharing>`_
 * `OPT-30B weights <https://drive.google.com/file/d/1_MBcgwTqHFboV0JkGWR03AOHusrxcHlu/view?usp=sharing>`_

Due to Meta's license on the OPT-175B model, we are not able to provide public links for downloading the preprocessed OPT-175B weights.
If you need the weights for other model sizes but have trouble following :ref:`the guide<process-weights>` to perform the conversion by yourself,
please join `Alpa slack <https://forms.gle/YEZTCrtZD6EAVNBQ7>`_ to request a copy from the Alpa developer team.


Run and Benchmark Generation in the Command Line
================================================

The code of this tutorial is under `examples/opt_serving <https://github.com/alpa-projects/alpa/tree/main/examples/opt_serving>`_.

- Run generation using the 125M model with PyTorch/HuggingFace backend on a single GPU:

  .. code:: shell

    cd opt_serving/benchmark
    python3 benchmark_text_gen.py --model facebook/opt-125m --debug


- Run generation using the 125M model with JAX backend on a single GPU:

  .. code:: shell

    python3 benchmark_text_gen.py --model jax/opt-125m --path [PATH_TO_WEIGHT] --debug


- Run model-parallel generation using the 2.7B model with Alpa on multiple GPUs:

  .. code:: shell

    # Start ray on the node
    ray start --head

    python3 benchmark_text_gen.py --model alpa/opt-2.7b --path [PATH_TO_WEIGHT] --debug


- Run distributed generation using the 175B model with Alpa on a cluster of GPU nodes.
  Note you will need >350GB total GPU memory in the entire cluster to successfully run the inference.

  Before running the command below, start Ray on the cluster following `this guide <https://docs.ray.io/en/latest/cluster/cloud.html#manual-cluster>`_. You can check the cluster status by ``ray status``. You should be able to see all GPUs and all nodes in the output.

  .. code:: shell

    python3 benchmark_text_gen.py --model alpa/opt-175b --path [PATH_TO_WEIGHT] --debug

Launch a Web Server to Serve the Models
===========================================


Code structure
==============


License
=======
The use of the OPT pretrained weights is subject to the `license: bigscience-bloom-rail-1.0 <https://huggingface.co/spaces/bigscience/license>`_ by BigScience.
