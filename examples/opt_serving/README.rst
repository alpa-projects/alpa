===========================
Serving OPT-175B using Alpa
===========================

This tutorial shows how to setup a serving system to serve the largest available pretrained language model `OPT-175B <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_.


Overview
========
As a serving system, Alpa offers the following unique advantages:

* **Support commodity hardware**: With Alpa, you can serve OPT-175B using your in-house GPU cluster, without needing the latest generations of A100 80GB GPUs nor fancy InfiniBand connections -- no hardware constraints!

* **Flexible parallelism strategies**: Alpa will automatically figure out the appropriate model-parallel strategies based on your cluster setup.

In this example, we use Alpa to serve the open-source OPT model, supporting all sizes ranging from 125M to 175B. Specifically, Alpa provides:
* A backend to perform efficient model-parallel inference for the large OPT models.
* A web frontend to collect and batch inference requests from users.

.. note::

  The pre-trained OPT model weights can be obtained from [Metaseq](https://github.com/facebookresearch/metaseq), subject to their license.

.. note:: 

  You will need at least 350GB memory to to serve the OPT-175B model. For example, you can use 4 x AWS p3.16xlarge instances, which provide 4 (instance) x 8 (GPU/instance) x 16 (GB/GPU) = 512 GB memory.
    
  You can also follow this guide to setup a serving system to serve smaller versions of OPT, such as OPT-66B, OPT-30B, etc. 
    
  Pick an appropriate size from `OPT weight downloading page <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_ based on your available resources.

Demo
====
Use huggingface/transformers interface and Alpa backend for distributed inference on a Ray cluster.

.. code:: python

  from transformers import AutoTokenizer
  from examples.opt_serving.model.wrapper import get_model

  # Load the tokenizer. We have to use the 30B version because
  # other versions have some issues. The 30B version works for all OPT models.
  tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
  tokenizer.add_bos_token = False

  # Load the model
  model = get_model(model_name="alpa/opt-2.7b",
                    device="cuda",
                    path="/home/ubuntu/opt_weights/")

  # Generate
  prompt = "Paris is the capital city of"

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
  output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
  generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

  print(generated_string)


Requirements
============
1. Install Alpa following the `installation guide <https://alpa-projects.github.io/install.html>`_.

2. Install additional requirements for serving:

  .. code:: shell

    pip3 install transformers flask cython omegaconf

    # Install torch corresponding to your CUDA version, e.g., for CUDA 11.3:
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


3. Compile several cython files for faster data processing:

  .. code:: shell
  
    cd examples/opt_serving && bash build.sh
  
  
Get Alpa-compatible OPT Weights
===============================
There are two ways to obtain Alpa-compatible OPT weights: proprocessing the weights by yourself or downloading a copy of preprocessed weights provided by the Alpa team. 

.. _process-weights:

Preprocess weights into numpy formats by yourself
-------------------------------------------------
We provide detailed instructions below on how to convert the original OPT-175B weights into Alpa-compatible formats. You can follow the same procedures to get Alpa-compatible weights for other model sizes.

0. Download and verify the weights
  First, download Metaseq's original 992 shards, verify the `MD5 of the shards <https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/assets/opt175b_md5sum_shards.csv>`_  of the shards, and put the shards under a folder, say, ``PATH_TO_992_SHARDS/``.

1. Consolidate the weights from 992 shards into one single checkpoint
  Use the script `step_1_consolidate_992_shards_to_singleton.py <scripts/step_1_consolidate_992_shards.py>`_ to consolidate the 992 shards into one single checkpoint.
  
  ..code:: shell
  
    cd scripts && python step_1_consolidate_992_shards_to_singleton.py --read-prefix PATH_TO_992_SHARDS/checkpoint_last --save_prefix PATH_TO_SAVE_CHECKPOINT
  
  The consolidated checkpoint will be save at ``PATH_TO_SAVE_CHECKPOINT``.
  
  .. note::
  
    The above script will require a peak memory (RAM) usage as large as twice of the model size. 
    For example, if you are performing consolidation for the 175B model, it will approximately have a peak memory usage of 175B x 2 bytes x 2 = 700GB. 
    Please make sure you RAM can support such a peak memory usage.
    
  .. note::
  
    The above script will save the consolidated checkpoint at ``PATH_TO_SAVE_CHECKPOINT``, hence will require at least 350GB disk space.
  

2. Convert the model into Alpa-compatible formats
  
    Alpa ingests weights simply from numpy formats. Use the script `step_2_convert_to_numpy_weights.py <scripts/step_2_convert_to_numpy_weights.py>`_ to convert the 
    single checkpoint into numpy formats:
    
    ..code:: shell
  
      cd scripts && python step_2_convert_to_numpy_weights.py --ckpt_path PATH_TO_SAVE_CHECKPOINT --output-folder PATH_TO_SAVE_CHECKPOINT
    
    The weights will be saved at the folder specified as ``PATH_TO_SAVE_CHECKPOINT``.
    
  .. note::
  
    The above script also require at least 350GB disk space to write the numpy-formatted weights.
    
    
Download Alpa-compatible weights
--------------------------------
Alternatively, we provide links to download the preprocessed 125M, 2.7B, 30B model weights below. 

 * `OPT-125M weights <https://drive.google.com/file/d/1Ps7DFD80wNO7u2t39YCYcBX-9XwypGzl/view?usp=sharing>`_
 * `OPT-2.7B weights <https://drive.google.com/file/d/1ayIaKRhxF9osZWgcFG-3vSkjcepSWdQd/view?usp=sharing>`_ 
 * `OPT-30B weights <https://drive.google.com/file/d/1_MBcgwTqHFboV0JkGWR03AOHusrxcHlu/view?usp=sharing>`_
   
Due to Meta's license on the OPT-175B model, we are not able to provide public links for downloading the preprocessed OPT-175B weights. 
If you need the weights for other model sizes but have trouble following the guide :ref:`Preprocess weights into numpy formats by yourself<process-weights>`, please join `Alpa slack <https://forms.gle/YEZTCrtZD6EAVNBQ7`_ to request a copy from the Alpa developer team. 


Run and Benchmark Generation in the Command Line
================================================

Run generation using the 125M model with PyTorch/HuggingFace backend:

.. code:: shell

  cd benchmark
  python3 benchmark_text_gen.py --model facebook/opt-125m


Run generation using the 125M model with JAX backend in debug model to see the generated text:

.. code:: shell

  python3 benchmark_text_gen.py --model jax/opt-125m --path [PATH_TO_WEIGHT] --debug


Run model-parallel generation using the 2.7B model with Alpa:

.. code:: shell

  ray start --head

  python3 benchmark_text_gen.py --model alpa/opt-2.7b --path [PATH_TO_WEIGHT] --debug
Run distributed generation with the 175B model using Alpa. Note you will need >350GB total GPU memory in the entire cluster to successfully run the inference.


.. code:: shell

  # Remember to start Ray on the entire cluster before running the generation
  python3 benchmark_text_gen.py --model alpa/opt-175b --path [PATH_TO_WEIGHT] --debug

Launch a Web Server to Serve the OPT Models
===========================================

Launch the web server:

.. code:: shell

  # Serve the OPT-175B model at port 10001
  python3 interactive_hosted.py --model alpa/opt-175b --port 10001 --path [PATH_TO_WEIGHT]


Then open `·https://[IP-ADDRESS]:10001`` in your browser to try out the model!

Code structure
==============

* `examples/opt_serving/benchmark <benchmark>`_: Benchmark scripts for generation in the command line.
* `examples/opt_serving/dataset <dataset>`_: Data loaders for serving. 
* `examples/opt_serving/service <service>`_: Model serving web server.
* `examples/opt_serving/generator.py <generator.py>`_: Backend for web server.
* `examples/opt_serving/interactive_hosted.py <interactive_hosted.py>`_: Web server entry point.

License
=======
The use of the OPT pretrained weights are subject to the [Model License](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md) by Metaseq.
