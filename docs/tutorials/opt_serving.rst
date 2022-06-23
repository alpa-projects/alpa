Serving OPT-175B using Alpa
===========================

This tutorial provides guides to setup a serving system to serve the largest available pretrained language model OPT-175B.

As a serving system, Alpa provides the following unique advantages:

- **Support commodity hardware**: With Alpa, you can serve OPT-175B using your in-house GPU cluster, without needing the latest generations of A100 80GB GPUs nor fancy InfiniBand connections -- no hardware constraints!

- **Flexible parallelism strategies**: Alpa will automatically figure out the appropriate model-parallel strategies based on your cluster setup.

In this example, we use Alpa to serve the open-source OPT model, supporting all sizes ranging from 125M to 175B.
Specifically, Alpa provides:

- A **backend** to perform model-parallel distributed inference for the large OPT models;

- A **web frontend** to collect and batch inference requests from users.

.. note::

    The pre-trained OPT model weights can be obtained from `Metaseq download page <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_. Usages of
    the pretrained model weights are subject to their `license <https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md>`_ .

.. note::

    You will need at least 350GB memory to to serve the OPT-175B model. For example, you can use 4 x AWS p3.16xlarge instance, which provide 4 instance x 8 (GPU/instance) x 16 (GB/GPU) = 512 GB memory.
    You can also follow this guide to setup a serving system to serve smaller versions of OPT, such as OPT-66B, OPT-30B, etc.
    Pick an appropriate size from `OPT weight release page <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_ based on your available resources.

Demo
----
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
  prompt = "Paris is the capital city of "
  
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
  output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
  generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)
  
  print(generated_string)

Requirements
------------
1. Install Alpa following the `installation guide <https://alpa-projects.github.io/install.html>`_.

2. Install additional requirements for serving:

.. code:: bash

    pip3 install transformers flask cython

    # Install torch corresponding to your CUDA version, e.g., for CUDA 11.3:
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

3. Compile several cython files for faster data processing:

.. code:: bash

    cd examples/opt_serving && bash build.sh

Get OPT Weights
---------------
There are two ways you can obtain the pretrained OPT weights.

1. You can download the original OPT weights released by `Metaseq <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_,
then use our script `convert_to_numpy_weight.py <scripts/convert_to_numpy_weights.p>`_ to convert it into Alpa-compatible formats.

2. We provide links to download the preprocessed 125M and 2.7B model below. For other sizes of OPT, please join `Alpa slack <https://forms.gle/YEZTCrtZD6EAVNBQ7>`_ to request a copy from the Alpa developer team.

   - `OPT-125M weights <https://drive.google.com/file/d/1Ps7DFD80wNO7u2t39YCYcBX-9XwypGzl/view?usp=sharing>`_
   - `OPT-2.7B weights <https://drive.google.com/file/d/1ayIaKRhxF9osZWgcFG-3vSkjcepSWdQd/view?usp=sharing>`_

Run and Benchmark Generation in The Command Line
------------------------------------------------

- Run generation using the 125M model with PyTorch/HuggingFace backend:

.. code:: bash

    cd benchmark
    python3 benchmark_text_gen.py --model facebook/opt-125m --path [PATH_TO_WEIGHT]

- Run generation using the 125M model with JAX backend in debug model to output the generated text：

.. code:: bash

    python3 benchmark_text_gen.py --model jax/opt-125m --path [PATH_TO_WEIGHT] --debug

- Run model-parallel generation using the 2.7B model with Alpa:

.. code:: bash

    ray start --head
    python3 benchmark_text_gen.py --model alpa/opt-2.7b --path [PATH_TO_WEIGHT] --debug

- Run distributed generation with the 175B model using Alpa. Note you will need >350GB total GPU memory in the entire cluster to successfully run the inference.

.. code:: bash

    # Remember to start Ray on the entire cluster before running the generation
    python3 benchmark_text_gen.py --model alpa/opt-175b --path [PATH_TO_WEIGHT] --debug

Launch a Web Server to Serve the OPT models
-------------------------------------------

Launch the web server:

.. code:: bash

    # Serve the OPT-175B model at port 10001
    python3 interactive_hosted.py --model alpa/opt-175b --port 10001 --path [PATH_TO_WEIGHT]

Then open ``https://[IP-ADDRESS]:10001`` in your browser to try out the model!


License
-------

The use of the OPT pretrained weights are subject to the `Model Licence <https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md>`_ by Metaseq.
