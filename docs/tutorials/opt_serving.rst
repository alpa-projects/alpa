Serving OPT-175B using Alpa
===========================

This tutorial provides guides to setup a serving system to serve the largest available pretrained language model OPT-175B.


As a serving system, Alpa provides the following unique advantages:

- **Support commodity hardware**: With Alpa, you can serve OPT-175B using your in-house GPU cluster, without needing the latest generations of A100 80GB GPUs nor fancy InfiniBand connections -- no hardware constraints!

- **Flexible parallelism strategies**: Alpa will automatically figure out the appropriate model-parallelism strategies based on your cluster setup.


In this example, we use Alpa to serve the open-source OPT model, supporting all sizes ranging from 125M to 175B.
Specifically, Alpa provides:

- A **backend** to perform model-parallel distributed inference for the large OPT models;

- A **web frontend** to collect and batch inference requests from users.

.. note::

    The trained OPT model weights can be obtained from `Metaseq download page <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_. Usages of
    the pretrained model weights are subject to their `license <https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md>`_ .

.. note::

    You will need at least 350GB memory to to serve the OPT-175B model. You can also follow this guide to setup a serving system to serve smaller versions of OPT,
    such as OPT-66B, OPT-30B, etc. Pick an appropriate size from `OPT weight release page <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>`_ based on
    your available resources.


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


Run Generation in Command Line
------------------------------

For a small model that can fit into one GPU, such as the OPT-125M, we can run single-GPU generation using either PyTorch backend or JAX backend.
For examples:

1. Run generation using the 125M OPT model with PyTorch/HuggingFace backend:

.. code:: bash

    cd benchmark
    python3 benchmark_text_gen.py --model facebook/opt-125m --path [PATH_TO_WEIGHT]

2. Run generation using the OPT-125M model with JAX backend in debug model to output the generated text：

.. code:: bash

    python3 benchmark_text_gen.py --model jax/opt-125m --path [PATH_TO_WEIGHT] --debug

3. Run model-parallel generation using the 2.7B model with Alpa:

.. code:: bash

    ray start --head
    python3 benchmark_text_gen.py --model alpa/opt-2.7b --path [PATH_TO_WEIGHT] --debug

4. Run distributed generation with the 175B model using Alpa; Note you will need >350Gb total GPU memory in the entire cluster to successfully run the inference.

.. code:: bash

    # Remember to start ray on the entire cluster before running the generation
    python3 benchmark_text_gen.py --model alpa/opt-175b --path [PATH_TO_WEIGHT] --debug

Launch a web server to serve the OPT models
-------------------------------------------

Launch the web server:

.. code:: bash

    # Serve the OPT-175B model at port 10001
    python3 interactive_hosted.py --model alpa/opt-175b --port 10001 --path [PATH_TO_WEIGHT]

Then open ``https://[IP-ADDRESS]:10001`` in your browser to try out the model!


License
-------

The Use of the OPT pretrained weights are subject to the `Model Licence <https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md>`_ by Metaseq.
