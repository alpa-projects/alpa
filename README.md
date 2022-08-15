# Alpa
[**Documentation**](https://alpa-projects.github.io) |
[**Slack**](https://forms.gle/YEZTCrtZD6EAVNBQ7)

[![CI](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml)
[![Build Jaxlib](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml)

Alpa is a system for training and serving large-scale neural networks.

Scaling neural networks to hundreds of billions of parameters has enabled dramatic breakthroughs such as GPT-3, but training and serving these large-scale neural networks require complicated distributed system techniques.
Alpa aims to automate large-scale distributed training and serving with just a few lines of code.

The key features of Alpa include:  

💻 **Automatic Parallelization**. Alpa automatically parallelizes users' single-device code on distributed clusters with data, operator, and pipeline parallelism. 

🚀 **Excellent Performance**. Alpa achieves linear scaling on training models with billions of parameters on distributed clusters.

✨ **Tight Integration with Machine Learning Ecosystem**. Alpa is backed by open-source, high-performance, and production-ready libraries such as [Jax](https://github.com/google/jax), [XLA](https://www.tensorflow.org/xla), and [Ray](https://github.com/ray-project/ray).


## 🌟Try Alpa-served OPT-175B!
Alpa provides a free, unlimited [OPT-175B](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT) text generation service. Try our service at [https://opt.alpa.ai/](https://opt.alpa.ai/) 
and share your [prompting results](examples/opt_serving/service/img.png)!

Join [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7) and let us know any new features you want!

## Quick Start
Use Alpa's decorator ``@parallelize`` to scale your single-device training code to distributed clusters.
```python
import alpa

# Parallelize the training step in Jax by simply using a decorator
@alpa.parallelize
def train_step(model_state, batch):
    def loss_func(params):
        out = model_state.forward(params, batch["x"])
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(model_state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now automatically runs on your designated cluster
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```

Check out the [Alpa Documentation](https://alpa-projects.github.io) site for installation instructions, tutorials, examples, and more.

## Installation
The quickest way to get started with Alpa is via pip. We push the latest Alpa wheels to the [PyPI index](https://pypi.org/project/alpa/#description) and the Alpa-modified jaxlib to 
our [GitHub-hosted index](https://alpa-projects.github.io/wheels.html).

```bash
# Install alpa
pip install alpa

# install alpa-jaxlib compatible with CUDA >= 11.1 and cuDNN >= 8.0.5,
pip install jaxlib==0.3.5+cuda111.cudnn805 -f https://alpa-projects.github.io/wheels.html

# You can install for other CUDA versions via:
pip install jaxlib==0.3.5+cuda{cuda_version}.cudnn{cudnn_version} -f https://alpa-projects.github.io/wheels.html
```
All supported CUDA and cuDNN versions are listed on the [index page](https://alpa-projects.github.io/wheels.html).

After installation, validate the installation works as expected:
```bash
# Start the ray cluster first
ray start --head
python -m alpa.test_install
```

You can also install Alpa from source for development or other CUDA versions. 
Follow this [detailed guide](https://alpa.ai/install.html#method-2-install-from-source) to install Alpa from source or troubleshooting if you meet any errors during the process.  

## Examples
Check the following list of examples to get started with training and serving large models.
- [Model parallelism on Mnist](https://github.com/alpa-projects/alpa/tree/main/examples/mnist)
- [Train large language models using both inter-operator and intra-operator parallelism](https://github.com/alpa-projects/alpa/tree/main/examples/language_model)
- [Serve OPT-175B model on your own cluster](https://github.com/alpa-projects/alpa/tree/main/examples/opt_serving)
- [Finetune OPT-175B language models](https://github.com/alpa-projects/alpa/tree/main/examples/opt_finetune)

## Learning more
- [Prof. Ion Stoica introduces the Alpa system](https://www.youtube.com/watch?v=qzYoMldlyoA)
- [Alpa OSDI 2022 paper](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)
- [Google AI blog](https://ai.googleblog.com/2022/05/alpa-automated-model-parallel-deep.html)
- [Alpa talk slides](https://docs.google.com/presentation/d/1CQ4S1ff8yURk9XmL5lpQOoMMlsjw4m0zPS6zYDcyp7Y/edit?usp=sharing)
- [Alpa ICML 2022 Big Model Tutorial slide deck](https://sites.google.com/view/icml-2022-big-model/home)
- [Alpa ICML 2022 Big Model Tutorial video recording](https://icml.cc/virtual/2022/tutorial/18440)

## Getting Involved
- Please read the [contributor guide](https://alpa-projects.github.io/developer/developer_guide.html) if you are interested in contributing to Alpa. 
- Connect to Alpa contributors via the [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7).

## License
Alpa is licensed under the [Apache-2.0 license](https://github.com/alpa-projects/alpa/blob/main/LICENSE).
