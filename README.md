# Alpa
[**Documentation**](https://alpa-projects.github.io) |
[**Slack**](https://forms.gle/YEZTCrtZD6EAVNBQ7)

[![CI](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml)
[![Build Jaxlib](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml)

Alpa is a system for training large-scale neural networks.
Scaling neural networks to hundreds of billions of parameters has enabled dramatic breakthroughs such as GPT-3, but training these large-scale neural networks requires complicated distributed training techniques.
Alpa aims to automate large-scale distributed training with just a few lines of code.

The key features of Alpa include:  

💻 **Automatic Parallelization**. Alpa automatically parallelizes users' single-device code on distributed clusters with data, operator, and pipeline parallelism. 

🚀 **Excellent Performance**. Alpa achieves linear scaling on training models with billions of parameters on distributed clusters.

✨ **Tight Integration with Machine Learning Ecosystem**. Alpa is backed by open-source, high-performance, and production-ready libraries such as [Jax](https://github.com/google/jax), [XLA](https://www.tensorflow.org/xla), and [Ray](https://github.com/ray-project/ray)

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

## More Information
- [Alpa paper](https://arxiv.org/pdf/2201.12023.pdf) (OSDI'22)
- [Google AI blog](https://ai.googleblog.com/2022/05/alpa-automated-model-parallel-deep.html)
- [Alpa talk slides](https://docs.google.com/presentation/d/1CQ4S1ff8yURk9XmL5lpQOoMMlsjw4m0zPS6zYDcyp7Y/edit?usp=sharing)
- [Big model tutorial](https://sites.google.com/view/icml-2022-big-model/home) (ICML'22)

## Getting Involved
- Please read the [contributor guide](https://alpa-projects.github.io/developer/developer_guide.html) if you are interested in contributing to Alpa. 
- Connect to Alpa contributors via the [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7).

## License
Alpa is licensed under the [Apache-2.0 license](https://github.com/alpa-projects/alpa/blob/main/LICENSE).
