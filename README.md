# Alpa
[**Documentation**](https://alpa-projects.github.io) |
[**Slack**](https://forms.gle/YEZTCrtZD6EAVNBQ7)

[![Build Jaxlib and Jax](https://github.com/alpa-projects/alpa/actions/workflows/build_jax.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/build_jax.yml)
[![CI](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml)

Alpa is a system for training large-scale neural networks.
Scaling neural networks to hundreds of billions of parameters has enabled dramatic breakthroughs such as GPT-3, but training these large-scale neural networks requires complicated distributed training techniques.
Alpa aims to automate large-scale distributed training with just a few lines of code.

The key capabilities of Alpa include:
💻 **Automatic Parallelization**. Alpa automatically parallelizes computational graphs with data, operator, and pipeline parallelism.
🚀 **Excellent Performance**. Alpa achieves linear scaling on training models with billions of parameters on distributed clusters.
✨ **Tight Integration with High-performance Deep Learning Ecosystem**. Alpa is backed by [Jax](https://github.com/google/jax), [XLA](https://www.tensorflow.org/xla) and [Ray](https://github.com/ray-project/ray)

## Quick Start
Use Alpa's decorator ``@parallelize`` to scale your single-device training code to distributed clusters.

```python
import alpa

# Parallelize the training step in Jax
@alpa.parallelize
def train_step(model_state, batch):
    def loss_func(params):
        out = model_state.forward(params, batch["x"])
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(model_state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now runs on your designated cluster
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```

Check out the [Alpa Documentation](https://alpa-projects.github.io) site for installation instructions, tutorials, examples, and more.

## More Information
- [Alpa paper](https://arxiv.org/pdf/2201.12023.pdf) (OSDI'22)
- [Google AI Blog](https://ai.googleblog.com/2022/05/alpa-automated-model-parallel-deep.html)

## Getting Involved
- Please read the [contributor guide](https://alpa-projects.github.io/developer/developer_guide.html) if you are interested in contributing to Alpa. 
- Please connect to Alpa contributors via the [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7).

## License
Alpa is licensed under the [Apache-2.0 license](https://github.com/alpa-projects/alpa/blob/main/LICENSE).
