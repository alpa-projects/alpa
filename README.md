Alpa
=======
[**Documentation**](https://alpa-projects.github.io) |
[**Slack**](https://forms.gle/YEZTCrtZD6EAVNBQ7)


[![Build Jaxlib and Jax](https://github.com/alpa-projects/alpa/actions/workflows/build_jax.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/build_jax.yml)
[![Unittest](https://github.com/alpa-projects/alpa/actions/workflows/unittest.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/unittest.yml)

Alpa automatically parallelizes tensor computation graphs and runs them on a distributed cluster.

Quick Start
-----------

Use Alpa's single line API ``@parallelize`` to scale your single-node training code to distributed clusters, even though 
your model is much bigger than a single device memory.
```python
import alpa

@alpa.parallelize
def train_step(model_state, batch):
    def loss_func(params):
        out = model_state.forward(params, batch["x"])
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now automatically runs on your designated cluster.
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```

Check out the [Alpa Documentation](https://alpa-projects.github.io) site for installation instructions, tutorials, examples, and more.

More Information
----------------
- [Alpa paper](https://arxiv.org/pdf/2201.12023.pdf) (OSDI'22)
- [Blog]()

Contributing
------------
Please read the [contributor guide](https://alpa-projects.github.io/developer/developer_guide.html) if you are interested in contributing to Alpa. 
Please connect to Alpa contributors via the [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7).


License
-------
Alpa is licensed under the [Apache-2.0 license](https://github.com/alpa-projects/alpa/blob/main/LICENSE).
