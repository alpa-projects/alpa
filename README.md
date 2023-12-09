**Note: Alpa is not actively maintained currently. It is available as a research artifact. The core algorithm in Alpa has been merged into XLA and it is under maintenance. https://github.com/openxla/xla/tree/main/xla/hlo/experimental/auto_sharding**


<div align="center">
<img src="https://github.com/alpa-projects/alpa/blob/main/docs/logo/alpa-logo-cropped.png" alt="logo" width="250"></img>
<br></br>
</div>

[![CI](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml)
[![Build Jaxlib](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml)

[**Documentation**](https://alpa-projects.github.io) | [**Slack**](https://forms.gle/YEZTCrtZD6EAVNBQ7)

Alpa is a system for training and serving large-scale neural networks.

Scaling neural networks to hundreds of billions of parameters has enabled dramatic breakthroughs such as GPT-3, but training and serving these large-scale neural networks require complicated distributed system techniques.
Alpa aims to automate large-scale distributed training and serving with just a few lines of code.

The key features of Alpa include:  

💻 **Automatic Parallelization**. Alpa automatically parallelizes users' single-device code on distributed clusters with data, operator, and pipeline parallelism. 

🚀 **Excellent Performance**. Alpa achieves linear scaling on training models with billions of parameters on distributed clusters.

✨ **Tight Integration with Machine Learning Ecosystem**. Alpa is backed by open-source, high-performance, and production-ready libraries such as [Jax](https://github.com/google/jax), [XLA](https://www.tensorflow.org/xla), and [Ray](https://github.com/ray-project/ray).

## Serving
The code below shows how to use huggingface/transformers interface and Alpa distributed backend for large model inference.
Detailed documentation is in [Serving OPT-175B using Alpa](https://alpa-projects.github.io/tutorials/opt_serving.html).

```python
from transformers import AutoTokenizer
from llm_serving.model.wrapper import get_model

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
tokenizer.add_bos_token = False

# Load the model. Alpa automatically downloads the weights to the specificed path
model = get_model(model_name="alpa/opt-2.7b", path="~/opt_weights/")

# Generate
prompt = "Paris is the capital city of"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

print(generated_string)
```

## Training
Use Alpa's decorator ``@parallelize`` to scale your single-device training code to distributed clusters.
Check out the [documentation](https://alpa-projects.github.io) site and
[examples](https://github.com/alpa-projects/alpa/tree/main/examples) folder
for installation instructions, tutorials, examples, and more.

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

## Learning more
- [Papers](docs/publications/publications.rst)
- [Google AI blog](https://ai.googleblog.com/2022/05/alpa-automated-model-parallel-deep.html)
- [OSDI 2022 talk slides](https://docs.google.com/presentation/d/1CQ4S1ff8yURk9XmL5lpQOoMMlsjw4m0zPS6zYDcyp7Y/edit?usp=sharing)
- [ICML 2022 big model tutorial](https://sites.google.com/view/icml-2022-big-model/home)
- [GTC 2023 talk video](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51337/)

## Getting Involved
- Connect to Alpa developers via the [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7).
- Please read the [contributor guide](https://alpa-projects.github.io/developer/developer_guide.html) if you are interested in contributing code.

## License
Alpa is licensed under the [Apache-2.0 license](https://github.com/alpa-projects/alpa/blob/main/LICENSE).
