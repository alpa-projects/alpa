### Changes
- change `self.bpe` and `self.bpe.bpe` into `huggingface.GPT2Tokenizer`
- change backend from torch to JAX, disable torch.distributed initialization.

### TODO:
- remove `cfg`, make a simple version
- remove `metaseq.task`