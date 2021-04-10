ParaNum
=======
ParaNum automatically parallelizes your python numerical computing code and neural networks
with a simple decorator.


Install
=======
This repo depends on our private fork of jax and tensorflow.

1. Clone repos
```bash
git clone git@github.com:merrymercy/model-parallel.git
git clone git@github.com:merrymercy/jax.git
git clone git@github.com:merrymercy/tensorflow-paranum.git
```
2. Install dependencies
The dependencies include numpy, scipy, cython, cuda, cudnn
3. Install Jax
```bash
export TF_PATH=~/tensorflow-paranum  # update this with your path
python3 build/build.py --enable_cuda --dev-install
cd dist
pip3 install -e .
```
4. Install ParaNum
```bash
pip3 install -e .
```

Organization
============
- `examples`: public examples
- `paranum`: the python interface of the library
- `playground`: private experimental scripts

