ParaNum
=======
ParaNum automatically parallelizes your python numerical computing code and neural networks
with a simple decorator.


Install
=======
This repo depends on our private fork of jax and tensorflow.

- Step 1. Clone repos
  ```bash
  git clone git@github.com:merrymercy/model-parallel.git
  git clone git@github.com:merrymercy/jax.git
  git clone git@github.com:merrymercy/tensorflow-paranum.git
  ```
- Step 2. Install dependencies  
  - CUDA Toolkit: cuda and cudnn
  - Python packages:
    ```bash
    pip3 instll numpy, scipy, flax
    ```
  - ILP Solver:
    ```bash
    sudo apt install coinor-cbc
    pip3 install pulp
    ```
      
- Step 3. Build and install jaxlib
  ```bash
  cd jax
  export TF_PATH=~/tensorflow-paranum  # update this with your path
  python3 build/build.py --enable_cuda --dev_install --tf_path=$TF_PATH
  cd dist
  pip3 install -e .
  ```
- Step 4. Install Jax
  ```bash
  cd jax
  pip3 install -e .
  ```
- Step 5. Install ParaNum
  ```bash
  cd model-parallel
  pip3 install -e .
  ```

Note:
All installations are in development mode, so you can modify python code and it will take effect immediately.
To modify c++ code in tensorflow, you only need to run the command below in Step 3 to recompile jaxlib.
```
python3 build/build.py --enable_cuda --dev_install --tf_path=$TF_PATH
```

Organization
============
- `examples`: public examples
- `paranum`: the python interface of the library
- `playground`: private experimental scripts
- `tests`: unit tests

