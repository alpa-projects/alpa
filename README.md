Parax
=======
Parax automatically parallelizes your python numerical computing code and neural networks
with a simple decorator.


Install
=======
This repo depends on our private fork of jax and tensorflow.

- Step 1. Clone repos
  ```bash
  git clone git@github.com:parax-project/parax.git
  git clone git@github.com:parax-project/jax-parax.git
  git clone git@github.com:parax-project/tensorflow-parax.git
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
  cd jax-parax
  export TF_PATH=~/tensorflow-parax  # update this with your path
  python3 build/build.py --enable_cuda --dev_install --tf_path=$TF_PATH
  cd dist
  pip3 install -e .
  ```

- Step 4. Install Jax
  ```bash
  cd jax
  pip3 install -e .
  ```

- Step 5. Install Parax
  ```bash
  cd parax
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
- `parax`: the python interface of the library
- `playground`: private experimental scripts
- `tests`: unit tests

