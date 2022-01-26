Alpa
=======
Alpa automatically parallelizes your python numerical computing code and neural networks
with a simple decorator.


Requirements
============

```
CuDNN >= 8.1
CUDA >= 11.1
python >= 3.7
```

Install
=======
This repo depends on our private fork of jax and tensorflow.

- Step 1. Clone repos
  ```bash
  git clone git@github.com:alpa-project/alpa.git
  git clone git@github.com:alpa-project/jax-alpa.git
  git clone git@github.com:alpa-project/tensorflow-alpa.git
  ```

- Step 2. Install dependencies
  - CUDA Toolkit: cuda and cudnn
  - Python packages:
    ```bash
    pip3 install cmake numpy scipy flax numba pybind11 cupy-cuda111
    
    # Install the latest Ray, replace `LINK_TO_WHEEL` with links provided:
    # https://docs.ray.io/en/latest/installation.html#daily-releases-nightlies
    pip3 install -U ray[default]@LINK_TO_WHEEL.whl
    ```
  - ILP Solver:
    ```bash
    sudo apt install coinor-cbc glpk-utils
    pip3 install pulp
    ```

- Step 3. Build and install jaxlib
  ```bash
  cd jax-alpa
  export TF_PATH=~/tensorflow-alpa  # update this with your path
  python3 build/build.py --enable_cuda --dev_install --tf_path=$TF_PATH
  cd dist
  pip3 install -e .
  ```

- Step 4. Install jax
  ```bash
  cd jax-alpa
  pip3 install -e .
  ```

- Step 5. Install Parax
  ```bash
  cd alpa
  pip3 install -e .
  ```

- Step 6. Build XLA pipeline marker custom call (See [here](alpa/pipeline_parallel/xla_custom_call_marker/README.md))
  ```bash
  cd alpa/pipeline_parallel/xla_custom_call_marker
  bash build.sh
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
- `alpa`: the python interface of the library
- `playground`: private experimental scripts
- `tests`: unit tests


Formatting & Linting
============
Install prospector and yapf via:
```bash
pip install prospector yapf
```

Use yapf to automatically format the code:
```bash
./format.sh
```

Then use prospector to run linting for the folder ``alpa/``:
```bash
prospector alpa/
```

Style guidelines:
- We follow Google Python Style Guide: https://google.github.io/styleguide/pyguide.html.
- **Avoid using backslash line continuation as much as possible.** yapf will not format well lines with backslash line continuation.
  Make use of [Python’s implicit line joining inside parentheses, brackets and braces.](http://docs.python.org/reference/lexical_analysis.html#implicit-line-joining)
  If necessary, you can add an extra pair of parentheses around an expression.
