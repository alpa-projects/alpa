# Alpa Docker
This directory contains Alpa's docker infrastructure. Alpa uses docker to provide environment to build and release Python wheels and to perform unit tests.
Most docker files in this directory depend on [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/).

Below we provide instructions on 
- How to build Alpa-modified jaxlib in a docker container 
- How to run Alpa in a docker container

More docker examples can be found in the directory of [Alpa CI/CD](../.github/workflows).

## Build Jaxlib-alpa wheels using Docker
We provide a Docker image to build the Alpa-modified jaxlib wheels inside a container. 


### Steps
First, figure out the CUDA and Python versions you want to use to build jaxlib. Current we support the following versions:
- CUDA: 11.1, 11.2, 11.3
- Python: 3.7, 3.8, 3.9

Suppose we want to build the jaxlib-alpa with CUDA 11.1 and Python 3.8.
#### Build the docker image
```python
# create a folder to save the output wheels
cd alpa/docker && mkdir -p dist

# build the image using the chosen CUDA version
docker build -t build-jaxlib-image -f build_jaxlib.Dockerfile . --build-arg JAX_CUDA_VERSION=11.1
```

#### Build the wheels inside a container
```bash
# create a subfolder for the specific wheel version. 
mkdir -p dist/cuda111

# build the wheel in a container using the selected Python and CUDA versions
docker run --tmpfs /build:exec --rm -v $(pwd)/dist:/dist build-jaxlib-image 3.8 cuda 11.1 master

# Move the output wheel
mv -f dist/*.whl dist/cuda111/
```
Check out the wheel under the folder ``alpa/build/dist/cuda111/``.

## Run Alpa in a docker container
You can run Alpa inside a docker container. Below are steps on how to run Alpa in a docker container in the interactive mode.

First, build a docker image based on the provided dockerfile:
```bash 
docker build -t run-alpa-image -f run_alpa.Dockerfile . 
```

Second, build a container from the image and enter the container's interactive shell:
```bash
docker run --gpus all --rm --shm-size=10.24gb -it run-alpa-image
```

Third, check alpa installation is correct:
```bash
conda activate alpa
# Start ray:
ray start --head
# Test Alpa can run correctly:
python -m alpa.test_install
```

Alternatively, you can skip the interactive shell, and pass commands or job scripts via the `docker run` command to the container.
