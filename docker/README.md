# Build Jaxlib-alpa wheels using Docker
We provide a Docker image to build the Alpa-modified Jaxlib wheel inside a container. The modified Jaxlib contains 
the intra-op parallelism ILP solver and other related implementations, which are necessary to Alpa.

If you are developing new features for Alpa, follow the [build guide](https://alpa-projects.github.io/install/from_source.html#install-from-source) 
to build from source.

## Steps
First, please figure out the CUDA and Python version you want to use to build JaxLib. Current supported versions are below:
- CUDA: 11.1, 11.2, 11.3
- Python: 3.7, 3.8, 3.9

Supposed we want to build the Jaxlib-alpa with CUDA 11.1 and Python 3.8.
### Build the docker image
```python
# build a folder to save the output wheels
cd alpa/docker && mkdir -p dist

# build the image using chosen CUDA version

docker build -t build-jaxlib-image -f build_jaxlib.Dockerfile . --build-arg JAX_CUDA_VERSION=11.1
```

### Build the wheels inside a container
```python
# create a subfolder for the specific wheel version. 
mkdir -p dist/cuda111

# build the wheel in a container using the selected Python and CUDA versions
docker run --tmpfs /build:exec --rm -v $(pwd)/dist:/dist build-jaxlib-image 3.8 cuda 11.1 master

# Move the output wheel
mv -f dist/*.whl dist/cuda111/
```
Check out the wheel under the folder ``alpa/build/dist/cuda111/``.