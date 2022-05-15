mkdir -p build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3) -Dpybind11_ROOT=$(pybind11-config --cmakedir)
# Set `CUDA_HOME` to /usr/local/cuda only if it's not already set
export CUDA_HOME="${CUDA_HOME:=/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include/:$CPATH
make
