mkdir -p build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3) -Dpybind11_ROOT=$(pybind11-config --cmakedir)
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/targets/x86_64-linux/include/:$CPATH
make
