# XLA Pipeline Marker Custom Call

To build the custom call for pipeline marker in XLA:
~~~bash
mkdir build
cd build
cmake .. -Dpybind11_ROOT=$(pybind11-config --cmakedir)
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/targets/x86_64-linux/include/:$CPATH
make
~~~
