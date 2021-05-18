# XLA Pipeline Marker Demo

To build and test:
~~~bash
mkdir build
cd build
cmake .. -Dpybind11_ROOT=$(pybind11-config --cmakedir)
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/targets/x86_64-linux/
make
cp *.so ..
cd ..
python test_pipeline_marker.py
~~~