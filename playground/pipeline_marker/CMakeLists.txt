cmake_minimum_required(VERSION 3.12)
project(gpu_custom_call_test LANGUAGES CXX CUDA)

set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 CONFIG)

add_library(kernelcu STATIC
    kernel.cu
)

set_target_properties(kernelcu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    # CUDA_SEPARABLE_COMPILATION ON
)


add_library(gpu_custom_call_test MODULE
    module.cpp
)

set_target_properties(gpu_custom_call_test PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(gpu_custom_call_test PRIVATE kernelcu)
target_link_libraries(gpu_custom_call_test PRIVATE pybind11::module)
