cmake_minimum_required(VERSION 3.12)
project(xla_pipeline_marker LANGUAGES CXX CUDA)

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


add_library(xla_pipeline_marker MODULE
    module.cpp
)

set_target_properties(xla_pipeline_marker PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(xla_pipeline_marker PRIVATE kernelcu)
target_link_libraries(xla_pipeline_marker PRIVATE pybind11::module)
