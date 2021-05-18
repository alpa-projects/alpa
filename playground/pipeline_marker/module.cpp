#include<pybind11/pybind11.h>
#include"kernel.h"
#include<string>

namespace py = pybind11;

py::bytes createHelper(
    kernel::runtime::memoryManager &m, size_t size) {
    kernel::swapHelper h = kernel::swapHelper{size, &m};
    return pybind11::bytes(
        std::string(reinterpret_cast<const char*>(&h), sizeof(kernel::swapHelper))
        );
}

PYBIND11_MODULE(gpu_custom_call_test, m) {
    m.doc() = "pybind gpu custom call test";
    // TODO: more general, not only f32
    m.def(
        "offload_f32", [](){
            const char *name = "xla._CUSTOM_CALL_TARGET";
            return py::capsule((void *) &kernel::offloadToHost, name);
        }, "return a capsule"
    );
    m.def(
        "upload_f32", [](){
            const char *name = "xla._CUSTOM_CALL_TARGET";
            return py::capsule((void *) &kernel::uploadToDevice, name);
        }
    );
    py::class_<kernel::runtime::memoryManager>(m, "runtimeManager")
        .def(py::init<>())
        .def("getHelper", &createHelper)
        .def("freeAll", &kernel::runtime::memoryManager::freeAll)
        ;
}