#include<pybind11/pybind11.h>
#include"kernel.h"
#include<string>

namespace py = pybind11;

PYBIND11_MODULE(gpu_custom_call_test, m) {
    m.doc() = "pybind gpu custom call test";
    // TODO: more general, not only f32
    m.def(
        "pipeline_marker", [](){
            const char *name = "xla._CUSTOM_CALL_TARGET";
            return py::capsule((void *) &kernel::pipelineMarker, name);
        }, "return a capsule"
    );
}