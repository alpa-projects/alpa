#include<pybind11/pybind11.h>
#include"kernel.h"
#include<string>

namespace py = pybind11;

PYBIND11_MODULE(xla_pipeline_marker, m) {
    m.doc() = "Pipeline marker in XLA.";
    m.def(
        "xla_pipeline_marker", [](){
            const char *name = "xla._CUSTOM_CALL_TARGET";
            return py::capsule((void *) &kernel::pipelineMarker, name);
        }, "return a capsule"
    );
    m.def(
        "identity", [](){
            const char *name = "xla._CUSTOM_CALL_TARGET";
            return py::capsule((void *) &kernel::identity, name);
        }, "return a capsule"
    );
}
