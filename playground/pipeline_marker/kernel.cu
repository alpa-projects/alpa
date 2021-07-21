#include"kernel.h"
#include<stdio.h>

namespace kernel{
void pipelineMarker(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const int64_t *sizes = reinterpret_cast<const int64_t *>(opaque);
    size_t n_inputs = opaque_len / sizeof(int64_t);
    for (size_t i = 0; i < n_inputs; i++) {
        const float *input = reinterpret_cast<const float *>(buffers[i]);
        float *output = reinterpret_cast<float *>(buffers[i + n_inputs]);
        cudaMemcpy(output, input, sizes[i], cudaMemcpyDeviceToDevice);
    }
}
};  //end namespace kernel
