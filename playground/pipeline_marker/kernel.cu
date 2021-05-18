#include"kernel.h"
#include<stdio.h>

namespace kernel{
void pipelineMarker(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const int64_t *sizes = reinterpret_cast<const int64_t *> opaque;
    for (size_t i = 0; i < opaque_len; i++) {
        printf("%lld %lld\n", i, sizes[i])
    }
    const float *x = reinterpret_cast<const float *>(buffers[0]);
    float *result = reinterpret_cast<float *>(buffers[1]);
    cudaMemset(result, 0, 1);
}
};  //end namespace kernel
