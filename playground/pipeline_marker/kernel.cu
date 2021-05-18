#include"kernel.h"
#include<stdio.h>

namespace kernel{

void offloadToHost(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const float *x = reinterpret_cast<const float *>(buffers[0]);
    float *key = reinterpret_cast<float *>(buffers[1]);
    const swapHelper *s = reinterpret_cast<const swapHelper*>(opaque);
    auto allocInfo = s->manager->allocMemory(s->TensorSize);

    cudaMemcpy((float *)allocInfo.second,
        x,
        s->TensorSize,
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy((float *)key,
        &allocInfo.first,
        sizeof(float),
        cudaMemcpyHostToDevice);
}

void uploadToDevice(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const float *key = reinterpret_cast<const float*>(buffers[0]);
    float *result = reinterpret_cast<float *>(buffers[1]);
    const swapHelper *s = reinterpret_cast<const swapHelper *>(opaque);
    float host_key;
    cudaMemcpy(&host_key, key, sizeof(float), cudaMemcpyDeviceToHost);

    auto entry = s->manager->getMemInfo(host_key);

    cudaMemcpy(result,
        entry.first,
        entry.second,
        cudaMemcpyHostToDevice
    );
}

};  //end namespace kernel
