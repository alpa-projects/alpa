#include <cuda_runtime_api.h>

namespace kernel {

void identity(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

}  // namespace kernel
