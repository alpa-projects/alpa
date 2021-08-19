#include<cuda_runtime_api.h>
#include<map>
#include<utility>
#include<cstdlib>

namespace kernel{
void pipelineMarker(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);
};  // end namespace kernel
