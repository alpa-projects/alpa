#include<cuda_runtime_api.h>
#include<map>
#include<utility>
#include<cstdlib>

namespace kernel{

using allocRet_t = std::pair<float, void*>;

namespace runtime{
    class memoryManager {
        using entry_t = std::pair<void*, size_t>;
        public:
            memoryManager() {}
            allocRet_t allocMemory(size_t MemSize) {
                void *ptr = malloc(MemSize);
                memoryMap[counter] = std::make_pair(ptr, MemSize);
                return std::make_pair(counter++, ptr);
            }
            entry_t getMemInfo(float key) {
                return memoryMap[key];
            }
            void freeMemory(float key) {
                auto iter = memoryMap.find(key);
                if (iter != memoryMap.end()) free(iter->second.first);
            }
            void freeAll() {
                for(auto item = memoryMap.begin();item != memoryMap.end();item++) {
                    free(item->second.first);
                }
            }
        private:
            std::map<float, entry_t> memoryMap;
            float counter = 0;
    };
};  //end namespace runtime


struct swapHelper {
    size_t TensorSize;
    runtime::memoryManager *manager;
};

void offloadToHost(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);
void uploadToDevice(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);
};  // end namespace kernel
