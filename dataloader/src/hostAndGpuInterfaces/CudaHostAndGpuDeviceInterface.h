#ifndef VERSION_CUDAHOSTANDGPUDEVICEINTERFACE_H
#define VERSION_CUDAHOSTANDGPUDEVICEINTERFACE_H

#include "resource.h"

#include <cuda_runtime.h>

// TODO: When i migrate to multi-gpu training, i will have to account for numa nodes on server cpus.
// TODO: Not an issue just yet, though.

class CudaHostAndGpuDeviceInterface final : public HostAndGpuDeviceInterface {
public:
    CudaHostAndGpuDeviceInterface();

    uint8_t *hostMemoryChangeSizeAndInvalidateMemory(size_t size) override;

    uint8_t *gpuMemoryIncreasePoolSizeAndInvalidateMemory(size_t size) override;

    void freeEverything() override;

    Fence insertNextFenceIntoStream() override;

    void synchronizeFenceWithConsumerStream(Fence fence, ConsumerStream consumerStream) override;

    void synchronizeFenceWithHostDevice(Fence fence) override;

    void copyFromHostToGpuMemory(const uint8_t *host, uint8_t *gpu, uint32_t size) override;

private:
    void setGpuMemoryPoolReleaseThreshold(size_t bytes) const;

    cudaMemPool_t pool = {};
    uint8_t *hostData = nullptr;
    uint8_t *gpuData = nullptr;
    cudaStream_t stream = {};

    std::unordered_map<uint64_t, cudaEvent_t> fenceToEventMap;
    std::atomic_uint64_t eventIndex;
};

#endif //VERSION_CUDAHOSTANDGPUDEVICEINTERFACE_H