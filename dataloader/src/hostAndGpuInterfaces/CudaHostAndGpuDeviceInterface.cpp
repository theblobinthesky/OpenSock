#include "CudaHostAndGpuDeviceInterface.h"

CudaHostAndGpuDeviceInterface::CudaHostAndGpuDeviceInterface() {
    if (const auto err = cudaDeviceGetDefaultMemPool(&pool, /*device=*/0); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaDeviceGetDefaultMemPool failed: {}", cudaGetErrorString(err)));
    }

    setGpuMemoryPoolReleaseThreshold(1);

    if (const cudaError_t err = cudaStreamCreate(&stream); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaStreamCreate failed: {}", cudaGetErrorString(err)));
    }
}

uint8_t *CudaHostAndGpuDeviceInterface::hostMemoryChangeSizeAndInvalidateMemory(const size_t size) {
    if (!stream) {
        if (const cudaError_t err = cudaStreamCreate(&stream); err != cudaSuccess) {
            throw std::runtime_error(std::format("cudaStreamCreate failed: {}", cudaGetErrorString(err)));
        }
    } else {
        cudaStreamSynchronize(stream);
    }

    freeEverything();

    if (const cudaError_t err = cudaHostAlloc(&hostData, size, cudaHostAllocWriteCombined); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaHostAlloc failed to allocate {} MB: {}", size / 1024 / 1024, cudaGetErrorString(err)));
    }

    return hostData;
}

uint8_t *CudaHostAndGpuDeviceInterface::gpuMemoryIncreasePoolSizeAndInvalidateMemory(const size_t size) {
    LOG_DEBUG("Increasing gpu memory pool to {}", size);
    setGpuMemoryPoolReleaseThreshold(size);

    if (!stream) {
        if (const cudaError_t err = cudaStreamCreate(&stream); err != cudaSuccess) {
            throw std::runtime_error(std::format("cudaStreamCreate failed: {}", cudaGetErrorString(err)));
        }
    }

    if (gpuData != nullptr) {
        if (const auto err = cudaFreeAsync(gpuData, stream); err != cudaSuccess) {
            throw std::runtime_error(std::format("cudaFreeAsync warm-up free failed: {}", cudaGetErrorString(err)));
        }
    }

    if (const auto err = cudaMallocAsync(&gpuData, size, stream); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaMallocAsync warm-up alloc failed: {}", cudaGetErrorString(err)));
    }

    return gpuData;
}

void CudaHostAndGpuDeviceInterface::freeEverything() {
    if (hostData != nullptr) {
        if (const cudaError_t err = cudaFreeHost(hostData); err != cudaSuccess) {
            throw std::runtime_error(std::format("cudaFreeHost failed to free data: {}", cudaGetErrorString(err)));
        }
    }

    // The gpu memory is handled by a cuda memory pool.
    // Beyond destroying the stream, no additional cleanup is required.

    if (stream) {
        if (const auto err = cudaStreamDestroy(stream); err != cudaSuccess) {
            LOG_WARNING("cudaStreamDestroy failed.");
        }
        stream = {};
    }

    hostData = nullptr;
    gpuData = nullptr;
}

Fence CudaHostAndGpuDeviceInterface::insertNextFenceIntoStream() {
    cudaEvent_t event;
    if (const cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaEventCreate failed: {}", cudaGetErrorString(err)));
    }

    if (const cudaError_t err = cudaEventRecord(event, stream); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorString(err)));
    }

    auto fence = eventIndex.fetch_add(1);
    fenceToEventMapMutex.lock();
    fenceToEventMap.emplace(fence, event);
    fenceToEventMapMutex.unlock();
    return {fence};
}

void CudaHostAndGpuDeviceInterface::synchronizeFenceWithConsumerStream(const Fence fence,
                                                                       const ConsumerStream consumerStream) {
    fenceToEventMapMutex.lock();
    const auto found = fenceToEventMap.find(fence.id);
    if (found == fenceToEventMap.end()) {
        throw std::runtime_error("Fence is invalid.");
    }
    fenceToEventMapMutex.unlock();

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto consumer = reinterpret_cast<cudaStream_t>(consumerStream.id);
    cudaEvent_t event = found->second;
    if (const cudaError_t err = cudaStreamWaitEvent(consumer, event); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaStreamWaitEvent failed: {}", cudaGetErrorString(err)));
    }

    fenceToEventMapMutex.lock();
    fenceToEventMap.erase(found);
    fenceToEventMapMutex.unlock();

    if (cudaEventDestroy(event) != cudaSuccess) {
        throw std::runtime_error("cudaEventDestroy failed.");
    }
}

void CudaHostAndGpuDeviceInterface::synchronizeFenceWithHostDevice(const Fence fence) {
    fenceToEventMapMutex.lock();
    const auto found = fenceToEventMap.find(fence.id);
    if (found == fenceToEventMap.end()) {
        throw std::runtime_error("Fence is invalid.");
    }
    fenceToEventMapMutex.unlock();

    const cudaEvent_t event = found->second;

    fenceToEventMapMutex.lock();
    fenceToEventMap.erase(found);
    fenceToEventMapMutex.unlock();

    if (const cudaError_t err = cudaEventSynchronize(event); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaEventSynchronize failed: {}", cudaGetErrorString(err)));
    }

    if (cudaEventDestroy(event) != cudaSuccess) {
        throw std::runtime_error("cudaEventDestroy failed.");
    }
}

void CudaHostAndGpuDeviceInterface::copyFromHostToGpuMemory(const uint8_t *host, uint8_t *gpu, const uint32_t size) {
    if (const cudaError_t err = cudaMemcpyAsync(gpu, host, size, cudaMemcpyHostToDevice, stream); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaMemcpyAsync failed: {}", cudaGetErrorString(err)));
    }
}

void CudaHostAndGpuDeviceInterface::setGpuMemoryPoolReleaseThreshold(size_t bytes) const {
    if (const auto err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &bytes);
        err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold) failed: {}", cudaGetErrorString(err)));
    }
}
