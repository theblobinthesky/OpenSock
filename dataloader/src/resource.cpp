#include "resource.h"
#include <format>

SharedPtr<ResourcePool> ResourcePool::instance;
std::atomic_int ResourcePool::currentClientId = std::atomic_int(-1);

SharedPtr<ResourcePool> ResourcePool::getInstance() {
    if (!instance) {
        instance = SharedPtr(new ResourcePool(), true);
    }

    return SharedPtr(instance);
}

void ResourcePool::reserveAtLeast(const size_t totalSize) {
    getInstance()->_reserve(totalSize);
}

Allocation ResourcePool::allocate(const size_t size) {
    uint8_t *hostPtr = hostData + offset;
    uint8_t *gpuPtr = gpuData + offset;
    offset += size;

    if (offset > totalSize) {
        throw std::runtime_error(std::format("Tried to allocate {} bytes beyond the size {} of the memory arena.",
                                             offset - totalSize, totalSize));
    }

    offset = offset % totalSize;
    return {
        .host = hostPtr,
        .gpu = gpuPtr,
        .size = size
    };
}

uint8_t *ResourcePool::getGpuData() const {
    return gpuData;
}

ResourcePool::ResourcePool() : hostData(null), gpuData(null), totalSize(0), allocSize(0), offset(0) {
}

ResourcePool::~ResourcePool() {
    if (hostData) {
        cudaFreeHost(hostData);
        cudaFree(gpuData);
    }
}

bool ResourcePool::acquire(const int clientId) {
    const bool clientChanged = currentClientId != clientId;

    if (clientChanged) {
        currentClientId = clientId;
        offset = 0;
    }

    return clientChanged;
}

void ResourcePool::_reserve(const size_t totalSize) {
    if (this->allocSize >= totalSize) {
        this->totalSize = totalSize;
        return;
    }

    // TODO: Free if necessary......
    // THIS IS BAD IT CAN'T UPSIZE YET!!!!

    this->totalSize = totalSize;
    this->allocSize = std::max(this->allocSize, totalSize);

    cudaError_t err = cudaMallocHost(&hostData, totalSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaHostMalloc failed");
    }

    err = cudaMalloc(&gpuData, totalSize);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaMalloc failed to allocate {} MB", totalSize / 1024 / 1024));
    }
}

MultipleAllocations::MultipleAllocations(const Allocation &totalAllocation) : totalAllocation(totalAllocation),
                                                                              offset(0) {
}

MultipleAllocations::operator bool() const {
    return totalAllocation.host != null;
}

Allocation MultipleAllocations::allocate(const size_t size) {
    const Allocation allocation = {
        .host = totalAllocation.host + offset,
        .gpu = totalAllocation.gpu + offset,
        .size = size
    };

    gpuAllocations.push_back(allocation.gpu);
    offset += size;
    return allocation;
}

std::vector<uint8_t *> MultipleAllocations::getGpuAllocations() const {
    return gpuAllocations;
}

ResourceClient::ResourceClient(const int clientId, const size_t numBarriers) : clientId(clientId),
                                                                               pool(ResourcePool::getInstance()),
                                                                               stream{} {
    if (const cudaError_t err = cudaStreamCreate(&stream);
        err != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed.");
    }

    for (size_t i = 0; i < numBarriers; i++) {
        cudaEvent_t barrierEvent;
        if (cudaEventCreate(&barrierEvent) != cudaSuccess) {
            throw std::runtime_error("cudaEventCreate failed.");
        }

        barriers.push_back(barrierEvent);
    }
}

ResourceClient::~ResourceClient() {
    if (const auto err = cudaStreamDestroy(stream); err != cudaSuccess) {
        std::printf("cudaStreamDestroy failed.\n");
        std::terminate();
    }

    for (cudaEvent_t barrier: barriers) {
        cudaEventDestroy(barrier);
    }
}

bool ResourceClient::acquire() {
    std::unique_lock lock(mutex);
    return pool->acquire(clientId);
}

MultipleAllocations ResourceClient::allocate(const size_t totalSize) {
    std::unique_lock lock(mutex);
    if (ResourcePool::currentClientId.load() != clientId) {
        // Do not allocate if the client has not acquired the pool.
        return MultipleAllocations({});
    }

    return MultipleAllocations(pool->allocate(totalSize));
}

#define RETURN_IF_INACTIVE() if (ResourcePool::currentClientId.load() != clientId) return

void ResourceClient::copy(uint8_t *gpuBuffer,
                          const uint8_t *buffer,
                          const uint32_t size) {
    std::unique_lock lock(mutex);
    RETURN_IF_INACTIVE();

    if (cudaMemcpyAsync(gpuBuffer, buffer, size,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync failed.");
    }
}

void ResourceClient::insertBarrier(const size_t barrierIdx) {
    std::unique_lock lock(mutex);
    RETURN_IF_INACTIVE();

    if (cudaEventRecord(barriers[barrierIdx], stream) != cudaSuccess) {
        throw std::runtime_error("cudaEventRecord failed.");
    }
}

void ResourceClient::sync(const size_t barrierIdx) {
    std::unique_lock lock(mutex);
    RETURN_IF_INACTIVE();

    if (cudaEventSynchronize(barriers[barrierIdx]) != cudaSuccess) {
        throw std::runtime_error("cudaEventSynchronize failed.");
    }
}

int ResourceClient::getClientId() const {
    return clientId;
}
