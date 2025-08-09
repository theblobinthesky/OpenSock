#ifndef RESOURCE_H
#define RESOURCE_H
#include "utils.h"
#include <vector>
#include <cuda_runtime.h>

class ResourceClient;

struct Allocation {
    uint8_t *host;
    uint8_t *gpu;
    size_t size;
};

class ResourcePool {
public:
    static SharedPtr<ResourcePool> getInstance();
    static void reserveAtLeast(size_t totalSize);

    ~ResourcePool();

    bool acquire(int clientId);

    Allocation allocate(size_t size);

    static std::atomic_int currentClientId;

    [[nodiscard]] uint8_t *getGpuData() const;

private:
    explicit ResourcePool();
    PREVENT_COPY_OR_MOVE(ResourcePool)

    void _reserve(size_t newTotalSize);

    static SharedPtr<ResourcePool> instance;

    uint8_t *hostData;
    uint8_t *gpuData;
    size_t totalSize;
    size_t allocSize;
    size_t offset;
    std::mutex allocateMutex;
};

class MultipleAllocations {
 public:
    explicit MultipleAllocations(const Allocation &totalAllocation);

    MultipleAllocations(const MultipleAllocations &allocs) = delete;

    MultipleAllocations(MultipleAllocations &&allocs) = default;

    MultipleAllocations &operator=(MultipleAllocations &&allocs) noexcept = default;

    explicit operator bool() const;

    // TODO: Make thread safe.
    Allocation allocate(size_t size);

    // TODO: Make thread safe.
    [[nodiscard]] std::vector<uint8_t *> getGpuAllocations() const;

private:
    Allocation totalAllocation;
    size_t offset;
    std::vector<uint8_t *> gpuAllocations;
};

class ResourceClient {
public:
    explicit ResourceClient(int clientId, size_t numBarriers);

    ~ResourceClient();

    bool acquire();

    [[nodiscard]] MultipleAllocations allocate(size_t totalSize);

    void copy(uint8_t *gpuBuffer, const uint8_t *buffer, uint32_t size);

    void insertBarrier(size_t barrierIdx);

    void sync(size_t barrierIdx);

    [[nodiscard]] int getClientId() const;

private:
    int clientId;
    SharedPtr<ResourcePool> pool;
    cudaStream_t stream;
    std::vector<cudaEvent_t> barriers;
    std::mutex mutex;
};

#endif //RESOURCE_H
