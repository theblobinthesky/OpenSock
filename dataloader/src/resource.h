#ifndef RESOURCE_H
#define RESOURCE_H
#include "utils.h"
#include <vector>
#include <cuda_runtime.h>
#include <set>

class ResourceClient;

struct Allocation {
    uint8_t *host;
    uint8_t *gpu;
    size_t size;

    Allocation operator+(const int other) const {
        return {
            .host = host + other,
            .gpu = gpu + other,
            .size = size
        };
    }
};

// The host memory can downsize to the target size, as the dataloader has full control over it.
// It can never waste pinned memory, if the implementation chooses not to.
class HostMemoryInterface {
public:
    virtual ~HostMemoryInterface() = default;

    virtual uint8_t *changeSizeAndInvalidateMemory(size_t size) = 0;

    virtual void freeEverything() = 0;
};

// The gpu memory never downsizes or invalidates memory, as the dataloader yields control to jax.
// It wastes some gpu memory some of the time, but compatability with jax is more important.
class GpuMemoryInterface {
public:
    virtual ~GpuMemoryInterface() = default;

    virtual uint8_t *increasePoolSizeAndInvalidateMemory(size_t size) = 0;

    virtual void freeEverything() = 0;
};

// Allow for AMD, TPU etc. support later down the line.

class MirroredAllocator {
public:
    MirroredAllocator(HostMemoryInterface *hostIf, GpuMemoryInterface *gpuIf);

    ~MirroredAllocator();

    void reserveAtLeast(size_t newNumItems, size_t newItemSize);

    bool allocate(Allocation &alloc);

    void free(const uint8_t *gpuPtr);

    void handOff(const uint8_t *gpuPtr);

    [[nodiscard]] bool isDrainingOldGeneration() const;

private:
    HostMemoryInterface *hostIf;
    GpuMemoryInterface *gpuIf;

    uint8_t *hostData;
    uint8_t *gpuData;

    size_t numItems = 0;
    size_t itemSize = 0;

    std::set<const uint8_t *> allocAndHandOffGpuData;
    std::vector<Allocation> freeList;

    std::atomic_int32_t genIdx = 0;
    std::atomic_int32_t newGenIdx = 0;
    std::mutex allocateMutex;

public:
    std::mutex memoryMutex;
    std::condition_variable memoryNotify;
};


// TODO: As i migrate to multi-gpu training, i will have to account for numa nodes on server cpus.
// TODO: Not an issue just yet, though.
class CudaHostMemoryInterface final : public HostMemoryInterface {
public:
    uint8_t *changeSizeAndInvalidateMemory(size_t size) override;

    void freeEverything() override;

private:
    uint8_t *data = null;
};

class CudaGpuMemoryInterface final : public GpuMemoryInterface {
public:
    CudaGpuMemoryInterface();

    uint8_t *increasePoolSizeAndInvalidateMemory(size_t size) override;

    void freeEverything() override;

private:
    void setPoolReleaseThreshold(size_t bytes) const;

    cudaMemPool_t pool = {};
    uint8_t *data = null;
};

class ResourcePool {
public:
    static SharedPtr<ResourcePool> getInstance();

    static std::atomic_int currentClientId;

    ~ResourcePool();

    bool acquire(int clientId, size_t numItems, size_t itemSize);

    [[nodiscard]] MirroredAllocator *getAllocator() noexcept;

    [[nodiscard]] cudaStream_t getCudaStream() const noexcept;

private:
    explicit ResourcePool();

    PREVENT_COPY_OR_MOVE(ResourcePool)

    static SharedPtr<ResourcePool> instance;
    CudaHostMemoryInterface hostMemoryIf;
    CudaGpuMemoryInterface gpuMemoryIf;
    MirroredAllocator allocator;
    cudaStream_t stream;
};

// TODO: Make one general purpose bump allocator i can use everywhere.
class ContiguousAllocation {
public:
    explicit ContiguousAllocation(const Allocation &totalAllocation);

    ContiguousAllocation(const ContiguousAllocation &allocs) = delete;

    ContiguousAllocation(ContiguousAllocation &&allocs) = default;

    ContiguousAllocation &operator=(ContiguousAllocation &&allocs) noexcept = default;

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

    bool acquire(size_t numItems, size_t itemSize);

    // TODO: Maybe make the Allocation a bump allocator by default. So no conversion is necessary?
    [[nodiscard]] bool allocate(ContiguousAllocation &alloc);

    void copy(uint8_t *gpuBuffer, const uint8_t *buffer, uint32_t size);

    void insertBarrier(size_t barrierIdx);

    void sync(size_t barrierIdx);

    [[nodiscard]] int getClientId() const;

private:
    int clientId;
    SharedPtr<ResourcePool> pool;
    std::vector<cudaEvent_t> barriers;
    std::mutex mutex;
};

#endif //RESOURCE_H
