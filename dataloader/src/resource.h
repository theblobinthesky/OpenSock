#ifndef RESOURCE_H
#define RESOURCE_H
#include "utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <condition_variable>
#include <queue>
#include <set>
#include "dataloader.h"

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

    explicit operator bool() const {
        return host != null;
    }
};

class HostAndGpuDeviceInterface {
public:
    virtual ~HostAndGpuDeviceInterface() = default;

    // The host memory can downsize to the target size, as the dataloader has full control over it.
    // It can never waste pinned memory, if the implementation chooses not to.
    virtual uint8_t *hostMemoryChangeSizeAndInvalidateMemory(size_t size) = 0;

    // The gpu memory never downsizes or invalidates memory, as the dataloader yields control to jax.
    // It wastes some gpu memory some of the time, but compatability with jax is more important.
    virtual uint8_t *gpuMemoryIncreasePoolSizeAndInvalidateMemory(size_t size) = 0;

    virtual void freeEverything() = 0;

    virtual uint64_t insertNextFenceIntoStream() = 0;

    virtual void synchronizeFenceWithConsumerStream(uint64_t fence, uintptr_t consumerStream) = 0;

    virtual void synchronizeFenceWithHostDevice(uint64_t fence) = 0;

    virtual void copyFromHostToGpuMemory(const uint8_t *host, uint8_t *gpu, const uint32_t size) = 0;
};

// Allow for AMD, TPU etc. support later down the line.

class MirroredAllocator {
public:
    explicit MirroredAllocator(HostAndGpuDeviceInterface *device);

    ~MirroredAllocator();

    void reserveAtLeast(size_t newNumItems, size_t newItemSize);

    bool allocate(Allocation &alloc);

    void free(const uint8_t *gpuPtr);

    void handOff(const uint8_t *gpuPtr);

    [[nodiscard]] bool isDrainingOldGeneration() const;

private:
    HostAndGpuDeviceInterface *device;

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
    std::atomic<uint64_t> memoryNotify;
};


// TODO: When i migrate to multi-gpu training, i will have to account for numa nodes on server cpus.
// TODO: Not an issue just yet, though.

// TODO: Maybe some of these methods need to be protected wrt. concurrent access.
class CudaHostAndGpuDeviceInterface final : public HostAndGpuDeviceInterface {
public:
    CudaHostAndGpuDeviceInterface();

    uint8_t *hostMemoryChangeSizeAndInvalidateMemory(size_t size) override;

    uint8_t *gpuMemoryIncreasePoolSizeAndInvalidateMemory(size_t size) override;

    void freeEverything() override;

    uint64_t insertNextFenceIntoStream() override;

    void synchronizeFenceWithConsumerStream(uint64_t fence, uintptr_t consumerStream) override;

    void synchronizeFenceWithHostDevice(uint64_t fence) override;

    void copyFromHostToGpuMemory(const uint8_t *host, uint8_t *gpu, uint32_t size) override;

private:
    void setGpuMemoryPoolReleaseThreshold(size_t bytes) const;

    cudaMemPool_t pool = {};
    uint8_t *hostData = null;
    uint8_t *gpuData = null;
    cudaStream_t stream;

    std::unordered_map<uint64_t, cudaEvent_t> fenceToEventMap;
    std::atomic_uint64_t eventIndex;
};

struct PrefetchItem {
    int32_t datasetStartingOffset;
    std::vector<uint8_t *> gpuAllocations;
    std::vector<uint64_t> fences;

    bool operator<(const PrefetchItem &other) const;
};

class ResourcePool {
public:
    static SharedPtr<ResourcePool> getInstance();

    static std::atomic_uint64_t acquiredClientId;

    void acquire(uint64_t clientId, size_t numItems, size_t itemSize);

    void synchronizeConsumerStream(uint64_t fence, uintptr_t consumerStream);

    void synchronizeHostDevice(uint64_t fence);

    [[nodiscard]] MirroredAllocator *getAllocator() noexcept;

private:
    explicit ResourcePool();

    PREVENT_COPY_OR_MOVE(ResourcePool)

    void threadMain(size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount);

    // Everything related to memory.
    static SharedPtr<ResourcePool> instance;
    CudaHostAndGpuDeviceInterface device;
    MirroredAllocator allocator;

    // Everything related to data sources.
    DataLoader &dl;

    // Everything related to threading.
    std::priority_queue<PrefetchItem> prefetchCache;
    std::condition_variable prefetchCacheNotify;
    std::mutex prefetchCacheMutex;

    // The thread pool must be last, so it's destroyed first before all other members.
    ThreadPool threadPool;
};

class ResourceClient {
public:
    explicit ResourceClient(uint64_t clientId, size_t numItems, size_t itemSize);

    void acquire();

    // TODO: Remove.
    [[nodiscard]] bool allocate(BumpAllocator<Allocation> &subAllocator);

    [[nodiscard]] int getClientId() const;

private:
    uint64_t clientId;
    size_t numItems;
    size_t itemSize;

    SharedPtr<ResourcePool> pool;
    std::mutex mutex;
};

#endif //RESOURCE_H
