#ifndef RESOURCE_H
#define RESOURCE_H
#include "utils.h"
#include <vector>
#include <condition_variable>
#include <latch>
#include <queue>
#include <set>
#include "dataloader.h"

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
        return host != nullptr;
    }
};

// Allow for AMD, TPU etc. support later down the line.
// None of these interface methods not need to be protected by a mutex.
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

    virtual Fence insertNextFenceIntoStream() = 0;

    virtual void synchronizeFenceWithConsumerStream(Fence fence, ConsumerStream consumerStream) = 0;

    virtual void synchronizeFenceWithHostDevice(Fence fence) = 0;

    virtual void copyFromHostToGpuMemory(const uint8_t *host, uint8_t *gpu, uint32_t size) = 0;
};

class MirroredAllocator {
public:
    explicit MirroredAllocator(const std::shared_ptr<HostAndGpuDeviceInterface> &device);

    ~MirroredAllocator();

    void reset();

    void reserveAtLeast(size_t newNumItems, size_t newItemSize);

    bool allocate(Allocation &alloc);

    void free(const uint8_t *gpuPtr);

    void handOff(const uint8_t *gpuPtr);

    [[nodiscard]] std::mutex &getMutex();

    [[nodiscard]] std::condition_variable &getDrainCv();

    [[nodiscard]] bool isDrained() const;

private:
    std::shared_ptr<HostAndGpuDeviceInterface> device;

    uint8_t *hostData = nullptr;
    uint8_t *gpuData = nullptr;

    size_t numItems = 0;
    size_t itemSize = 0;

    std::set<const uint8_t *> allocAndHandOffGpuData;
    std::vector<Allocation> freeList;

    std::mutex mutex;
    std::condition_variable drainCv;

public:
    std::atomic<uint64_t> memoryNotify;
};

struct PrefetchItem {
    int32_t datasetStartingOffset;
    std::vector<uint8_t *> gpuAllocations;
    std::vector<Fence> fences;
    std::vector<Shape> shapes;
    std::vector<bool> hasMetaTensor;
    std::vector<DType> metaDType;

    bool operator<(const PrefetchItem &other) const;
};

class DataLoader;

class ResourcePool {
public:
    // Explicitly-managed singleton. Lives for process lifetime unless shutdown() is called.
    static ResourcePool &get();

    // Does not lazily initialize the pool, only to destroy it immediately.
    // Makes us compatible with sanitizers like address sanitizer.
    static void shutdownLazily();

    static void setDeviceForNewResourcePool(const std::shared_ptr<HostAndGpuDeviceInterface> &device);

    // Explicit shutdown to release threads and device memory.
    void shutdown();

    PrefetchItem acquireAndGetNextBatch(const std::shared_ptr<DataLoader> &dataLoader);

    void synchronizeConsumerStream(Fence fence, ConsumerStream consumerStream) const;

    void synchronizeHostDevice(Fence fence) const;

    [[nodiscard]] MirroredAllocator *getAllocator() noexcept;

    [[nodiscard]] uint64_t getGenIdx() const noexcept;

private:
    explicit ResourcePool(const std::shared_ptr<HostAndGpuDeviceInterface> &device);

    ~ResourcePool();

    PREVENT_COPY_OR_MOVE(ResourcePool)

    void wakeupAllThreads();

    void threadMain(size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount, uint64_t initialGenIdx);

    static std::shared_ptr<HostAndGpuDeviceInterface> deviceForNewResourcePool;
    static ResourcePool *singleton;

    // Everything related to memory.
    std::shared_ptr<HostAndGpuDeviceInterface> device;
    MirroredAllocator allocator;

    // Everything related to data sources.
    std::shared_ptr<DataLoader> dl;
    std::atomic_uint64_t genIdx;

    // Everything related to threading.
    std::unique_ptr<std::latch> genChangeAssemble;
    std::unique_ptr<std::latch> genChangeSendOff;
    std::priority_queue<PrefetchItem> prefetchCache;
    std::condition_variable prefetchCacheNotify;
    std::mutex prefetchCacheMutex;

    // The thread pool must be last, so it's destroyed first before all other members.
    ThreadPool threadPool;
};

#endif //RESOURCE_H
