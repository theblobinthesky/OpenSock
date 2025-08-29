#include "resource.h"
#include <format>

#include "datadecoder.h"
#include "dataio.h"

MirroredAllocator::MirroredAllocator(HostAndGpuDeviceInterface *device) : device(device) {
}

MirroredAllocator::~MirroredAllocator() {
    device->freeEverything();
}

void MirroredAllocator::reset() {
    // Free device/host resources first.
    device->freeEverything();
    // Then clear allocator bookkeeping.
    std::unique_lock lock(mutex);
    allocAndHandOffGpuData.clear();
    freeList.clear();
    numItems = 0;
    itemSize = 0;
    hostData = nullptr;
    gpuData = nullptr;
}

void MirroredAllocator::reserveAtLeast(const size_t newNumItems, const size_t newItemSize) {
    std::unique_lock lock(mutex);

    LOG_DEBUG("reserveAtLeast newNumItems={}, newItemSize={}", newNumItems, newItemSize);
    if (const size_t requiredSize = newNumItems * newItemSize; requiredSize > numItems * itemSize) {
        hostData = device->hostMemoryChangeSizeAndInvalidateMemory(requiredSize);
        gpuData = device->gpuMemoryIncreasePoolSizeAndInvalidateMemory(requiredSize);
    }

    numItems = newNumItems;
    itemSize = newItemSize;

    freeList.clear();
    for (size_t i = 0; i < numItems; i++) {
        freeList.push_back(Allocation{
            .host = hostData + i * itemSize,
            .gpu = gpuData + i * itemSize,
            .size = itemSize
        });
    }
}

bool MirroredAllocator::allocate(Allocation &alloc) {
    std::unique_lock lock(mutex);

    if (freeList.empty()) {
        // Error: Tried to allocate in an empty pool.
        return false;
    }

    alloc = freeList.back();
    freeList.pop_back();

    if (allocAndHandOffGpuData.contains(alloc.gpu)) {
        throw std::runtime_error(std::format("Tried to allocate space ({} host, {}) gpu that has not yet been freed.",
                                             reinterpret_cast<uint64_t>(alloc.host),
                                             reinterpret_cast<uint64_t>(alloc.gpu)));
    }


    return true;
}

void MirroredAllocator::free(const uint8_t *gpuPtr) {
    std::unique_lock lock(mutex);
    LOG_DEBUG("deleter called");
    const auto gpuDataFound = allocAndHandOffGpuData.find(gpuPtr);
    if (gpuDataFound == allocAndHandOffGpuData.end()) {
        throw std::runtime_error("Tried to free space that was never allocated or already freed.");
    }

    allocAndHandOffGpuData.erase(gpuDataFound);

    const size_t idx = (gpuPtr - gpuData) / itemSize;
    freeList.push_back(Allocation{
        .host = hostData + idx * itemSize,
        .gpu = gpuData + idx * itemSize,
        .size = itemSize
    });
    lock.unlock();

    if (isDrained()) {
        drainCv.notify_all();
    }

    memoryNotify.fetch_add(1);
    memoryNotify.notify_all();
}

void MirroredAllocator::handOff(const uint8_t *gpuPtr) {
    std::unique_lock lock(mutex);
    allocAndHandOffGpuData.emplace(gpuPtr);
}

std::mutex &MirroredAllocator::getMutex() {
    return mutex;
}

std::condition_variable &MirroredAllocator::getDrainCv() {
    return drainCv;
}

bool MirroredAllocator::isDrained() const {
    return allocAndHandOffGpuData.empty();
}

CudaHostAndGpuDeviceInterface::CudaHostAndGpuDeviceInterface() {
    if (const auto err = cudaDeviceGetDefaultMemPool(&pool, /*device=*/0); err != cudaSuccess) {
        throw std::runtime_error("cudaDeviceGetDefaultMemPool failed");
    }

    setGpuMemoryPoolReleaseThreshold(1);

    if (const cudaError_t err = cudaStreamCreate(&stream); err != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed.");
    }
}

uint8_t *CudaHostAndGpuDeviceInterface::hostMemoryChangeSizeAndInvalidateMemory(const size_t size) {
    if (!stream) {
        if (const cudaError_t err = cudaStreamCreate(&stream); err != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed.");
        }
    } else {
        cudaStreamSynchronize(stream);
    }

    freeEverything();

    if (const cudaError_t err = cudaHostAlloc(&hostData, size, cudaHostAllocWriteCombined); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaHostAlloc failed to allocate {} MB", size / 1024 / 1024));
    }

    return hostData;
}

uint8_t *CudaHostAndGpuDeviceInterface::gpuMemoryIncreasePoolSizeAndInvalidateMemory(const size_t size) {
    LOG_DEBUG("Increasing gpu memory pool to {}", size);
    setGpuMemoryPoolReleaseThreshold(size);

    if (!stream) {
        if (const cudaError_t err = cudaStreamCreate(&stream); err != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed.");
        }
    }

    if (gpuData != nullptr) {
        if (const auto err = cudaFreeAsync(gpuData, stream); err != cudaSuccess) {
            throw std::runtime_error("cudaFreeAsync warm-up free failed");
        }
    }

    if (const auto err = cudaMallocAsync(&gpuData, size, stream); err != cudaSuccess) {
        throw std::runtime_error("cudaMallocAsync warm-up alloc failed");
    }

    return gpuData;
}

void CudaHostAndGpuDeviceInterface::freeEverything() {
    if (hostData != nullptr) {
        if (const cudaError_t err = cudaFreeHost(hostData); err != cudaSuccess) {
            throw std::runtime_error("cudaFreeHost failed to free data.");
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
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        throw std::runtime_error("cudaEventCreate failed.");
    }

    if (cudaEventRecord(event, stream) != cudaSuccess) {
        throw std::runtime_error("cudaEventRecord failed.");
    }

    auto fence = eventIndex.fetch_add(1);
    fenceToEventMap.emplace(fence, event);
    return {fence};
}

void CudaHostAndGpuDeviceInterface::synchronizeFenceWithConsumerStream(const Fence fence,
                                                                       const ConsumerStream consumerStream) {
    const auto found = fenceToEventMap.find(fence.id);
    if (found == fenceToEventMap.end()) {
        throw std::runtime_error("Fence is invalid.");
    }

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto consumer = reinterpret_cast<cudaStream_t>(consumerStream.id);
    cudaEvent_t event = found->second;
    if (cudaStreamWaitEvent(consumer, event) != cudaSuccess) {
        throw std::runtime_error("cudaStreamWaitEvent failed.");
    }

    fenceToEventMap.erase(found);
    if (cudaEventDestroy(event) != cudaSuccess) {
        throw std::runtime_error("cudaEventDestroy failed.");
    }
}

void CudaHostAndGpuDeviceInterface::synchronizeFenceWithHostDevice(const Fence fence) {
    const auto found = fenceToEventMap.find(fence.id);
    if (found == fenceToEventMap.end()) {
        throw std::runtime_error("Fence is invalid.");
    }

    const cudaEvent_t event = found->second;
    if (cudaEventSynchronize(event) != cudaSuccess) {
        throw std::runtime_error("cudaEventSynchronize failed.");
    }

    fenceToEventMap.erase(found);
    if (cudaEventDestroy(event) != cudaSuccess) {
        throw std::runtime_error("cudaEventDestroy failed.");
    }
}

void CudaHostAndGpuDeviceInterface::copyFromHostToGpuMemory(const uint8_t *host, uint8_t *gpu, const uint32_t size) {
    if (cudaMemcpyAsync(gpu, host, size, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync failed.");
    }
}

void CudaHostAndGpuDeviceInterface::setGpuMemoryPoolReleaseThreshold(size_t bytes) const {
    if (const auto err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &bytes);
        err != cudaSuccess) {
        throw std::runtime_error("cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold) failed");
    }
}

ResourcePool &ResourcePool::get() {
    // Intentional leak to avoid implicit destruction order issues; use shutdown() to free resources explicitly.
    static auto *instance = new ResourcePool();
    return *instance;
}

MirroredAllocator *ResourcePool::getAllocator() noexcept {
    return &allocator;
}

uint64_t ResourcePool::getGenIdx() const noexcept {
    return genIdx.load();
}

bool PrefetchItem::operator<(const PrefetchItem &other) const {
    // Make sure to sort the priority queue such that the smallest elements have priority.
    return datasetStartingOffset > other.datasetStartingOffset;
}

ResourcePool::ResourcePool() : allocator(&device), dl(nullptr),
                               threadPool(
                                   [this](const size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount) {
                                       this->threadMain(threadIdx, desiredThreadCount);
                                   },
                                   0,
                                   [this] {
                                       // Allow the threads to continue exection.
                                       genChangeSendOff->count_down();
                                   },
                                   [this] { wakeupAllThreads(); }) {
}

ResourcePool::~ResourcePool() {
    threadPool.resize(0);
    prefetchCacheNotify.notify_all();
}

void ResourcePool::shutdown() {
    // Stop threads and wake any waiters.
    threadPool.resize(0);

    // Wait for allocator to drain if there is a current dataloader.
    if (dl != nullptr) {
        std::unique_lock lock(allocator.getMutex());
        allocator.getDrainCv().wait(lock, [this] { return allocator.isDrained(); });
    }

    // Best-effort cleanup of state.
    prefetchCache = std::priority_queue<PrefetchItem>();
    allocator.reset();
}

// Wake up any threads waiting on our condvars/atomics during pool resize.
void ResourcePool::wakeupAllThreads() {
    // Wake threads waiting on prefetch order / cache availability.
    prefetchCacheNotify.notify_all();

    // Wake threads waiting on allocator memory availability.
    allocator.memoryNotify.fetch_add(1);
    allocator.memoryNotify.notify_all();

    // TODO (i don't think this is necessary): genChangeSendOff->count_down();
}

void ResourcePool::threadMain(const size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount) {
#define IS_SHUTDOWN_REQUIRED() (threadIdx >= desiredThreadCount.load())
#define IS_GENCHANGE_REQUIRED() (rememberedGenIdx != genIdx.load())
#define IS_SHUTDOWN_OR_GENCHANGE_REQUIRED() (IS_SHUTDOWN_REQUIRED() || IS_GENCHANGE_REQUIRED())

#define DO_SHUTDOWN_IF_NECESSARY() \
    if (IS_SHUTDOWN_REQUIRED()) { return; }

#define DO_SHUTDOWN_OR_GENCHANGE_IF_NECESSARY() \
    if (IS_SHUTDOWN_REQUIRED()) { return; } \
    if (IS_GENCHANGE_REQUIRED()) { continue; }

    LOG_DEBUG("thread {} started", threadIdx);

    const auto temporaryArena = std::make_unique<uint8_t[]>(dl->outputBatchMemorySize);
    auto temporaryAllocator = BumpAllocator(temporaryArena.get(), dl->outputBatchMemorySize);

    uint64_t rememberedGenIdx = genIdx.load();

    while (!IS_SHUTDOWN_REQUIRED()) {
        if (IS_GENCHANGE_REQUIRED()) {
            genChangeAssemble->count_down();
            rememberedGenIdx = genIdx.load();
            genChangeSendOff->wait();
            continue;
        }


        const auto [startingOffset, batchPaths] = dl->batchedDataset.getNextInFlightBatch();
        const std::vector<Head> &heads = dl->batchedDataset.getDataset().getHeads();


        // For each head, load all batch items into one contigous cpu array.
        std::vector<CpuAllocation> hostAllocations;
        temporaryAllocator.reset();
        for (size_t headIdx = 0; headIdx < heads.size(); headIdx++) {
            hostAllocations.push_back(
                loadFilesFromHeadIntoContigousBatch(temporaryAllocator, batchPaths, heads, headIdx)
            );
        }
        DO_SHUTDOWN_OR_GENCHANGE_IF_NECESSARY()


        // At this point, the thread has loaded the batch into cpu ram.
        // As soon as pinned memory is available, we start copying to the gpu.
        // The assumption is, that resource load, decompression and decode takes
        // much longer than host->gpu copy.


        // Make sure threads don't steal memory from high priority batches that need to be copied immediately.
        std::unique_lock lock(prefetchCacheMutex);
        prefetchCacheNotify.wait(lock, [this, startingOffset, &rememberedGenIdx, threadIdx, &desiredThreadCount] {
            return IS_SHUTDOWN_OR_GENCHANGE_REQUIRED() ||
                   startingOffset <= dl->batchedDataset.getLastWaitingBatch().load()
                   + static_cast<int>(dl->batchSize * dl->prefetchSize);
        });
        lock.unlock();
        DO_SHUTDOWN_OR_GENCHANGE_IF_NECESSARY()


        // Grab a (host,gpu) memory pair.
        auto allocations = BumpAllocator(Allocation{}, 0);
        while (!IS_SHUTDOWN_OR_GENCHANGE_REQUIRED()) {
            const uint64_t old = allocator.memoryNotify.load();

            Allocation allocation{};
            const bool success = allocator.allocate(allocation);
            allocations = BumpAllocator(allocation, allocation.size);
            if (success) {
                break;
            }

            allocator.memoryNotify.wait(old);
        }
        DO_SHUTDOWN_OR_GENCHANGE_IF_NECESSARY()


        assert(allocations.getArena().host != nullptr);
        std::memcpy(allocations.getArena().host, temporaryAllocator.getArena(), dl->outputBatchMemorySize);
        std::vector<uint8_t *> gpuAllocations;
        std::vector<Fence> fences;
        for (size_t headIdx = 0; headIdx < heads.size(); headIdx++) {
            const CpuAllocation &nonPinnedCpuAllocation = hostAllocations[headIdx];
            const auto &[host, gpu, _] = allocations.allocate(nonPinnedCpuAllocation.batchBufferSize);

            // Start async upload to gpu memory as soon as possible.
            // TODO: This also locked a mutex. Maybe this is still necessary?
            device.copyFromHostToGpuMemory(host, gpu, nonPinnedCpuAllocation.batchBufferSize);
            gpuAllocations.push_back(gpu);
            // TODO: This needs to be synchronized by mutex?
            fences.push_back(device.insertNextFenceIntoStream());
        }
        DO_SHUTDOWN_OR_GENCHANGE_IF_NECESSARY()


        // Make sure threads submit their batches in dataset order.
        std::unique_lock lock2(prefetchCacheMutex);
        prefetchCacheNotify.wait(lock2, [this, startingOffset, &rememberedGenIdx, threadIdx, &desiredThreadCount] {
            return IS_SHUTDOWN_OR_GENCHANGE_REQUIRED() ||
                   startingOffset - static_cast<int>(dl->batchSize)
                   == dl->batchedDataset.getLastWaitingBatch().load();
        });
        dl->batchedDataset.markBatchWaiting(startingOffset);

        prefetchCache.push({
            .datasetStartingOffset = startingOffset,
            .gpuAllocations = gpuAllocations,
            .fences = fences,
        });
        prefetchCacheNotify.notify_all();
    }

    LOG_DEBUG("thread {} shutting down; desiredThreadCount={}", threadIdx, desiredThreadCount.load());
}

PrefetchItem ResourcePool::acquireAndGetNextBatch(const std::shared_ptr<DataLoader> &dataLoader) {
    if (!dataLoader) {
        throw std::runtime_error("DataLoader for the next batch cannot be null.");
    }

    if (dl == nullptr || dl->id != dataLoader->id) {
        LOG_DEBUG("resource pool acquired with clientId: {}", dataLoader->id);

        // Make sure, in time, all threads exit their generational loops
        ++genIdx;

        // The protocol allows for a clean state transition:
        // Wait for all threads to exit their generational loops.
        // Only then can you safely reinitialize everything.
        genChangeSendOff = std::make_unique<std::latch>(1);
        if (dl != nullptr) {
            genChangeAssemble = std::make_unique<std::latch>(dl->numThreads);
            // Wait for all threads to stop working.
            wakeupAllThreads();
            genChangeAssemble->wait();

            // Wait for all memory to be returned from the dlpack consumer (i.e. jax, tf., pytorch).
            {
                std::unique_lock lock(allocator.getMutex());
                allocator.getDrainCv().wait(lock, [this] { return allocator.isDrained(); });
            }

            // Reset and reinitialize everything.
            dl->batchedDataset.forgetInFlightBatches();
        }

        // Reset and reinitialize.
        dl = dataLoader;
        prefetchCache = std::priority_queue<PrefetchItem>();
        allocator.reserveAtLeast(dataLoader->prefetchSize, dataLoader->outputBatchMemorySize);
        threadPool.resize(dataLoader->numThreads);
    }

    // Fetch the next batch.
    std::unique_lock lock(prefetchCacheMutex);
    prefetchCacheNotify.wait(lock, [this] { return !prefetchCache.empty(); });

    const auto prefetchItem = prefetchCache.top();
    prefetchCache.pop();
    LOG_DEBUG("acquireAndGetNextBatch datasetStartingOffset={}", prefetchItem.datasetStartingOffset);
    dl->batchedDataset.popWaitingBatch(prefetchItem.datasetStartingOffset);

    lock.unlock();

    return prefetchItem;
}

void ResourcePool::synchronizeConsumerStream(const Fence fence, const ConsumerStream consumerStream) {
    device.synchronizeFenceWithConsumerStream(fence, consumerStream);
}

void ResourcePool::synchronizeHostDevice(const Fence fence) {
    device.synchronizeFenceWithHostDevice(fence);
}
