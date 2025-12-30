#include "resource.h"
#include <format>

#include "dataio.h"
#include "hostAndGpuInterfaces/CudaHostAndGpuDeviceInterface.h"

MirroredAllocator::MirroredAllocator(const std::shared_ptr<HostAndGpuDeviceInterface> &device) : device(device) {
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

std::shared_ptr<HostAndGpuDeviceInterface> ResourcePool::deviceForNewResourcePool;

ResourcePool *ResourcePool::singleton;

ResourcePool &ResourcePool::get() {
    // Intentional leak to avoid implicit destruction order issues; use shutdown() to free resources explicitly.
    if (!singleton) {
        if (!deviceForNewResourcePool) {
            // This needs to be in here, because the augmentations, compression etc. are tested without sanitizers.
            // The sanitizers don't play well with CUDA, or any GPU library in general.
            // So we cannot just eagerly initialize them in library.cpp, which leads to this unfortunate code.
            deviceForNewResourcePool = std::make_shared<CudaHostAndGpuDeviceInterface>();
        }
        singleton = new ResourcePool(deviceForNewResourcePool);
    }
    return *singleton;
}

void ResourcePool::shutdownLazily() {
    if (singleton) {
        get().shutdown();
    }
}

void ResourcePool::setDeviceForNewResourcePool(const std::shared_ptr<HostAndGpuDeviceInterface> &device) {
    deviceForNewResourcePool = device;
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

ResourcePool::ResourcePool(const std::shared_ptr<HostAndGpuDeviceInterface> &device)
    : device(device), allocator(device), dl(nullptr),
      threadPool(
          [this](const size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount, uint64_t initialGenIdx) {
              this->threadMain(threadIdx, desiredThreadCount, initialGenIdx);
          },
          0,
          [this] {
              // Allow the threads to continue exection.
              genChangeSendOff->count_down();
          },
          [this] { wakeupAllThreads(); }) {
}

ResourcePool::~ResourcePool() {
    threadPool.resize(0, genIdx.load());
    prefetchCacheNotify.notify_all();
}

void ResourcePool::shutdown() {
    // Stop threads and wake any waiters.
    threadPool.resize(0, genIdx.load());

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

void ResourcePool::threadMain(const size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount,
                              const uint64_t initialGenIdx) {
#define IS_SHUTDOWN_REQUIRED() (threadIdx >= desiredThreadCount.load())
#define IS_GENCHANGE_REQUIRED() (rememberedGenIdx != genIdx.load())
#define IS_SHUTDOWN_OR_GENCHANGE_REQUIRED() (IS_SHUTDOWN_REQUIRED() || IS_GENCHANGE_REQUIRED())

#define DO_SHUTDOWN_IF_NECESSARY() \
    if (IS_SHUTDOWN_REQUIRED()) { return; }

#define DO_SHUTDOWN_OR_GENCHANGE_IF_NECESSARY() \
    if (IS_SHUTDOWN_REQUIRED()) { return; } \
    if (IS_GENCHANGE_REQUIRED()) { continue; }

    LOG_DEBUG("thread {} started", threadIdx);
    uint64_t rememberedGenIdx = initialGenIdx;

    const auto inputBuffer = std::make_unique<uint8_t[]>(dl->maxInputBatchMemorySize);
    auto inputArena = BumpAllocator(inputBuffer.get(), dl->maxInputBatchMemorySize);

    const auto outputBuffer = std::make_unique<uint8_t[]>(dl->outputBatchMemorySize);
    auto outputArena = BumpAllocator(outputBuffer.get(), dl->outputBatchMemorySize);

    const size_t augPipeBufSize = dl->augPipe->getMaximumRequiredBufferSize();
    const auto augPipeBuffer1 = std::make_unique<uint8_t[]>(augPipeBufSize);
    const auto augPipeBuffer2 = std::make_unique<uint8_t[]>(augPipeBufSize);

    while (!IS_SHUTDOWN_REQUIRED()) {
        if (IS_GENCHANGE_REQUIRED()) {
            genChangeAssemble->count_down();
            LOG_DEBUG("thread {} waiting for genchange", threadIdx);
            rememberedGenIdx = genIdx.load();
            genChangeSendOff->wait();
            continue;
        }


        const auto [startingOffset, batchPaths] = dl->batchedDataset.getNextInFlightBatch();
        auto &bds = dl->batchedDataset;
        auto &ds = bds.getDataset();
        const std::vector<ItemKey> &itemKeys = ds->getDataSource()->getItemKeys();

        // For each head, load all batch items into one contigous cpu array.
        std::vector<CpuAllocation> inputAllocsOnHost;
        std::vector<uint8_t *> outputAllocsOnHost;
        inputArena.reset();
        outputArena.reset();
        for (size_t i = 0; i < itemKeys.size(); i++) {
            // The convention is that data sources always read into cpu memory, even if no
            // augmentations are applied. This is because we cannot guarantee,
            // that the decoders never read from pinned memory.
            // Reading from pinned memory can be very slow, as it is not always backed by RAM.
            inputAllocsOnHost.push_back(ds->getDataSource()->loadItemSliceIntoContigousBatch(
                inputArena, batchPaths, i, dl->maxBytesOfInputPerItemOfItemKey[i]
            ));
            outputAllocsOnHost.push_back(outputArena.allocate(dl->bytesOfOutputOfItemKey[i]));
        }
        DO_SHUTDOWN_OR_GENCHANGE_IF_NECESSARY()


        // TODO: Move augmentations further up.
        // TODO: Technically, no pair has to be available while we are still decoding and augmenting.
        bool lastSchemaIsLeakingMemory = false;
        DataProcessingSchema lastSchema;
        for (size_t i = 0; i < itemKeys.size(); i++) {
            const auto itemKey = itemKeys[i];
            const CpuAllocation &inputAlloc = inputAllocsOnHost[i];
            uint8_t *outputAlloc = outputAllocsOnHost[i];
            switch (itemKey.type) {
                case ItemType::NONE:
                    if (itemKey.probeResult.shape != inputAlloc.shapes[i]) {
                        throw std::runtime_error("Item type NONE requires the shape to be constant across files.");
                    }

                    // TODO (Speed): Zero copy.
                    memcpy(outputAlloc, inputAlloc.batchBuffer.uint8, dl->bytesOfOutputOfItemKey[i]);
                    std::printf("probe.shape==%lu\n", dl->bytesOfOutputOfItemKey[i]);
                    break;
                case ItemType::POINTS:
                    throw std::runtime_error("done");
                    ASSERT(!lastSchema.inputShapesPerAug.empty());
                    dl->augPipe->augmentWithPoints(
                        inputAlloc.shapes,
                        itemKey.probeResult.dtype,
                        inputAlloc.batchBuffer.uint8,
                        augPipeBuffer1.get(), augPipeBuffer2.get(),
                        outputAlloc,
                        lastSchema
                    );
                    break;
                case ItemType::RASTER:
                    if (lastSchemaIsLeakingMemory) {
                        dl->augPipe->freeProcessingSchema(lastSchema);
                    }
                    lastSchemaIsLeakingMemory = true;
                    lastSchema = dl->augPipe->getProcessingSchema(inputAlloc.shapes, startingOffset);
                    dl->augPipe->augmentWithRaster(
                        itemKey.probeResult.dtype,
                        inputAlloc.batchBuffer.uint8,
                        outputAlloc,
                        augPipeBuffer1.get(), augPipeBuffer2.get(),
                        dl->maxBytesOfInputPerItemOfItemKey[i],
                        lastSchema
                    );
                    break;
                default:
                    throw std::runtime_error("Unsupported spatial hint in worker thread loop.");
            }
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


        ASSERT(allocations.getArena().host != nullptr);
        std::vector<uint8_t *> gpuAllocations;
        std::vector<Fence> fences;
        for (size_t itemKeyIdx = 0; itemKeyIdx < itemKeys.size(); itemKeyIdx++) {
            const auto &batchBuffer = outputAllocsOnHost[itemKeyIdx];
            const size_t batchBufferSize = dl->bytesOfOutputOfItemKey[itemKeyIdx];
            const auto &[host, gpu, _2] = allocations.allocate(batchBufferSize);

            // Start async upload to gpu memory as soon as possible.
            // TODO: This also locked a mutex. Maybe this is still necessary?

            std::memcpy(host, batchBuffer, batchBufferSize);
            // TODO: Think about pinned memory availabilty. I think this is right, but recheck.

            device->copyFromHostToGpuMemory(host, gpu, batchBufferSize);
            gpuAllocations.push_back(gpu);
            // TODO: This needs to be synchronized by mutex?
            fences.push_back(device->insertNextFenceIntoStream());
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
        LOG_DEBUG("Waiting for Gen change send-off.");
        genChangeSendOff = std::make_unique<std::latch>(1);
        LOG_DEBUG("Gen change sent-off!");
        if (dl != nullptr) {
            LOG_DEBUG("Assembling {} threads.", dl->numThreads);
            genChangeAssemble = std::make_unique<std::latch>(dl->numThreads);
            // Wait for all threads to stop working.
            wakeupAllThreads();
            genChangeAssemble->wait();
            LOG_DEBUG("Assembled threads.");

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
        threadPool.resize(dataLoader->numThreads, genIdx.load());
    }

    // Fetch the next batch.
    std::unique_lock lock(prefetchCacheMutex);
    prefetchCacheNotify.wait(lock, [this] { return !prefetchCache.empty(); });

    const PrefetchItem prefetchItem = prefetchCache.top();
    prefetchCache.pop();
    LOG_DEBUG("acquireAndGetNextBatch datasetStartingOffset={}", prefetchItem.datasetStartingOffset);
    dl->batchedDataset.popWaitingBatch(prefetchItem.datasetStartingOffset);

    lock.unlock();

    return prefetchItem;
}

void ResourcePool::synchronizeConsumerStream(const Fence fence, const ConsumerStream consumerStream) const {
    device->synchronizeFenceWithConsumerStream(fence, consumerStream);
}

void ResourcePool::synchronizeHostDevice(const Fence fence) const {
    device->synchronizeFenceWithHostDevice(fence);
}
