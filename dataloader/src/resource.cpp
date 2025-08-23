#include "resource.h"
#include <format>

#include "data.h"
#include "dataset.h"

MirroredAllocator::MirroredAllocator(HostAndGpuDeviceInterface *device)
    : device(device), hostData(nullptr), gpuData(nullptr) {
}

MirroredAllocator::~MirroredAllocator() {
    device->freeEverything();
}

void MirroredAllocator::reserveAtLeast(const size_t newNumItems, const size_t newItemSize) {
    std::unique_lock lock(allocateMutex); // TODO: Remove?
    newGenIdx += 1;

    LOG_DEBUG("reserveAtLeast");
    cudaStreamSynchronize(ResourcePool::getInstance()->getCudaStream());
    // TODO: This is ugly, but necessary atp bc. otherwise we might not be done async copying when we hand off to jax.
    // TODO: Find a better option than this, so we don't stall quite as much.
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

    // TODO: Maybe allocate lazily, so we don't slow down program startup.
}

bool MirroredAllocator::allocate(Allocation &alloc) {
    std::unique_lock lock(allocateMutex); // TODO: Remove?
    LOG_DEBUG("allocGpuData.size()=={}", allocAndHandOffGpuData.size());
    if (isDrainingOldGeneration()) {
        if (allocAndHandOffGpuData.empty()) {
            LOG_DEBUG("draining; case genIdx=newGenIdx");
            genIdx += 1;
        } else {
            // Error: Tried to allocate while draining old generation of allocations.
            LOG_DEBUG("draining; case error {}", allocAndHandOffGpuData.size());

            return false;
        }
    }

    if (freeList.empty()) {
        // Error: Tried to allocate in an empty pool.
        return false;
    }

    alloc = freeList.back();
    freeList.pop_back();

    // TODO: Remove later?
    if (allocAndHandOffGpuData.contains(alloc.gpu)) {
        throw std::runtime_error(std::format("Tried to allocate space ({} host, {}) gpu that has not yet been freed.",
                                             reinterpret_cast<uint64_t>(alloc.host),
                                             reinterpret_cast<uint64_t>(alloc.gpu)));
    }


    return true;
}

void MirroredAllocator::free(const uint8_t *gpuPtr) {
    std::unique_lock lock(allocateMutex); // TODO: Remove?
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

    memoryNotify.fetch_add(1);
    memoryNotify.notify_all();
}

void MirroredAllocator::handOff(const uint8_t *gpuPtr) {
    std::unique_lock lock(allocateMutex); // TODO: Remove?
    allocAndHandOffGpuData.emplace(gpuPtr);
}

bool MirroredAllocator::isDrainingOldGeneration() const {
    return newGenIdx != genIdx;
}

CudaHostAndGpuDeviceInterface::CudaHostAndGpuDeviceInterface() {
    if (const auto err = cudaDeviceGetDefaultMemPool(&pool, /*device=*/0); err != cudaSuccess) {
        throw std::runtime_error("cudaDeviceGetDefaultMemPool failed");
    }

    setGpuMemoryPoolReleaseThreshold(1);

    if (const cudaError_t err = cudaStreamCreate(&stream);
        err != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed.");
    }
}

uint8_t *CudaHostAndGpuDeviceInterface::hostMemoryChangeSizeAndInvalidateMemory(const size_t size) {
    cudaStreamSynchronize(stream);

    freeEverything();

    if (const cudaError_t err = cudaHostAlloc(&hostData, size, cudaHostAllocWriteCombined); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaHostAlloc failed to allocate {} MB", size / 1024 / 1024));
    }

    return hostData;
}

uint8_t *CudaHostAndGpuDeviceInterface::gpuMemoryIncreasePoolSizeAndInvalidateMemory(const size_t size) {
    setGpuMemoryPoolReleaseThreshold(size);

    if (gpuData != null) {
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
    if (hostData != null) {
        if (const cudaError_t err = cudaFreeHost(hostData); err != cudaSuccess) {
            throw std::runtime_error("cudaFreeHost failed to free data.");
        }
    }

    // The gpu memory is handled by a cuda memory pool.
    // Beyond destroying the stream, no additional cleanup is required.

    if (const auto err = cudaStreamDestroy(stream); err != cudaSuccess) {
        LOG_ERROR("cudaStreamDestroy failed.");
        std::terminate();
    }
}

uint64_t CudaHostAndGpuDeviceInterface::insertNextFenceIntoStream() {
    cudaEvent_t event;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        throw std::runtime_error("cudaEventCreate failed.");
    }

    if (cudaEventRecord(event, stream) != cudaSuccess) {
        throw std::runtime_error("cudaEventRecord failed.");
    }

    auto fence = eventIndex.fetch_add(1);
    fenceToEventMap.emplace(fence, event);
    return fence;
}

void CudaHostAndGpuDeviceInterface::synchronizeFenceWithConsumerStream(const uint64_t fence,
                                                                       const uintptr_t consumerStream) {
    const auto found = fenceToEventMap.find(fence);
    if (found == fenceToEventMap.end()) {
        throw std::runtime_error("Fence is invalid.");
    }

    const auto consumer = reinterpret_cast<cudaStream_t>(consumerStream);
    const cudaEvent_t event = found->second;
    if (cudaStreamWaitEvent(consumer, event) != cudaSuccess) {
        throw std::runtime_error("cudaStreamWaitEvent failed.");
    }

    fenceToEventMap.erase(found);
    if (cudaEventDestroy(event) != cudaSuccess) {
        throw std::runtime_error("cudaEventDestroy failed.");
    } // TODO: this is illegal as we need to sync to the event multiple times. can only delete in the very end...
}

void CudaHostAndGpuDeviceInterface::synchronizeFenceWithHostDevice(const uint64_t fence) {
    const auto found = fenceToEventMap.find(fence);
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
    } // TODO: this is illegal as we need to sync to the event multiple times. can only delete in the very end...
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


SharedPtr<ResourcePool> ResourcePool::instance;
std::atomic_uint64_t ResourcePool::acquiredClientId = std::atomic_uint64_t(-1);

SharedPtr<ResourcePool> ResourcePool::getInstance() {
    if (!instance) {
        instance = SharedPtr(new ResourcePool(), true);
    }

    return {instance};
}

MirroredAllocator *ResourcePool::getAllocator() noexcept {
    return &allocator;
}

bool PrefetchItem::operator<(const PrefetchItem &other) const {
    // Make sure to sort the priority queue such that the smallest elements have priority.
    return datasetStartingOffset > other.datasetStartingOffset;
}

ResourcePool::ResourcePool() : allocator(&device),
                               threadPool(
                                   [this](const size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount) {
                                       this->threadMain(threadIdx, desiredThreadCount);
                                   }, 0) {
}

ResourcePool::~ResourcePool() {
}

void ResourcePool::threadMain(const size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount) {
#define IS_SHUTDOWN_REQUIRED() !(threadIdx < desiredThreadCount.load())
#define DO_SHUTDOWN_IF_NECESSARY() if (IS_SHUTDOWN_REQUIRED()) { break; }

    const auto temporaryArena = std::make_unique<uint8_t *>(new uint8_t[dl.outputBatchMemorySize]);
    auto temporaryAllocator = BumpAllocator(*temporaryArena, dl.outputBatchMemorySize);

    uint64_t rememberedClientId = acquiredClientId;

    while (IS_SHUTDOWN_REQUIRED()) {
        const auto [startingOffset, genIdx, batchPaths] = dl.batchedDataset.getNextInFlightBatch();
        // TODO: const size_t barrierIdx = lastBarrierIdx++ % dl.prefetchSize;
        const std::vector<Head> &heads = dl.batchedDataset.getDataset().getHeads();

        // Make sure threads submit their batches in dataset order.
        std::unique_lock lock(prefetchCacheMutex);
        prefetchCacheNotify.wait(lock, [this, startingOffset, genIdx, threadIdx, &desiredThreadCount] {
            // Make sure to leave the conditional variable when shutdown is already enabled.
            return IS_SHUTDOWN_REQUIRED()
                   || dl.batchedDataset.getGenIdx().load() != genIdx
                   || startingOffset - static_cast<int>(dl.batchSize) == dl.batchedDataset.getLastWaitingBatch().load();
        });
        DO_SHUTDOWN_IF_NECESSARY()


        // For each head, load all batch items into one contigous cpu array.
        std::vector<CpuAllocation> hostAllocations;
        temporaryAllocator.reset();
        for (size_t headIdx = 0; headIdx < heads.size(); headIdx++) {
            hostAllocations.push_back(
                loadFilesFromHeadIntoContigousBatch(temporaryAllocator, batchPaths, heads, headIdx)
            );
        }
        DO_SHUTDOWN_IF_NECESSARY()

        // At this point, the thread has loaded the batch into cpu ram.
        // As soon as pinned memory is available, we start copying to the gpu.
        // The assumption is, that resource load, decompression and decode takes
        // much longer than host->gpu copy.


        // Grab a host,gpu memory pair.
        auto allocations = BumpAllocator(Allocation{}, 0);
        if (!dl.resourceClient.allocate(allocations)) {
            // TODO: I think this if can be deleted too.
            while (!IS_SHUTDOWN_REQUIRED() && !allocations.getArena()) {
                // TODO: Is allocations.getArena() necessary? I think the break makes this redundant.
                const uint64_t old = allocator.memoryNotify.load();

                // TODO: I don't think we're handling acquired correctly here, as this would just infinity loop?
                if (dl.resourceClient.allocate(allocations)) {
                    break;
                }

                allocator.memoryNotify.wait(old);
            }
        }
        DO_SHUTDOWN_IF_NECESSARY()

        // TODO: Make sure to not touch the allocation if the current resource client has lost access to the pool.
        // TODO: Otherwise, you'd get null pointer exceptions.
        std::memcpy(allocations.getArena().host, temporaryAllocator.getArena(), dl.outputBatchMemorySize);
        std::vector<uint8_t *> gpuAllocations;
        std::vector<uint64_t> fences;
        if (allocations.getArena()) {
            for (size_t headIdx = 0; headIdx < heads.size(); headIdx++) {
                const CpuAllocation &nonPinnedCpuAllocation = hostAllocations[headIdx];
                const auto &[host, gpu, _] = allocations.allocate(nonPinnedCpuAllocation.batchBufferSize);

                // Start async upload to gpu memory as soon as possible.
                // TODO: This also checked for inactive clients. Needs to be done manually, now.
                // TODO: This also locked a mutex. Maybe this is still necessary?
                device.copyFromHostToGpuMemory(host, gpu, nonPinnedCpuAllocation.batchBufferSize);
                gpuAllocations.push_back(gpu);
                // TODO: This also checked for inactive clients. Needs to be done manually, now.
                // TODO: This needs to be synchronized by mutex?
                fences.push_back(device.insertNextFenceIntoStream());
            }
        }

        DO_SHUTDOWN_IF_NECESSARY()


        if (dl.batchedDataset.getGenIdx().load() == genIdx) {
            prefetchCache.push({
                .datasetStartingOffset = startingOffset,
                .gpuAllocations = gpuAllocations
                .fences = fences,
            });

            dl.batchedDataset.markBatchWaiting(startingOffset);
        }

        lock.unlock();

        prefetchCacheNotify.notify_all();
    }
}

void ResourcePool::acquire(const uint64_t clientId, const size_t numItems, const size_t itemSize) {
    if (acquiredClientId != clientId) {
        LOG_DEBUG("resource pool acquired with clientId: {}", clientId);
        acquiredClientId = clientId;
        allocator.reserveAtLeast(numItems, itemSize);


        // TODO:
        std::unique_lock lock(prefetchCacheMutex);

        dl.batchedDataset.forgetInFlightBatches();
        while (!prefetchCache.empty()) {
            prefetchCache.pop(); // Clear is not supported
        }

        prefetchCacheNotify.notify_all();
    }
}

void ResourcePool::synchronizeConsumerStream(const uint64_t fence, const uintptr_t consumerStream) {
    device.synchronizeFenceWithConsumerStream(fence, consumerStream);
}

void ResourcePool::synchronizeHostDevice(const uint64_t fence) {
    device.synchronizeFenceWithHostDevice(fence);
}

ResourceClient::ResourceClient(const uint64_t clientId, const size_t numItems, const size_t itemSize)
    : clientId(clientId), numItems(numItems), itemSize(itemSize),
      pool(ResourcePool::getInstance()) {
}

void ResourceClient::acquire() {
    std::unique_lock lock(mutex);
    pool->acquire(clientId, numItems, itemSize);
}

bool ResourceClient::allocate(BumpAllocator<Allocation> &subAllocator) {
    std::unique_lock lock(mutex);
    if (ResourcePool::acquiredClientId.load() != clientId) {
        // Do not allocate if the client has not acquired the pool.
        subAllocator = BumpAllocator<Allocation>({}, 0);
        return false;
    }

    Allocation alloc = {};
    const bool success = pool->getAllocator()->allocate(alloc);
    subAllocator = BumpAllocator(alloc, alloc.size);
    return success;
}

#define RETURN_IF_INACTIVE() if (ResourcePool::acquiredClientId.load() != clientId) return

int ResourceClient::getClientId() const {
    return clientId;
}
