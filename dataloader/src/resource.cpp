#include "resource.h"
#include <format>

MirroredAllocator::MirroredAllocator(HostMemoryInterface *hostIf,
                                     GpuMemoryInterface *gpuIf) : hostIf(hostIf), gpuIf(gpuIf), hostData(nullptr),
                                                                  gpuData(nullptr) {
}

MirroredAllocator::~MirroredAllocator() {
    hostIf->freeEverything();
    gpuIf->freeEverything();
}

void MirroredAllocator::reserveAtLeast(const size_t newNumItems, const size_t newItemSize) {
    std::unique_lock lock(allocateMutex); // TODO: Remove?
    newGenIdx += 1;

    LOG_DEBUG("reserveAtLeast");
    cudaStreamSynchronize(ResourcePool::getInstance()->getCudaStream());
    // TODO: This is ugly, but necessary atp bc. otherwise we might not be done async copying when we hand off to jax.
    // TODO: Find a better option than this, so we don't stall quite as much.
    if (const size_t requiredSize = newNumItems * newItemSize; requiredSize > numItems * itemSize) {
        hostData = hostIf->changeSizeAndInvalidateMemory(requiredSize);
        gpuData = gpuIf->increasePoolSizeAndInvalidateMemory(requiredSize);
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

    const size_t i = (gpuPtr - gpuData) / itemSize;
    freeList.push_back(Allocation{
        .host = hostData + i * itemSize,
        .gpu = gpuData + i * itemSize,
        .size = itemSize
    });

    memoryNotify.notify_all();
}

void MirroredAllocator::handOff(const uint8_t *gpuPtr) {
    std::unique_lock lock(allocateMutex); // TODO: Remove?
    allocAndHandOffGpuData.emplace(gpuPtr);
}

bool MirroredAllocator::isDrainingOldGeneration() const {
    return newGenIdx != genIdx;
}

uint8_t *CudaHostMemoryInterface::changeSizeAndInvalidateMemory(const size_t size) {
    cudaStreamSynchronize(ResourcePool::getInstance()->getCudaStream());

    freeEverything();

    if (const cudaError_t err = cudaHostAlloc(&data, size, cudaHostAllocWriteCombined); err != cudaSuccess) {
        throw std::runtime_error(std::format("cudaHostAlloc failed to allocate {} MB", size / 1024 / 1024));
    }

    return data;
}

void CudaHostMemoryInterface::freeEverything() {
    if (data != null) {
        if (const cudaError_t err = cudaFreeHost(data); err != cudaSuccess) {
            throw std::runtime_error("cudaFreeHost failed to free data.");
        }
    }
}

CudaGpuMemoryInterface::CudaGpuMemoryInterface() {
    if (const auto err = cudaDeviceGetDefaultMemPool(&pool, /*device=*/0); err != cudaSuccess) {
        throw std::runtime_error("cudaDeviceGetDefaultMemPool failed");
    }

    setPoolReleaseThreshold(1);
}

uint8_t *CudaGpuMemoryInterface::increasePoolSizeAndInvalidateMemory(const size_t size) {
    setPoolReleaseThreshold(size);

    const auto stream = ResourcePool::getInstance()->getCudaStream();
    if (data != null) {
        if (const auto err = cudaFreeAsync(data, stream); err != cudaSuccess) {
            throw std::runtime_error("cudaFreeAsync warm-up free failed");
        }
    }

    if (const auto err = cudaMallocAsync(&data, size, stream); err != cudaSuccess) {
        throw std::runtime_error("cudaMallocAsync warm-up alloc failed");
    }

    return data;
}

void CudaGpuMemoryInterface::freeEverything() {
    // Beyond destroying the stream, nothing is necessary here.
}

void CudaGpuMemoryInterface::setPoolReleaseThreshold(const size_t bytes) const {
    if (const auto err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, const_cast<size_t *>(&bytes));
        err != cudaSuccess) {
        throw std::runtime_error("cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold) failed");
    }
}

SharedPtr<ResourcePool> ResourcePool::instance;
std::atomic_int ResourcePool::currentClientId = std::atomic_int(-1);

SharedPtr<ResourcePool> ResourcePool::getInstance() {
    if (!instance) {
        instance = SharedPtr(new ResourcePool(), true);
    }

    return {instance};
}

MirroredAllocator *ResourcePool::getAllocator() noexcept {
    return &allocator;
}

cudaStream_t ResourcePool::getCudaStream() const noexcept {
    return stream;
}

ResourcePool::ResourcePool() : allocator(&hostMemoryIf, &gpuMemoryIf), stream(null) {
    if (const cudaError_t err = cudaStreamCreate(&stream);
        err != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed.");
    }
}

ResourcePool::~ResourcePool() {
    if (const auto err = cudaStreamDestroy(stream); err != cudaSuccess) {
        LOG_DEBUG("cudaStreamDestroy failed.");
        std::terminate();
    }
}

void ResourcePool::threadMain(const size_t threadIdx, const std::atomic_uint32_t &desiredThreadCount) {
    while (threadIdx < desiredThreadCount.load()) {
        prefetchSemaphore.acquire();
        if (shutdown) break;

        const auto [startingOffset, genIdx, batchPaths] = batchedDataset.getNextInFlightBatch();
        const size_t barrierIdx = lastBarrierIdx++ % prefetchSize;
        const std::vector<Head> &heads = batchedDataset.getDataset().getHeads();

        // For each head, load all batch items into one contigous cpu array.
        auto allocations = BumpAllocator(Allocation{}, 0);
        if (!resourceClient.allocate(allocations)) {
            MirroredAllocator &allocator = *ResourcePool::getInstance()->getAllocator();
            while (!allocations.getArena()) {
                std::unique_lock lock(allocator.memoryMutex);
                allocator.memoryNotify.wait(lock, [this, &allocations] {
                    return shutdown || resourceClient.allocate(allocations);
                });
            }
        }


        // Make sure to not touch the allocation if the current resource client has lost access to the pool.
        // Otherwise, you'd get null pointer exceptions.
        std::vector<uint8_t *> gpuAllocations;
        if (allocations.getArena()) {
            for (size_t headIdx = 0; headIdx < heads.size(); headIdx++) {
                const auto [host, gpu, size] = loadFiles(allocations, batchPaths, heads, headIdx);

                // Start async upload to gpu memory as soon as possible.
                resourceClient.copy(gpu, host, size);
                gpuAllocations.push_back(gpu);
            }
        }

        resourceClient.insertBarrier(barrierIdx);

        // Make sure threads submit their batches in dataset order.
        std::unique_lock lock(prefetchCacheMutex);
        prefetchCacheNotify.wait(lock, [this, startingOffset, genIdx] {
            // Make sure to leave the conditional variable when shutdown is already enabled.
            return shutdown
                   || batchedDataset.getGenIdx().load() != genIdx
                   || startingOffset - static_cast<int>(batchSize) == batchedDataset.getLastWaitingBatch().load();
        });
        if (shutdown) break;

        if (batchedDataset.getGenIdx().load() == genIdx) {
            prefetchCache.push({
                .datasetStartingOffset = startingOffset,
                .barrierIdx = barrierIdx,
                .gpuAllocations = gpuAllocations
            });

            batchedDataset.markBatchWaiting(startingOffset);
        }

        lock.unlock();

        prefetchCacheNotify.notify_all();
    }
}

bool ResourcePool::acquire(const int clientId, const size_t numItems, const size_t itemSize) {
    const bool clientChanged = currentClientId != clientId;

    if (clientChanged) {
        LOG_DEBUG("resource pool acquired with clientId: {}", clientId);
        currentClientId = clientId;
        allocator.reserveAtLeast(numItems, itemSize);
    }

    return clientChanged;
}

ResourceClient::ResourceClient(const int clientId, const size_t numBarriers) : clientId(clientId),
                                                                               pool(ResourcePool::getInstance()) {
    for (size_t i = 0; i < numBarriers; i++) {
        cudaEvent_t barrierEvent;
        if (cudaEventCreate(&barrierEvent) != cudaSuccess) {
            throw std::runtime_error("cudaEventCreate failed.");
        }

        barriers.push_back(barrierEvent);
    }
}

ResourceClient::~ResourceClient() {
    for (cudaEvent_t barrier: barriers) {
        cudaEventDestroy(barrier);
    }
}

bool ResourceClient::acquire(const size_t numItems, const size_t itemSize) {
    std::unique_lock lock(mutex);
    return pool->acquire(clientId, numItems, itemSize);
}

bool ResourceClient::allocate(BumpAllocator<Allocation> &subAllocator) {
    std::unique_lock lock(mutex);
    if (ResourcePool::currentClientId.load() != clientId) {
        // Do not allocate if the client has not acquired the pool.
        subAllocator = BumpAllocator<Allocation>({}, 0);
        return false;
    }

    Allocation alloc = {};
    const bool success = pool->getAllocator()->allocate(alloc);
    subAllocator = BumpAllocator(alloc, alloc.size);
    return success;
}

#define RETURN_IF_INACTIVE() if (ResourcePool::currentClientId.load() != clientId) return

void ResourceClient::copy(uint8_t *gpuBuffer,
                          const uint8_t *buffer,
                          const uint32_t size) {
    std::unique_lock lock(mutex);
    RETURN_IF_INACTIVE();

    if (cudaMemcpyAsync(gpuBuffer, buffer, size,
                        cudaMemcpyHostToDevice, ResourcePool::getInstance()->getCudaStream()) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync failed.");
    }
}

void ResourceClient::insertBarrier(const size_t barrierIdx) {
    // std::unique_lock lock(mutex);
    RETURN_IF_INACTIVE();

    if (cudaEventRecord(barriers[barrierIdx], ResourcePool::getInstance()->getCudaStream()) != cudaSuccess) {
        throw std::runtime_error("cudaEventRecord failed.");
    }
}

void ResourceClient::sync(const size_t barrierIdx) {
    // std::unique_lock lock(mutex);
    RETURN_IF_INACTIVE();

    if (cudaEventSynchronize(barriers[barrierIdx]) != cudaSuccess) {
        throw std::runtime_error("cudaEventSynchronize failed.");
    }
}

int ResourceClient::getClientId() const {
    return clientId;
}
