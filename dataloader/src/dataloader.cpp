#include "dataloader.h"
#include "utils.h"
#include "data.h"
#include <filesystem>
#include <dlpack/dlpack.h>

#define CHS 3

namespace py = pybind11;
namespace fs = std::filesystem;

bool PrefetchItem::operator<(const PrefetchItem &other) const {
    // Make sure to sort the priority queue such that the smallest elements have priority.
    return datasetStartingOffset > other.datasetStartingOffset;
}

std::atomic_int DataLoader::nextId;

DataLoader::DataLoader(
    Dataset &_dataset,
    const int _batchSize,
    const int _numThreads,
    const int _prefetchSize
) : id(nextId.fetch_add(1)),
    batchedDataset(std::move(_dataset), _batchSize),
    batchSize(_batchSize),
    numThreads(_numThreads),
    prefetchSize(_prefetchSize),
    // The memory of the last getNextBatch cannot be invalidated until the next getNextBatch call.
    // Therefore, make sure the semaphore can't wrap around the buffer directly after the first call.
    prefetchSemaphore(_prefetchSize - 1),
    outputBatchMemorySize(0),
    resourceClient(id, _prefetchSize),
    threadPool([this] { this->threadMain(); }, _numThreads) {
    if (batchSize <= 0 || _numThreads <= 0 || _prefetchSize <= 0) {
        throw std::runtime_error(
            "Batch size, the number of threads and the prefetch size need to be strictly positive.");
    }

    if (_prefetchSize < _numThreads) {
        throw std::runtime_error(
            "Prefetch size must be larger or equal than the number of threads.");
    }

    numberOfBatches = (batchedDataset.getDataset().getEntries().size() + batchSize - 1) / batchSize;

    for (const Head &head: batchedDataset.getDataset().getHeads()) {
        size_t itemMemorySize = head.getBytesPerItem();
        for (const int dim: head.getShape()) {
            itemMemorySize *= dim;
        }

        outputBatchMemorySize += batchSize * itemMemorySize;
    }

    resourceClient.acquire(prefetchSize, outputBatchMemorySize);

    threadPool.start();
}

DataLoader::~DataLoader() {
    shutdown = true;
    prefetchSemaphore.disable();
    prefetchCacheNotify.notify_all();
}

void DataLoader::threadMain() {
    while (!shutdown) {
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

// ReSharper disable once CppParameterMayBeConstPtrOrRef
void deleter(DLManagedTensor *self) {
    ResourcePool::getInstance()->getAllocator()->free(static_cast<uint8_t *>(self->dl_tensor.data));
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete self;
}

py::dict DataLoader::getNextBatch() {
    if (resourceClient.acquire(prefetchSize, outputBatchMemorySize)) {
        std::unique_lock lock(prefetchCacheMutex);

        batchedDataset.forgetInFlightBatches();
        while (!prefetchCache.empty()) prefetchCache.pop(); // Clear is not supported

        prefetchSemaphore.releaseAll();
        prefetchCacheNotify.notify_all();
    }

    std::unique_lock lock(prefetchCacheMutex);
    prefetchCacheNotify.wait(lock, [this] { return !prefetchCache.empty(); });

    const auto [startingOffset, barrierIdx, gpuAllocations] = prefetchCache.top();
    prefetchCache.pop();
    batchedDataset.popWaitingBatch(startingOffset);

    lock.unlock();
    prefetchSemaphore.release();

    debugLog("Loading from; startingOffset: %d, genIdx: %d\n", startingOffset, batchedDataset.getGenIdx().load());

    py::dict pyBatch;
    const auto &heads = batchedDataset.getDataset().getHeads();
    for (size_t i = 0; i < heads.size(); i++) {
        const Head &head = heads[i];
        uint8_t *gpuAllocation = gpuAllocations[i];
        ResourcePool::getInstance()->getAllocator()->handOff(gpuAllocation);

        const std::vector<int> &shape = head.getShape();
        const int ndim = static_cast<int>(shape.size()) + 1;

        const auto shapeArr = new int64_t[ndim]{};
        shapeArr[0] = static_cast<int64_t>(batchSize);
        for (size_t s = 0; s < shape.size(); s++) {
            shapeArr[s + 1] = shape[s];
        }

        uint8_t itemCode;
        switch (head.getItemFormat()) {
            case ItemFormat::FLOAT: itemCode = kDLFloat;
                break;
            case ItemFormat::UINT: itemCode = kDLUInt;
                break;
            default:
                throw std::runtime_error("Invalid item format.");
        }

        const auto *dlMngTensor = new DLManagedTensor{
            .dl_tensor = {
                .data = gpuAllocation,
                .device = {
                    .device_type = kDLCUDA,
                    .device_id = 0 // TODO: This hardcodes GPU0.
                },
                .ndim = ndim,
                .dtype = {
                    .code = itemCode,
                    .bits = static_cast<uint8_t>(8 * head.getBytesPerItem()),
                    .lanes = 1
                },
                .shape = shapeArr,
                .strides = null,
                .byte_offset = 0
            },
            .manager_ctx = this,
            .deleter = &deleter
        };

        ResourcePool::getInstance().acquire();
        pyBatch[head.getDictName().c_str()] = py::capsule(dlMngTensor, "dltensor");
    }

    /*for (size_t i = 0; i < prefetchCache.size(); i++) {
        const auto &pair = prefetchCache[i];
        if (pair.barrierIdx == barrierIdx) {
            debugLog("Idiot index ++ (%lu, %lu, idx %lu, size %lu) \n",
                        pair.barrierIdx,
                        barrierIdx, i, prefetchCache.size() + 1);
        }
    }*/

    resourceClient.sync(barrierIdx);

    return pyBatch;
}

size_t DataLoader::getNumberOfBatches() const {
    return numberOfBatches;
}
