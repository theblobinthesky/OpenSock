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
    const Dataset &_dataset,
    const int _batchSize,
    const int _numThreads,
    const int _prefetchSize
) : id(nextId.fetch_add(1)),
    dataset(_dataset),
    batchSize(_batchSize),
    numThreads(_numThreads),
    prefetchSize(_prefetchSize),
    // The memory of the last getNextBatch cannot be invalidated until the next getNextBatch call.
    // Therefore, make sure the semaphore can't wrap around the buffer directly after the first call.
    prefetchSemaphore(_prefetchSize - 1),
    lastStartingOffset(-static_cast<int>(batchSize)),
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

    numberOfBatches = (dataset.getEntries().size() + batchSize - 1) / batchSize;

    for (const Head &head: dataset.getHeads()) {
        size_t itemMemorySize = sizeof(float);
        for (const int dim: head.getShape()) {
            itemMemorySize *= dim;
        }

        outputBatchMemorySize += batchSize * itemMemorySize;
    }

    ResourcePool::reserveAtLeast(outputBatchMemorySize * _prefetchSize);
    // resourceClient.acquire();

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

        datasetMutex.lock();
        const auto [startingOffset, genIdx, batchPaths] = dataset.getNextBatchI(batchSize);
        datasetMutex.unlock();

        const size_t barrierIdx = lastBarrierIdx++ % prefetchSize;
        const std::vector<Head> &heads = dataset.getHeads();

        // For each head, load all batch items into one contigous cpu array.
        MultipleAllocations allocations = resourceClient.allocate(outputBatchMemorySize);

        // Make sure to not touch the allocation if the current resource client has lost access to the pool.
        // Otherwise, you'd get null pointer exceptions.
        if (allocations) {
            for (size_t headIdx = 0; headIdx < heads.size(); headIdx++) {
                const Head &head = heads[headIdx];

                Allocation allocation = {};
                switch (head.getFilesType()) {
                    case FileType::JPG:
                        allocation = loadJpgFiles(allocations, batchPaths, heads, headIdx);
                        break;
                    case FileType::NPY:
                        allocation = loadNpyFiles(allocations, batchPaths, heads, headIdx);
                        break;
                    default:
                        throw std::runtime_error("Cannot load an unsupported file type.");
                }

                // Start async upload to gpu memory as soon as possible.
                resourceClient.copy(allocation.gpu, allocation.host, allocation.size);
            }
        }

        resourceClient.insertBarrier(barrierIdx);

        // Make sure threads submit their batches in dataset order.
        std::unique_lock lock(prefetchCacheMutex);
        prefetchCacheNotify.wait(lock, [this, startingOffset, genIdx] {
            // Make sure to leave the conditional variable when shutdown is already enabled.
            return shutdown
                   || dataset.getGenIdx().load() != genIdx
                   || startingOffset - static_cast<int>(batchSize) == lastStartingOffset.load();
        });
        if (shutdown) break;

        if (dataset.getGenIdx().load() == genIdx) {
            prefetchCache.push({
                .datasetStartingOffset = startingOffset,
                .barrierIdx = barrierIdx,
                .gpuAllocations = allocations.getGpuAllocations() // Maybe std::move ?
            });

            lastStartingOffset = startingOffset;
        }

        lock.unlock();

        prefetchCacheNotify.notify_all();
    }
}

void deleter(DLManagedTensor *self) {
    ResourcePool::getInstance().release();
}

py::dict DataLoader::getNextBatch() {
    if (resourceClient.acquire()) {
        std::unique_lock lock(prefetchCacheMutex);
        dataset.resetByNumBatches(prefetchCache.size(), batchSize);
        lastStartingOffset = dataset.getOffset() - static_cast<int>(batchSize);
        while (!prefetchCache.empty()) prefetchCache.pop(); // Clear is not supported
        lock.unlock();

        prefetchSemaphore.releaseAll();
        prefetchCacheNotify.notify_all();
    }

    std::unique_lock lock(prefetchCacheMutex);
    prefetchCacheNotify.wait(lock, [this] { return !prefetchCache.empty(); });

    const auto [_, barrierIdx, gpuAllocations] = prefetchCache.top();
    prefetchCache.pop();

    lock.unlock();
    prefetchSemaphore.release();

    py::dict pyBatch;
    const auto &heads = dataset.getHeads(); // TODO: Prevent copy.
    for (size_t i = 0; i < heads.size(); i++) {
        const Head &head = heads[i];
        uint8_t *gpuAllocation = gpuAllocations[i];

        const std::vector<int> &shape = head.getShape();
        const int ndim = static_cast<int>(shape.size()) + 1;

        const auto shapeArr = new int64_t[ndim]{};
        shapeArr[0] = static_cast<int64_t>(batchSize);
        for (size_t s = 0; s < shape.size(); s++) {
            shapeArr[s + 1] = shape[s];
        }

        int64_t lastStride = 1;
        const auto stridesArr = new int64_t[shape.size() + 1]{};
        for (int s = static_cast<int>(shape.size()); s >= 0; s--) {
            stridesArr[s] = lastStride;
            lastStride *= shapeArr[s];
        }

        const auto *dlMngTensor = new DLManagedTensor{
            .dl_tensor = {
                .data = gpuAllocation,
                .device = {
                    .device_type = kDLCUDA,
                    .device_id = 0 // This hardcodes GPU0.
                },
                .ndim = ndim,
                .dtype = {
                    .code = kDLFloat,
                    .bits = 32,
                    .lanes = 1
                },
                .shape = shapeArr,
                .strides = null,
                .byte_offset = 0
            },
            .manager_ctx = null,
            .deleter = &deleter
        };

        ResourcePool::getInstance().acquire();
        pyBatch[head.getDictName().c_str()] = py::capsule(
            dlMngTensor, "dltensor", [](void *ptr) {
                const auto *dlManagedTensor = static_cast<DLManagedTensor *>(ptr);
                delete[] dlManagedTensor->dl_tensor.shape;
                delete[] dlManagedTensor->dl_tensor.strides;
                delete dlManagedTensor;
            });
    }

    /*for (size_t i = 0; i < prefetchCache.size(); i++) {
        const auto &pair = prefetchCache[i];
        if (pair.barrierIdx == barrierIdx) {
            std::printf("Idiot index ++ (%lu, %lu, idx %lu, size %lu) \n",
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
