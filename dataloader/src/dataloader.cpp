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
    outputBatchMemorySize = alignUp(outputBatchMemorySize, 16);

    resourceClient.acquire(prefetchSize, outputBatchMemorySize);

    threadPool.start();
}

DataLoader::~DataLoader() {
    shutdown = true;
    prefetchSemaphore.disable();
    prefetchCacheNotify.notify_all();
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

    LOG_DEBUG("Loading from; startingOffset: {}, genIdx: {}", startingOffset, batchedDataset.getGenIdx().load());

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
            LOG_DEBUG("Idiot index ++ ({}, {}, idx {}, size {})",
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
