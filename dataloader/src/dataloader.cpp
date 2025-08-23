#include "dataloader.h"
#include "utils.h"
#include "data.h"
#include <filesystem>
#include <dlpack/dlpack.h>

namespace py = pybind11;

std::atomic_uint64_t DataLoader::nextId;

size_t getOutputBatchMemorySize(const BatchedDataset &batchedDataset, const size_t batchSize) {
    size_t outputBatchMemorySize = 0;
    for (const Head &head: batchedDataset.getDataset().getHeads()) {
        size_t itemMemorySize = head.getBytesPerItem();
        for (const int dim: head.getShape()) {
            itemMemorySize *= dim;
        }

        outputBatchMemorySize += batchSize * itemMemorySize;
    }

    return alignUp(outputBatchMemorySize, 16);
}

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
    outputBatchMemorySize(getOutputBatchMemorySize(batchedDataset, batchSize)),
    resourceClient(id, prefetchSize, outputBatchMemorySize) {
    if (batchSize <= 0 || _numThreads <= 0 || _prefetchSize <= 0) {
        throw std::runtime_error(
            "Batch size, the number of threads and the prefetch size need to be strictly positive.");
    }

    if (_prefetchSize < _numThreads) {
        throw std::runtime_error(
            "Prefetch size must be larger or equal than the number of threads.");
    }

    resourceClient.acquire();

    threadPool.start();
}

DataLoader::~DataLoader() {
    prefetchCacheNotify.notify_all();
}

DLWrapper::DLWrapper(const uint64_t fence,
                     const int deviceId,
                     DLManagedTensor *dlManagedTensor)
    : fence(fence), deviceId(deviceId), dlManagedTensor(dlManagedTensor) {
}

pybind11::capsule DLWrapper::__dlpack__(const pybind11::object &consumerStreamObject) const {
    const auto resourcePool = ResourcePool::getInstance();

    if (consumerStreamObject.is_none()) {
        resourcePool->synchronizeHostDevice(fence);
    } else {
        const auto consumerStream = static_cast<uintptr_t>(py::int_(consumerStreamObject));
        resourcePool->synchronizeConsumerStream(fence, consumerStream);
    }

    return pybind11::capsule(dlManagedTensor, "dltensor");
}

std::pair<int, int> DLWrapper::__dlpack_device__() const {
    return {kDLCUDA, deviceId};
}


// ReSharper disable once CppParameterMayBeConstPtrOrRef
void deleter(DLManagedTensor *self) {
    ResourcePool::getInstance().release(); // These link ***.
    ResourcePool::getInstance()->getAllocator()->free(static_cast<uint8_t *>(self->dl_tensor.data));
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete static_cast<DLWrapper *>(self->manager_ctx);
    delete self;
}

py::dict DataLoader::getNextBatch() {
    resourceClient.acquire(prefetchSize, outputBatchMemorySize);

    std::unique_lock lock(prefetchCacheMutex);
    prefetchCacheNotify.wait(lock, [this] { return !prefetchCache.empty(); });

    const auto [startingOffset, gpuAllocations, fences] = prefetchCache.top();
    prefetchCache.pop();
    batchedDataset.popWaitingBatch(startingOffset);

    lock.unlock();

    LOG_DEBUG("Loading from; startingOffset: {}, genIdx: {}", startingOffset, batchedDataset.getGenIdx().load());

    py::dict pyBatch;
    const auto &heads = batchedDataset.getDataset().getHeads();
    for (size_t i = 0; i < heads.size(); i++) {
        const Head &head = heads[i];
        uint8_t *gpuAllocation = gpuAllocations[i];
        uint64_t fence = fences[i];
        ResourcePool::getInstance()->getAllocator()->handOff(gpuAllocation);

        const std::vector<int> &shape = head.getShape();
        const int ndim = static_cast<int>(shape.size()) + 1;

        auto *const shapeArr = new int64_t[ndim]{};
        shapeArr[0] = static_cast<int64_t>(batchSize);
        std::memcpy(shapeArr + 1, shape.data(), shape.size() * sizeof(size_t));

        uint8_t itemCode;
        switch (head.getItemFormat()) {
            case ItemFormat::FLOAT: itemCode = kDLFloat;
                break;
            case ItemFormat::UINT: itemCode = kDLUInt;
                break;
            default:
                throw std::runtime_error("Invalid item format.");
        }

        constexpr int deviceType = 2; // TODO: This hardcodes CUDA.
        constexpr int deviceId = 0; // TODO: This hardcodes GPU0.
        auto *dlManagedTensor = new DLManagedTensor{
            .dl_tensor = {
                .data = gpuAllocation,
                .device = {
                    .device_type = kDLCUDA,
                    .device_id = deviceId
                },
                .ndim = ndim,
                .dtype = {
                    .code = itemCode,
                    .bits = static_cast<uint8_t>(8 * head.getBytesPerItem()), //NOLINT(readability-magic-numbers)
                    .lanes = 1
                },
                .shape = shapeArr,
                .strides = null,
                .byte_offset = 0
            },
            .manager_ctx = null,
            .deleter = &deleter
        };

        DLWrapper *wrapper = new DLWrapper(fence, deviceType, deviceId, dlManagedTensor);
        dlManagedTensor->manager_ctx = wrapper;

        ResourcePool::getInstance().acquire(); // These link ***.
        pyBatch[head.getDictName().c_str()] = py::capsule(wrapper, "dlwrapper");
    }

    return pyBatch;
}
