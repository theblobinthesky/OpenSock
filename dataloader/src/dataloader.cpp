#include "dataloader.h"
#include "utils.h"
#include <filesystem>
#include <dlpack/dlpack.h>

namespace py = pybind11;

std::atomic_uint64_t DataLoader::nextId;
std::mutex DataLoader::concurrencyMutex;

size_t getOutputBatchMemorySize(const BatchedDataset &batchedDataset, const size_t batchSize) {
    size_t outputBatchMemorySize = 0;
    for (const ItemKey &itemKey: batchedDataset.getDataset().getDataSource()->getItemKeys()) {
        const ProbeResult &probe = itemKey.probeResult;
        uint64_t itemMemorySize = probe.bytesPerItem;
        for (const auto dim: probe.shape) {
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
    outputBatchMemorySize(getOutputBatchMemorySize(batchedDataset, batchSize)) {
    if (batchSize <= 0 || _numThreads <= 0 || _prefetchSize <= 0) {
        throw std::runtime_error(
            "Batch size, the number of threads and the prefetch size need to be strictly positive.");
    }

    LOG_INFO("DataLoader with id={} initialized.", id);
}

DLWrapper::DLWrapper(const Fence fence,
                     const int deviceType,
                     const int deviceId,
                     DLManagedTensor *dlManagedTensor)
    : fence(fence), deviceType(deviceType), deviceId(deviceId), dlManagedTensor(dlManagedTensor) {
}

pybind11::capsule DLWrapper::getDLpackCapsule(const pybind11::object &consumerStreamObject) const {
    auto &resourcePool = ResourcePool::get();

    if (consumerStreamObject.is_none()) {
        resourcePool.synchronizeHostDevice(fence);
    } else {
        const auto consumerStream = static_cast<uintptr_t>(py::int_(consumerStreamObject));
        resourcePool.synchronizeConsumerStream(fence, ConsumerStream{consumerStream});
    }

    return pybind11::capsule(dlManagedTensor, "dltensor");
}

std::pair<int, int> DLWrapper::getDLpackDevice() const {
    return {deviceType, deviceId};
}


// ReSharper disable once CppParameterMayBeConstPtrOrRef
void deleter(DLManagedTensor *self) {
    ResourcePool::get().getAllocator()->free(static_cast<uint8_t *>(self->dl_tensor.data));
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete static_cast<DLWrapper *>(self->manager_ctx);
    delete self;
}

py::dict DataLoader::getNextBatch() {
    std::unique_lock lock(concurrencyMutex);
    const auto [datasetStartingOffset, gpuAllocations, fences]
            = ResourcePool::get().acquireAndGetNextBatch(shared_from_this());

    LOG_DEBUG("Loading from; datasetStartingOffset: {}, genIdx: {}",
              datasetStartingOffset, ResourcePool::get().getGenIdx());

    py::dict pyBatch;
    const auto &itemKeys = batchedDataset.getDataset().getDataSource()->getItemKeys();
    for (size_t i = 0; i < itemKeys.size(); i++) {
        const ItemKey &itemKey = itemKeys[i];
        const ProbeResult &probe = itemKey.probeResult;
        uint8_t *gpuAllocation = gpuAllocations[i];
        const Fence fence = fences[i];
        ResourcePool::get().getAllocator()->handOff(gpuAllocation);

        const std::vector<uint32_t> &shape = probe.shape;
        const int ndim = static_cast<int>(shape.size()) + 1;

        auto *const shapeArr = new int64_t[ndim]{};
        shapeArr[0] = static_cast<int64_t>(batchSize);
        for (size_t s = 0; s < shape.size(); s++) {
            shapeArr[s + 1] = static_cast<int64_t>(shape[s]);
        }

        uint8_t itemCode;
        switch (probe.format) {
            case ItemFormat::FLOAT: itemCode = kDLFloat;
                break;
            case ItemFormat::UINT: itemCode = kDLUInt;
                break;
            default:
                throw std::runtime_error("Invalid item format.");
        }

        constexpr int deviceType = kDLCUDA; // TODO: This hardcodes CUDA.
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
                    .bits = static_cast<uint8_t>(8 * probe.bytesPerItem), //NOLINT(readability-magic-numbers)
                    .lanes = 1
                },
                .shape = shapeArr,
                .strides = nullptr,
                .byte_offset = 0
            },
            .manager_ctx = nullptr,
            .deleter = &deleter
        };

        // ReSharper disable once CppDFAMemoryLeak
        auto *wrapper = new DLWrapper(fence, deviceType, deviceId, dlManagedTensor);
        dlManagedTensor->manager_ctx = wrapper;

        pyBatch[itemKey.keyName.c_str()] = py::cast(wrapper, py::return_value_policy::reference);
    }

    return pyBatch;
}
