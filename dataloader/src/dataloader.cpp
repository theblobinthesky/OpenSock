#include "dataloader.h"
#include "utils.h"
#include <filesystem>
#include <dlpack/dlpack.h>

namespace py = pybind11;

std::atomic_uint64_t DataLoader::nextId;
std::mutex DataLoader::concurrencyMutex;

void precomputeItemSizesInMemory(
    const BatchedDataset &bds,
    const DataAugmentationPipe &augPipe,
    const size_t batchSize,
    std::vector<size_t> &maxBytesOfEveryInput,
    std::vector<size_t> &outputSizesPerBatchOfItem,
    std::vector<size_t> &outputMetadataSizesPerBatchOfItem,
    size_t &maxInputBatchMemorySize,
    size_t &outputBatchMemorySize
) {
    const auto &itemKeys = bds.getDataset().getDataSource()->getItemKeys();
    const auto rasterMaxInputShapeSize = getShapeSize(augPipe.getRasterMaxInputShape());
    const auto augPipeRasterOutputShapeSize = getShapeSize(augPipe.getStaticOutputShape());

    outputSizesPerBatchOfItem.clear();
    outputMetadataSizesPerBatchOfItem.clear();
    maxInputBatchMemorySize = outputBatchMemorySize = 0;
    for (const auto &itemKey: itemKeys) {
        const auto probeDTypeWidth = getWidthOfDType(itemKey.probeResult.dtype);
        size_t maxBytesOfInputPerItem = probeDTypeWidth;
        size_t bytesOfOutputPerItem = probeDTypeWidth;
        size_t bytesOfMetaOutputPerItem;

        switch (itemKey.type) {
            case ItemType::NONE: {
                const size_t probeShapeSize = getShapeSize(itemKey.probeResult.shape);
                maxBytesOfInputPerItem *= probeShapeSize;
                bytesOfOutputPerItem *= probeShapeSize;
                bytesOfMetaOutputPerItem = 0;
            }
            break;
            case ItemType::RASTER:
                maxBytesOfInputPerItem *= rasterMaxInputShapeSize;
                bytesOfOutputPerItem *= augPipeRasterOutputShapeSize;
                bytesOfMetaOutputPerItem = 0;
                break;
            case ItemType::POINTS:
                maxBytesOfInputPerItem *= augPipe.getMaxNumPoints();
                bytesOfOutputPerItem *= augPipe.getMaxNumPoints();

                // Points need a metadata tensor of lengths with shape (b,).
                bytesOfMetaOutputPerItem = batchSize * getWidthOfDType(PointsLengthsTensorDType);
                break;
            default:
                throw std::runtime_error("Item sizes cannot be computed because an item type is unknown.");
        }

        maxBytesOfEveryInput.push_back(maxBytesOfInputPerItem);
        maxInputBatchMemorySize += maxBytesOfInputPerItem;

        const auto outputSize = batchSize * bytesOfOutputPerItem;
        outputSizesPerBatchOfItem.push_back(outputSize);
        outputBatchMemorySize += outputSize;

        const auto outputMetadataSize = batchSize * bytesOfMetaOutputPerItem;
        outputMetadataSizesPerBatchOfItem.push_back(outputMetadataSize);
        outputBatchMemorySize += outputMetadataSize;
    }

    maxInputBatchMemorySize = alignUp(maxInputBatchMemorySize, 16);
    outputBatchMemorySize = alignUp(outputBatchMemorySize, 16);
}

DataLoader::DataLoader(
    Dataset &_dataset,
    const size_t _batchSize,
    const size_t _numThreads,
    const size_t _prefetchSize,
    DataAugmentationPipe &_augPipe
) : id(nextId.fetch_add(1)),
    batchedDataset(std::move(_dataset), _batchSize),
    augPipe(std::move(_augPipe)),
    batchSize(_batchSize),
    numThreads(_numThreads),
    prefetchSize(_prefetchSize) {
    if (batchSize == 0 || _numThreads == 0 || _prefetchSize == 0) {
        throw std::runtime_error(
            "Batch size, the number of threads and the prefetch size need to be strictly positive.");
    }

    precomputeItemSizesInMemory(
        batchedDataset, augPipe, batchSize,
        maxBytesOfEveryInput, outputSizesPerBatchOfItem, outputMetadataSizesPerBatchOfItem,
        maxInputBatchMemorySize, outputBatchMemorySize
    );

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

uint8_t getItemCode(const DType dtype) {
    switch (dtype) {
        case DType::UINT8: return kDLUInt;
        case DType::INT32: return kDLInt;
        case DType::FLOAT32: return kDLFloat;
        default:
            throw std::runtime_error("DType cannot be converted to item code.");
    }
}

DLWrapper *get(
    const uint32_t batchSize,
    const Shape &shape,
    uint8_t *gpuAllocation,
    const DType dtype,
    const Fence &fence
) {
    const int ndim = static_cast<int>(shape.size()) + 1;
    // ReSharper disable once CppDFAMemoryLeak
    auto *const shapeArr = new int64_t[ndim]{};
    shapeArr[0] = static_cast<int64_t>(batchSize);
    for (size_t s = 0; s < shape.size(); s++) {
        shapeArr[s + 1] = static_cast<int64_t>(shape[s]);
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
                .code = getItemCode(dtype),
                .bits = static_cast<uint8_t>(8 * getWidthOfDType(dtype)), //NOLINT(readability-magic-numbers)
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

    return wrapper;
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

        const Shape &shape = probe.shape; // TODO: This needs to contain the batch dimension.
        auto wrapper = get(batchSize, shape, gpuAllocation, probe.dtype, fence);

        // TODO
        /* switch (itemKey.type) {
            case ItemType::NONE:
            case ItemType::RASTER:
                break;
            case ItemType::POINTS:
                break;
            default:
                throw std::runtime_error("Next batch encountered unknown spatial hint..");
        } // TODO*/
        pyBatch[itemKey.keyName.c_str()] = py::cast(wrapper, py::return_value_policy::reference);
    }

    return pyBatch;
}
