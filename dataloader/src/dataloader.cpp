#include "dataloader.h"
#include "utils.h"
#include <filesystem>
#include <dlpack/dlpack.h>

namespace py = pybind11;

std::atomic_uint64_t DataLoader::nextId;
std::mutex DataLoader::concurrencyMutex;

void precomputeItemSizesInMemory(
    const BatchedDataset &bds,
    const std::shared_ptr<DataAugmentationPipe> &augPipe,
    const size_t batchSize,
    std::vector<size_t> &maxBytesOfInputPerItemOfItemKey,
    std::vector<size_t> &bytesOfOutputOfItemKey,
    std::vector<size_t> &bytesOfMetadataOutputPerItemOfItemKey,
    size_t &maxInputBatchMemorySize,
    size_t &outputBatchMemorySize
) {
    const auto &itemKeys = bds.getDataset()->getDataSource()->getItemKeys();
    const auto &rasterMaxInputShape = augPipe->getRasterMaxInputShape();
    const auto rasterMaxInputShapeNoBatch = std::span(rasterMaxInputShape).subspan(1);
    const auto rasterMaxInputShapeNoBatchSize = getShapeSize(rasterMaxInputShapeNoBatch);
    const auto augPipeRasterOutputShapeSize = getShapeSize(augPipe->getStaticOutputShape());

    bytesOfOutputOfItemKey.clear();
    bytesOfMetadataOutputPerItemOfItemKey.clear();
    maxInputBatchMemorySize = outputBatchMemorySize = 0;
    for (const auto &itemKey: itemKeys) {
        const auto probeDTypeWidth = getWidthOfDType(itemKey.probeResult.dtype); // TODO: Revert
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
                maxBytesOfInputPerItem *= rasterMaxInputShapeNoBatchSize;
                bytesOfOutputPerItem *= augPipeRasterOutputShapeSize;
                bytesOfMetaOutputPerItem = 0;
                break;
            case ItemType::POINTS:
                maxBytesOfInputPerItem *= augPipe->getMaxNumPoints();
                bytesOfOutputPerItem *= augPipe->getMaxNumPoints();

                // Points need a metadata tensor of lengths with shape (b,).
                bytesOfMetaOutputPerItem = getWidthOfDType(PointsLengthsTensorDType);
                break;
            default:
                throw std::runtime_error("Item sizes cannot be computed because an item type is unknown.");
        }

        maxBytesOfInputPerItemOfItemKey.push_back(maxBytesOfInputPerItem);
        maxInputBatchMemorySize += batchSize * maxBytesOfInputPerItem;

        const size_t bytesOfOutput = batchSize * bytesOfOutputPerItem;
        bytesOfOutputOfItemKey.push_back(bytesOfOutput);
        outputBatchMemorySize += bytesOfOutput;

        bytesOfMetadataOutputPerItemOfItemKey.push_back(bytesOfMetaOutputPerItem);
        outputBatchMemorySize += batchSize * bytesOfMetaOutputPerItem;
    }

    maxInputBatchMemorySize = alignUp(maxInputBatchMemorySize, 16);
    outputBatchMemorySize = alignUp(outputBatchMemorySize, 16);
}

DataLoader::DataLoader(
    const std::shared_ptr<Dataset> &_dataset,
    const size_t _batchSize,
    const size_t _numThreads,
    const size_t _prefetchSize,
    const std::shared_ptr<DataAugmentationPipe> &_augPipe
) : id(nextId.fetch_add(1)),
    batchedDataset(_dataset, _batchSize),
    augPipe(_augPipe),
    batchSize(_batchSize),
    numThreads(_numThreads),
    prefetchSize(_prefetchSize) {
    if (batchSize == 0 || _numThreads == 0 || _prefetchSize == 0) {
        throw std::runtime_error(
            "Batch size, the number of threads and the prefetch size need to be strictly positive.");
    }

    precomputeItemSizesInMemory(
        batchedDataset, augPipe, batchSize,
        maxBytesOfInputPerItemOfItemKey, bytesOfOutputOfItemKey, bytesOfMetadataOutputPerItemOfItemKey,
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
    const auto &resourcePool = ResourcePool::get();

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
        case DType::FLOAT16:
        case DType::FLOAT32:
            return kDLFloat;
        default:
            throw std::runtime_error("DType cannot be converted to item code.");
    }
}

DLWrapper *getWrappedTensor(
    const uint32_t batchSize,
    const Shape &shape,
    uint8_t *gpuAllocation, // NOLINT(*-non-const-parameter)
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

std::pair<py::dict, py::dict> DataLoader::getNextBatch() {
    std::unique_lock lock(concurrencyMutex);
    const auto [datasetStartingOffset, gpuAllocations, fences]
            = ResourcePool::get().acquireAndGetNextBatch(shared_from_this());

    LOG_DEBUG("Loading from; datasetStartingOffset: {}, genIdx: {}",
              datasetStartingOffset, ResourcePool::get().getGenIdx());

    py::dict pyBatch;
    py::dict pyMetadata;
    const auto &itemKeys = batchedDataset.getDataset()->getDataSource()->getItemKeys();

    size_t allocationIdx = 0;
    for (const auto &itemKey: itemKeys) {
        const ProbeResult &probe = itemKey.probeResult;
        uint8_t *gpuAllocation = gpuAllocations[allocationIdx];
        const Fence fence = fences[allocationIdx];
        ResourcePool::get().getAllocator()->handOff(gpuAllocation);
        allocationIdx++;

        Shape shape;
        switch (itemKey.type) {
            case ItemType::NONE:
                shape = probe.shape;
                break;
            case ItemType::RASTER:
                shape = augPipe->getStaticOutputShape();
                break;
            case ItemType::POINTS:
                /*shape = {
                };*/
                throw std::runtime_error("crashiiii..");
            default:
                throw std::runtime_error("getNextBatch encountered unknown item key type.");
        }

        auto wrapper = getWrappedTensor(batchSize, shape, gpuAllocation, probe.dtype, fence);
        pyBatch[itemKey.keyName.c_str()] = py::cast(wrapper, py::return_value_policy::reference);

        // For POINTS type, also return the metadata tensor with original lengths
        /*if (itemKey.type == ItemType::POINTS) {
            uint8_t *metaGpuAllocation = gpuAllocations[allocationIdx];
            const Fence metaFence = fences[allocationIdx];
            ResourcePool::get().getAllocator()->handOff(metaGpuAllocation);
            allocationIdx++;

            // Metadata shape is (batchSize,) containing original point counts
            constexpr Shape metaShape = {};  // Empty shape means scalar per batch item -> (batchSize,)
            auto metaWrapper = getWrappedTensor(batchSize, metaShape, metaGpuAllocation, PointsLengthsTensorDType, metaFence);
            const std::string metaKeyName = itemKey.keyName + "_lengths";
            pyMetadata[metaKeyName.c_str()] = py::cast(metaWrapper, py::return_value_policy::reference);
        }*/
    }

    return {pyBatch, pyMetadata};
}
