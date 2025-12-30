#ifndef DATALOADER_H
#define DATALOADER_H

#include "dataio.h"
#include "resource.h"
#include "pybind11_includes.h"
#include <atomic>
#include <dlpack/dlpack.h>

class ResourceClient;

class DLWrapper {
public:
    DLWrapper(Fence fence, int deviceType, int deviceId, DLManagedTensor *dlManagedTensor);

    [[nodiscard]] pybind11::capsule getDLpackCapsule(const pybind11::object &consumerStreamObject) const;

    [[nodiscard]] std::pair<int, int> getDLpackDevice() const;

private:
    Fence fence;
    int deviceType;
    int deviceId;
    DLManagedTensor *dlManagedTensor;
};

class DataLoader : public std::enable_shared_from_this<DataLoader> {
public:
    DataLoader(
        const std::shared_ptr<Dataset> &_dataset,
        size_t _batchSize,
        size_t _numThreads,
        size_t _prefetchSize,
        const std::shared_ptr<DataAugmentationPipe> &_augPipe
    );

    DataLoader(const DataLoader &) = delete;

    DataLoader(DataLoader &&) = delete;

    std::pair<pybind11::dict, pybind11::dict> getNextBatch();

    static std::atomic_uint64_t nextId;
    static std::mutex concurrencyMutex; // TODO: Maybe this is not necessary. Maybe remove this later.
    uint64_t id;
    BatchedDataset batchedDataset;
    std::shared_ptr<DataAugmentationPipe> augPipe;
    const size_t batchSize;
    const size_t numThreads;
    const size_t prefetchSize;

    // Metadata about inputs/outputs:
    std::vector<size_t> maxBytesOfInputPerItemOfItemKey;
    std::vector<size_t> bytesOfOutputOfItemKey;
    std::vector<size_t> bytesOfMetadataOutputPerItemOfItemKey;
    size_t maxInputBatchMemorySize{};
    size_t outputBatchMemorySize{};
};

#endif //DATALOADER_H
