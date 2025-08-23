#ifndef DATALOADER_H
#define DATALOADER_H
#include "dataset.h"
#include "resource.h"
#include <atomic>
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

// TODO: Register this in library.
class DLWrapper {
public:
    DLWrapper(uint64_t fence, int deviceId, DLManagedTensor *dlManagedTensor);

    pybind11::capsule __dlpack__(const pybind11::object &consumerStreamObject) const;

    std::pair<int, int> __dlpack_device__() const;

private:
    uint64_t fence;
    int deviceId;
    DLManagedTensor *dlManagedTensor;
};

class DataLoader {
public:
    DataLoader(
        Dataset &_dataset,
        int _batchSize,
        int _numThreads,
        int _prefetchSize
    );

    DataLoader(const DataLoader &) = delete;

    DataLoader(DataLoader &&) = delete;

    ~DataLoader();

    pybind11::dict getNextBatch();

    // private: TODO
    static std::atomic_uint64_t nextId;
    uint64_t id;
    BatchedDataset batchedDataset;
    const size_t batchSize;
    const size_t numThreads;
    const size_t prefetchSize;

    size_t outputBatchMemorySize;
    ResourceClient resourceClient;
};

#endif //DATALOADER_H
