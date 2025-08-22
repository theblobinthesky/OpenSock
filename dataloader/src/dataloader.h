#ifndef DATALOADER_H
#define DATALOADER_H
#include "dataset.h"
#include "resource.h"
#include <vector>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <pybind11/pybind11.h>

#define CHS 3

struct PrefetchItem {
    int32_t datasetStartingOffset;
    size_t barrierIdx;
    std::vector<uint8_t *> gpuAllocations;

    bool operator<(const PrefetchItem& other) const;
};

class DataLoader {
public:
    DataLoader(
        Dataset& _dataset,
        int _batchSize,
        int _numThreads,
        int _prefetchSize
    );

    DataLoader(const DataLoader &dl) = delete;

    DataLoader(DataLoader &&dl) = delete;

    ~DataLoader();

    pybind11::dict getNextBatch();

    [[nodiscard]] size_t getNumberOfBatches() const;

private:
    static std::atomic_int nextId;
    int id;
    BatchedDataset batchedDataset;
    const size_t batchSize;
    const size_t numThreads;
    const size_t prefetchSize;
    size_t numberOfBatches;
    Semaphore prefetchSemaphore;
    std::priority_queue<PrefetchItem> prefetchCache;
    std::condition_variable prefetchCacheNotify;
    std::atomic_int lastBarrierIdx;
    std::mutex prefetchCacheMutex;
    size_t outputBatchMemorySize;
    ResourceClient resourceClient;
};

#endif //DATALOADER_H
