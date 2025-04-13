#ifndef DATALOADER_H
#define DATALOADER_H
#include "dataset.h"
#include "resource.h"
#include <cstddef>
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
        const Dataset& _dataset,
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
    Dataset dataset;
    const size_t batchSize;
    const size_t numThreads;
    const size_t prefetchSize;
    size_t numberOfBatches;
    Semaphore prefetchSemaphore;
    std::mutex datasetMutex;
    std::priority_queue<PrefetchItem> prefetchCache;
    std::atomic_int32_t lastStartingOffset;
    std::condition_variable prefetchCacheNotify;
    std::atomic_int lastBarrierIdx;
    std::mutex prefetchCacheMutex;
    size_t outputBatchMemorySize;
    ResourceClient resourceClient;
    ThreadPool threadPool;
    std::atomic_bool shutdown;

    void threadMain();
};

#endif //DATALOADER_H
