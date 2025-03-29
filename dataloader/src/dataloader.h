#ifndef DATALOADER_H
#define DATALOADER_H
#include "dataset.h"
#include <cstddef>
#include <thread>
#include <vector>
#include <semaphore>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <pybind11/pybind11.h>

#define CHS 3

// TODO: Align everything!
class PinnedMemoryArena {
public:
    PinnedMemoryArena();

    explicit PinnedMemoryArena(size_t _total_size);

    PinnedMemoryArena &operator=(PinnedMemoryArena &&arena) noexcept;

    PinnedMemoryArena(const PinnedMemoryArena &arena) = delete;

    PinnedMemoryArena(PinnedMemoryArena &&arena) noexcept;

    ~PinnedMemoryArena();

    uint8_t *allocate(size_t size);

    size_t getOffset() const;

private:
    uint8_t *data{};
    size_t total_size;
    size_t offset;
};

class Semaphore {
public:
    explicit Semaphore(int initial);

    void acquire();

    void release();

    void disable();

private:
    std::counting_semaphore<> semaphore;
    std::atomic_int numTokensUsed;
    std::atomic_bool disabled;
};

class ListOfAllocations {
public:
    explicit ListOfAllocations(PinnedMemoryArena *memoryArena);

    ListOfAllocations &operator=(ListOfAllocations &&allocs) noexcept;

    ListOfAllocations(const ListOfAllocations &allocs) = delete;

    ListOfAllocations(ListOfAllocations &&arena) noexcept;

    uint8_t *allocate(size_t size);

    PinnedMemoryArena *memoryArena;
    std::vector<uint8_t *> ptrs;
    std::vector<size_t> sizes;
};

class ThreadPool {
public:
    explicit ThreadPool(const std::function<void()> &_threadMain,
                        size_t _threadCount);

    void start();

    ThreadPool &operator=(ThreadPool &&pool) noexcept = delete;

    ThreadPool(const ThreadPool &pool) = delete;

    ThreadPool(ThreadPool &&pool) noexcept = delete;

    ~ThreadPool() noexcept;

private:
    std::function<void()> threadMain;
    size_t threadCount;
    std::vector<std::thread> threads;
    std::atomic_uint32_t shutdownCounter;
    std::mutex shutdownMutex;
    std::condition_variable shutdownNotify;

    void extendedThreadMain();
};

class DataLoader {
public:
    DataLoader(
        Dataset _dataset,
        int _batchSize,
        pybind11::function _createDatasetFunction,
        int _numThreads,
        int _prefetchSize
    );

    DataLoader(const DataLoader &dl) = delete;
    DataLoader(DataLoader &&dl) = delete;

    ~DataLoader();

    pybind11::dict getNextBatch();

    [[nodiscard]] size_t getNumberOfBatches() const;

private:
    Dataset dataset;
    size_t batchSize;
    size_t numberOfBatches;
    pybind11::function createDatasetFunction;
    Semaphore prefetchSemaphore;
    std::mutex datasetMutex;
    std::vector<ListOfAllocations> prefetchCache;
    std::condition_variable prefetchCacheNotify;
    std::mutex prefetchCacheMutex;
    PinnedMemoryArena memoryArena;
    size_t outputBatchMemorySize;
    ThreadPool threadPool;
    std::atomic_bool shutdown;

    void threadMain();
};

#endif //DATALOADER_H
