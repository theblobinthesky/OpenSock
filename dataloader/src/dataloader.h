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
#include <cuda_runtime.h>

#define CHS 3

struct Allocation {
    uint8_t *host;
    uint8_t *gpu;
};

class MemoryArena {
public:
    static void initialize(size_t totalSize);

    static MemoryArena *getInstance();

    MemoryArena &operator=(MemoryArena &&arena) = delete;

    MemoryArena &operator=(const MemoryArena &) = delete;

    MemoryArena(const MemoryArena &arena) = delete;

    MemoryArena(MemoryArena &&arena) = delete;

    Allocation allocate(size_t size);

    void free();

    void destroy();

    [[nodiscard]] size_t getOffset(const Allocation &allocation) const;

    [[nodiscard]] uint8_t *getGpuData() const;

private:
    explicit MemoryArena(size_t _totalSize);

    void freeAll();

    static MemoryArena *memoryArena;
    uint8_t *hostData;
    uint8_t *gpuData;
    size_t totalSize;
    size_t offset;
    size_t extRefCounter{};
    std::atomic_bool destroyed;
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
    ListOfAllocations() = default;

    ListOfAllocations &operator=(ListOfAllocations &&allocs) noexcept;

    ListOfAllocations(const ListOfAllocations &allocs) = delete;

    ListOfAllocations(ListOfAllocations &&arena) noexcept;

    Allocation allocate(size_t size);

    [[nodiscard]] uint32_t getOffset(size_t i) const;

    std::vector<Allocation> allocations;
    std::vector<size_t> sizes;
};

class ThreadPool {
public:
    explicit ThreadPool(const std::function<void(size_t)> &_threadMain,
                        size_t _threadCount);

    void start();

    ThreadPool &operator=(ThreadPool &&pool) noexcept = delete;

    ThreadPool(const ThreadPool &pool) = delete;

    ThreadPool(ThreadPool &&pool) noexcept = delete;

    ~ThreadPool() noexcept;

private:
    std::function<void(size_t)> threadMain;
    size_t threadCount;
    std::vector<std::thread> threads;
    std::atomic_uint32_t shutdownCounter;
    std::mutex shutdownMutex;
    std::condition_variable shutdownNotify;

    void extendedThreadMain(size_t threadIdx);
};

class GPUState {
public:
    explicit GPUState(size_t numStreams);

    ~GPUState();

    void copy(size_t streamIndex, uint8_t *gpuBuffer, uint8_t *buffer,
              uint32_t size) const;

    void sync(size_t streamIndex) const;

private:
    std::vector<cudaStream_t> streams;
};

struct ThreadAllocationsPair {
    size_t threadIdx;
    ListOfAllocations allocations;
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
    const size_t batchSize;
    size_t numberOfBatches;
    Semaphore prefetchSemaphore;
    std::mutex datasetMutex;
    std::vector<ThreadAllocationsPair> prefetchCache;
    std::condition_variable prefetchCacheNotify;
    std::mutex prefetchCacheMutex;
    size_t outputBatchMemorySize;
    GPUState gpu;
    ThreadPool threadPool;
    std::atomic_bool shutdown;

    void threadMain(size_t threadIdx);
};

#endif //DATALOADER_H
