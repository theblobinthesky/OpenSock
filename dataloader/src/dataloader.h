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

enum class MemoryArenaType {
    PINNED_HOST, GPU_DEVICE
};

// TODO: Align everything!
class MemoryArena {
public:
    MemoryArena();

    explicit MemoryArena(MemoryArenaType type, size_t _total_size);

    MemoryArena &operator=(MemoryArena &&arena) noexcept;

    MemoryArena(const MemoryArena &arena) = delete;

    MemoryArena(MemoryArena &&arena) noexcept;

    ~MemoryArena();

    uint8_t *allocate(size_t size);

    [[nodiscard]] uint8_t *getData() const;

private:
    MemoryArenaType type;
    uint8_t *data;
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
    explicit ListOfAllocations(MemoryArena *memoryArena);

    ListOfAllocations &operator=(ListOfAllocations &&allocs) noexcept;

    ListOfAllocations(const ListOfAllocations &allocs) = delete;

    ListOfAllocations(ListOfAllocations &&arena) noexcept;

    uint8_t *allocate(size_t size);

    [[nodiscard]] uint32_t getOffset(size_t i) const;

    MemoryArena *memoryArena;
    std::vector<uint8_t *> ptrs;
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
    GPUState(size_t numStreams);

    ~GPUState();

    void copy(size_t streamIndex, uint8_t *gpuBuffer, uint8_t *buffer, uint32_t size) const;

    void sync(size_t streamIndex);

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
    pybind11::function createDatasetFunction;
    Semaphore prefetchSemaphore;
    std::mutex datasetMutex;
    std::vector<ThreadAllocationsPair> prefetchCache;
    std::condition_variable prefetchCacheNotify;
    std::mutex prefetchCacheMutex;
    MemoryArena pinnedMemoryArena;
    MemoryArena gpuMemoryArena;
    size_t outputBatchMemorySize;
    GPUState gpu;
    ThreadPool threadPool;
    std::atomic_bool shutdown;

    void threadMain(size_t threadIdx);
};

#endif //DATALOADER_H
