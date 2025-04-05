#include "dataloader.h"
#include "utils.h"
#include "image.h"
#include "cnpy.h"
#include <filesystem>
#include <dlpack/dlpack.h>

#define CHS 3

namespace py = pybind11;
namespace fs = std::filesystem;

MemoryArena *MemoryArena::memoryArena;

void MemoryArena::initialize(const size_t totalSize) {
    if (memoryArena == null) {
        memoryArena = new MemoryArena(totalSize);
    }
}

MemoryArena *MemoryArena::getInstance() {
    if (memoryArena == null) {
        throw std::runtime_error("MemoryArena is not initialized.");
    }

    return memoryArena;
}

// TODO: Make sure to leave more space in a future version of this allocator
// so threads don't contend for the same cache line potentially.
Allocation MemoryArena::allocate(const size_t size) {
    uint8_t *hostPtr = hostData + offset;
    uint8_t *gpuPtr = gpuData + offset;
    offset += size;

    if (offset > totalSize) {
        throw std::runtime_error(std::format("Tried to allocate {} bytes beyond the size {} of the memory arena.",
                                             offset - totalSize, totalSize));
    }

    offset = offset % totalSize;
    extRefCounter++;
    return {
        .host = hostPtr,
        .gpu = gpuPtr
    };
}

void MemoryArena::free() {
    extRefCounter--;

    if (destroyed && extRefCounter == 0) {
        // TODO
    }
}

void MemoryArena::destroy() {
    if (extRefCounter == 0) {
        destroyed = true;
    }
}

MemoryArena::MemoryArena(const size_t _totalSize)
    : hostData(null), gpuData(null), totalSize(_totalSize), offset(0) {
    cudaError_t err = cudaMallocHost(&hostData, totalSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaHostMalloc failed");
    }

    err = cudaMalloc(&gpuData, totalSize);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "cudaMalloc failed to allocate " + std::to_string(
                totalSize / 1024 / 1024) + "MB");
    }
}

void MemoryArena::freeAll() {
    cudaFreeHost(hostData);
    cudaFree(gpuData);
}

size_t MemoryArena::getOffset(const Allocation &allocation) const {
    return allocation.host - hostData;
}

uint8_t *MemoryArena::getGpuData() const {
    return gpuData;
}

ListOfAllocations::ListOfAllocations(const size_t totalSize) : totalAllocation({}), offset(0) {
    totalAllocation = MemoryArena::getInstance()->allocate(totalSize);
}

ListOfAllocations &ListOfAllocations::operator=(
    ListOfAllocations &&allocs) noexcept {
    totalAllocation = allocs.totalAllocation;
    allocations = std::move(allocs.allocations);
    sizes = std::move(allocs.sizes);

    return *this;
}

ListOfAllocations::ListOfAllocations(
    ListOfAllocations &&allocs) noexcept : totalAllocation(allocs.totalAllocation),
                                           offset(0),
                                           allocations(std::move(allocs.allocations)),
                                           sizes(std::move(allocs.sizes)) {
}

Allocation ListOfAllocations::allocate(const size_t size) {
    const Allocation allocation = {
        .host = totalAllocation.host + offset,
        .gpu = totalAllocation.gpu + offset
    };

    offset += size;

    allocations.push_back(allocation);
    sizes.push_back(size);
    return allocation;
}

uint32_t ListOfAllocations::getOffset(const size_t i) const {
    return allocations[i].host - totalAllocation.host;
}

Semaphore::Semaphore(const int initial)
    : semaphore(initial), numTokensUsed(initial) {
}

void Semaphore::acquire() {
    if (!disabled) {
        semaphore.acquire();
        ++numTokensUsed;
    }
}

void Semaphore::release() {
    semaphore.release();
    --numTokensUsed;
}

void Semaphore::disable() {
    semaphore.release(numTokensUsed);
    numTokensUsed = 0;
    disabled = true;
}

ThreadPool::ThreadPool(const std::function<void()> &_threadMain,
                       const size_t _threadCount) : threadMain(_threadMain),
                                                    threadCount(_threadCount),
                                                    shutdownCounter(0) {
}

void ThreadPool::start() {
    for (size_t i = 0; i < threadCount; i++) {
        threads.emplace_back(&ThreadPool::extendedThreadMain, this);
    }
}

ThreadPool::~ThreadPool() noexcept {
    for (auto &thread: threads) {
        if (thread.joinable()) thread.join();
    }
}

void ThreadPool::extendedThreadMain() {
    if (threadMain) threadMain();

    ++shutdownCounter;
    shutdownNotify.notify_all();
    std::unique_lock lock(shutdownMutex);
    shutdownNotify.wait(
        lock, [this] {
            return shutdownCounter >= threads.size();
        });
}

GPUState::GPUState(const size_t numBarriers) : stream{} {
    if (const cudaError_t err = cudaStreamCreate(&stream);
        err != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed.");
    }

    for (size_t i = 0; i < numBarriers; i++) {
        cudaEvent_t barrierEvent;
        if (cudaEventCreate(&barrierEvent) != cudaSuccess) {
            throw std::runtime_error("cudaEventCreate failed.");
        }

        barriers.push_back(barrierEvent);
    }
}

GPUState::~GPUState() {
    if (const auto err = cudaStreamDestroy(stream); err != cudaSuccess) {
        std::printf("cudaStreamDestroy failed.\n");
        std::terminate();
    }

    for (cudaEvent_t barrier: barriers) {
        cudaEventDestroy(barrier);
    }
}

void GPUState::copy(uint8_t *gpuBuffer,
                    const uint8_t *buffer,
                    const uint32_t size) const {
    if (cudaMemcpyAsync(gpuBuffer, buffer, size,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync failed.");
    }
}

void GPUState::insertBarrier(const size_t barrierIdx) const {
    if (cudaEventRecord(barriers[barrierIdx], stream) != cudaSuccess) {
        throw std::runtime_error("cudaEventRecord failed.");
    }
}

void GPUState::sync(const size_t barrierIdx) const {
    if (cudaEventSynchronize(barriers[barrierIdx]) != cudaSuccess) {
        throw std::runtime_error("cudaEventSynchronize failed.");
    }
}

DataLoader::DataLoader(
    Dataset _dataset,
    const int _batchSize,
    const int _numThreads,
    const int _prefetchSize
) : dataset(std::move(_dataset)),
    batchSize(_batchSize),
    numThreads(_numThreads),
    prefetchSize(_prefetchSize),
    prefetchSemaphore(_prefetchSize),
    outputBatchMemorySize(0),
    gpu(_prefetchSize),
    threadPool([this] { this->threadMain(); },
               _numThreads)
// TODO: This initialized incorrectly before the defensive programming.
{
    if (batchSize <= 0 || _numThreads <= 0 || _prefetchSize <= 0) {
        throw std::runtime_error(
            "Batch size, the number of threads and the prefetch size need to be strictly positive.");
    }

    if (_prefetchSize < _numThreads) {
        throw std::runtime_error(
            "Prefetch size must be larger or equal than the number of threads.");
    }

    numberOfBatches = (dataset.getEntries().size() + batchSize - 1) /
                      batchSize;

    for (const Head &head: dataset.getHeads()) {
        size_t itemMemorySize = sizeof(float);
        for (const int dim: head.getShape()) {
            itemMemorySize *= dim;
        }

        outputBatchMemorySize += batchSize * itemMemorySize;
    }

    MemoryArena::initialize(outputBatchMemorySize * _prefetchSize);

    threadPool.start();
}

DataLoader::~DataLoader() {
    shutdown = true;
    prefetchSemaphore.disable();
}

void DataLoader::threadMain() {
    while (!shutdown) {
        prefetchSemaphore.acquire();
        if (shutdown) {
            break;
        }

        datasetMutex.lock();
        std::vector<std::vector<std::string> > batchPaths = dataset.getNextBatch(batchSize);
        datasetMutex.unlock();

        const size_t barrierIdx = lastBarrierIdx++ % prefetchSize;
        const std::vector<Head> &heads = dataset.getHeads();

        // For each subdirectory, load all images and assemble into one numpy array.
        ListOfAllocations allocations(outputBatchMemorySize);
        for (size_t i = 0; i < heads.size(); i++) {
            const Head &head = heads[i];

            const size_t itemSize = head.getShapeSize();
            prefetchCacheMutex.lock();
            const size_t batchBufferSize = batchSize * itemSize * sizeof(float);
            const auto batchAlloc = allocations.allocate(batchBufferSize);
            const auto batchBuffer = reinterpret_cast<float *>(batchAlloc.host);
            prefetchCacheMutex.unlock();

            switch (head.getFilesType()) {
                case FileType::JPG: {
                    const auto batchBufferOld = new uint8_t[batchSize * itemSize];

                    for (size_t j = 0; j < batchSize; j++) {
                        ImageData imgData = readJpegFile(batchPaths[j][i]);
                        resizeImage(imgData, batchBufferOld + j * itemSize,
                                    IMAGE_WIDTH(head), IMAGE_HEIGHT(head));
                    }

                    for (size_t b = 0; b < batchSize; b++) {
                        for (size_t y = 0; y < IMAGE_HEIGHT(head); y++) {
                            for (size_t j = 0; j < IMAGE_WIDTH(head); j++) {
                                for (size_t c = 0; c < 3; c++) {
                                    const size_t idx = b * itemSize
                                                       + y * IMAGE_WIDTH(head) * 3
                                                       + j * 3
                                                       + c;
                                    batchBuffer[idx] = static_cast<float>(batchBufferOld[idx]) / 255.0f;
                                }
                            }
                        }
                    }

                    delete[] batchBufferOld;
                }
                break;
                case FileType::NPY: {
                    for (size_t j = 0; j < batchSize; j++) {
                        // Load the NPY file from the path
                        cnpy::NpyArray arr = cnpy::npy_load(batchPaths[j][i]);
                        // Ensure the file contains float data
                        if (arr.word_size != sizeof(float)) {
                            throw std::runtime_error(
                                std::format("NPY file {} has word size {} and does not contain float32 data.",
                                            batchPaths[j][i], arr.word_size));
                        }
                        // Compute the total number of elements in the array
                        size_t npy_num_elements = 1;
                        for (const unsigned long d: arr.shape) {
                            npy_num_elements *= d;
                        }
                        if (npy_num_elements != itemSize) {
                            throw std::runtime_error(
                                "NPY file " + batchPaths[j][i] +
                                " has mismatched size. Expected " +
                                std::to_string(itemSize) + ", got " +
                                std::to_string(npy_num_elements));
                        }
                        // Copy the float data into the batch buffer
                        const auto *npy_data = arr.data<float>();
                        std::memcpy(batchBuffer + j * itemSize, npy_data, itemSize * sizeof(float));
                    }
                }
                break;
                default: {
                    throw std::runtime_error("Should never get here!");
                }
            }

            // Start async upload to gpu memory as soon as possible.
            gpu.copy(batchAlloc.gpu, reinterpret_cast<uint8_t *>(batchBuffer),
                     batchBufferSize);
        }

        gpu.insertBarrier(barrierIdx);

        prefetchCacheMutex.lock();
        prefetchCache.push_back({
            .barrierIdx = barrierIdx,
            .allocations = std::move(allocations)
        });
        prefetchCacheMutex.unlock();

        prefetchCacheNotify.notify_one();
    }
}

void deleter(DLManagedTensor *self) {
    MemoryArena::getInstance()->free();
}

py::dict DataLoader::getNextBatch() {
    std::unique_lock lock(prefetchCacheMutex);
    prefetchCacheNotify.wait(
        lock, [this] { return !prefetchCache.empty(); });

    const auto [barrierIdx, allocations] = std::move(prefetchCache.front());
    prefetchCache.erase(prefetchCache.begin());

    lock.unlock();
    prefetchSemaphore.release();

    py::dict pyBatch;
    const auto &heads = dataset.getHeads(); // TODO: Prevent copy.
    for (size_t i = 0; i < heads.size(); i++) {
        const Head &head = heads[i];
        const uint32_t batchOffset = allocations.getOffset(i);

        const std::vector<int> &shape = head.getShape();
        const int ndim = static_cast<int>(shape.size()) + 1;

        const auto shapeArr = new int64_t[ndim]{};
        shapeArr[0] = static_cast<int64_t>(batchSize);
        for (size_t s = 0; s < shape.size(); s++) {
            shapeArr[s + 1] = shape[s];
        }

        int64_t lastStride = 1;
        const auto stridesArr = new int64_t[shape.size() + 1]{};
        for (int s = static_cast<int>(shape.size()); s >= 0; s--) {
            stridesArr[s] = lastStride;
            lastStride *= shapeArr[s];
        }

        const auto *dlMngTensor = new DLManagedTensor{
            .dl_tensor = {
                .data = MemoryArena::getInstance()->getGpuData(),
                .device = {
                    .device_type = kDLCUDA,
                    .device_id = 0 // This hardcodes GPU0.
                },
                .ndim = ndim,
                .dtype = {
                    .code = kDLFloat,
                    .bits = 32,
                    .lanes = 1
                },
                .shape = shapeArr,
                .strides = null,
                .byte_offset = batchOffset
            },
            .manager_ctx = null,
            .deleter = &deleter
        };

        pyBatch[head.getDictName().c_str()] = py::capsule(
            dlMngTensor, "dltensor", [](void *ptr) {
                const auto *dlManagedTensor = static_cast<DLManagedTensor *>(ptr);
                delete[] dlManagedTensor->dl_tensor.shape;
                delete[] dlManagedTensor->dl_tensor.strides;
                delete dlManagedTensor;
            });
    }

    /*for (size_t i = 0; i < prefetchCache.size(); i++) {
        const auto &pair = prefetchCache[i];
        if (pair.barrierIdx == barrierIdx) {
            std::printf("Idiot index ++ (%lu, %lu, idx %lu, size %lu) \n",
                        pair.barrierIdx,
                        barrierIdx, i, prefetchCache.size() + 1);
        }
    }*/

    gpu.sync(barrierIdx);

    return pyBatch;
}

size_t DataLoader::getNumberOfBatches() const {
    return numberOfBatches;
}
