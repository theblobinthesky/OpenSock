#include "dataloader.h"
#include "image.h"
#include <filesystem>
#include <pybind11/numpy.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cuda_runtime.h>

#define CHS 3

namespace py = pybind11;
namespace fs = std::filesystem;

// TODO: Align everything!
PinnedMemoryArena::PinnedMemoryArena(): total_size(0),
                                        offset(0) {
}

PinnedMemoryArena::PinnedMemoryArena(const size_t _total_size) : total_size(
        _total_size), offset(0) {
    const cudaError_t err = cudaMallocHost(&data, _total_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed");
    }

    std::printf("Allocated %luMB of pinned memory.\n", _total_size / 1024 / 1024);
}

PinnedMemoryArena &PinnedMemoryArena::operator=(
    PinnedMemoryArena &&arena) noexcept {
    offset = arena.offset;
    total_size = arena.total_size;
    data = arena.data;

    arena.offset = 0;
    arena.total_size = 0;
    arena.data = null;
    return *this;
}

PinnedMemoryArena::PinnedMemoryArena(
    PinnedMemoryArena &&arena) noexcept : data(arena.data),
                                          total_size(arena.total_size),
                                          offset(arena.offset) {
    arena.data = null;
    arena.offset = arena.total_size = 0;
}

PinnedMemoryArena::~PinnedMemoryArena() {
    cudaFreeHost(data);
}

// TODO: Make sure to leave more space in a future version of this allocator
// so threads don't contend for the same cache line potentially.
uint8_t *PinnedMemoryArena::allocate(const size_t size) {
    uint8_t *ptr = data + offset;
    offset += size;

    if (offset > total_size) {
        throw std::bad_alloc();
    }

    offset = offset % total_size;
    return ptr;
}

size_t PinnedMemoryArena::getOffset() const {
    return offset;
}


ListOfAllocations::ListOfAllocations(
    PinnedMemoryArena *_memoryArena): memoryArena(_memoryArena) {
}

ListOfAllocations &ListOfAllocations::operator=(
    ListOfAllocations &&allocs) noexcept {
    memoryArena = allocs.memoryArena;
    ptrs = std::move(allocs.ptrs);
    sizes = std::move(allocs.sizes);

    allocs.memoryArena = null;

    return *this;
}

ListOfAllocations::ListOfAllocations(
    ListOfAllocations &&allocs) noexcept : memoryArena(allocs.memoryArena),
                                           ptrs(std::move(allocs.ptrs)),
                                           sizes(std::move(allocs.sizes)) {
    allocs.memoryArena = null;
}

uint8_t *ListOfAllocations::allocate(const size_t size) {
    const size_t prevOffset = memoryArena->getOffset();
    uint8_t *ptr = memoryArena->allocate(size);
    const size_t offset = memoryArena->getOffset();

    ptrs.push_back(ptr);
    sizes.push_back(offset - prevOffset);
    return ptr;
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

pybind11::array_t<float>
tensor_to_numpy(const Eigen::Tensor<float, 4> &tensor) {
    auto dims = tensor.dimensions();
    return py::array_t(
        {dims[0], dims[1], dims[2], dims[3]},
        {
            sizeof(float) * dims[1] * dims[2] * dims[3],
            sizeof(float) * dims[2] * dims[3],
            sizeof(float) * dims[3],
            sizeof(float)
        },
        tensor.data()
    );
}

bool existsEnvVar(const std::string &name) {
    return std::getenv(name.c_str()) != null;
}

DataLoader::DataLoader(
    Dataset _dataset,
    const int _batchSize,
    py::function _createDatasetFunction,
    const int _numThreads,
    const int _prefetchSize
) : dataset(std::move(_dataset)),
    batchSize(_batchSize),
    createDatasetFunction(std::move(_createDatasetFunction)),
    prefetchSemaphore(_prefetchSize),
    outputBatchMemorySize(0), threadPool([this] { threadMain(); }, _numThreads)
// TODO: This initialized incorreclty before the defensive programming.
{
    if (batchSize <= 0 || _numThreads <= 0 || _prefetchSize <= 0) {
        throw std::runtime_error(
            "Batch size, the number of threads and the prefetch size need to be strictly positive.");
    }

    if (!fs::exists(dataset.getRootDir()) || existsEnvVar(
            INVALID_DS_ENV_VAR)) {
        fs::remove_all(dataset.getRootDir());
        createDatasetFunction();
    }

    dataset.init();

    numberOfBatches = (dataset.getDataset().size() + batchSize - 1) /
                      batchSize;

    for (const Subdirectory &subDir: dataset.getSubDirs()) {
        outputBatchMemorySize += batchSize * subDir.getImageHeight() *
                subDir.
                getImageWidth() * 3 * sizeof(float);
    }

    const auto mas = outputBatchMemorySize * _prefetchSize;
    memoryArena = PinnedMemoryArena(mas);

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
        std::vector<std::vector<std::string> > batchPaths = dataset.
                getNextBatch(
                    batchSize);
        datasetMutex.unlock();

        const std::vector<Subdirectory> &subDirs = dataset.getSubDirs();

        // For each subdirectory, load all images and assemble into one numpy array.
        ListOfAllocations allocations(&memoryArena);
        for (size_t i = 0; i < subDirs.size(); i++) {
            const Subdirectory &subDir = subDirs[i];

            const size_t imageSize = subDir.calculateImageSize();
            prefetchCacheMutex.lock();
            uint8_t *batchBuffer = allocations.allocate(batchSize * imageSize);
            prefetchCacheMutex.unlock();

            // Load the remaining images.
            for (size_t j = 0; j < batchSize; j++) {
                ImageData imgData = readJpegFile(batchPaths[j][i]);
                resizeImage(imgData, batchBuffer + j * imageSize,
                            subDir.getImageWidth(),
                            subDir.getImageHeight());
            }
        }

        prefetchCacheMutex.lock();
        prefetchCache.push_back(std::move(allocations));
        prefetchCacheMutex.unlock();

        prefetchCacheNotify.notify_one();
    }
}

py::dict DataLoader::getNextBatch() {
    std::unique_lock lock(prefetchCacheMutex);
    prefetchCacheNotify.wait(
        lock, [this] { return !prefetchCache.empty(); });

    const ListOfAllocations batch = std::move(prefetchCache.front());
    prefetchCache.erase(prefetchCache.begin());
    lock.unlock();
    prefetchSemaphore.release();

    py::dict pyBatch;
    auto subDirs = dataset.getSubDirs(); // TODO: Prevent copy.
    for (size_t i = 0; i < subDirs.size(); i++) {
        const Subdirectory &subDir = subDirs[i];
        uint8_t *batchData = batch.ptrs[i];
        // const uint32_t *batchSize = batch.ptrs[i];

        // Create a capsule that will free the batchBuffer when the numpy array is garbage collected.
        py::capsule freeWhenDone(batchData, [](void *batchData) {
        });

        // Define the shape and strides for the numpy array.
        std::vector shape = {
            static_cast<ssize_t>(batchSize),
            static_cast<long>(subDir.getImageHeight()),
            static_cast<long>(subDir.getImageWidth()),
            static_cast<long>(CHS)
        };

        std::vector strides = {
            static_cast<ssize_t>(subDir.calculateImageSize()),
            static_cast<ssize_t>(subDir.getImageWidth() * CHS),
            static_cast<ssize_t>(CHS),
            static_cast<ssize_t>(1)
        };

        // Create the numpy array from the contiguous buffer.
        py::array npBatch(py::buffer_info(
                              batchData, /* Pointer to buffer */
                              sizeof(unsigned char),
                              /* Size of one scalar */
                              py::format_descriptor<unsigned
                                  char>::format(),
                              /* Format descriptor */
                              4, /* Number of dimensions */
                              shape, /* Buffer dimensions */
                              strides /* Strides (in bytes) */
                          ), freeWhenDone);

        // Store the assembled batch in the dictionary under the subdirectories dict name.
        pyBatch[subDir.getDictName().c_str()] = npBatch;
    }

    return pyBatch;
}

size_t DataLoader::getNumberOfBatches() const {
    return numberOfBatches;
}
