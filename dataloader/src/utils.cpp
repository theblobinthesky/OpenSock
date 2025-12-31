#include "utils.h"

#include <barrier>
#include <utility>

ThreadPool::ThreadPool(const NonblockingThreadMain &_threadMain, const size_t _threadCount,
                       WakeupForPoolResize _wakeupForPoolDownsize,
                       WakeupForPoolResize _wakeupForPoolShutdown)
    : threadMain([_threadMain](size_t, const std::atomic_uint32_t &, const uint64_t) { _threadMain(); }),
      desiredThreadCount(_threadCount),
      wakeupForPoolDownsize(std::move(_wakeupForPoolDownsize)),
      wakeupForPoolShutdown(std::move(_wakeupForPoolShutdown)) {
}

ThreadPool::ThreadPool(BlockingThreadMain _threadMain, const size_t _threadCount,
                       WakeupForPoolResize _wakeupForPoolDownsize,
                       WakeupForPoolResize _wakeupForPoolShutdown)
    : threadMain(std::move(_threadMain)),
      desiredThreadCount(_threadCount),
      wakeupForPoolDownsize(std::move(_wakeupForPoolDownsize)),
      wakeupForPoolShutdown(std::move(_wakeupForPoolShutdown)) {
}

void ThreadPool::resize(const size_t newThreadCount, const uint64_t initialGenIdx) {
    LOG_DEBUG("ThreadPool::resize from {} to {} begin", desiredThreadCount.load(), newThreadCount);

    // Use the actual number of running threads as the current size;
    // desiredThreadCount is an intent observed by workers, not a count of started threads.
    const size_t oldThreadCount = threads.size();
    desiredThreadCount = newThreadCount;

    // If we are downsizing, wake up any threads potentially blocked on
    // external condition variables/atomics so they can observe shutdown.
    if (wakeupForPoolDownsize && newThreadCount > 0) {
        wakeupForPoolDownsize();
    }
    if (wakeupForPoolShutdown && newThreadCount == 0) {
        wakeupForPoolShutdown();
    }

    // Downsize, if necessary.
    for (size_t i = newThreadCount; i < oldThreadCount; i++) {
        if (auto &thread = threads[i]; thread.joinable()) {
            thread.join();
        }
    }

    if (newThreadCount < oldThreadCount) {
        threads.resize(newThreadCount);
    }

    // Upsize, if necessary.
    for (size_t i = oldThreadCount; i < newThreadCount; i++) {
        threads.emplace_back(&ThreadPool::extendedThreadMain, this, i, initialGenIdx);
    }

    LOG_DEBUG("ThreadPool::resize end");
}

ThreadPool::~ThreadPool() noexcept {
    resize(0, 0);
}

void ThreadPool::extendedThreadMain(const size_t threadIdx, const uint64_t initialGenIdx) const {
    try {
        threadMain(threadIdx, desiredThreadCount, initialGenIdx);
    } catch (const std::exception &e) {
        LOG_ERROR("Thread main threw exception: {}", e.what());
        exit(1);
    }
}

void loadFileIntoBuffer(const std::string &path, size_t &inputSize, uint8_t *buf, const size_t bufferSize) {
    inputSize = 0;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("fopen failed.");
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        throw std::runtime_error("fseek failed.");
    }
    inputSize = ftell(f);
    if (inputSize > bufferSize) {
        fclose(f);
        throw std::runtime_error("Load file would overflow the buffer.");
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        throw std::runtime_error("fseek failed.");
    }

    if (!buf) {
        fclose(f);
        throw std::runtime_error("bug is null.");
    }

    if (fread(buf, 1, inputSize, f) != inputSize) {
        fclose(f);
        throw std::runtime_error("fread failed.");
    }

    fclose(f);
}
