#include "utils.h"

#include <barrier>
#include <utility>

ThreadPool::ThreadPool(BlockingThreadMain _threadMain, const size_t _threadCount,
                       WakeupForPoolResize _wakeupForPoolResize)
    : threadMain(std::move(_threadMain)),
      desiredThreadCount(_threadCount),
      wakeupForPoolResize(std::move(_wakeupForPoolResize)) {
}

ThreadPool::ThreadPool(const NonblockingThreadMain &_threadMain, const size_t _threadCount,
                       WakeupForPoolResize _wakeupForPoolResize)
    : threadMain([_threadMain](size_t, const std::atomic_uint32_t &) { _threadMain(); }),
      desiredThreadCount(_threadCount),
      wakeupForPoolResize(std::move(_wakeupForPoolResize)) {
}

void ThreadPool::start() {
    for (size_t i = 0; i < desiredThreadCount; i++) {
        threads.emplace_back(&ThreadPool::extendedThreadMain, this, i);
    }
}

void ThreadPool::resize(const size_t newThreadCount) {
    LOG_DEBUG("ThreadPool::resize from {} to {} begin", desiredThreadCount.load(), newThreadCount);

    const size_t oldThreadCount = desiredThreadCount.load();
    desiredThreadCount = newThreadCount;

    // If we are downsizing, wake up any threads potentially blocked on
    // external condition variables/atomics so they can observe shutdown.
    if (wakeupForPoolResize && newThreadCount < oldThreadCount) {
        wakeupForPoolResize();
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
        threads.emplace_back(&ThreadPool::extendedThreadMain, this, i);
    }

    LOG_DEBUG("ThreadPool::resize end");
}

ThreadPool::~ThreadPool() noexcept {
    resize(0);
}

void ThreadPool::extendedThreadMain(const size_t threadIdx) const {
    try {
        threadMain(threadIdx, desiredThreadCount);
    } catch (const std::exception &e) {
        LOG_ERROR("Thread main threw exception: {}", e.what());
    }
}
