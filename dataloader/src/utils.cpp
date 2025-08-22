#include "utils.h"

#include <barrier>
#include <utility>

Semaphore::Semaphore(const int initial)
    : semaphore(initial), numTokensUsed(0), disabled(false) {
}

void Semaphore::acquire() {
    if (!disabled) {
        ++numTokensUsed;
        semaphore.acquire();
    }
}

void Semaphore::release() {
    semaphore.release();
    --numTokensUsed;
}

void Semaphore::releaseAll() {
    semaphore.release(numTokensUsed);
    numTokensUsed = 0;
}

void Semaphore::disable() {
    disabled = true;
    releaseAll();
}

ThreadPool::ThreadPool(BlockingThreadMain _threadMain, const size_t _threadCount)
    : threadMain(std::move(_threadMain)),
      desiredThreadCount(_threadCount) {
}

ThreadPool::ThreadPool(const NonblockingThreadMain &_threadMain, const size_t _threadCount)
    : threadMain([_threadMain](size_t, const std::atomic_uint32_t &) { _threadMain(); }),
      desiredThreadCount(_threadCount) {
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

    // Downsize, if necessary.
    for (size_t i = newThreadCount; i < oldThreadCount; i++) {
        if (auto &thread = threads[i]; thread.joinable()) {
            thread.join();
        }
    }

    threads.resize(newThreadCount);

    // Upsize, if necessary.
    for (size_t i = desiredThreadCount; i < newThreadCount; i++) {
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
