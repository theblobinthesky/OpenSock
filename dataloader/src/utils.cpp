#include "utils.h"

#include <barrier>

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

ThreadPool::ThreadPool(const std::function<void()> &_threadMain,
                       const size_t _threadCount) : threadMain(_threadMain),
                                                    threadCount(_threadCount),
                                                    shutdownBarrier(static_cast<long>(_threadCount)) {
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
    try {
        threadMain();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Thread main threw exception: %s\n", e.what());
    }

    shutdownBarrier.arrive_and_wait();
}
