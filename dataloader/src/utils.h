#ifndef TYPES_H
#define TYPES_H
#include <atomic>
#include <barrier>
#include <functional>
#include <csignal>
#include <semaphore>
#include <string>
#include "spdlog/spdlog.h"

#define null nullptr

#ifdef ENABLE_DEBUG_PRINT
#define LOG_DEBUG(...)   spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...)    spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::info, __VA_ARGS__)
#define LOG_WARNING(...) spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...)   spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::err, __VA_ARGS__)
#else
#define LOG_DEBUG(...)
#define LOG_INFO(...)
#define LOG_WARNING(...)
#define LOG_ERROR(...)
#endif

#define PREVENT_MOVE(ClassName) \
    ClassName &operator=(ClassName &&) = delete; \
    ClassName(ClassName &&) = delete;

#define PREVENT_COPY(ClassName) \
    ClassName &operator=(const ClassName &) = delete; \
    ClassName(const ClassName &) = delete;

#define PREVENT_COPY_OR_MOVE(ClassName) \
    PREVENT_MOVE(ClassName) \
    PREVENT_COPY(ClassName)


template<typename T>
class SharedPtr {
public:
    SharedPtr() : refCounter(nullptr), ptr(nullptr), weakPtr(false) {
    }

    explicit SharedPtr(T *ptr, const bool weakPtr = false)
        : refCounter(new int{weakPtr ? 0 : 1}),
          ptr(ptr),
          weakPtr(weakPtr) {
        if (ptr == null) {
            throw std::runtime_error("Shared pointer cannot handle a null pointer.");
        }
    }

    ~SharedPtr() {
        release();
    }

    SharedPtr &operator=(const SharedPtr &other) = delete;

    SharedPtr(const SharedPtr &other) : refCounter(other.refCounter), ptr(other.ptr), weakPtr(false) {
        acquire();
    }

    SharedPtr &operator=(SharedPtr &&other) noexcept {
        if (this != &other) {
            // This instance might have held an old pointer.
            // Make sure to not leave it dangling.
            release();

            refCounter = other.refCounter;
            ptr = other.ptr;
            weakPtr = other.weakPtr;

            other.refCounter = null;
            other.ptr = null;
            other.weakPtr = false;
        }

        return *this;
    }

    SharedPtr(SharedPtr &&other) noexcept
        : refCounter(other.refCounter),
          ptr(other.ptr),
          weakPtr(other.weakPtr) {
        other.refCounter = null;
        other.ptr = null;
        other.weakPtr = false;
    }

    void acquire() const {
        if (refCounter != null) {
            ++(*refCounter);
        }
    }

    void release() {
        // Already freed memory cannot be freed twice.
        // Weak pointers are exempted from the reference counting mechanism.
        if (refCounter == null || weakPtr) return;

        --(*refCounter);

        if (*refCounter == 0) {
            // delete refCounter;
            // delete ptr;
            // TODO: Fix this!
        }

        refCounter = null;
        ptr = null;
    }

    T *get() {
        if (ptr == null) {
            throw std::runtime_error("SharedPtr::get failed because ptr is null.");
        }

        return ptr;
    }

    T &operator*() const {
        if (ptr == null) {
            throw std::runtime_error("SharedPtr::operator* failed because ptr is null.");
        }

        return *ptr;
    }

    T *operator->() const {
        if (ptr == null) {
            std::raise(SIGTRAP);
            // throw std::runtime_error("SharedPtr::operator-> failed because ptr is null.");
        }

        return ptr;
    }

    explicit operator bool() const {
        return ptr != null;
    }

private:
    int *refCounter;
    T *ptr;
    bool weakPtr;
};

using NonblockingThreadMain = std::function<void()>;

using BlockingThreadMain = std::function<void(size_t, const std::atomic_uint32_t &)>;

class ThreadPool {
public:
    explicit ThreadPool(BlockingThreadMain _threadMain, size_t _threadCount);

    explicit ThreadPool(const NonblockingThreadMain &_threadMain, size_t _threadCount);

    void start();

    void resize(size_t newThreadCount);

    ThreadPool &operator=(ThreadPool &&pool) noexcept = delete;

    ThreadPool(const ThreadPool &pool) = delete;

    ThreadPool(ThreadPool &&pool) noexcept = delete;

    ~ThreadPool() noexcept;

private:
    BlockingThreadMain threadMain;
    std::atomic_uint32_t desiredThreadCount;
    std::vector<std::thread> threads;

    void extendedThreadMain(size_t threadIdx) const;
};

template<typename T>
class BumpAllocator {
public:
    BumpAllocator(T arena, const size_t arenaSize) : arena(arena), offset(0), arenaSize(arenaSize) {
    }

    T allocate(const size_t size) {
        T t = arena + offset % arenaSize;
        offset += size;
        return t;
    }

    void reset() {
        offset = 0;
    }

    [[nodiscard]] T &getArena() {
        return arena;
    }

private:
    T arena;
    size_t offset;
    size_t arenaSize;
};

inline bool existsEnvVar(const std::string &name) {
    return std::getenv(name.c_str()) != null;
}

inline uint64_t alignUp(const uint64_t offset, const uint64_t alignTo) {
    return (offset + alignTo - 1) & ~(alignTo - 1);
}

#define INVALID_DS_ENV_VAR "INVALIDATE_DATASET"

#if !defined(__AVX2__)
#error This project requires avx2 support.
#endif

#endif
