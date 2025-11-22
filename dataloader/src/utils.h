#ifndef TYPES_H
#define TYPES_H
#include <atomic>
#include <barrier>
#include <functional>
#include <semaphore>
#include <string>
#include "spdlog/spdlog.h"

#ifdef ENABLE_DEBUG_PRINT
#define LOG_TRACE(...)   spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::trace, __VA_ARGS__)
#define LOG_DEBUG(...)   spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::debug, __VA_ARGS__)
#define LOG_DEBUG_FUN(FUN, ...) spdlog::log(spdlog::source_loc{__FILE__, __LINE__, (FUN)}, spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...)    spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::info, __VA_ARGS__)
#define LOG_WARNING(...) spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...)   spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::err, __VA_ARGS__)
#else
#define LOG_TRACE(...)
#define LOG_DEBUG(...)
#define LOG_DEBUG_FUN(FUN, ...)
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


using NonblockingThreadMain = std::function<void()>;

using BlockingThreadMain = std::function<void(size_t, const std::atomic_uint32_t &)>;

// Called by the pool when resizing down, so blocked workers can be woken up.
using WakeupForPoolResize = std::function<void()>;

class ThreadPool {
public:
    explicit ThreadPool(const NonblockingThreadMain &_threadMain, size_t _threadCount,
                        WakeupForPoolResize _wakeupForPoolDownsize = {},
                        WakeupForPoolResize _wakeupForPoolShutdown = {});

    explicit ThreadPool(BlockingThreadMain _threadMain, size_t _threadCount,
                        WakeupForPoolResize _wakeupForPoolDownsize = {},
                        WakeupForPoolResize _wakeupForPoolShutdown = {});

    void resize(size_t newThreadCount);

    ThreadPool &operator=(ThreadPool &&pool) noexcept = delete;

    ThreadPool(const ThreadPool &pool) = delete;

    ThreadPool(ThreadPool &&pool) noexcept = delete;

    ~ThreadPool() noexcept;

private:
    BlockingThreadMain threadMain;
    std::atomic_uint32_t desiredThreadCount;
    std::vector<std::thread> threads;
    WakeupForPoolResize wakeupForPoolDownsize;
    WakeupForPoolResize wakeupForPoolShutdown;

    void extendedThreadMain(size_t threadIdx) const;
};

template<typename T>
class BumpAllocator {
public:
    BumpAllocator(T arena, const size_t arenaSize) : arena(arena), arenaSize(arenaSize) {
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
    size_t offset = 0;
    size_t arenaSize = 0;
};

// Safe for us: environment never mutated after startup
// NOLINTNEXTLINE(concurrency-mt-unsafe)
inline bool existsEnvVar(const std::string &name) {
    return std::getenv(name.c_str()) != nullptr;
}

inline uint64_t alignUp(const uint64_t offset, const uint64_t alignTo) {
    return (offset + alignTo - 1) & ~(alignTo - 1);
}

#define INVALID_DS_ENV_VAR "INVALIDATE_DATASET"

#if !defined(__AVX2__)
#error This project requires avx2 support.
#endif

struct Fence {
    uint64_t id;
};

struct ConsumerStream {
    uint64_t id;
};

#endif
