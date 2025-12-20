#ifndef TYPES_H
#define TYPES_H
#include <atomic>
#include <barrier>
#include <functional>
#include <semaphore>
#include <string>
#include <random>
#include <chrono>
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

    [[nodiscard]] T getCurrent() {
        return arena + offset;
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

enum class ItemFormat {
    UINT,
    FLOAT
};

enum class DType {
    UINT8,
    INT32,
    FLOAT32
};

inline size_t getWidthByDType(const DType dtype) {
    switch (dtype) {
        case DType::UINT8:
            return 1;
        case DType::INT32:
        case DType::FLOAT32:
            return 4;
        default:
            throw std::runtime_error("DType not supported.");
    }
}

[[nodiscard]] inline double randomUniformDoubleBetween01(const uint64_t seed, const uint64_t subSeed) {
    std::default_random_engine engine;
    engine.seed(seed + subSeed * 0xffff);
    std::uniform_real_distribution<> uniform;
    return uniform(engine);
}

[[nodiscard]] inline size_t randomUniformSize(const uint64_t seed, const uint64_t subSeed, const size_t maxExclusive) {
    const double rand = randomUniformDoubleBetween01(seed, subSeed);
    return static_cast<size_t>(rand * static_cast<double>(static_cast<int64_t>(maxExclusive) - 1));
}

[[nodiscard]] inline size_t randomUniformBetween(const uint64_t seed, const uint64_t subSeed, const size_t minInclusive, const size_t maxExclusive) {
    return minInclusive + randomUniformSize(seed, subSeed, maxExclusive + 1 - minInclusive);
}

[[nodiscard]] inline size_t getIdx(const size_t b, const size_t i, const size_t k, const std::vector<uint32_t> &shape) {
    return b * shape[1] * shape[2]
           + i * shape[2]
           + k;
}

[[nodiscard]] inline size_t getIdx(const size_t b, const size_t i, const size_t j, const size_t k, const std::vector<uint32_t> &shape) {
    return b * shape[1] * shape[2] * shape[3]
           + i * shape[2] * shape[3]
           + j * shape[3]
           + k;
}

[[nodiscard]] inline uint32_t getShapeSize(const std::vector<uint32_t>& shape) {
    uint32_t size = 1;
    for (const uint32_t dim: shape) size *= dim;
    return size;
}

template <typename Func>
void dispatchWithType(const DType dtype, const uint8_t *in, uint8_t *out, Func fun) {
    switch (dtype) {
        case DType::UINT8:
            fun(in, out);
            break;
        case DType::INT32:
            fun(reinterpret_cast<const int32_t *>(in), reinterpret_cast<int32_t *>(out));
            break;
        case DType::FLOAT32:
            fun(reinterpret_cast<const float *>(in), reinterpret_cast<float *>(out));
            break;
        default:
            throw std::runtime_error("Dtype unsupported in augmentation.");
    }
}

#endif
