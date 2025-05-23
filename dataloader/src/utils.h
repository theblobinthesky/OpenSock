#ifndef TYPES_H
#define TYPES_H
#include <atomic>
#include <barrier>
#include <functional>
#include <condition_variable>
#include <csignal>
#include <latch>
#include <semaphore>
#include <string>

#define null nullptr

#if false
#define debugLog(...) std::printf(__VA_ARGS__)
#else
#define debugLog(...)
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
    SharedPtr(): refCounter(nullptr), ptr(nullptr), weakPtr(false) {
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

class Semaphore {
public:
    explicit Semaphore(int initial);

    PREVENT_COPY_OR_MOVE(Semaphore)

    void acquire();

    void release();

    void releaseAll();

    void disable();

    // private:
public:
    std::counting_semaphore<> semaphore;
    std::atomic_int numTokensUsed;
    std::atomic_bool disabled;
};

class ThreadPool {
public:
    explicit ThreadPool(const std::function<void()> &_threadMain,
                        size_t _threadCount);

    void start();

    ThreadPool &operator=(ThreadPool &&pool) noexcept = delete;

    ThreadPool(const ThreadPool &pool) = delete;

    ThreadPool(ThreadPool &&pool) noexcept = delete;

    ~ThreadPool() noexcept;

private:
    std::function<void()> threadMain;
    size_t threadCount;
    std::vector<std::thread> threads;
    std::barrier<> shutdownBarrier;

    void extendedThreadMain();
};

inline bool existsEnvVar(const std::string &name) {
    return std::getenv(name.c_str()) != null;
}

#define INVALID_DS_ENV_VAR "INVALIDATE_DATASET"


#endif
