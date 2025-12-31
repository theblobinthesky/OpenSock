#ifndef VERSION_COMPRESSION_H
#define VERSION_COMPRESSION_H
#include <queue>
#include <utility>
#include <string>

#include "utils.h"

constexpr uint64_t MAGIC_NUMBER = 0x6E61746976656461;
constexpr size_t FILE_FORMAT_VERSION = 1;
constexpr size_t MAX_SHAPE_SIZE = 8;

// If there is a tie in compression performance, smaller codecs are preferred over larger codecs.
enum class Codec: uint32_t {
    NONE = 0,
    ZSTD_LEVEL_3 = 1,
    ZSTD_LEVEL_7 = 2,
    ZSTD_LEVEL_22 = 3
};

constexpr Codec CodecVector[] = {Codec::ZSTD_LEVEL_3, Codec::ZSTD_LEVEL_7, Codec::ZSTD_LEVEL_22};

class CompressorOptions {
public:
    CompressorOptions(const size_t numThreads,
                      std::string inputDirectory, // TODO: Migrate to dataset later.
                      std::string outputDirectory, // TODO: Migrate to dataset later.
                      std::vector<uint32_t> shape,
                      const bool castToFP16,
                      std::vector<std::vector<int> > permutations,
                      const bool withBitshuffle,
                      const std::vector<Codec> &allowedCodecs,
                      const float toleranceForWorseCodec)
        : numThreads(numThreads),
          inputDirectory(std::move(inputDirectory)),
          outputDirectory(std::move(outputDirectory)),
          shape(std::move(shape)),
          castToFP16(castToFP16),
          permutations(std::move(permutations)),
          withBitshuffle(withBitshuffle),
          allowedCodecs(allowedCodecs),
          toleranceForWorseCodec(toleranceForWorseCodec) {
    }

    size_t numThreads;
    std::string inputDirectory; // TODO: Migrate to dataset later.
    std::string outputDirectory; // TODO: Migrate to dataset later.
    std::vector<uint32_t> shape;
    bool castToFP16;
    std::vector<std::vector<int> > permutations;
    bool withBitshuffle;
    std::vector<Codec> allowedCodecs;
    float toleranceForWorseCodec;
};

enum class CompressorFlags: uint64_t {
    CAST_TO_FP16 = 1 << 0,
    SHAPE_PERMUTE = 1 << 1,
    BITSHUFFLE = 1 << 2,
};

struct CompressorSettings {
    uint64_t magic = MAGIC_NUMBER;
    uint16_t version = FILE_FORMAT_VERSION;

    uint64_t flags{};
    Codec codec = Codec::NONE;

    uint32_t shape[MAX_SHAPE_SIZE]{};
    uint32_t shapeLength{};
    uint32_t permutation[MAX_SHAPE_SIZE]{};

    uint32_t compressedSize{};

    void setShape(const std::vector<uint32_t> &newShape) {
        shapeLength = newShape.size();
        for (size_t i = 0; i < shapeLength; i++) {
            shape[i] = newShape[i];
        }
    }

    void setPermutation(const std::vector<int> &newPermutation) {
        for (size_t i = 0; i < shapeLength; i++) {
            permutation[i] = newPermutation[i];
        }
    }

    [[nodiscard]] size_t getShapeSize() const noexcept {
        size_t size = 1;
        for (size_t i = 0; i < shapeLength; i++) size *= shape[i];
        return size;
    }

    [[nodiscard]] Shape getShape() const noexcept {
        Shape sh;
        sh.reserve(shapeLength);
        for (size_t i = 0; i < shapeLength; i++) {
            sh.push_back(shape[i]);
        }
        return sh;
    }

    [[nodiscard]] int getItemSize() const noexcept {
        return (flags & static_cast<uint64_t>(CompressorFlags::CAST_TO_FP16)) ? 2 : 4;
    }

    [[nodiscard]] bool isIdentityPermutation() const noexcept {
        for (size_t i = 0; i < shapeLength; i++) {
            if (permutation[i] != i) return false;
        }
        return true;
    }
};

struct CompressionItem {
    std::string fileName;
};

class Compressor {
public:
    explicit Compressor(const CompressorOptions &options);

    PREVENT_COPY_OR_MOVE(Compressor)

    ~Compressor();

    void start();

private:
    void compressArray(const std::vector<uint32_t> &shape,
                       uint8_t *dataIn,
                       uint8_t *dataScratch,
                       uint8_t *dataScratch2,
                       CompressorSettings &settingsOut,
                       uint8_t *&dataOut,
                       size_t &dataSizeOut) const;

    CompressorOptions options;
    size_t bufferSize; // TODO: This is a little laxxx....
    size_t arenaSize;
    BumpAllocator<uint8_t *> allocator;

    std::queue<CompressionItem> workQueue;
    std::atomic_uint32_t workQueueSize;

    std::mutex mutex;
    std::mutex workMutex;
    std::condition_variable workNotify;
    std::atomic_uint32_t shutdownCounter;

    // The thread pool must be last, so it's destroyed first before all other members.
    ThreadPool threadPool;

public:
    void threadMain();
};

class Decompressor {
public:
    [[nodiscard]] static size_t getMaximumRequiredRawFileBufferSize(const Shape &maxInputShape);

    [[nodiscard]] static size_t getMaximumRequiredScratchBufferSize(const Shape &maxInputShape);

    static CompressorSettings probeArray(const uint8_t *dataIn, size_t dataSize);

    static void decompressArray(
        const uint8_t *dataIn,
        uint8_t *dataScratch,
        uint8_t *dataScratch2,
        uint8_t *dataOut,
        const CompressorSettings &settings
    );
};

CompressorSettings probeCompressedFileSettings(const std::string &path);

#endif
