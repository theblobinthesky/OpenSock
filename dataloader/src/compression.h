#ifndef VERSION_COMPRESSION_H
#define VERSION_COMPRESSION_H
#include <queue>
#include <utility>

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
                      std::vector<int> shape,
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
    std::vector<int> shape;
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

    uint64_t flags;
    Codec codec;

    uint32_t shape[MAX_SHAPE_SIZE];
    uint32_t shapeSize;
    uint32_t permutation[MAX_SHAPE_SIZE];

    uint32_t compressedSize;

    void setShape(const std::vector<int> &newShape) {
        shapeSize = newShape.size();
        for (size_t i = 0; i < shapeSize; i++) {
            shape[i] = newShape[i];
        }
    }

    void setPermutation(const std::vector<int> &newPermutation) {
        for (size_t i = 0; i < shapeSize; i++) {
            permutation[i] = newPermutation[i];
        }
    }

    [[nodiscard]] size_t getShapeLength() const noexcept {
        size_t length = 1;
        for (size_t i = 0; i < shapeSize; i++) length *= shape[i];

        return length;
    }

    [[nodiscard]] int getItemSize() const noexcept {
        return flags & static_cast<uint64_t>(CompressorFlags::CAST_TO_FP16) ? 2 : 4;
    }

    [[nodiscard]] bool isIdentityPermutation() const noexcept {
        for (size_t i = 0; i < shapeSize; i++) {
            if (permutation[i] != i) return false;
        }
        return true;
    }
};

struct CompressionStatistics {
};

struct CompressionItem {
    std::string fileName;
};

class Compressor {
public:
    explicit Compressor(CompressorOptions _options);

    PREVENT_COPY_OR_MOVE(Compressor)

    ~Compressor();

    void start();

private:
    void compressArray(const std::vector<int> &shape,
                       uint8_t *dataIn,
                       uint8_t *dataScratch,
                       uint8_t *dataScratch2,
                       CompressorSettings &settingsOut,
                       uint8_t *&dataOut,
                       size_t &dataSizeOut) const;

    CompressorOptions options;
    ThreadPool threadPool;
    size_t bufferSize; // TODO: This is a little laxxx....
    size_t arenaSize;
    BumpAllocator<uint8_t *> allocator;

    std::queue<CompressionItem> workQueue;
    std::atomic_uint32_t workQueueSize;

    std::mutex mutex;
    std::mutex workMutex;
    std::condition_variable workNotify;
    std::atomic_uint32_t shutdownCounter;

public:
    void threadMain();
};

class Decompressor {
public:
    explicit Decompressor(std::vector<int> _shape);

    PREVENT_COPY_OR_MOVE(Decompressor)

    ~Decompressor();

    void decompress(const std::string &path, std::vector<uint8_t> &outData, std::vector<size_t> &outShape, int &outBytesPerItem) const;

private:
    static void decompressArray(uint8_t *dataIn, uint8_t *dataScratch,
                                const CompressorSettings &settings,
                                uint8_t *&dataOut,
                                size_t &dataSizeOut);

    std::vector<int> shape;
    size_t bufferSize;
    size_t arenaSize;
    BumpAllocator<uint8_t *> allocator;

    uint8_t *dataInBuffer;
    uint8_t *dataScratchBuffer;
};

#endif
