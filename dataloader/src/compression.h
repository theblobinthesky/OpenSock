#ifndef VERSION_COMPRESSION_H
#define VERSION_COMPRESSION_H
#include <queue>

#include "utils.h"

enum class Codec {
    ZSTD_LEVEL_3,
    ZSTD_LEVEL_7,
    ZSTD_LEVEL_22
};

struct CompressionOptions {
    bool withPermuteChannels;
    bool withBitshuffle;
    std::vector<Codec> allowedCodecs;
};

struct CompressionStatistics {

};

struct CompressionItem {
};

class Compressor {
public:
    Compressor(CompressionOptions options);

    void compressArray(std::vector<int> shape, uint8_t *dataIn, uint8_t *dataOut, size_t dataOutSize);

private:
    ThreadPool threadPool;
    std::queue<CompressionItem> workQueue;
    std::mutex workQueueMutex;
    CompressionOptions options;

    void applyShapePermutation(const std::vector<int> &shape, uint8_t *dataIn, std::vector<int> perm, uint8_t *dataOut);
    void applyBitshuffle(const std::vector<int> &shape, uint8_t *dataIn, std::vector<int> perm, uint8_t *dataOut);

public:
    void threadMain();
};

// Decompression should be stateless, as it integrates with the already multithreaded dataloader.
void decompressArray(std::vector<int> shape, uint8_t *dataIn, uint8_t *dataOut, size_t dataOutSize);

#endif
