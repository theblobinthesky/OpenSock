#include "compression.h"

#include <cstring>
#include <filesystem>
#include <format>
#include <immintrin.h>
#include <zstd.h>
#include "cnpy.h"

extern "C" {
#include "bitshuffle_core.h"
}

void loadNpyAsFloat32(const std::string &path, uint8_t *data) {
    cnpy::NpyArray arr = cnpy::npy_load(path);

    if (arr.word_size != sizeof(float)) {
        throw std::runtime_error("NPY file is not float32: " + path);
    }
    if (arr.fortran_order) {
        throw std::runtime_error("NPY file uses Fortran (column-major) ordering: " + path);
    }

    std::memcpy(data, arr.data<uint8_t>(), arr.num_bytes());
}

void writeBinaryFile(const char *path, const uint8_t *data, const size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        throw std::runtime_error(std::format("Failed to open file for writing: {}", path));
    }

    if (const size_t written = fwrite(data, 1, size, fp); written != size) {
        fclose(fp);
        throw std::runtime_error(std::format("Failed to write all data to file: {}", path));
    }

    fclose(fp);
}

size_t readBinaryFileIntoBuffer(const char *path, uint8_t *buffer, size_t bufferSize) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        throw std::runtime_error(std::format("Failed to open file for reading: {}", path));
    }

    // Determine file size
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        throw std::runtime_error(std::format("Failed to seek in file: {}", path));
    }

    const long fileSize = ftell(fp);
    if (fileSize < 0) {
        fclose(fp);
        throw std::runtime_error(std::format("Failed to get file size: {}", path));
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        throw std::runtime_error(std::format("Failed to seek in file: {}", path));
    }

    if (static_cast<size_t>(fileSize) > bufferSize) {
        fclose(fp);
        throw std::runtime_error(std::format(
            "Provided buffer too small: need {}, got {}", fileSize, bufferSize));
    }

    const size_t read = fread(buffer, 1, static_cast<size_t>(fileSize), fp);
    fclose(fp);

    if (read != static_cast<size_t>(fileSize)) {
        throw std::runtime_error(std::format("Failed to read full file: {}", path));
    }

    return read;
}

CompressorSettings probeCompressedFileSettings(const std::string &path) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error(std::format("Failed to open file for reading: {}", path));
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        throw std::runtime_error(std::format("Failed to seek in file: {}", path));
    }

    const long fileSize = ftell(fp);
    if (fileSize < 0) {
        fclose(fp);
        throw std::runtime_error(std::format("Failed to get file size: {}", path));
    }

    if (static_cast<size_t>(fileSize) < sizeof(CompressorSettings)) {
        fclose(fp);
        throw std::runtime_error(std::format(
            "File too small to contain settings: {} bytes < {}", static_cast<size_t>(fileSize),
            sizeof(CompressorSettings)));
    }

    if (fseek(fp, fileSize - static_cast<long>(sizeof(CompressorSettings)), SEEK_SET) != 0) {
        fclose(fp);
        throw std::runtime_error(std::format("Failed to seek to settings in file: {}", path));
    }

    CompressorSettings settings{};
    const size_t read = fread(&settings, 1, sizeof(CompressorSettings), fp);
    fclose(fp);
    if (read != sizeof(CompressorSettings)) {
        throw std::runtime_error(std::format("Failed to read settings from file: {}", path));
    }

    if (settings.magic != MAGIC_NUMBER) {
        throw std::runtime_error("The compressed array contains an incorrect magic number.");
    }

    return settings;
}

void castToFP16(const float *data, const size_t size, uint16_t *out) {
    size_t i = 0;
    for (; i < size; i += 8) {
        const __m256 float32s = _mm256_loadu_ps(data + i);
        const auto half32s = _mm256_cvtps_ph(float32s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), half32s);
    }

    for (; i < size; i++) {
        out[i] = _cvtss_sh(data[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
}

uint64_t getLength(const std::vector<uint32_t> &shape) {
    uint64_t length = 1;
    for (const uint32_t dim: shape) {
        length *= dim;
    }

    return length;
}

uint64_t getLength(const uint32_t shape[MAX_SHAPE_SIZE], const size_t shapeSize) {
    uint64_t length = 1;
    for (size_t i = 0; i < shapeSize; i++) {
        length *= shape[i];
    }

    return length;
}

// NOLINTBEGIN(clang-analyzer-core.uninitialized.ArraySubscript)
template<typename T>
void applyShapePermutationInternal(const uint32_t shape[MAX_SHAPE_SIZE], const uint32_t permutation[MAX_SHAPE_SIZE],
                                   const size_t shapeSize,
                                   const uint8_t *dataInFlat, uint8_t *dataOutFlat) {
    const T *__restrict dataIn = reinterpret_cast<const T *>(dataInFlat);
    T *__restrict dataOut = reinterpret_cast<T *>(dataOutFlat);

    uint32_t oldStrides[MAX_SHAPE_SIZE];
    oldStrides[shapeSize - 1] = 1;
    for (int i = static_cast<int>(shapeSize) - 2; i >= 0; i--) {
        oldStrides[i] = oldStrides[i + 1] * shape[i + 1];
    }

    uint32_t newShape[MAX_SHAPE_SIZE];
    for (size_t i = 0; i < shapeSize; i++) {
        newShape[i] = shape[permutation[i]];
    }

    uint32_t newStrides[MAX_SHAPE_SIZE];
    newStrides[shapeSize - 1] = 1;
    for (int i = static_cast<int>(shapeSize) - 2; i >= 0; i--) {
        newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    }

    uint32_t inversePermutation[MAX_SHAPE_SIZE];
    for (size_t i = 0; i < shapeSize; i++) {
        inversePermutation[permutation[i]] = i;
    }

    uint32_t remapCache[MAX_SHAPE_SIZE];
    for (size_t i = 0; i < shapeSize; i++) {
        remapCache[i] = newStrides[inversePermutation[i]];
    }

    const uint64_t length = getLength(shape, shapeSize);
    for (size_t i = 0; i < length; i++) {
        size_t newIdx = 0;
        for (size_t s = 0; s < shapeSize; s++) {
            ASSERT(shape[s] != 0);
            const size_t oldDimIdx = (i / oldStrides[s]) % shape[s];
            newIdx += oldDimIdx * remapCache[s];
        }

        dataOut[newIdx] = dataIn[i];
    }
}

// NOLINTEND(clang-analyzer-core.uninitialized.ArraySubscript)

void applyShapePermutation(const uint32_t shape[MAX_SHAPE_SIZE], const uint32_t permutation[MAX_SHAPE_SIZE],
                           const CompressorSettings &settings,
                           const uint8_t *dataIn, uint8_t *dataOut) {
    const size_t itemSize = settings.getItemSize();
    if (itemSize == 1) {
        applyShapePermutationInternal<uint8_t>(shape, permutation, settings.shapeLength, dataIn, dataOut);
    } else if (itemSize == 2) {
        applyShapePermutationInternal<uint16_t>(shape, permutation, settings.shapeLength, dataIn, dataOut);
    } else if (itemSize == 4) {
        applyShapePermutationInternal<uint32_t>(shape, permutation, settings.shapeLength, dataIn, dataOut);
    } else if (itemSize == 8) {
        applyShapePermutationInternal<uint64_t>(shape, permutation, settings.shapeLength, dataIn, dataOut);
    } else {
        throw std::runtime_error("Cannot apply shape permutation to item size other than 1, 2, 4 or 8.");
    }
}

void revertShapePermutation(const CompressorSettings &settings, const uint8_t *dataIn, uint8_t *dataOut) {
    uint32_t inverseShape[MAX_SHAPE_SIZE] = {};
    for (size_t i = 0; i < settings.shapeLength; i++) {
        inverseShape[i] = settings.shape[settings.permutation[i]];
    }

    uint32_t inversePermutation[MAX_SHAPE_SIZE] = {};
    for (size_t i = 0; i < settings.shapeLength; i++) {
        inversePermutation[settings.permutation[i]] = i;
    }

    applyShapePermutation(inverseShape, inversePermutation, settings, dataIn, dataOut);
}

void applyBitshuffle(const uint8_t *dataIn, uint8_t *dataOut, const size_t numItems, const size_t itemSize) {
    if (const auto ret = bshuf_bitshuffle(dataIn, dataOut, numItems, itemSize, 0); ret < 0) {
        throw std::runtime_error(std::format("Bitshuffle failed with code {}.", ret));
    }
}

void revertBitshuffle(const uint8_t *dataIn, uint8_t *dataOut, const size_t numItems, const size_t itemSize) {
    if (const auto ret = bshuf_bitunshuffle(dataIn, dataOut, numItems, itemSize, 0); ret < 0) {
        throw std::runtime_error(std::format("Bitunshuffle failed with code {}.", ret));
    }
}

size_t applyCodec(const Codec codec, const uint8_t *dataIn, uint8_t *dataOut, const size_t size) {
    int compressionLevel;
    switch (codec) {
        case Codec::ZSTD_LEVEL_3:
            compressionLevel = 3;
            break;
        case Codec::ZSTD_LEVEL_7:
            compressionLevel = 7;
            break;
        case Codec::ZSTD_LEVEL_22:
            compressionLevel = 22;
            break;
        default:
            throw std::runtime_error("Encountered unknown compression level.");
    }

    const size_t res = ZSTD_compress(dataOut, ZSTD_compressBound(size), dataIn, size, compressionLevel);
    if (ZSTD_isError(res)) {
        throw std::runtime_error(std::format("ZSTD compress failed: {}", ZSTD_getErrorName(res)));
    }

    return res;
}

size_t revertCodec(const Codec codec, const uint8_t *dataIn, uint8_t *dataOut, const size_t compressedSize) {
    if (codec != Codec::NONE) {
        const auto requiredSize = ZSTD_getFrameContentSize(dataIn, compressedSize);

        const size_t res = ZSTD_decompress(dataOut, requiredSize, dataIn, compressedSize);
        if (ZSTD_isError(res)) {
            throw std::runtime_error(std::format("ZSTD decompress failed: {}", ZSTD_getErrorName(res)));
        }

        return res;
    }

    throw std::runtime_error(std::format("Codec {} cannot be reverted.", static_cast<uint32_t>(codec)));
}

void applyAllShortOfCodec(uint8_t *dataIn,
                          uint8_t *dataScratch,
                          uint8_t *dataScratch2,
                          const CompressorSettings &settings,
                          uint8_t *&bitshuffle_out,
                          uint8_t *&bitshuffle_scratch) {
    const size_t length = settings.getShapeSize();
    const bool doesCastToFP16 = settings.flags & static_cast<uint64_t>(CompressorFlags::CAST_TO_FP16);

    uint8_t *fp16_out, *fp16_scratch;
    if (doesCastToFP16) {
        castToFP16(reinterpret_cast<float *>(dataIn), length, reinterpret_cast<uint16_t *>(dataScratch));
        fp16_out = dataScratch;
        fp16_scratch = dataScratch2;
    } else {
        memcpy(dataScratch, dataIn, length * sizeof(float));
        fp16_out = dataScratch;
        fp16_scratch = dataScratch2;
    }

    uint8_t *permute_out, *permute_scratch;
    if (!settings.isIdentityPermutation()) {
        applyShapePermutation(settings.shape, settings.permutation, settings, fp16_out, fp16_scratch);
        permute_out = fp16_scratch;
        permute_scratch = fp16_out;
    } else {
        permute_out = fp16_out;
        permute_scratch = fp16_scratch;
    }

    if (settings.flags & static_cast<uint64_t>(CompressorFlags::BITSHUFFLE)) {
        applyBitshuffle(permute_out, permute_scratch, length, settings.getItemSize());
        bitshuffle_out = permute_scratch;
        bitshuffle_scratch = permute_out;
    } else {
        bitshuffle_out = permute_out;
        bitshuffle_scratch = permute_scratch;
    }
}

size_t getMaxSizeRequiredByCodec(const std::vector<uint32_t> &shape) {
    const uint64_t size = getLength(shape) * sizeof(float); // TODO: only weak upper bound potentially
    return ZSTD_compressBound(size);
}

static std::vector<std::string> listAllFiles(const std::string &directoryPath) {
    std::vector<std::string> paths;

    for (const std::filesystem::directory_entry &entry:
         std::filesystem::recursive_directory_iterator(
             directoryPath)) {
        paths.push_back(entry.path());
    }

    return paths;
}

Compressor::Compressor(const CompressorOptions &_options) : options(_options),
                                                            bufferSize(
                                                                alignUp(getMaxSizeRequiredByCodec(options.shape), 16)
                                                                + alignUp(sizeof(CompressorSettings), 16)),
                                                            arenaSize(options.numThreads * 3 * bufferSize),
                                                            allocator(new uint8_t[arenaSize], arenaSize),
                                                            threadPool([this] { this->threadMain(); },
                                                                       options.numThreads) {
    for (const auto dim: options.shape) {
        if (dim == 0) {
            throw std::runtime_error("The dimension of a shape cannot be non-positive.");
        }
    }

    for (const auto &permutation: options.permutations) {
        if (permutation.size() != options.shape.size()) {
            throw std::runtime_error("A permutation does not have the same size as the shape.");
        }

        for (const auto mapTo: permutation) {
            if (mapTo < 0 || mapTo >= static_cast<int>(options.shape.size())) {
                throw std::runtime_error("A permutation cannot map to indices outside of 0..{shape.size}.");
            }
        }

        for (size_t i = 0; i < permutation.size(); i++) {
            for (size_t j = i + 1; j < permutation.size(); j++) {
                if (permutation[i] == permutation[j]) {
                    throw std::runtime_error("A permutation cannot map to the same index twice.");
                }
            }
        }
    }

    if (options.inputDirectory.ends_with("/")) {
        options.inputDirectory = options.inputDirectory.substr(0, options.inputDirectory.size() - 1);
    }
    if (options.outputDirectory.ends_with("/")) {
        options.outputDirectory = options.outputDirectory.substr(0, options.outputDirectory.size() - 1);
    }

    for (const auto files = listAllFiles(options.inputDirectory); const std::string &file: files) {
        if (file.ends_with(".npy")) {
            workQueue.push({file.substr(options.inputDirectory.size() + 1)});
        }
    }
    workQueueSize = workQueue.size();
    shutdownCounter = 0;
}

void Compressor::start() {
    threadPool.resize(options.numThreads, 0);

    std::unique_lock lock(workMutex);
    workNotify.wait(lock, [this] {
        return shutdownCounter == options.numThreads;
    });
}

Compressor::~Compressor() {
    delete[] allocator.getArena();
}

void Compressor::compressArray(const std::vector<uint32_t> &shape,
                               uint8_t *dataIn,
                               uint8_t *dataScratch,
                               uint8_t *dataScratch2,
                               CompressorSettings &settingsOut,
                               uint8_t *&dataOut,
                               size_t &dataSizeOut) const {
    settingsOut = {};
    settingsOut.setShape(options.shape);
    settingsOut.codec = Codec::ZSTD_LEVEL_3;
    settingsOut.flags = 0;
    if (options.castToFP16) {
        settingsOut.flags |= static_cast<uint64_t>(CompressorFlags::CAST_TO_FP16);
    }
    if (options.withBitshuffle) {
        settingsOut.flags |= static_cast<uint64_t>(CompressorFlags::BITSHUFFLE);
    }

    // Now search for the most appropriate compression:
    const uint64_t size = getLength(shape) * (options.castToFP16 ? 2 : 4);

    // Find the best permutation.
    uint8_t *bitshuffle_out, *bitshuffle_scratch;
    if (options.permutations.empty()) {
        // Set to identity.
        for (size_t i = 0; i < settingsOut.shapeLength; i++) {
            settingsOut.permutation[i] = i;
        }
    } else {
        size_t bestPermIdx = 0;
        size_t bestCompressedSize = -1;
        for (size_t p = 0; p < options.permutations.size(); p++) {
            const std::vector<int> &permutation = options.permutations[p];
            settingsOut.setPermutation(permutation);

            applyAllShortOfCodec(dataIn, dataScratch, dataScratch2, settingsOut, bitshuffle_out, bitshuffle_scratch);
            if (const size_t compressedSize = applyCodec(settingsOut.codec, bitshuffle_out, bitshuffle_scratch, size);
                compressedSize < bestCompressedSize) {
                bestPermIdx = p;
                bestCompressedSize = compressedSize;
            }
        }

        settingsOut.setPermutation(options.permutations[bestPermIdx]);

        if (!settingsOut.isIdentityPermutation()) {
            settingsOut.flags |= static_cast<uint64_t>(CompressorFlags::SHAPE_PERMUTE);
        }
    }

    applyAllShortOfCodec(dataIn, dataScratch, dataScratch2, settingsOut, bitshuffle_out, bitshuffle_scratch);


    // Find the best codec that is still fast.
    if (options.allowedCodecs.empty()) {
        settingsOut.codec = Codec::NONE;
        dataOut = bitshuffle_out;
        dataSizeOut = size;
    } else {
        size_t bestCompressedSize = -1;
        std::vector<size_t> codecCompressedSizes;

        for (const auto allowedCodec: options.allowedCodecs) {
            const size_t compressedSize = applyCodec(allowedCodec, bitshuffle_out, bitshuffle_scratch, size);
            codecCompressedSizes.push_back(compressedSize);

            if (compressedSize < bestCompressedSize) {
                bestCompressedSize = compressedSize;
            }
        }

        size_t fastestCodecWithinToleranceIdx = 0;
        size_t fastestCodecWithinToleranceCompressedSize = 0;
        for (size_t i = 0; i < options.allowedCodecs.size(); i++) {
            if (static_cast<double>(codecCompressedSizes[i] - bestCompressedSize)
                <= options.toleranceForWorseCodec * static_cast<double>(bestCompressedSize)) {
                fastestCodecWithinToleranceIdx = i;
                fastestCodecWithinToleranceCompressedSize = codecCompressedSizes[i];
                break;
            }
        }

        if (fastestCodecWithinToleranceIdx != options.allowedCodecs.size() - 1) {
            const Codec &fastestCodecWithinTolerance = options.allowedCodecs[fastestCodecWithinToleranceIdx];
            applyCodec(fastestCodecWithinTolerance, bitshuffle_out, bitshuffle_scratch, size);
        }
        memcpy(dataIn, bitshuffle_scratch, fastestCodecWithinToleranceCompressedSize);


        dataOut = dataIn;
        dataSizeOut = fastestCodecWithinToleranceCompressedSize;
    }

    settingsOut.compressedSize = dataSizeOut;
}

void Compressor::threadMain() {
    mutex.lock();
    uint8_t *dataIn = allocator.allocate(bufferSize);
    uint8_t *dataScratch = allocator.allocate(bufferSize);
    uint8_t *dataScratch2 = allocator.allocate(bufferSize);
    mutex.unlock();

    while (workQueueSize > 0) {
        mutex.lock();
        const auto [fileName] = workQueue.front();
        workQueue.pop();
        --workQueueSize;
        mutex.unlock();

        const std::string inputPath = std::format("{}/{}", options.inputDirectory, fileName);
        loadNpyAsFloat32(inputPath, dataIn);

        CompressorSettings settingsOut = {};
        uint8_t *dataOut;
        size_t dataSizeOut;
        compressArray(
            options.shape, dataIn, dataScratch, dataScratch2,
            settingsOut, dataOut, dataSizeOut
        );

        const size_t dataSizeOutAligned = alignUp(dataSizeOut, 16);
        *reinterpret_cast<CompressorSettings *>(dataOut + dataSizeOutAligned) = settingsOut;

        const std::string outputPath = std::format("{}/{}.compressed", options.outputDirectory, fileName);
        writeBinaryFile(outputPath.c_str(), dataOut, dataSizeOutAligned + sizeof(CompressorSettings));
    }

    ++shutdownCounter;
    workNotify.notify_all();
}

size_t Decompressor::getMaximumRequiredRawFileBufferSize(const Shape &maxInputShape) {
    return alignUp(getMaxSizeRequiredByCodec(maxInputShape), 16)
           + alignUp(sizeof(CompressorSettings), 16);
}

size_t Decompressor::getMaximumRequiredScratchBufferSize(const Shape &maxInputShape) {
    return getShapeSize(maxInputShape) * getWidthOfDType(DType::FLOAT64);
}

CompressorSettings Decompressor::probeArray(const uint8_t *dataIn, const size_t dataSize) {
    const auto settings = reinterpret_cast<const CompressorSettings *>(dataIn + dataSize - sizeof(CompressorSettings));
    if (settings->magic != MAGIC_NUMBER) {
        throw std::runtime_error("The compressed array contains an incorrect magic number.");
    }

    return *settings;
}

void Decompressor::decompressArray(
    const uint8_t *dataIn,
    uint8_t *dataScratch,
    uint8_t *dataScratch2,
    uint8_t *dataOut,
    const CompressorSettings &settings
) {
    if (settings.magic != MAGIC_NUMBER) {
        throw std::runtime_error("The compressed array contains an incorrect magic number.");
    }
    if (settings.version != FILE_FORMAT_VERSION) {
        throw std::runtime_error("The compressed array has an incorrect version.");
    }

    const uint8_t *src = dataIn;
    const bool hasShuffle = settings.flags & static_cast<uint64_t>(CompressorFlags::BITSHUFFLE);
    const bool hasPermute = settings.flags & static_cast<uint64_t>(CompressorFlags::SHAPE_PERMUTE);

    // Revert codec
    if (settings.codec != Codec::NONE) {
        uint8_t *dst = (hasShuffle || hasPermute) ? dataScratch : dataOut;
        revertCodec(settings.codec, src, dst, settings.compressedSize);
        src = dst;
    }

    // Revert Bitshuffle
    if (hasShuffle) {
        uint8_t *dst = hasPermute ? dataScratch2 : dataOut;
        revertBitshuffle(src, dst, settings.getShapeSize(), settings.getItemSize());
        src = dst;
    }

    // Revert Permutation
    if (hasPermute) {
        revertShapePermutation(settings, src, dataOut);
        src = dataOut;
    }

    if (src != dataOut) {
        std::memcpy(dataOut, src, settings.getShapeSize() * settings.getItemSize());
    }
}
