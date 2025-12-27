#include "NpyDataDecoder.h"
#include "cnpy.h"
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cstring>

ProbeResult NpyDataDecoder::probeFromMemory(uint8_t *inputData, const size_t inputSize) {
    if (inputSize < 12) throw std::runtime_error("NPY too small");

    size_t wordSize = 0;
    std::vector<size_t> shp;
    bool fortran = false;
    cnpy::parse_npy_header(inputData, wordSize, shp, fortran);

    if (fortran) {
        throw std::runtime_error("NPY fortran_order True not supported");
    }
    if (wordSize != 4) {
        throw std::runtime_error(std::format("NPY word size is {} not 4 bytes", wordSize));
    }

    Shape shape;
    shape.reserve(shp.size());
    for (const size_t d: shp) {
        if (d == 0) throw std::runtime_error("NPY shape contains zero dimension");
        shape.push_back(static_cast<uint32_t>(d));
    }

    return ProbeResult{
        .dtype = DType::FLOAT32,
        .shape = std::move(shape),
        .extension = "npy"
    };
}

DecodingResult NpyDataDecoder::loadFromMemory(const uint32_t bufferSize, uint8_t *inputData, const size_t inputSize,
                                              BumpAllocator<uint8_t *> &output) {
    if (inputSize < 12) throw std::runtime_error("NPY too small");

    size_t word_size = 0;
    std::vector<size_t> shp;
    bool fortran = false;
    cnpy::parse_npy_header(inputData, word_size, shp, fortran);

    if (fortran) {
        throw std::runtime_error("NPY fortran_order=True not supported");
    }
    if (word_size != 4) {
        throw std::runtime_error("NPY word size not float32 (4 bytes)");
    }

    // cnpy memory header uses 2-byte header length at offset 8; data follows header at offset 9+len
    const uint16_t header_len = *reinterpret_cast<const uint16_t *>(inputData + 8);
    const size_t data_offset = 9 + static_cast<size_t>(header_len);

    Shape shape;
    size_t shapeSize = 1;
    for (const size_t s: shp) {
        shape.push_back(static_cast<uint32_t>(s));
        shapeSize *= s;
    }

    const size_t n_bytes = shapeSize * getWidthOfDType(DType::FLOAT32);
    if (data_offset + n_bytes > inputSize) {
        throw std::runtime_error("NPY payload truncated");
    }

    uint8_t *out = output.allocate(bufferSize);
    std::memcpy(out, inputData + data_offset, n_bytes);

    return {
        .data = out,
        .shape = shape
    };
}

std::string NpyDataDecoder::getExtension() {
    return "npy";
}
