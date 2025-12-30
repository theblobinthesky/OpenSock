#include "NpyDataDecoder.h"
#include "cnpy.h"
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>

ProbeResult NpyDataDecoder::probeFromMemory(uint8_t *inputData, const size_t inputSize) {
    if (inputSize < 12) throw std::runtime_error("NPY input too small");

    size_t wordSize = 0;
    std::vector<size_t> shp;
    bool fortran = false;

    // cnpy parses the dictionary, but we must validate the basics
    cnpy::parse_npy_header(inputData, wordSize, shp, fortran);

    if (fortran) {
        throw std::runtime_error("NPY fortran_order True not supported");
    }
    if (wordSize != 4) {
        throw std::runtime_error("NPY word size is not 4 bytes (Float32 required)");
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
    if (inputSize < 12) throw std::runtime_error("NPY input too small");

    size_t word_size = 0;
    std::vector<size_t> shp;
    bool fortran = false;
    cnpy::parse_npy_header(inputData, word_size, shp, fortran);

    if (fortran) throw std::runtime_error("NPY fortran_order=True not supported");
    if (word_size != 4) throw std::runtime_error("NPY word size not float32 (4 bytes)");

    // Safely determine data offset handling NPY versions and endianness
    // V1: 6(magic) + 2(ver) + 2(len) = 10 byte preamble
    // V2: 6(magic) + 2(ver) + 4(len) = 12 byte preamble
    const uint8_t major_ver = inputData[6];
    LOG_TRACE("File has numpy version {}", major_ver);
    uint32_t header_len = 0;
    size_t preamble_len = 0;

    if (major_ver == 1) {
        uint16_t hlen_u16 = 0;
        std::memcpy(&hlen_u16, inputData + 8, sizeof(hlen_u16));
        header_len = hlen_u16;
        preamble_len = 10;
    } else if (major_ver == 2) {
        std::memcpy(&header_len, inputData + 8, sizeof(header_len));
        preamble_len = 12;
    } else {
        throw std::runtime_error("Unsupported NPY major version");
    }

    const size_t data_offset = preamble_len + header_len;

    // Calculate total size with overflow checks
    size_t total_elements = 1;
    for (const size_t s : shp) {
        if (s == 0) throw std::runtime_error("NPY shape contains zero dimension");
        if (total_elements > std::numeric_limits<size_t>::max() / s) {
            throw std::runtime_error("NPY shape overflow");
        }
        total_elements *= s;
    }

    const size_t n_bytes = total_elements * getWidthOfDType(DType::FLOAT32);

    // Strict bounds checking
    if (data_offset > inputSize || (inputSize - data_offset) < n_bytes) {
        throw std::runtime_error("NPY payload truncated or header length mismatch");
    }
    if (n_bytes > bufferSize) {
        throw std::runtime_error("Npy array sizes are not constant / buffer too small.");
    }

    Shape shape;
    shape.reserve(shp.size());
    for (const size_t s : shp) shape.push_back(static_cast<uint32_t>(s));

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
