#include "NpyDataDecoder.h"

#include <charconv>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
constexpr uint8_t NPY_MAGIC[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

uint16_t readU16LE(const uint8_t *p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t readU32LE(const uint8_t *p) {
    return static_cast<uint32_t>(p[0])
           | (static_cast<uint32_t>(p[1]) << 8)
           | (static_cast<uint32_t>(p[2]) << 16)
           | (static_cast<uint32_t>(p[3]) << 24);
}

void skipSpaces(const char *&p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
        ++p;
    }
}

size_t findHeaderKey(const std::string_view header, const std::string_view key) {
    size_t pos = header.find(key);
    while (pos != std::string_view::npos) {
        const size_t endPos = pos + key.size();
        if (pos >= 1 && endPos < header.size()) {
            const char q1 = header[pos - 1];
            const char q2 = header[endPos];
            if ((q1 == '\'' && q2 == '\'') || (q1 == '"' && q2 == '"')) {
                return pos - 1;
            }
        }
        pos = header.find(key, pos + 1);
    }
    throw std::runtime_error(std::format("NPY header missing key '{}'", key));
}

std::string_view extractStringValue(const std::string_view header, const std::string_view key) {
    size_t pos = findHeaderKey(header, key);

    pos = header.find(':', pos);
    if (pos == std::string_view::npos) {
        throw std::runtime_error(std::format("NPY header key '{}' missing ':'", key));
    }
    ++pos;

    const char *p = header.data() + pos;
    const char *end = header.data() + header.size();
    skipSpaces(p, end);
    if (p >= end || (*p != '\'' && *p != '"')) {
        throw std::runtime_error(std::format("NPY header key '{}' not a string", key));
    }

    const char quote = *p;
    ++p;
    const char *start = p;
    while (p < end && *p != quote) {
        ++p;
    }
    if (p >= end) {
        throw std::runtime_error(std::format("NPY header key '{}' unterminated string", key));
    }

    return {start, static_cast<size_t>(p - start)};
}

bool extractBoolValue(const std::string_view header, const std::string_view key) {
    size_t pos = findHeaderKey(header, key);

    pos = header.find(':', pos);
    if (pos == std::string_view::npos) {
        throw std::runtime_error(std::format("NPY header key '{}' missing ':'", key));
    }
    ++pos;

    const char *p = header.data() + pos;
    const char *end = header.data() + header.size();
    skipSpaces(p, end);

    constexpr std::string_view TRUE_STR = "True";
    constexpr std::string_view FALSE_STR = "False";
    if (static_cast<size_t>(end - p) >= TRUE_STR.size() && std::string_view(p, TRUE_STR.size()) == TRUE_STR) {
        return true;
    }
    if (static_cast<size_t>(end - p) >= FALSE_STR.size() && std::string_view(p, FALSE_STR.size()) == FALSE_STR) {
        return false;
    }

    throw std::runtime_error(std::format("NPY header key '{}' not a bool", key));
}

Shape extractShapeValue(const std::string_view header) {
    size_t pos = header.find("'shape'");
    if (pos == std::string_view::npos) {
        pos = header.find("\"shape\"");
    }
    if (pos == std::string_view::npos) {
        throw std::runtime_error("NPY header missing key 'shape'");
    }

    pos = header.find(':', pos);
    if (pos == std::string_view::npos) {
        throw std::runtime_error("NPY header key 'shape' missing ':'");
    }
    ++pos;

    const char *p = header.data() + pos;
    const char *end = header.data() + header.size();
    skipSpaces(p, end);
    if (p >= end || *p != '(') {
        throw std::runtime_error("NPY header key 'shape' not a tuple");
    }
    ++p;

    Shape shape;
    bool sawClosingParen = false;
    while (p < end) {
        skipSpaces(p, end);
        if (p < end && *p == ')') {
            ++p;
            sawClosingParen = true;
            break;
        }

        uint64_t dim = 0;
        const char *numStart = p;
        auto res = std::from_chars(numStart, end, dim);
        if (res.ec != std::errc()) {
            throw std::runtime_error("NPY header shape contains non-integer");
        }
        p = res.ptr;

        if (dim == 0) {
            throw std::runtime_error("NPY shape contains zero dimension");
        }
        if (dim > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("NPY shape dimension too large");
        }
        shape.push_back(static_cast<uint32_t>(dim));

        skipSpaces(p, end);
        if (p < end && *p == ',') {
            ++p;
            continue;
        }
        if (p < end && *p == ')') {
            ++p;
            sawClosingParen = true;
            break;
        }
    }

    if (!sawClosingParen) {
        throw std::runtime_error("NPY header shape missing ')' terminator");
    }

    return shape;
}

struct ParsedNpy {
    DType dtype;
    uint32_t itemSize;
    Shape shape;
    size_t dataOffset;
};

ParsedNpy parseNpy(const uint8_t *inputData, const size_t inputSize) {
    if (inputSize < 10) {
        throw std::runtime_error("NPY too small");
    }
    if (std::memcmp(inputData, NPY_MAGIC, sizeof(NPY_MAGIC)) != 0) {
        throw std::runtime_error("NPY magic header mismatch");
    }

    const uint8_t major = inputData[6];
    const uint8_t minor = inputData[7];
    (void)minor;

    size_t headerStart = 0;
    size_t headerLen = 0;
    if (major == 1) {
        if (inputSize < 10) throw std::runtime_error("NPY too small");
        headerLen = readU16LE(inputData + 8);
        headerStart = 10;
    } else if (major == 2 || major == 3) {
        if (inputSize < 12) throw std::runtime_error("NPY too small");
        headerLen = readU32LE(inputData + 8);
        headerStart = 12;
    } else {
        throw std::runtime_error(std::format("Unsupported NPY version {}", major));
    }

    const size_t headerEnd = headerStart + headerLen;
    if (headerEnd > inputSize) {
        throw std::runtime_error("NPY header truncated");
    }

    const std::string_view header(reinterpret_cast<const char *>(inputData + headerStart), headerLen);

    const bool fortran = extractBoolValue(header, "fortran_order");
    if (fortran) {
        throw std::runtime_error("NPY fortran_order=True not supported");
    }

    const std::string_view descr = extractStringValue(header, "descr");
    if (descr.size() < 3) {
        throw std::runtime_error("NPY dtype descr too short");
    }

    const char endian = descr[0];
    const char type = descr[1];
    uint32_t itemSize = 0;
    {
        const char *p = descr.data() + 2;
        const char *end = descr.data() + descr.size();
        uint64_t tmp = 0;
        const auto res = std::from_chars(p, end, tmp);
        if (res.ec != std::errc() || tmp == 0 || tmp > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("NPY dtype itemsize parse failed");
        }
        itemSize = static_cast<uint32_t>(tmp);
    }

    if (endian == '>' && itemSize > 1) {
        throw std::runtime_error("Big-endian NPY not supported");
    }

    DType dtype;
    if (type == 'u' && itemSize == 1) {
        dtype = DType::UINT8;
    } else if (type == 'i' && itemSize == 4) {
        dtype = DType::INT32;
    } else if (type == 'f' && itemSize == 2) {
        dtype = DType::FLOAT16;
    } else if (type == 'f' && itemSize == 4) {
        dtype = DType::FLOAT32;
    } else if (type == 'f' && itemSize == 8) {
        dtype = DType::FLOAT64;
    } else {
        throw std::runtime_error(std::format("Unsupported NPY dtype descr '{}'", descr));
    }

    Shape shape = extractShapeValue(header);

    return {
        .dtype = dtype,
        .itemSize = itemSize,
        .shape = std::move(shape),
        .dataOffset = headerEnd
    };
}

uint64_t getNumElements(const Shape &shape) {
    uint64_t num = 1;
    for (const uint32_t dim: shape) {
        if (dim == 0) {
            throw std::runtime_error("NPY shape contains zero dimension");
        }
        if (num > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error("NPY element count overflow");
        }
        num *= dim;
    }
    return num;
}
} // namespace

ProbeResult NpyDataDecoder::probeFromMemory(uint8_t *inputData, const size_t inputSize) {
    const ParsedNpy npy = parseNpy(inputData, inputSize);
    return {
        .dtype = npy.dtype,
        .shape = npy.shape,
        .extension = "npy"
    };
}

DecodingResult NpyDataDecoder::loadFromMemory(
    const uint32_t bufferSize, uint8_t *inputData, const size_t inputSize,
    BumpAllocator<uint8_t *> &output,
    uint8_t *__restrict__, uint8_t *__restrict__
) {
    const ParsedNpy npy = parseNpy(inputData, inputSize);

    if (npy.itemSize != getWidthOfDType(npy.dtype)) {
        throw std::runtime_error("NPY dtype width mismatch");
    }

    const uint64_t numElems = getNumElements(npy.shape);
    const uint64_t nBytes64 = numElems * static_cast<uint64_t>(npy.itemSize);
    if (nBytes64 > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("NPY payload too large");
    }

    const size_t nBytes = static_cast<size_t>(nBytes64);
    if (nBytes > bufferSize) {
        throw std::runtime_error("NPY payload larger than provided output buffer");
    }

    if (npy.dataOffset + nBytes > inputSize) {
        throw std::runtime_error("NPY payload truncated");
    }

    uint8_t *out = output.allocate(bufferSize);
    std::memcpy(out, inputData + npy.dataOffset, nBytes);

    return {
        .data = out,
        .shape = npy.shape
    };
}

std::string NpyDataDecoder::getExtension() {
    return "npy";
}
