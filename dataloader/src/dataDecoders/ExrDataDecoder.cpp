#include "ExrDataDecoder.h"

#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <Imath/ImathBox.h>
#include <cstring>
#include <stdexcept>

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;

class MemoryIStream final : public IStream {
public:
    MemoryIStream(const uint8_t *inputData, const size_t inputSize)
        : IStream("file_name"), inputData(inputData), inputSize(inputSize) {
    }

    bool read(char *c, const int n) override {
        if (pos + n > inputSize) {
            throw std::runtime_error("EXR InputStream Out of bounds.");
        }

        std::memcpy(c, inputData + pos, n);
        pos += n;
        return pos < inputSize;
    }

    uint64_t tellg() override {
        return pos;
    }

    void seekg(const uint64_t _pos) override {
        if (_pos > inputSize) {
            throw std::runtime_error("EXR InputStream seek out of bounds.");
        }
        pos = _pos;
    }

private:
    uint64_t pos = 0;
    const uint8_t *inputData;
    const size_t inputSize;
};

void readExr(const uint8_t *inputData, const size_t inputSize,
             uint32_t &width, uint32_t &height, float *outputData) {
    MemoryIStream stream(inputData, inputSize);
    InputFile file(stream);
    const Header &hdr = file.header();
    const Box2i dw = hdr.dataWindow();

    const int w = dw.max.x - dw.min.x + 1;
    const int h = dw.max.y - dw.min.y + 1;
    width = w;
    height = h;

    if (outputData == nullptr) {
        return;
    }

    const ChannelList &channels = hdr.channels();
    if (!channels.findChannel("R") || !channels.findChannel("G") || !channels.findChannel("B")) {
        throw std::runtime_error("EXR file missing required RGB channels.");
    }

    FrameBuffer fb;
    constexpr size_t xStride = sizeof(float) * 3;
    const size_t yStride = xStride * static_cast<size_t>(w);

    const auto base = reinterpret_cast<char *>(outputData);
    const ptrdiff_t xOffset = static_cast<ptrdiff_t>(dw.min.x) * static_cast<ptrdiff_t>(xStride);
    const ptrdiff_t yOffset = static_cast<ptrdiff_t>(dw.min.y) * static_cast<ptrdiff_t>(yStride);

    fb.insert("R", Slice(FLOAT,
                         base - xOffset - yOffset + 0 * sizeof(float),
                         xStride, yStride, 1, 1, 0.0F));
    fb.insert("G", Slice(FLOAT,
                         base - xOffset - yOffset + 1 * sizeof(float),
                         xStride, yStride, 1, 1, 0.0F));
    fb.insert("B", Slice(FLOAT,
                         base - xOffset - yOffset + 2 * sizeof(float),
                         xStride, yStride, 1, 1, 0.0F));

    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);
}

ProbeResult ExrDataDecoder::probeFromMemory(uint8_t *inputData, const size_t inputSize) {
    uint32_t width, height;
    readExr(inputData, inputSize, width, height, nullptr);

    return {
        .dtype = DType::FLOAT32,
        .shape = std::vector<uint32_t>{height, width, 3},
        .extension = "exr"
    };
}

DecodingResult ExrDataDecoder::loadFromMemory(
    const uint32_t bufferSize, uint8_t *inputData, const size_t inputSize,
    BumpAllocator<uint8_t *> &output,
    uint8_t *__restrict__, uint8_t *__restrict__
) {
    uint8_t *outputData = output.allocate(bufferSize);
    uint32_t width, height;
    readExr(inputData, inputSize, width, height, reinterpret_cast<float *>(outputData));

    return {
        .data = outputData,
        .shape = {height, width, 3}
    };
}

std::string ExrDataDecoder::getExtension() {
    return "exr";
}
