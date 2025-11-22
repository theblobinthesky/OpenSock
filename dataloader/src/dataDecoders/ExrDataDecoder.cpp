#include "ExrDataDecoder.h"

#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <Imath/ImathBox.h>

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
        return pos != inputSize - 1;
    }

    uint64_t tellg() override {
        return pos;
    }

    void seekg(const uint64_t _pos) override {
        pos = _pos;
    }

private:
    uint64_t pos = 0;
    const uint8_t *inputData;
    const size_t inputSize;
};

void readExr(const uint8_t *inputData, const size_t inputSize,
             uint32_t &width, uint32_t &height, uint8_t *outputData) {
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

CpuAllocation loadExrFiles(BumpAllocator<uint8_t *> &cpuAllocator,
                           const std::vector<std::vector<std::string> > &batchPaths,
                           const std::vector<Head> &heads, const size_t headIdx) {
    const Head &head = heads[headIdx];
    const auto batchAllocation = getBatchAllocation(head, cpuAllocator, batchPaths.size(), 4);
    const auto [shapeSize, batchBufferSize, batchBuffer] = batchAllocation;

    const int outW = static_cast<int>(IMAGE_WIDTH(head));
    const int outH = static_cast<int>(IMAGE_HEIGHT(head));

    for (size_t j = 0; j < batchPaths.size(); ++j) {
        const std::string &path = batchPaths[j][headIdx];

        int inW;
        int inH;
        std::vector<float> rgb;
        readExrRGBInterleaved(path, rgb, inW, inH);

        float *out = batchBuffer.float32 + j * shapeSize;

        if (inW == outW && inH == outH) {
            // Same size: direct copy
            std::memcpy(out, rgb.data(), static_cast<size_t>(outW) * outH * 3 * sizeof(float));
        } else {
            // Resize in linear space
            resizeImageFloatRGB(rgb.data(), inW, inH, out, outW, outH);
        }
    }

    return batchAllocation;
}

ProbeResult ExrDataDecoder::probeFromMemory(uint8_t *inputData, const size_t inputSize) {
    uint32_t width, height;
    readExr(inputData, inputSize, width, height, nullptr);

    return {
        .format = ItemFormat::FLOAT,
        .bytesPerItem = 4,
        .shape = std::vector<uint32_t>{height, width, 3},
        .extension = "exr"
    };
}

uint8_t *ExrDataDecoder::loadFromMemory(const ProbeResult &settings,
                                        uint8_t *inputData, const size_t inputSize, BumpAllocator<uint8_t *> &output) {
    uint8_t *outputData = output.allocate(settings.getShapeSize());
    uint32_t width, height;
    readExr(inputData, inputSize, width, height, outputData);

    return outputData;
}

std::string ExrDataDecoder::getExtension() {
    return "exr";
}
