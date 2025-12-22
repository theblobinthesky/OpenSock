#include "ResizeAugmentation.h"

#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

ResizeAugmentation::ResizeAugmentation(const uint32_t height, const uint32_t width) : height(height), width(width) {
}

bool ResizeAugmentation::isOutputShapeDetStaticExceptForBatchDim() {
    return true;
}

static bool isInputShapeSupported(const std::vector<uint32_t> &inputShape) {
    return isInputShapeBHWN(inputShape, 3);
}

DataOutputSchema ResizeAugmentation::getDataOutputSchema(const std::vector<uint32_t> &inputShape,
                                                         const uint64_t itemSeed) const {
    std::vector<uint32_t> outputShape;
    if (isInputShapeSupported(inputShape)) {
        outputShape = {
            inputShape[0],
            height,
            width,
            inputShape[3]
        };
    }
    auto *itemSettings = new ResizeSettings{
        .originalHeight = inputShape[1],
        .originalWidth = inputShape[2],
        .skip = inputShape == outputShape
    };

    return {
        .outputShape = std::move(outputShape),
        .itemSettings = itemSettings
    };
}

void ResizeAugmentation::freeItemSettings(void *itemSettings) const {
    delete static_cast<ResizeSettings *>(itemSettings);
}

std::vector<uint32_t> ResizeAugmentation::getMaxOutputShapeAxesIfSupported(
    const std::vector<uint32_t> &inputShape) const {
    if (isInputShapeSupported(inputShape)) {
        return {
            inputShape[0],
            height,
            width,
            inputShape[3]
        };
    }
    return {};
}

template<typename T>
void resizePoints(
    const std::vector<uint32_t> &shape,
    const T *inputData, T *outputData,
    ResizeSettings *itemSettings,
    const uint32_t height, const uint32_t width
) {
    for (size_t b = 0; b < shape[0]; b++) {
        for (size_t i = 0; i < shape[1]; i++) {
            const size_t idx = getIdx(b, i, 0, shape);
            outputData[idx + 0] = static_cast<T>(
                height * static_cast<double>(inputData[idx + 0]) / static_cast<double>(itemSettings->originalHeight));
            outputData[idx + 1] = static_cast<T>(
                width * static_cast<double>(inputData[idx + 1]) / static_cast<double>(itemSettings->originalWidth));
        }
    }
}

static bool shouldBeSkipped(void *itemSettings) {
    const auto settings = static_cast<ResizeSettings *>(itemSettings);
    return settings->skip;
}

bool ResizeAugmentation::isAugmentWithPointsSkipped(const std::vector<uint32_t> &, DType, void *itemSettings) {
   return shouldBeSkipped(itemSettings);
}

void ResizeAugmentation::augmentWithPoints(
    const std::vector<uint32_t> &shape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        resizePoints(shape, input, output, static_cast<ResizeSettings *>(itemSettings), height, width);
    });
}

bool ResizeAugmentation::isAugmentWithRasterSkipped(
    const std::vector<uint32_t> &, const std::vector<uint32_t> &,
    DType, void *itemSettings
) {
    return shouldBeSkipped(itemSettings);
}

void ResizeAugmentation::augmentWithRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *
) {
    const size_t inputSize = getShapeSize(inputShape) / inputShape[0];
    const size_t outputSize = getShapeSize(outputShape) / inputShape[0];

    switch (dtype) {
        case DType::UINT8:
            for (size_t b = 0; b < inputShape[0]; b++) {
                stbir_resize_uint8_srgb(
                    inputData + inputSize * b,
                    static_cast<int>(inputShape[2]),
                    static_cast<int>(inputShape[1]),
                    static_cast<int>(inputShape[2]) * 3,
                    outputData + outputSize * b,
                    static_cast<int>(width),
                    static_cast<int>(height),
                    static_cast<int>(width) * 3, STBIR_RGB
                );
            }
            break;
        case DType::FLOAT32:
            for (size_t b = 0; b < inputShape[0]; b++) {
                stbir_resize_float_linear(
                    reinterpret_cast<const float *>(inputData) + inputSize * b,
                    static_cast<int>(inputShape[2]),
                    static_cast<int>(inputShape[1]),
                    static_cast<int>(inputShape[2]) * 3 * static_cast<int>(sizeof(float)),
                    reinterpret_cast<float *>(outputData) + outputSize * b,
                    static_cast<int>(width),
                    static_cast<int>(height),
                    static_cast<int>(width) * 3 * static_cast<int>(sizeof(float)),
                    STBIR_RGB
                );
            }
            break;
        case DType::INT32:
        default:
            throw std::runtime_error("Dtype unsupported in augmentation.");
    }
}
