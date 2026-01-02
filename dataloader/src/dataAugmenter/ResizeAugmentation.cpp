#include "ResizeAugmentation.h"

#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

ResizeAugmentation::ResizeAugmentation(const uint32_t height, const uint32_t width) : height(height), width(width) {
}

bool ResizeAugmentation::isOutputShapeDetStaticExceptForBatchDim() {
    return true;
}

bool ResizeAugmentation::isOutputShapeDetStaticGivenStaticInputShape() {
    return true;
}

static bool isInputShapeSupported(const std::vector<uint32_t> &inputShape) {
    return isInputShapeHWN(inputShape, 3);
}

DataOutputSchema
ResizeAugmentation::getDataOutputSchema(const std::vector<uint32_t> &inputShape, const uint64_t) const {
    std::vector<uint32_t> outputShape;
    if (isInputShapeSupported(inputShape)) {
        outputShape = {
            height,
            width,
            inputShape[2]
        };
    }
    auto *itemProp = new ResizeProp{
        .originalHeight = inputShape[0],
        .originalWidth = inputShape[1],
        .skip = inputShape == outputShape
    };

    return {
        .outputShape = std::move(outputShape),
        .itemProp = itemProp
    };
}

void ResizeAugmentation::freeItemProp(ItemProp &itemProp) const {
    delete static_cast<ResizeProp *>(itemProp);
}

std::vector<uint32_t> ResizeAugmentation::getMaxOutputShapeAxesIfSupported(
    const std::vector<uint32_t> &inputShape) const {
    if (isInputShapeSupported(inputShape)) {
        return {
            height,
            width,
            inputShape[2]
        };
    }
    return {};
}

template<typename T>
void resizePoints(
    const std::vector<uint32_t> &shape,
    const T *inputData, T *outputData,
    ResizeProp *itemProp,
    const uint32_t height, const uint32_t width
) {
    for (size_t i = 0; i < shape[0]; i++) {
        const size_t idx = 2 * i; // getIdx(i, 0, shape);
        outputData[idx + 0] = static_cast<T>(
            height * static_cast<double>(inputData[idx + 0]) / static_cast<double>(itemProp->originalHeight));
        outputData[idx + 1] = static_cast<T>(
            width * static_cast<double>(inputData[idx + 1]) / static_cast<double>(itemProp->originalWidth));
    }
}

static bool shouldBeSkipped(void *itemProp) {
    const ResizeProp *settings = static_cast<ResizeProp *>(itemProp);
    return settings->skip;
}

bool ResizeAugmentation::isAugmentWithPointsSkipped(const Shape &, DType, ItemProp &itemProp) {
    return shouldBeSkipped(itemProp);
}

void ResizeAugmentation::augmentWithPoints(
    const Shape &shape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    ItemProp &itemProp
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        resizePoints(shape, input, output, static_cast<ResizeProp *>(itemProp), height, width);
    });
}

bool ResizeAugmentation::isAugmentWithRasterSkipped(
    const Shape &, const Shape &,
    DType, ItemProp &itemProp
) {
    return shouldBeSkipped(itemProp);
}

void ResizeAugmentation::augmentWithRaster(
    const Shape &inputShape,
    const Shape &,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    ItemProp &
) {
    switch (dtype) {
        case DType::UINT8:
            stbir_resize_uint8_srgb(
                inputData,
                static_cast<int>(inputShape[1]),
                static_cast<int>(inputShape[0]),
                static_cast<int>(inputShape[1]) * 3,
                outputData,
                static_cast<int>(width),
                static_cast<int>(height),
                static_cast<int>(width) * 3, STBIR_RGB
            );
            break;
        case DType::FLOAT32:
            stbir_resize_float_linear(
                reinterpret_cast<const float *>(inputData),
                static_cast<int>(inputShape[1]),
                static_cast<int>(inputShape[0]),
                static_cast<int>(inputShape[1]) * 3 * static_cast<int>(sizeof(float)),
                reinterpret_cast<float *>(outputData),
                static_cast<int>(width),
                static_cast<int>(height),
                static_cast<int>(width) * 3 * static_cast<int>(sizeof(float)),
                STBIR_RGB
            );
            break;
        case DType::INT32:
        default:
            throw std::runtime_error("Dtype unsupported in augmentation.");
    }
}
