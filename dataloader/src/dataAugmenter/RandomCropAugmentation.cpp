#include "RandomCropAugmentation.h"

RandomCropAugmentation::RandomCropAugmentation(
    const uint32_t minCropHeight, const uint32_t minCropWidth,
    const uint32_t maxCropHeight, const uint32_t maxCropWidth
) : minCropHeight(minCropHeight), minCropWidth(minCropWidth),
    maxCropHeight(maxCropHeight), maxCropWidth(maxCropHeight) {

    if (minCropHeight > maxCropHeight || minCropWidth > maxCropWidth) {
        throw std::runtime_error("Min crop dimensions must be smaller than max crop dimensions.");
    }

    if (maxCropHeight == 0 || maxCropWidth == 0) {
        throw std::runtime_error("Random crop augmentations needs nonzero crop-height and crop-width.");
    }
}

bool RandomCropAugmentation::isOutputShapeDetStaticExceptForBatchDim() {
    return minCropHeight == maxCropHeight && minCropWidth == maxCropWidth;
}

static bool isInputShapeSupported(const std::vector<uint32_t> &inputShape) {
    return inputShape.size() == 3;
}

DataOutputSchema RandomCropAugmentation::getDataOutputSchema(const std::vector<uint32_t> &inputShape, const uint64_t itemSeed) const {
    uint32_t cropHeight = randomUniformBetween(itemSeed, 0, minCropHeight, maxCropHeight);
    uint32_t cropWidth = randomUniformBetween(itemSeed, 0, minCropWidth, maxCropWidth);

    std::vector<uint32_t> outputShape;
    if (isInputShapeSupported(inputShape)) {
        outputShape = {
            cropHeight,
            cropWidth,
            inputShape[2]
        };
    }

    auto *itemProp = new RandomCropProp{
        .top = randomUniformSize(itemSeed, 2, inputShape[1] - cropHeight),
        .left = randomUniformSize(itemSeed, 3, inputShape[2] - cropWidth),
        .height = cropHeight,
        .width = cropWidth,
        .skip = inputShape == outputShape
    };

    return {
        .outputShape = std::move(outputShape),
        .itemProp = itemProp
    };
}

void RandomCropAugmentation::freeItemProp(ItemProp &itemProp) const {
    delete static_cast<RandomCropProp *>(itemProp);
}

std::vector<uint32_t> RandomCropAugmentation::getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) const {
    if (isInputShapeSupported(inputShape)) {
        return {
            maxCropHeight,
            maxCropWidth,
            inputShape[2]
        };
    }
    return {};
}

template<typename T>
void randomCropPoints(
    const std::vector<uint32_t> &shape,
    const T *inputData, T *outputData,
    const RandomCropProp *itemProp
) {
    for (size_t b = 0; b < shape[0]; b++) {
        for (size_t i = 0; i < shape[1]; i++) {
            const size_t idx = getIdx(b, i, 0, shape);
            outputData[idx + 0] = inputData[idx + 0] - itemProp->top;
            outputData[idx + 1] = inputData[idx + 1] - itemProp->left;
        }
    }
}

static bool shouldBeSkipped(void *itemProp) {
    const RandomCropProp *settings = static_cast<RandomCropProp *>(itemProp);
    return settings->skip;
}

bool RandomCropAugmentation::isAugmentWithPointsSkipped(const Shape &, DType, ItemProp &itemProp) {
    return shouldBeSkipped(itemProp);
}

void RandomCropAugmentation::augmentWithPoints(
    const Shape &shape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    ItemProp &itemProp
) {
    ASSERT(shape[2] == 2);
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        randomCropPoints(
            shape, input, output,
            static_cast<RandomCropProp *>(itemProp)
        );
    });
}

template<typename T>
void randomCropRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const T *inputData, T *outputData,
    const RandomCropProp *itemProp
) {
    const size_t clampedHeight = std::min(inputShape[0], itemProp->height);
    const size_t clampedWidth = std::min(inputShape[1], itemProp->width);

    std::printf("inputShape==%s\n", formatVector(inputShape).c_str());

    for (size_t i = 0; i < clampedHeight; i++) {
        for (size_t j = 0; j < clampedWidth; j++) {
            for (size_t k = 0; k < inputShape[2]; k++) {
                size_t inpIdx = getIdx(itemProp->top + i, itemProp->left + j, k, inputShape);
                outputData[getIdx(i, j, k, outputShape)] = inputData[inpIdx];
            }
        }
    }
}

bool RandomCropAugmentation::isAugmentWithRasterSkipped(
    const Shape &, const Shape &,
    DType, ItemProp &itemProp
) {
    return shouldBeSkipped(itemProp);
}

void RandomCropAugmentation::augmentWithRaster(
    const Shape &inputShape,
    const Shape &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    ItemProp &itemProp
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        randomCropRaster(
            inputShape, outputShape,
            input, output,
            static_cast<RandomCropProp *>(itemProp)
        );
    });
}
