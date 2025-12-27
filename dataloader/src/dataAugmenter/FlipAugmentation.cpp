#include "FlipAugmentation.h"

FlipAugmentation::FlipAugmentation(
    const float verticalFlipProbability,
    const float horizontalFlipProbability
) : verticalFlipProbability(verticalFlipProbability),
    horizontalFlipProbability(horizontalFlipProbability) {
    if (horizontalFlipProbability == 0 && verticalFlipProbability == 0) {
        throw std::runtime_error("Flip augmentation is cannot be disabled");
    }
}

bool FlipAugmentation::isOutputShapeDetStaticExceptForBatchDim() {
    return false;
}

static bool isInputShapeSupported(const std::vector<uint32_t> &inputShape) {
    return inputShape.size() == 4;
}

DataOutputSchema FlipAugmentation::getDataOutputSchema(const std::vector<uint32_t> &inputShape,
                                                       const uint64_t itemSeed) const {
    std::vector<uint32_t> outputShape;
    if (isInputShapeSupported(inputShape)) {
        outputShape = inputShape;
    }

    const bool doesFlipVertical = randomUniformDoubleBetween01(itemSeed, 0) <= verticalFlipProbability;
    const bool doesFlipHorizontal = randomUniformDoubleBetween01(itemSeed, 1) <= horizontalFlipProbability;
    auto *itemProp = new FlipProp{
        .doesVerticalFlip = doesFlipVertical,
        .doesHorizontalFlip = doesFlipHorizontal,
        .originalHeight = inputShape[1],
        .originalWidth = inputShape[2]
    };

    return {
        .outputShape = std::move(outputShape),
        .itemProp = itemProp
    };
}

void FlipAugmentation::freeItemProp(ItemProp &itemProp) const {
    delete static_cast<FlipProp *>(itemProp);
}

std::vector<uint32_t>
FlipAugmentation::getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) const {
    if (isInputShapeSupported(inputShape)) {
        return inputShape;
    }
    return {};
}

template<typename T>
void flipPoints(
    const Shape &shape,
    const T *inputData, T *outputData,
    FlipProp *itemProp
) {
    if (itemProp->doesVerticalFlip && itemProp->doesHorizontalFlip) {
        for (size_t i = 0; i < shape[0]; i++) {
            const size_t idx = getIdx(i, 0, shape);
            outputData[idx + 0] = itemProp->originalHeight - 1 - inputData[idx + 0];
            outputData[idx + 1] = itemProp->originalWidth - 1 - inputData[idx + 1];
        }
    } else if (itemProp->doesVerticalFlip) {
        for (size_t i = 0; i < shape[0]; i++) {
            const size_t idx = getIdx(i, 0, shape);
            outputData[idx + 0] = itemProp->originalHeight - 1 - inputData[idx + 0];
            outputData[idx + 1] = inputData[idx + 1];
        }
    } else if (itemProp->doesHorizontalFlip) {
        for (size_t i = 0; i < shape[0]; i++) {
            const size_t idx = getIdx(i, 0, shape);
            outputData[idx + 0] = inputData[idx + 0];
            outputData[idx + 1] = itemProp->originalWidth - 1 - inputData[idx + 1];
        }
    }
}

static bool shouldBeSkipped(const ItemProp itemProp) {
    const auto settings = static_cast<FlipProp *>(itemProp);
    return !settings->doesHorizontalFlip && !settings->doesVerticalFlip;
}

bool FlipAugmentation::isAugmentWithPointsSkipped(const Shape &, DType, ItemProp &itemProp) {
    return shouldBeSkipped(itemProp);
}

void FlipAugmentation::augmentWithPoints(
    const Shape &shape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    ItemProp &itemProp
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        flipPoints(
            shape,
            input, output,
            static_cast<FlipProp *>(itemProp)
        );
    });
}

template<typename Func>
void loopOverRaster(const Shape &shape, Func func) {
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2]; k++) {
                func(i, j, k);
            }
        }
    }
}

template<typename T>
void flipRaster(
    const Shape &inputShape,
    const Shape &outputShape,
    const T *inputData, T *outputData,
    const FlipProp *itemProp
) {
    assert(inputShape == outputShape);
    if (itemProp->doesVerticalFlip && itemProp->doesHorizontalFlip) {
        loopOverRaster(inputShape, [&](const size_t i, const size_t j, const size_t k) {
            size_t outIdx = getIdx(inputShape[1] - 1 - i, inputShape[2] - 1 - j, k, inputShape);
            outputData[outIdx] = inputData[getIdx(i, j, k, inputShape)];
        });
    } else if (itemProp->doesVerticalFlip) {
        loopOverRaster(inputShape, [&](const size_t i, const size_t j, const size_t k) {
            size_t outIdx = getIdx(inputShape[1] - 1 - i, j, k, inputShape);
            outputData[outIdx] = inputData[getIdx(i, j, k, inputShape)];
        });
    } else if (itemProp->doesHorizontalFlip) {
        loopOverRaster(inputShape, [&](const size_t i, const size_t j, const size_t k) {
            size_t outIdx = getIdx(i, inputShape[2] - 1 - j, k, inputShape);
            outputData[outIdx] = inputData[getIdx(i, j, k, inputShape)];
        });
    }
}

bool FlipAugmentation::isAugmentWithRasterSkipped(
    const Shape &,
    const Shape &,
    DType, ItemProp &itemProp
) {
    return shouldBeSkipped(itemProp);
}

void FlipAugmentation::augmentWithRaster(
    const Shape &inputShape,
    const Shape &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    ItemProp &itemProp
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        flipRaster(
            inputShape, outputShape,
            input, output, static_cast<FlipProp *>(itemProp)
        );
    });
}
