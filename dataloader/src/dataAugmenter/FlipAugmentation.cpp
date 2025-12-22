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
    auto *itemSettings = new FlipItemSettings{
        .doesVerticalFlip = doesFlipVertical,
        .doesHorizontalFlip = doesFlipHorizontal,
        .originalHeight = inputShape[1],
        .originalWidth = inputShape[2]
    };

    return {
        .outputShape = std::move(outputShape),
        .itemSettings = itemSettings
    };
}

void FlipAugmentation::freeItemSettings(void *itemSettings) const {
    delete static_cast<FlipItemSettings *>(itemSettings);
}

std::vector<uint32_t>
FlipAugmentation::getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) const {
    if (isInputShapeSupported(inputShape)) {
        return inputShape;
    }
    return {};
}

template<typename Func>
void loopOverPoints(const Shape &shape, Func func) {
    for (size_t b = 0; b < shape[0]; b++) {
        for (size_t i = 0; i < shape[1]; i++) {
            func(b, i);
        }
    }
}

template<typename T>
void flipPoints(
    const std::vector<uint32_t> &shape,
    const T *inputData, T *outputData,
    FlipItemSettings *itemSettings
) {
    if (itemSettings->doesVerticalFlip && itemSettings->doesHorizontalFlip) {
        loopOverPoints(shape, [&](const size_t b, const size_t i) {
            const size_t idx = getIdx(b, i, 0, shape);
            outputData[idx + 0] = itemSettings->originalHeight - 1 - inputData[idx + 0];
            outputData[idx + 1] = itemSettings->originalWidth - 1 - inputData[idx + 1];
        });
    } else if (itemSettings->doesVerticalFlip) {
        loopOverPoints(shape, [&](const size_t b, const size_t i) {
            const size_t idx = getIdx(b, i, 0, shape);
            outputData[idx + 0] = itemSettings->originalHeight - 1 - inputData[idx + 0];
            outputData[idx + 1] = inputData[idx + 1];
        });
    } else if (itemSettings->doesHorizontalFlip) {
        loopOverPoints(shape, [&](const size_t b, const size_t i) {
            const size_t idx = getIdx(b, i, 0, shape);
            outputData[idx + 0] = inputData[idx + 0];
            outputData[idx + 1] = itemSettings->originalWidth - 1 - inputData[idx + 1];
        });
    }
}

static bool shouldBeSkipped(void *itemSettings) {
    const auto settings = static_cast<FlipItemSettings *>(itemSettings);
    return !settings->doesHorizontalFlip && !settings->doesVerticalFlip;
}

bool FlipAugmentation::isAugmentWithPointsSkipped(const std::vector<uint32_t> &, DType, void *itemSettings) {
    return shouldBeSkipped(itemSettings);
}

void FlipAugmentation::augmentWithPoints(
    const std::vector<uint32_t> &shape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        flipPoints(
            shape,
            input, output,
            static_cast<FlipItemSettings *>(itemSettings)
        );
    });
}

template<typename Func>
void loopOverRaster(const Shape &shape, Func func) {
    for (size_t b = 0; b < shape[0]; b++) {
        for (size_t i = 0; i < shape[1]; i++) {
            for (size_t j = 0; j < shape[2]; j++) {
                for (size_t k = 0; k < shape[3]; k++) {
                    func(b, i, j, k);
                }
            }
        }
    }
}

template<typename T>
void flipRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const T *inputData, T *outputData,
    FlipItemSettings *itemSettings
) {
    assert(inputShape == outputShape);
    if (itemSettings->doesVerticalFlip && itemSettings->doesHorizontalFlip) {
        loopOverRaster(inputShape, [&](const size_t b, const size_t i, const size_t j, const size_t k) {
            size_t outIdx = getIdx(b, inputShape[1] - 1 - i, inputShape[2] - 1 - j, k, inputShape);
            outputData[outIdx] = inputData[getIdx(b, i, j, k, inputShape)];
        });
    } else if (itemSettings->doesVerticalFlip) {
        loopOverRaster(inputShape, [&](const size_t b, const size_t i, const size_t j, const size_t k) {
            size_t outIdx = getIdx(b, inputShape[1] - 1 - i, j, k, inputShape);
            outputData[outIdx] = inputData[getIdx(b, i, j, k, inputShape)];
        });
    } else if (itemSettings->doesHorizontalFlip) {
        loopOverRaster(inputShape, [&](const size_t b, const size_t i, const size_t j, const size_t k) {
            size_t outIdx = getIdx(b, i, inputShape[2] - 1 - j, k, inputShape);
            outputData[outIdx] = inputData[getIdx(b, i, j, k, inputShape)];
        });
    }
}

bool FlipAugmentation::isAugmentWithRasterSkipped(
    const std::vector<uint32_t> &,
    const std::vector<uint32_t> &,
    DType, void *itemSettings
) {
    return shouldBeSkipped(itemSettings);
}

void FlipAugmentation::augmentWithRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        flipRaster(
            inputShape, outputShape,
            input, output, static_cast<FlipItemSettings *>(itemSettings)
        );
    });
}
