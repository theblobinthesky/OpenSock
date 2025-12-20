#include "FlipAugmentation.h"

FlipAugmentation::FlipAugmentation(
    const bool flipHorizontal, const float horizontalFlipProbability,
    const bool flipVertical, const float verticalFlipProbability
) : flipHorizontal(flipHorizontal), horizontalFlipProbability(horizontalFlipProbability),
    flipVertical(flipVertical), verticalFlipProbability(verticalFlipProbability) {
}

bool FlipAugmentation::isOutputShapeStaticExceptForBatch() {
    return false;
}

DataOutputSchema FlipAugmentation::getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) {
    std::vector<uint32_t> outputShape;
    if (isInputShapeBHWN(inputShape, 2)) {
        outputShape = inputShape;
    }

    const bool doesFlipHorizontal = randomUniformDoubleBetween01(itemSeed, 0) <= horizontalFlipProbability;
    const bool doesFlipVertical = randomUniformDoubleBetween01(itemSeed, 1) <= verticalFlipProbability;
    auto *itemSettings = new FlipItemSettings{
        .doesHorizontalFlip = doesFlipHorizontal,
        .doesVerticalFlip = doesFlipVertical
    };

    return {
        .outputShape = std::move(outputShape),
        .itemSettings = itemSettings
    };
}

void FlipAugmentation::freeItemSettings(void *itemSettings) const {
    delete static_cast<FlipItemSettings *>(itemSettings);
}

std::vector<uint32_t> FlipAugmentation::getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) {
    return inputShape;
}

template<typename T>
void flipPoints(
    const std::vector<uint32_t> &shape,
    const T *inputData, T *outputData,
    FlipItemSettings *itemSettings
) {
    if (itemSettings->doesVerticalFlip) {
        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                const size_t inpIdx = i * shape[1] * 2 + j * 2 + 0;
                const size_t outIdx = (shape[0] - 1 - i) * shape[1] * 2 + j * 2 + 0;
                outputData[outIdx] = inputData[inpIdx];
            }
        }
    }

    if (itemSettings->doesHorizontalFlip) {
        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                const size_t inpIdx = i * shape[1] * 2 + j * 2 + 0;
                const size_t outIdx = i * shape[1] * 2 + (shape[1] - 1 - j) * 2 + 0;
                outputData[outIdx] = inputData[inpIdx];
            }
        }
    }
}

bool FlipAugmentation::augmentWithPoints(
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

    return true;
}

template<typename T>
void flipRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const T *inputData, T *outputData,
    FlipItemSettings *itemSettings
) {
    if (itemSettings->doesVerticalFlip) {
        for (size_t i = 0; i < inputShape[0]; i++) {
            for (size_t j = 0; j < inputShape[1]; j++) {
                const size_t idx = i * inputShape[1] * 2 + j * 2 + 0;
                outputData[idx] = outputShape[0] - 1 - inputData[idx];
            }
        }
    }

    if (itemSettings->doesHorizontalFlip) {
        for (size_t i = 0; i < inputShape[0]; i++) {
            for (size_t j = 0; j < inputShape[1]; j++) {
                const size_t idx = i * inputShape[1] * 2 + j * 2 + 1;
                outputData[idx] = outputShape[1] - 1 - inputData[idx];
            }
        }
    }
}

bool FlipAugmentation::augmentWithRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        flipRaster(
            inputShape, outputShape,
            input, output,
            static_cast<FlipItemSettings *>(itemSettings)
        );
    });

    return true;
}
