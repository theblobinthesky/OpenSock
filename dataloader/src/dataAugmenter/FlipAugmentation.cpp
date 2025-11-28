#include "FlipAugmentation.h"

FlipAugmentation::FlipAugmentation(
    const bool flipHorizontal, const float horizontalFlipProbability,
    const bool flipVertical, const float verticalFlipProbability
) : flipHorizontal(flipHorizontal), horizontalFlipProbability(horizontalFlipProbability),
    flipVertical(flipVertical), verticalFlipProbability(verticalFlipProbability) {
}

bool FlipAugmentation::isOutputShapeStatic() {
    return false;
}

std::vector<size_t> FlipAugmentation::getOutputShapeIfSupported(const std::vector<size_t> &inputShape) {
    if (!isInputShapeBHWN(inputShape, 2)) {
        return std::vector<size_t>{};
    }
    return inputShape;
}

void *FlipAugmentation::getItemSettings(const uint64_t itemSeed) const {
    const bool doesFlipHorizontal = randomUniformDoubleBetween01(itemSeed, 0) <= horizontalFlipProbability;
    const bool doesFlipVertical = randomUniformDoubleBetween01(itemSeed, 1) <= verticalFlipProbability;
    return new FlipItemSettings{.doesHorizontalFlip = doesFlipHorizontal, .doesVerticalFlip = doesFlipVertical};
}

void FlipAugmentation::freeItemSettings(void *itemSettings) const {
    delete static_cast<FlipItemSettings *>(itemSettings);
}

template<typename T>
void augmentWithChannelFirstHelper(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
    const T *inputData, T *outputData,
    FlipItemSettings *itemSettings
) {
    if (itemSettings->doesVerticalFlip) {
        for (size_t i = 0; i < inputShape[0]; i++) {
            for (size_t j = 0; j < inputShape[1]; j++) {
                const size_t inpIdx = i * inputShape[1] * 2 + j * 2 + 0;
                const size_t outIdx = (outputShape[0] - 1 - i) * inputShape[1] * 2 + j * 2 + 0;
                outputData[outIdx] = inputData[inpIdx];
            }
        }
    }

    if (itemSettings->doesHorizontalFlip) {
        for (size_t i = 0; i < inputShape[0]; i++) {
            for (size_t j = 0; j < inputShape[1]; j++) {
                const size_t inpIdx = i * inputShape[1] * 2 + j * 2 + 0;
                const size_t outIdx = i * inputShape[1] * 2 + (outputShape[1] - 1 - j) * 2 + 0;
                outputData[outIdx] = inputData[inpIdx];
            }
        }
    }
}

void FlipAugmentation::augmentWithChannelFirst(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
    const ItemFormat format,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    switch (format) {
        case ItemFormat::UINT:
            augmentWithChannelFirstHelper<uint8_t>(
                inputShape, outputShape,
                inputData, outputData,
                static_cast<FlipItemSettings *>(itemSettings)
            );
            break;
        case ItemFormat::FLOAT:
            augmentWithChannelFirstHelper<float>(
                inputShape, outputShape,
                reinterpret_cast<const float *>(inputData), reinterpret_cast<float *>(outputData),
                static_cast<FlipItemSettings *>(itemSettings)
            );
            break;
        default:
            throw std::runtime_error("Item format unsupported in FlipAugmentation.");
    }
}

template<typename T>
void augmentWithChannelLastHelper(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
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

void FlipAugmentation::augmentWithChannelLast(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
    const ItemFormat format,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    switch (format) {
        case ItemFormat::UINT:
            augmentWithChannelLastHelper<uint8_t>(
                inputShape, outputShape,
                inputData, outputData,
                static_cast<FlipItemSettings *>(itemSettings)
            );
            break;
        case ItemFormat::FLOAT:
            augmentWithChannelLastHelper<float>(
                inputShape, outputShape,
                reinterpret_cast<const float *>(inputData), reinterpret_cast<float *>(outputData),
                static_cast<FlipItemSettings *>(itemSettings)
            );
            break;
        default:
            throw std::runtime_error("Item format unsupported in FlipAugmentation.");
    }
}
