#include "RandomCropAugmentation.h"

RandomCropAugmentation::RandomCropAugmentation(
    const size_t cropHeight, const size_t cropWidth
) : cropHeight(cropHeight), cropWidth(cropWidth) {
    if (cropHeight == 0 || cropWidth == 0) {
        throw std::runtime_error("Random crop augmentations needs nonzero crop-height and crop-width.");
    }
}

bool RandomCropAugmentation::isOutputShapeStaticExceptForBatch() {
    return true;
}

std::vector<size_t> RandomCropAugmentation::getOutputShapeIfSupported(const std::vector<size_t> &inputShape) {
    if (inputShape.size() != 4) {
        return std::vector<size_t>{};
    }

    return {
        inputShape[0],
        cropHeight,
        cropWidth,
        inputShape[3]
    };
}

void *RandomCropAugmentation::getItemSettings(const std::vector<size_t> &inputShape, const uint64_t itemSeed) const {
    return new RandomCropSettings{
        .left = randomUniformSize(itemSeed, 0, inputShape[1] - cropHeight),
        .top = randomUniformSize(itemSeed, 1, inputShape[2] - cropWidth)
    };
}

void RandomCropAugmentation::freeItemSettings(void *itemSettings) const {
    delete static_cast<RandomCropSettings *>(itemSettings);
}

template<typename T>
void randomCropPoints(
    const std::vector<size_t> &shape,
    const T *inputData, T *outputData,
    RandomCropSettings *itemSettings
) {
    for (size_t b = 0; b < shape[0]; b++) {
        for (size_t i = 0; i < shape[1]; i++) {
            for (size_t k = 0; k < shape[2]; k++) {
                const size_t idx = getIdx(b, i, 0, shape);
                outputData[idx + 0] = inputData[idx + 0] - itemSettings->left;
                outputData[idx + 1] = inputData[idx + 1] - itemSettings->top;
            }
        }
    }
}

bool RandomCropAugmentation::augmentWithPoints(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    assert(inputShape[2] == 2);
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        randomCropPoints(
            inputShape,
            input, output,
            static_cast<RandomCropSettings *>(itemSettings)
        );
    });

    return true;
}

template<typename T>
void randomCropRaster(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
    const T *inputData, T *outputData,
    RandomCropSettings *itemSettings,
    const size_t cropHeight, const size_t cropWidth
) {
    const size_t clampedHeight = std::min(inputShape[1], cropHeight);
    const size_t clampedWidth = std::min(inputShape[2], cropWidth);

    for (size_t b = 0; b < outputShape[0]; b++) {
        for (size_t i = 0; i < clampedHeight; i++) {
            for (size_t j = 0; j < clampedWidth; j++) {
                for (size_t k = 0; k < outputShape[3]; k++) {
                    size_t inpIdx = getIdx(b, itemSettings->left + i, itemSettings->top + j, k, inputShape);
                    outputData[getIdx(b, i, j, k, outputShape)] = inputData[inpIdx];
                }
            }
        }
    }
}

bool RandomCropAugmentation::augmentWithRaster(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        randomCropRaster(
            inputShape, outputShape,
            input, output,
            static_cast<RandomCropSettings *>(itemSettings),
            cropHeight, cropWidth
        );
    });

    return true;
}
