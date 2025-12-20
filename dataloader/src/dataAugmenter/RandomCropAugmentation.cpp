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

bool RandomCropAugmentation::isOutputShapeStaticExceptForBatch() {
    return false;
}

DataOutputSchema RandomCropAugmentation::getDataOutputSchema(const std::vector<uint32_t> &inputShape, const uint64_t itemSeed) {
    uint32_t cropHeight = randomUniformBetween(itemSeed, 0, minCropHeight, maxCropHeight);
    uint32_t cropWidth = randomUniformBetween(itemSeed, 0, minCropWidth, maxCropWidth);

    std::vector<uint32_t> outputShape;
    if (inputShape.size() == 4) {
        outputShape = {
            inputShape[0],
            cropHeight,
            cropWidth,
            inputShape[3]
        };
    }

    auto *itemSettings = new RandomCropSettings{
        .left = randomUniformSize(itemSeed, 2, inputShape[1] - cropHeight),
        .top = randomUniformSize(itemSeed, 3, inputShape[2] - cropWidth),
        .height = cropHeight,
        .width = cropWidth
    };

    return {
        .outputShape = std::move(outputShape),
        .itemSettings = itemSettings
    };
}

void RandomCropAugmentation::freeItemSettings(void *itemSettings) const {
    delete static_cast<RandomCropSettings *>(itemSettings);
}

template<typename T>
void randomCropPoints(
    const std::vector<uint32_t> &shape,
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
    const std::vector<uint32_t> &shape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    assert(shape[2] == 2);
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        randomCropPoints(
            shape, input, output,
            static_cast<RandomCropSettings *>(itemSettings)
        );
    });

    return true;
}

template<typename T>
void randomCropRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const T *inputData, T *outputData,
    RandomCropSettings *itemSettings
) {
    const size_t clampedHeight = std::min(inputShape[1], itemSettings->height);
    const size_t clampedWidth = std::min(inputShape[2], itemSettings->width);

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
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        randomCropRaster(
            inputShape, outputShape,
            input, output,
            static_cast<RandomCropSettings *>(itemSettings)
        );
    });

    return true;
}
