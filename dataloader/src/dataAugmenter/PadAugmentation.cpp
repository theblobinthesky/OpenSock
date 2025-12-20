#include "PadAugmentation.h"

PadAugmentation::PadAugmentation(
    const uint32_t padHeight, const uint32_t padWidth,
    const PadSettings padSettings
) : padHeight(padHeight), padWidth(padWidth), padSettings(padSettings) {
}

bool PadAugmentation::isOutputShapeStaticExceptForBatch() {
    return true;
}

std::vector<uint32_t> getOutputShape(const std::vector<uint32_t> &inputShape, uint32_t padHeight, uint32_t padWidth) {
    std::vector<uint32_t> outputShape;
    if (inputShape.size() == 4 && padHeight <= inputShape[1] && padWidth <= inputShape[2]) {
        outputShape = {
            inputShape[0],
            padHeight,
            padWidth,
            inputShape[3]
        };
    }
    return outputShape;
}

DataOutputSchema PadAugmentation::getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t) {
    return {
        .outputShape = getOutputShape(inputShape, padHeight, padWidth),
        .itemSettings = nullptr
    };
}

void PadAugmentation::freeItemSettings(void *itemSettings) const {
    // Nothing.
}

std::vector<uint32_t> PadAugmentation::getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) {
    return {
        inputShape[0],
        padHeight,
        padWidth,
        inputShape[3]
    };
}

bool PadAugmentation::augmentWithPoints(
    const std::vector<uint32_t> &,
    const DType,
    const uint8_t *__restrict__, uint8_t *__restrict__,
    void *
) {
    // Padding only applies to rasters.
    return false;
}

template<typename T>
void padRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const T *inputData, T *outputData,
    const PadSettings padSettings
) {
    size_t left = 0, top = 0;
    switch (padSettings) {
        case PadSettings::PAD_TOP_LEFT: {
            left = outputShape[1] - inputShape[1];
            top = outputShape[0] - inputShape[0];
        } break;
        case PadSettings::PAD_TOP_RIGHT: {
            left = 0;
            top = outputShape[0] - inputShape[0];
        } break;
        case PadSettings::PAD_BOTTOM_LEFT: {
            left = outputShape[1] - inputShape[1];
            top = 0;
        } break;
        case PadSettings::PAD_BOTTOM_RIGHT: {
            left = 0;
            top = 0;
        } break;
    }

    // Copy raster. TODO: Vectorise.
    for (size_t b = 0; b < inputShape[0]; b++) {
        for (size_t i = 0; i < inputShape[1]; i++) {
            for (size_t j = 0; j < inputShape[2]; j++) {
                for (size_t k = 0; k < inputShape[3]; k++) {
                    outputData[getIdx(b, i + left, j + top, k, outputShape)] = inputData[getIdx(b, i, j, k, inputShape)];
                }
            }
        }
    }
}

bool PadAugmentation::augmentWithRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        padRaster(
            inputShape, outputShape,
            input, output,
            padSettings
        );
    });

    return true;
}
