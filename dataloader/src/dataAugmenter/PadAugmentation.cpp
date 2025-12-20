#include "PadAugmentation.h"

PadAugmentation::PadAugmentation(
    const size_t padHeight, const size_t padWidth,
    const PadSettings padSettings
) : padHeight(padHeight), padWidth(padWidth), padSettings(padSettings) {
}

bool PadAugmentation::isOutputShapeStaticExceptForBatch() {
    return true;
}

std::vector<size_t> PadAugmentation::getOutputShapeIfSupported(const std::vector<size_t> &inputShape) {
    if (inputShape.size() != 4 || padHeight > inputShape[1] || padWidth > inputShape[2]) {
        return std::vector<size_t>{};
    }

    return {
        inputShape[0],
        padHeight,
        padWidth,
        inputShape[3]
    };
}

void *PadAugmentation::getItemSettings(const uint64_t) const {
    return nullptr;
}

void PadAugmentation::freeItemSettings(void *itemSettings) const {
    // Nothing.
}

bool PadAugmentation::augmentWithPoints(
    const std::vector<size_t> &,
    const std::vector<size_t> &,
    const DType,
    const uint8_t *__restrict__, uint8_t *__restrict__,
    void *
) {
    // Padding only applies to rasters.
    return false;
}

template<typename T>
void padRaster(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
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
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
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
