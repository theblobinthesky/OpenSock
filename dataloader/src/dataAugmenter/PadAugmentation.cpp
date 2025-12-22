#include "PadAugmentation.h"

PadAugmentation::PadAugmentation(
    const uint32_t padHeight, const uint32_t padWidth,
    const PadSettings padSettings
) : padHeight(padHeight), padWidth(padWidth), padSettings(padSettings) {
}

bool PadAugmentation::isOutputShapeDetStaticExceptForBatchDim() {
    return true;
}

static bool isInputShapeSupported(const std::vector<uint32_t> &inputShape, const uint32_t padHeight,
                                  const uint32_t padWidth) {
    return inputShape.size() == 4 && inputShape[1] <= padHeight && inputShape[2] <= padWidth;
}

DataOutputSchema PadAugmentation::getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t) const {
    std::vector<uint32_t> outputShape;
    if (isInputShapeSupported(inputShape, padHeight, padWidth)) {
        outputShape = {
            inputShape[0],
            padHeight,
            padWidth,
            inputShape[3]
        };
    }

    return {
        .outputShape = outputShape,
        .itemSettings = nullptr
    };
}

void PadAugmentation::freeItemSettings(void *itemSettings) const {
    // Nothing.
}

std::vector<uint32_t> PadAugmentation::getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) const {
    if (isInputShapeSupported(inputShape, padHeight, padWidth)) {
        return {
            inputShape[0],
            padHeight,
            padWidth,
            inputShape[3]
        };
    }
    return {};
}

bool PadAugmentation::isAugmentWithPointsSkipped(const std::vector<uint32_t> &, DType, void *) {
    // Padding only applies to rasters.
    return true;
}

void PadAugmentation::augmentWithPoints(
    const std::vector<uint32_t> &,
    const DType,
    const uint8_t *__restrict__, uint8_t *__restrict__,
    void *
) {
    // Padding only applies to rasters.
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
        }
        break;
        case PadSettings::PAD_TOP_RIGHT: {
            left = 0;
            top = outputShape[0] - inputShape[0];
        }
        break;
        case PadSettings::PAD_BOTTOM_LEFT: {
            left = outputShape[1] - inputShape[1];
            top = 0;
        }
        break;
        case PadSettings::PAD_BOTTOM_RIGHT: {
            left = 0;
            top = 0;
        }
        break;
    }

    // Copy raster. TODO: Vectorise.
    for (size_t b = 0; b < inputShape[0]; b++) {
        for (size_t i = 0; i < inputShape[1]; i++) {
            for (size_t j = 0; j < inputShape[2]; j++) {
                for (size_t k = 0; k < inputShape[3]; k++) {
                    outputData[getIdx(b, i + left, j + top, k, outputShape)] = inputData[
                        getIdx(b, i, j, k, inputShape)];
                }
            }
        }
    }
}

bool PadAugmentation::isAugmentWithRasterSkipped(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape, DType, void *
) {
    return inputShape == outputShape;
}

void PadAugmentation::augmentWithRaster(
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
}
