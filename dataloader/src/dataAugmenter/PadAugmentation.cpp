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
        .itemProp = nullptr
    };
}

void PadAugmentation::freeItemProp(ItemProp &) const {
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

bool PadAugmentation::isAugmentWithPointsSkipped(const Shape &, DType, ItemProp &) {
    // Padding only applies to rasters
    return true;
}

void PadAugmentation::augmentWithPoints(
    const Shape &,
    const DType,
    const uint8_t *__restrict__, uint8_t *__restrict__,
    ItemProp &
) {
    // Padding only applies to rasters.
}

template<typename T>
void padRaster(
    const Shape &inputShape,
    const Shape &outputShape,
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

    // Copy raster.
    for (size_t i = 0; i < inputShape[0]; i++) {
        for (size_t j = 0; j < inputShape[1]; j++) {
            for (size_t k = 0; k < inputShape[2]; k++) {
                outputData[getIdx(i + left, j + top, k, outputShape)] = inputData[getIdx(i, j, k, inputShape)];
            }
        }
    }
}

bool PadAugmentation::isAugmentWithRasterSkipped(
    const Shape &inputShape,
    const Shape &outputShape, DType, ItemProp &) {
    return inputShape == outputShape;
}

void PadAugmentation::augmentWithRaster(
    const Shape &inputShape,
    const Shape &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    ItemProp &
) {
    dispatchWithType(dtype, inputData, outputData, [&](auto *input, auto *output) {
        padRaster(
            inputShape, outputShape,
            input, output,
            padSettings
        );
    });
}
