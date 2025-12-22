#include "augmentation.h"

#include <utility>
#include <ranges>

bool isInputShapeBHWN(const std::vector<uint32_t> &inputShape, const size_t N) {
    if (inputShape.size() != 4) {
        return false;
    }

    return inputShape[3] == N;
}

uint32_t calcMaxRequiredBufferSize(
    const std::vector<IDataAugmentation *> &dataAugmentations,
    const std::vector<uint32_t> &maxAlongAllAxesInputShape,
    const uint32_t maxBytesPerElement,
    const uint32_t maxNumPoints
) {
    uint32_t maxShapeSize = getShapeSize(maxAlongAllAxesInputShape);
    std::vector<uint32_t> lastMaxShape = maxAlongAllAxesInputShape;
    for (size_t i = 0; i < dataAugmentations.size(); i++) {
        const auto dataAugmentation = dataAugmentations[i];
        const auto outputMaxShape = dataAugmentation->getMaxOutputShapeAxesIfSupported(lastMaxShape);
        if (outputMaxShape.empty()) {
            throw std::runtime_error(std::format("Augmentation {} pipline does not support maximum input shape {}",
                                                 i, formatVector(lastMaxShape)));
        }

        lastMaxShape = outputMaxShape;
        maxShapeSize = std::max(maxShapeSize, getShapeSize(lastMaxShape));
    }

    return std::max(
        maxShapeSize, maxAlongAllAxesInputShape[0] * maxNumPoints
        * static_cast<uint32_t>(maxAlongAllAxesInputShape.size() - 2)
    ) * maxBytesPerElement;
}

DataAugmentationPipe::DataAugmentationPipe(
    std::vector<IDataAugmentation *> dataAugmentations,
    const std::vector<uint32_t> &maxInputShape, const uint32_t maxNumPoints,
    const uint32_t maxBytesPerElement
) : dataAugs(std::move(dataAugmentations)), buffer1(nullptr), buffer2(nullptr),
    maximumRequiredBufferSize(calcMaxRequiredBufferSize(dataAugs, maxInputShape, maxBytesPerElement, maxNumPoints)) {
    if (!isOutputShapeStatic()) {
        throw std::runtime_error("Augmentation pipe needs to output a static shape.");
    }
}

size_t DataAugmentationPipe::getMaximumRequiredBufferSize() const {
    return maximumRequiredBufferSize;
}

void DataAugmentationPipe::setBuffer(uint8_t *_buffer1, uint8_t *_buffer2) {
    buffer1 = _buffer1;
    buffer2 = _buffer2;
}

DataProcessingSchema DataAugmentationPipe::getProcessingSchema(const std::vector<uint32_t> &inputShape,
                                                               const uint64_t itemSeed) const {
    std::vector<uint32_t> lastShape = inputShape;
    std::vector<void *> itemSettingsList;
    std::vector<Shape> inputShapes, outputShapes;

    for (size_t i = 0; i < dataAugs.size(); i++) {
        const auto dataAugmentation = dataAugs[i];
        const auto [outputShape, itemSettings] = dataAugmentation->getDataOutputSchema(lastShape, itemSeed);
        if (outputShape.empty()) {
            throw std::runtime_error(std::format("Augmenter {} does not support input shape {}.",
                                                 i, formatVector(lastShape)));
        }

        inputShapes.push_back(lastShape);
        outputShapes.push_back(outputShape);
        lastShape = outputShape;
        itemSettingsList.push_back(itemSettings);
    }

    return {
        .outputShape = lastShape,
        .itemSettingsList = itemSettingsList,
        .dataAugInputShapes = inputShapes,
        .dataAugOutputShapes = outputShapes
    };
}

void DataAugmentationPipe::freeProcessingSchema(const DataProcessingSchema &processingSchema) const {
    for (size_t i = 0; i < dataAugs.size(); i++) {
        dataAugs[i]->freeItemSettings(processingSchema.itemSettingsList[i]);
    }
}

template<typename Func, typename SkipCheck>
void dispatchInLoop(
    const std::vector<IDataAugmentation *> &dataAugs,
    uint8_t *__restrict__ buffer1, uint8_t *__restrict__ buffer2,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const Shape &inputShape, const Shape &outputShape, const DType dtype,
    SkipCheck skipCheck, Func func
) {
    uint8_t *swapBuffer1 = buffer1;
    uint8_t *swapBuffer2 = buffer2;
    bool outputBufferIsUntouched = true;

    for (size_t i = 0; i < dataAugs.size(); i++) {
        bool doesLaterAugSwap = false;
        for (size_t j = i + 1; j < dataAugs.size(); j++) {
            if (!skipCheck(j)) {
                doesLaterAugSwap = true;
                break;
            }
        }

        if (!skipCheck(i)) {
            func(
                i, i == 0 || outputBufferIsUntouched ? inputData : swapBuffer1,
                i == dataAugs.size() - 1 || !doesLaterAugSwap ? outputData : swapBuffer2
            );

            const auto tmp = swapBuffer1;
            swapBuffer1 = swapBuffer2;
            swapBuffer2 = tmp;
            outputBufferIsUntouched = false;
        }
    }

    if (outputBufferIsUntouched) {
        if (inputShape != outputShape) {
            throw std::runtime_error(std::format(
                "Pipeline did not modify buffers, but input shape {} != output shape {}",
                formatVector(inputShape), formatVector(outputShape)
            ));
        }
        memcpy(outputData, inputData, getShapeSize(inputShape) * getWidthByDType(dtype));
    }
}

void DataAugmentationPipe::augmentWithPoints(
    const std::vector<uint32_t> &shape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const DataProcessingSchema &schema
) const {
    if (isUnsignedDType(dtype)) {
        throw std::runtime_error("Augmentation of points cannot happen in unsigned type.");
    }

    dispatchInLoop(dataAugs, buffer1, buffer2, inputData, outputData, shape, shape, dtype,
                   [&](const size_t i) {
                       const auto dataAug = dataAugs[i];
                       void *itemSettings = schema.itemSettingsList[i];
                       return dataAug->isAugmentWithPointsSkipped(shape, dtype, itemSettings);
                   },
                   [&](const size_t i, const uint8_t *input, uint8_t *output) {
                       const auto dataAugmentation = dataAugs[i];
                       void *itemSettings = schema.itemSettingsList[i];
                       return dataAugmentation->augmentWithPoints(shape, dtype, input, output, itemSettings);
                   });
}

void DataAugmentationPipe::augmentWithRaster(
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const DataProcessingSchema &schema
) const {
    const Shape &inputShape = schema.dataAugInputShapes[0];
    const Shape &outputShape = schema.dataAugOutputShapes[schema.dataAugOutputShapes.size() - 1];
    dispatchInLoop(dataAugs, buffer1, buffer2, inputData, outputData, inputShape, outputShape, dtype,
                   [&](const size_t i) {
                       const auto dataAug = dataAugs[i];
                       void *itemSettings = schema.itemSettingsList[i];
                       return dataAug->isAugmentWithRasterSkipped(
                           schema.dataAugInputShapes[i],
                           schema.dataAugOutputShapes[i],
                           dtype, itemSettings
                       );
                   },
                   [&](const size_t i, const uint8_t *input, uint8_t *output) {
                       const auto dataAug = dataAugs[i];
                       void *itemSettings = schema.itemSettingsList[i];
                       return dataAug->augmentWithRaster(
                           schema.dataAugInputShapes[i],
                           schema.dataAugOutputShapes[i],
                           dtype, input, output, itemSettings
                       );
                   });
}

bool DataAugmentationPipe::isOutputShapeStatic() const {
    for (IDataAugmentation *dataAugmentation: dataAugs) {
        if (dataAugmentation->isOutputShapeDetStaticExceptForBatchDim()) {
            return true;
        }
    }
    return false;
}
