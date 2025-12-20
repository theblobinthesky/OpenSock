#include "augmentation.h"

#include <utility>

bool isInputShapeBHWN(const std::vector<size_t> &inputShape) {
    return inputShape.size() == 4;
}

bool isInputShapeBHWN(const std::vector<size_t> &inputShape, const size_t N) {
    if (inputShape.size() != 4) {
        return false;
    }

    return inputShape[3] == N;
}

uint32_t calcMaxRequiredBufferSize(
    const std::vector<IDataAugmentation *> &dataAugmentations,
    const std::vector<uint32_t> &maxAlongAllAxesInputShape,
    const uint32_t maxBytesPerElement
) {
    uint32_t maxShapeSize = getShapeSize(maxAlongAllAxesInputShape);
    std::vector<uint32_t> lastMaxShape = maxAlongAllAxesInputShape;
    for (const auto dataAugmentation: dataAugmentations) {
        lastMaxShape = dataAugmentation->getMaxOutputShapeAxesIfSupported(lastMaxShape);
        if (lastMaxShape.empty()) {
            throw std::runtime_error("Augmentation pipline does not support maximum input shape.");
        }

        maxShapeSize = std::max(maxShapeSize, getShapeSize(lastMaxShape));
    }

    return maxShapeSize * maxBytesPerElement;
}

DataAugmentationPipe::DataAugmentationPipe(
    std::vector<IDataAugmentation *> _dataAugmentations,
    const std::vector<uint32_t> &maxInputShape,
    const uint32_t maxBytesPerElement
) : dataAugs(std::move(_dataAugmentations)), buffer1(nullptr), buffer2(nullptr) {
    if (!isOutputShapeStatic()) {
        throw std::runtime_error("Augmentation pipe needs to output a static shape.");
    }
    maximumRequiredBufferSize = calcMaxRequiredBufferSize(dataAugs, maxInputShape, maxBytesPerElement);
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

    for (const auto dataAugmentation: dataAugs) {
        const auto [outputShape, itemSettings] = dataAugmentation->getDataOutputSchema(lastShape, itemSeed);
        lastShape = outputShape;
        itemSettingsList.push_back(itemSettings);

        if (lastShape.empty()) {
            throw std::runtime_error("Augmentation pipeline does not support input shape.");
        }
    }

    return {
        .outputShape = lastShape,
        .itemSettingsLists = itemSettingsList
    };
}

void DataAugmentationPipe::freeProcessingSchema(const DataProcessingSchema &processingSchema) const {
    for (size_t i = 0; i < dataAugs.size(); i++) {
        dataAugs[i]->freeItemSettings(processingSchema.itemSettingsLists[i]);
    }
}

template<typename Func>
void dispatchInLoop(
    const std::vector<IDataAugmentation *> &dataAugs,
    uint8_t *__restrict__ buffer1, uint8_t *__restrict__ buffer2,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    Func func
) {
    uint8_t *input = buffer1;
    uint8_t *output = buffer2;

    for (size_t i = 0; i < dataAugs.size(); i++) {
        if (func(i, i == 0 ? inputData : input, i == dataAugs.size() - 1 ? outputData : output)) {
            const auto tmp = input;
            input = output;
            output = tmp;
        }
    }
}

void DataAugmentationPipe::augmentWithPoints(
    const std::vector<uint32_t> &shape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const std::vector<void *> &itemSettingsList
) const {
    dispatchInLoop(dataAugs, buffer1, buffer2, inputData, outputData,
                   [&](const size_t i, const uint8_t *input, uint8_t *output) {
                       const auto dataAugmentation = dataAugs[i];
                       void *itemSettings = itemSettingsList[i];
                       return dataAugmentation->augmentWithPoints(shape, dtype, input, output, itemSettings);
                   });
}

void DataAugmentationPipe::augmentWithRaster(
    const std::vector<uint32_t> &inputShape,
    const std::vector<uint32_t> &outputShape,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const std::vector<void *> &itemSettingsList
) const {
    dispatchInLoop(dataAugs, buffer1, buffer2, inputData, outputData,
                   [&](const size_t i, const uint8_t *input, uint8_t *output) {
                       const auto dataAug = dataAugs[i];
                       void *itemSettings = itemSettingsList[i];
                       return dataAug->augmentWithRaster(inputShape, outputShape, dtype, input, output, itemSettings);
                   });
}

bool DataAugmentationPipe::isOutputShapeStatic() const {
    for (IDataAugmentation *dataAugmentation: dataAugs) {
        if (dataAugmentation->isOutputShapeStaticExceptForBatch()) {
            return true;
        }
    }
    return false;
}
