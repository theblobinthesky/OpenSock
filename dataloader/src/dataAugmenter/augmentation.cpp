#include "augmentation.h"

#include <utility>

bool isInputShapeBHWN(const std::vector<size_t> &inputShape, const size_t N) {
    if (inputShape.size() != 4) {
        return false;
    }

    return inputShape[3] == N;
}

DataAugmentationPipe::DataAugmentationPipe(std::vector<IDataAugmentation *> dataAugmentations)
    : dataAugmentations(std::move(dataAugmentations)) {
}

bool DataAugmentationPipe::isOutputShapeStatic() const {
    for (IDataAugmentation *dataAugmentation: dataAugmentations) {
        if (dataAugmentation->isOutputShapeStatic()) {
            return true;
        }
    }

    return false;
}

std::vector<size_t> DataAugmentationPipe::getOutputShapeIfSupported(const std::vector<size_t> &inputShape) const {
    std::vector<size_t> tempShape = inputShape;
    for (IDataAugmentation *dataAugmentation: dataAugmentations) {
        tempShape = dataAugmentation->getOutputShapeIfSupported(tempShape);
        if (tempShape.empty()) {
            return std::vector<size_t>{};
        }
    }

    return tempShape;
}

[[nodiscard]] void *DataAugmentationPipe::getItemSettings(uint64_t itemSeed) const {

}

void freeItemSettings(void *itemSettings) const {
}

void augmentWithChannelFirst(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
    ItemFormat format,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
);

void augmentWithChannelLast(
    const std::vector<size_t> &inputShape,
    const std::vector<size_t> &outputShape,
    ItemFormat format,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    void *itemSettings
);
