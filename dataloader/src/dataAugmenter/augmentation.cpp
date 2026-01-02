#include "augmentation.h"

#include <utility>
#include <ranges>

bool isInputShapeHWN(const std::vector<uint32_t> &inputShape, const size_t N) {
    if (inputShape.size() != 3) {
        return false;
    }

    return inputShape[2] == N;
}

void calcMaxRequiredBufferSize(
    const std::vector<std::shared_ptr<IDataAugmentation> > &dataAugmentations,
    const std::vector<uint32_t> &maxAlongAllAxesInputShape,
    const uint32_t maxBytesPerElement,
    const uint32_t maxNumPoints,
    uint32_t &maxRequiredBufferSize,
    uint32_t &maxIntermediateSizeForItem
) {
    const uint32_t batchSize = maxAlongAllAxesInputShape[0];
    uint32_t maxShapeSize = getShapeSize(maxAlongAllAxesInputShape);

    auto lastMaxShape = std::vector(maxAlongAllAxesInputShape.begin() + 1, maxAlongAllAxesInputShape.end());
    for (size_t i = 0; i < dataAugmentations.size(); i++) {
        const std::shared_ptr<IDataAugmentation> dataAugmentation = dataAugmentations[i];
        const auto outputMaxShape = dataAugmentation->getMaxOutputShapeAxesIfSupported(lastMaxShape);
        if (outputMaxShape.empty()) {
            throw std::runtime_error(std::format("Augmentation {} pipline does not support maximum input shape {}",
                                                 i, formatVector(lastMaxShape)));
        }

        lastMaxShape = outputMaxShape;
        maxShapeSize = std::max(maxShapeSize, batchSize * getShapeSize(lastMaxShape));
    }

    const size_t numAxes = maxAlongAllAxesInputShape.size();
    maxRequiredBufferSize = std::max(maxShapeSize,
                                     batchSize * maxNumPoints * static_cast<uint32_t>(numAxes - 2)
                            ) * maxBytesPerElement;
    maxIntermediateSizeForItem = maxRequiredBufferSize / batchSize;
}

Shape calcStaticOutputShape(
    const std::vector<std::shared_ptr<IDataAugmentation> > &dataAugs,
    const Shape &inputShape
) {
    auto lastShape = std::vector(inputShape.begin() + 1, inputShape.end());
    for (const auto &dataAug: dataAugs) {
        lastShape = dataAug->getDataOutputSchema(lastShape, 0).outputShape;
    }
    return lastShape;
}

DataAugmentationPipe::DataAugmentationPipe(
    std::vector<std::shared_ptr<IDataAugmentation> > dataAugmentations,
    const Shape &rasterMaxInputShape, const uint32_t maxNumPoints, const uint32_t maxBytesPerElement
) : dataAugs(std::move(dataAugmentations)), maxRequiredBufferSize(0), maxIntermediateSizeForItem(0),
    rasterMaxInputShape(rasterMaxInputShape), maxNumPoints(maxNumPoints),
    staticOutputShape(calcStaticOutputShape(dataAugs, rasterMaxInputShape)) {
    if (!isOutputShapeStatic()) {
        throw std::runtime_error("Augmentation pipe needs to output a static shape.");
    }

    if (maxBytesPerElement > 16) {
        throw std::runtime_error("The #bytes per element can't be >16.");
    }

    calcMaxRequiredBufferSize(
        dataAugs,
        rasterMaxInputShape, maxBytesPerElement, maxNumPoints,
        maxRequiredBufferSize,
        maxIntermediateSizeForItem
    );
}

size_t DataAugmentationPipe::getMaximumRequiredBufferSize() const {
    return maxRequiredBufferSize;
}

Shape DataAugmentationPipe::getStaticOutputShape() const {
    return staticOutputShape;
}

DataProcessingSchema DataAugmentationPipe::getProcessingSchema(const Shapes &inputShapes,
                                                               const uint64_t itemSeed) const {
    Shape outerOutputShape;
    std::vector<ItemProps> itemPropsPerAug(dataAugs.size());
    std::vector<Shapes> inputShapesPerAug(dataAugs.size());
    std::vector<Shapes> outputShapesPerAug(dataAugs.size());

    for (size_t b = 0; b < inputShapes.size(); b++) { // NOLINT(*-loop-convert)
        // Compute changing shapes for each batch slice seperately.
        std::vector<uint32_t> lastShapeForSlice = inputShapes[b];
        std::vector<void *> itemPropsForSlice;
        std::vector<Shape> inputShapesForSlice, outputShapesForSlice;

        for (size_t i = 0; i < dataAugs.size(); i++) {
            const std::shared_ptr<IDataAugmentation> dataAugmentation = dataAugs[i];
            auto [outputShape, itemProp] = dataAugmentation->getDataOutputSchema(lastShapeForSlice, itemSeed);
            if (outputShape.empty()) {
                throw std::runtime_error(std::format("Augmenter {} does not support input shape {}.",
                                                     i, formatVector(lastShapeForSlice)));
            }

            inputShapesForSlice.push_back(lastShapeForSlice);
            outputShapesForSlice.push_back(outputShape);
            lastShapeForSlice = outputShape;
            itemPropsForSlice.push_back(itemProp);
        }

        outerOutputShape = lastShapeForSlice;

        // Accumulate them into their respective augmentation list.
        for (size_t i = 0; i < dataAugs.size(); i++) {
            itemPropsPerAug[i].push_back(itemPropsForSlice[i]);
            inputShapesPerAug[i].push_back(inputShapesForSlice[i]);
            outputShapesPerAug[i].push_back(outputShapesForSlice[i]);
        }
    }

    const uint32_t batchSize = inputShapes.size();
    Shape outputShapeWithBatch = {batchSize};
    outputShapeWithBatch.insert(outputShapeWithBatch.end(), outerOutputShape.begin(), outerOutputShape.end());

    return {
        .outputShape = std::move(outputShapeWithBatch), // All are required to be the same output shape.
        .itemPropsPerAug = std::move(itemPropsPerAug),
        .inputShapesPerAug = std::move(inputShapesPerAug),
        .outputShapesPerAug = std::move(outputShapesPerAug)
    };
}

void DataAugmentationPipe::freeProcessingSchema(const DataProcessingSchema &processingSchema) const {
    for (size_t i = 0; i < dataAugs.size(); i++) {
        for (const auto &b: processingSchema.itemPropsPerAug) {
            auto tmp = b[i];
            dataAugs[i]->freeItemProp(tmp);
        }
    }
}

template<typename Func, typename SkipCheck>
void dispatchInLoop(
    const std::vector<std::shared_ptr<IDataAugmentation> > &dataAugs,
    uint8_t *__restrict__ buffer1, uint8_t *__restrict__ buffer2,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const Shapes &inputShapes, const Shape &outputShape, const DType dtype,
    const size_t maxBytesOfInputPerItemOfItemKey,
    const uint32_t maxIntermediateSizeOfItem,
    SkipCheck skipCheck, Func func,
    const bool forceEqualityIfSkip
) {
    const auto outputShapeNoBatch = std::span(outputShape).subspan(1);

    const size_t B = inputShapes.size();
    std::vector<uint8_t *> swapBuffers1(B);
    std::vector<uint8_t *> swapBuffers2(B);
    std::vector<bool> outputBufferIsUntouchedForSlice(B);
    for (size_t i = 0; i < B; i++) {
        swapBuffers1[i] = buffer1 + i * maxIntermediateSizeOfItem;
        swapBuffers2[i] = buffer2 + i * maxIntermediateSizeOfItem;
        outputBufferIsUntouchedForSlice[i] = true;
    }

    const uint32_t outputSize = getShapeSize(outputShapeNoBatch) * getWidthOfDType(dtype);

    for (size_t i = 0; i < dataAugs.size(); i++) {
        for (size_t b = 0; b < B; b++) {
            bool doesLaterAugSwap = false;
            for (size_t j = i + 1; j < dataAugs.size(); j++) {
                if (!skipCheck(b, j)) {
                    doesLaterAugSwap = true;
                    break;
                }
            }

            if (!skipCheck(b, i)) {
                const uint8_t *sliceOfInputForAug = (i == 0 || outputBufferIsUntouchedForSlice[b])
                                                        ? inputData + b * maxBytesOfInputPerItemOfItemKey
                                                        : swapBuffers1[b];
                uint8_t *sliceOfOutputForAug = (i == dataAugs.size() - 1 || !doesLaterAugSwap)
                                                   ? outputData + b * outputSize
                                                   : swapBuffers2[b];
                func(b, i, sliceOfInputForAug, sliceOfOutputForAug);

                const auto tmp = swapBuffers1[b];
                swapBuffers1[b] = swapBuffers2[b];
                swapBuffers2[b] = tmp;
                outputBufferIsUntouchedForSlice[b] = false;
            }
        }
    }

    for (size_t b = 0; b < B; b++) {
        if (outputBufferIsUntouchedForSlice[b]) {
            if (forceEqualityIfSkip && std::span(inputShapes[b]) != outputShapeNoBatch) {
                throw std::runtime_error(std::format(
                    "Pipeline did not modify buffers, but input shape {} != output shape {}",
                    formatVector(inputShapes[b]), formatVector(outputShape)
                ));
            }
            memcpy(
                outputData + b * outputSize,
                inputData + b * maxBytesOfInputPerItemOfItemKey,
                outputSize
            );
        }
    }
}

void DataAugmentationPipe::augmentWithPoints(
    const Shapes &shapes,
    const DType dtype,
    const uint8_t *__restrict__ inputData,
    uint8_t *__restrict__ outputData,
    int32_t *__restrict__ metaOutputData,
    uint8_t *__restrict__ buffer1, uint8_t *__restrict__ buffer2,
    const DataProcessingSchema &schema
) const {
    if (isUnsignedDType(dtype)) {
        throw std::runtime_error("Augmentation of points cannot happen in unsigned type.");
    }

    if (shapes.empty()) {
        throw std::runtime_error("Cannot augment an empty list of points.");
    }

    const uint32_t lastDim = shapes[0].back();
    for (const auto &shape: shapes) {
        if (shape.size() != 2) {
            throw std::runtime_error(std::format(
                "Augmentation can only apply to shapes (n_i, d), but got {}",
                formatVector(shape)
            ));
        }
        if (shape.back() != lastDim) {
            throw std::runtime_error("Augmentation can only apply to shapes (n_i, d), but d is not constant.");
        }
    }

    const Shape outputShape = {
        static_cast<uint32_t>(shapes.size()),
        maxNumPoints,
        shapes[0][1]
    };

    std::vector<size_t> maxBytesOfInputPerItemOfItemKey;
    const std::span outputShapeNoBatch = std::span(outputShape).subspan(1);
    const size_t maxBytesOfInput = getShapeSize(outputShapeNoBatch) * getWidthOfDType(dtype);

    dispatchInLoop(dataAugs, buffer1, buffer2,
                   inputData, outputData,
                   shapes, outputShape,
                   dtype, maxBytesOfInput,
                   maxIntermediateSizeForItem,
                   [&](const size_t b, const size_t i) {
                       const auto dataAug = dataAugs[i];
                       void *itemProp = schema.itemPropsPerAug[i][b];
                       return dataAug->isAugmentWithPointsSkipped(shapes[b], dtype, itemProp);
                   },
                   [&](const size_t b, const size_t i, const uint8_t *input, uint8_t *output) {
                       const auto dataAugmentation = dataAugs[i];
                       void *itemProp = schema.itemPropsPerAug[i][b];
                       return dataAugmentation->augmentWithPoints(shapes[b], dtype, input, output, itemProp);
                   }, false);

    if (metaOutputData) {
        // Fill metadata buffer with true lengths.
        for (size_t i = 0; i < shapes.size(); i++) {
            const uint32_t n = shapes[i][0];
            metaOutputData[i] = static_cast<int32_t>(n);
        }
    }
}

void DataAugmentationPipe::augmentWithRaster(
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    uint8_t *__restrict__ buffer1, uint8_t *__restrict__ buffer2,
    const size_t maxBytesOfInputPerItemOfItemKey,
    const DataProcessingSchema &schema
) const {
    // TODO: Shapes can be different!!!!!!!!!!!!!

    const Shapes &inputShapes = schema.inputShapesPerAug[0];
    dispatchInLoop(dataAugs, buffer1, buffer2,
                   inputData, outputData,
                   inputShapes, schema.outputShape,
                   dtype, maxBytesOfInputPerItemOfItemKey,
                   maxIntermediateSizeForItem,
                   [&](const size_t b, const size_t i) {
                       const auto dataAug = dataAugs[i];
                       void *itemProp = schema.itemPropsPerAug[i][b];
                       return dataAug->isAugmentWithRasterSkipped(
                           schema.inputShapesPerAug[i][b],
                           schema.outputShapesPerAug[i][b],
                           dtype, itemProp
                       );
                   },
                   [&](const size_t b, const size_t i, const uint8_t *input, uint8_t *output) {
                       const auto dataAug = dataAugs[i];
                       void *itemProp = schema.itemPropsPerAug[i][b];
                       return dataAug->augmentWithRaster(
                           schema.inputShapesPerAug[i][b],
                           schema.outputShapesPerAug[i][b],
                           dtype, input, output, itemProp
                       );
                   }, true);
}

Shape DataAugmentationPipe::getRasterMaxInputShape() const {
    return rasterMaxInputShape;
}

[[nodiscard]] uint32_t DataAugmentationPipe::getMaxNumPoints() const {
    return maxNumPoints;
}

bool DataAugmentationPipe::isOutputShapeStatic() const {
    int lastStaticOutputIdx = -1;
    for (int i = static_cast<int>(dataAugs.size() - 1); i >= 0; i--) {
        if (dataAugs[i]->isOutputShapeDetStaticExceptForBatchDim()) {
            lastStaticOutputIdx = i;
            break;
        }
    }

    if (lastStaticOutputIdx == -1) {
        return false;
    }

    for (size_t i = static_cast<size_t>(lastStaticOutputIdx); i < dataAugs.size(); i++) {
        if (!dataAugs[i]->isOutputShapeDetStaticGivenStaticInputShape()) {
            return false;
        }
    }
    return true;
}
