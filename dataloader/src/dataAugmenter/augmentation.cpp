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
    const std::vector<IDataAugmentation *> &dataAugmentations,
    const std::vector<uint32_t> &maxAlongAllAxesInputShape,
    const uint32_t maxBytesPerElement,
    const uint32_t maxNumPoints,
    uint32_t &maxRequiredBufferSize
) {
    const uint32_t batchSize = maxAlongAllAxesInputShape[0];

    uint32_t maxShapeSize = getShapeSize(maxAlongAllAxesInputShape);
    auto lastMaxShape = std::vector(maxAlongAllAxesInputShape.begin() + 1, maxAlongAllAxesInputShape.end());
    for (size_t i = 0; i < dataAugmentations.size(); i++) {
        const auto dataAugmentation = dataAugmentations[i];
        const auto outputMaxShape = dataAugmentation->getMaxOutputShapeAxesIfSupported(lastMaxShape);
        if (outputMaxShape.empty()) {
            throw std::runtime_error(std::format("Augmentation {} pipline does not support maximum input shape {}",
                                                 i, formatVector(lastMaxShape)));
        }

        lastMaxShape = outputMaxShape;
        maxShapeSize = std::max(maxShapeSize, batchSize * getShapeSize(lastMaxShape));
    }

    maxRequiredBufferSize = std::max(
               maxShapeSize,
               batchSize * maxNumPoints * static_cast<uint32_t>(maxAlongAllAxesInputShape.size() - 2)
           ) * maxBytesPerElement;
}

Shape calcStaticOutputShape(
    const std::vector<IDataAugmentation *> &dataAugs,
    const Shape &inputShape
) {
    Shape lastShape = inputShape;
    for (const auto dataAug: dataAugs) {
        lastShape = dataAug->getDataOutputSchema(lastShape, 0).outputShape;
    }
    return lastShape;
}

DataAugmentationPipe::DataAugmentationPipe(
    std::vector<IDataAugmentation *> dataAugmentations,
    const Shape &maxInputShape, const uint32_t maxNumPoints,
    const uint32_t maxBytesPerElement
) : dataAugs(std::move(dataAugmentations)), buffer1(nullptr), buffer2(nullptr),
    maxInputShape(maxInputShape), maxNumPoints(maxNumPoints),
    maxRequiredBufferSize(0), staticOutputShape(calcStaticOutputShape(dataAugs, maxInputShape)) {
    if (!isOutputShapeStatic()) {
        throw std::runtime_error("Augmentation pipe needs to output a static shape.");
    }

    calcMaxRequiredBufferSize(
        dataAugs,
        maxInputShape, maxBytesPerElement, maxNumPoints,
        maxRequiredBufferSize
    );
}

size_t DataAugmentationPipe::getMaximumRequiredBufferSize() const {
    return maxRequiredBufferSize;
}

Shape DataAugmentationPipe::getStaticOutputShape() const {
    return staticOutputShape;
}

void DataAugmentationPipe::setBuffer(uint8_t *_buffer1, uint8_t *_buffer2) {
    buffer1 = _buffer1;
    buffer2 = _buffer2;
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
            const auto dataAugmentation = dataAugs[i];
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
    Shape outputShapeWithBatch =  { batchSize };
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
    const std::vector<IDataAugmentation *> &dataAugs,
    uint8_t *__restrict__ buffer1, uint8_t *__restrict__ buffer2,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const Shapes &inputShapes, const Shape &outputShape, const DType dtype,
    const uint32_t batchStride,
    SkipCheck skipCheck, Func func
) {
    const auto outputShapeNoBatch = std::vector(outputShape.begin() + 1, outputShape.end());

    const size_t B = inputShapes.size();
    std::vector<uint8_t *> swapBuffers1(B);
    std::vector<uint8_t *> swapBuffers2(B);
    std::vector<bool> outputBufferIsUntouchedForSlice(B);
    for (size_t i = 0; i < B; i++) {
        swapBuffers1[i] = buffer1;
        swapBuffers2[i] = buffer2;
        outputBufferIsUntouchedForSlice[i] = true;
    }

    const uint32_t outputSize = getShapeSize(outputShape) * getWidthOfDType(dtype);

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
                                                        ? inputData + b * outputSize
                                                        : swapBuffers1[b] + b * batchStride;
                uint8_t *sliceOfOutputForAug = (i == dataAugs.size() - 1 || !doesLaterAugSwap)
                                                   ? outputData + b * outputSize
                                                   : swapBuffers2[b] + b * batchStride;
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
            if (inputShapes[b] != outputShapeNoBatch) {
                throw std::runtime_error(std::format(
                    "Pipeline did not modify buffers, but input shape {} != output shape {}",
                    formatVector(inputShapes[b]), formatVector(outputShape)
                ));
            }
            memcpy(
                outputData + b * outputSize,
                inputData + b * outputSize,
                getShapeSize(inputShapes[b]) * getWidthOfDType(dtype)
            );
        }
    }
}

void DataAugmentationPipe::augmentWithPoints(
    const Shapes &shapes,
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const DataProcessingSchema &schema
) const {
    if (isUnsignedDType(dtype)) {
        throw std::runtime_error("Augmentation of points cannot happen in unsigned type.");
    }

    if (shapes.empty()) {
        throw std::runtime_error("Cannot augment an empty list of points.");
    }

    const uint32_t lastBatch = shapes[0][0];
    const uint32_t lastDim = shapes[0].back();
    for (const auto &shape: shapes) {
        if (shape.size() != 3) {
            throw std::runtime_error("Augmentation can only apply to shapes (b, n_i, d); n_i may vary across items.");
        }
        if (shape[0] != lastBatch) {
            throw std::runtime_error("Augmentation can only apply to shapes (b, n_i, d), but b is not constant.");
        }
        if (shape.back() != lastDim) {
            throw std::runtime_error("Augmentation can only apply to shapes (b, n_i, d), but d is not constant.");
        }
    }

    // TODO: Shapes can be different!!!!!!!!!!!!!
    const Shape outputShape = {
        shapes[0][0],
        maxNumPoints,
        shapes[0][2]
    };

    dispatchInLoop(dataAugs, buffer1, buffer2,
                   inputData, outputData,
                   shapes, outputShape,
                   dtype, maxRequiredBufferSize,
                   [&](const size_t b, const size_t i) {
                       const auto dataAug = dataAugs[i];
                       void *itemProp = schema.itemPropsPerAug[i][b];
                       return dataAug->isAugmentWithPointsSkipped(shapes[b], dtype, itemProp);
                   },
                   [&](const size_t b, const size_t i, const uint8_t *input, uint8_t *output) {
                       const auto dataAugmentation = dataAugs[i];
                       void *itemProp = schema.itemPropsPerAug[i][b];
                       return dataAugmentation->augmentWithPoints(shapes[b], dtype, input, output, itemProp);
                   });
}

void DataAugmentationPipe::augmentWithRaster(
    const DType dtype,
    const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
    const DataProcessingSchema &schema
) const {
    // TODO: Shapes can be different!!!!!!!!!!!!!

    const Shapes &inputShapes = schema.inputShapesPerAug[0];
    dispatchInLoop(dataAugs, buffer1, buffer2,
                   inputData, outputData,
                   inputShapes, schema.outputShape,
                   dtype, maxRequiredBufferSize,
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
                   });
}

const Shape &DataAugmentationPipe::getMaxInputShape() const {
    return maxInputShape;
}

uint32_t DataAugmentationPipe::getMaxNumPoints() const {
    return maxNumPoints;
}

bool DataAugmentationPipe::isOutputShapeStatic() const {
    for (IDataAugmentation *dataAugmentation: dataAugs) {
        if (dataAugmentation->isOutputShapeDetStaticExceptForBatchDim()) {
            return true;
        }
    }
    return false;
}
