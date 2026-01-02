#ifndef VERSION_DATAAUGMENTER_H
#define VERSION_DATAAUGMENTER_H

#include <vector>
#include "utils.h"

bool isInputShapeHWN(const std::vector<uint32_t> &inputShape, size_t N);

struct DataOutputSchema {
    Shape outputShape; // Empty, if input shape is not supported.
    ItemProp itemProp;
};

class IDataAugmentation {
public:
    virtual ~IDataAugmentation() = default;

    virtual bool isOutputShapeDetStaticExceptForBatchDim() = 0;

    virtual bool isOutputShapeDetStaticGivenStaticInputShape() = 0;

    [[nodiscard]] virtual DataOutputSchema getDataOutputSchema(const Shape &inputShape, uint64_t itemSeed) const = 0;

    // TODO: I don't love the manual free api here, but we'll go with it for now.
    virtual void freeItemProp(ItemProp &itemProp) const = 0;

    [[nodiscard]] virtual Shape getMaxOutputShapeAxesIfSupported(const Shape &inputShape) const = 0;

    // Returns true if computation is skipped.
    [[nodiscard]] virtual bool isAugmentWithPointsSkipped(
        const Shape &shape,
        DType dtype, ItemProp &itemProp
    ) = 0;

    // Never read from outputData.
    virtual void augmentWithPoints(
        const Shape &shape,
        DType dtype,
        const uint8_t *__restrict__ inputData,
        uint8_t *__restrict__ outputData,
        ItemProp &itemProp
    ) = 0;

    // Returns true if computation is skipped.
    [[nodiscard]] virtual bool isAugmentWithRasterSkipped(
        const Shape &inputShape,
        const Shape &outputShape,
        DType dtype, ItemProp &itemProp
    ) = 0;

    // Never read from outputData.
    virtual void augmentWithRaster(
        const Shape &inputShape,
        const Shape &outputShape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        ItemProp &itemProp
    ) = 0;
};

struct DataProcessingSchema {
    Shape outputShape;
    std::vector<ItemProps> itemPropsPerAug;
    std::vector<Shapes> inputShapesPerAug;
    std::vector<Shapes> outputShapesPerAug;
};

// Usage: Initialize, ask for maximium required buffer size, pass buffers, use.
class DataAugmentationPipe {
public:
    explicit DataAugmentationPipe(
        std::vector<std::shared_ptr<IDataAugmentation> > dataAugmentations,
        const Shape &rasterMaxInputShape, uint32_t maxNumPoints, uint32_t maxBytesPerElement
    );

    [[nodiscard]] size_t getMaximumRequiredBufferSize() const;

    // Returns no batch dimension.
    [[nodiscard]] Shape getStaticOutputShape() const;

    [[nodiscard]] DataProcessingSchema getProcessingSchema(const Shapes &inputShapes, uint64_t itemSeed) const;

    // TODO: I don't love the manual free api here, but we'll go with it for now.
    void freeProcessingSchema(const DataProcessingSchema &processingSchema) const;

    // Shapes are necessary, as the number of points can vary for each item in the batch.
    void augmentWithPoints(
        const Shapes &shapes,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        int32_t *metaOutputData,
        uint8_t *__restrict__ buffer1, uint8_t *__restrict__ buffer2,
        const DataProcessingSchema &schema
    ) const;

    void augmentWithRaster(
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        uint8_t *__restrict__ buffer1, uint8_t *__restrict__ buffer2,
        size_t maxBytesOfInputPerItemOfItemKey,
        const DataProcessingSchema &schema
    ) const;

    // Returns with batch dimension.
    [[nodiscard]] Shape getRasterMaxInputShape() const;

    [[nodiscard]] uint32_t getMaxNumPoints() const;

private:
    std::vector<std::shared_ptr<IDataAugmentation> > dataAugs;
    uint32_t maxRequiredBufferSize;
    uint32_t maxIntermediateSizeForItem;

    Shape rasterMaxInputShape;
    uint32_t maxNumPoints;
    Shape staticOutputShape;

    [[nodiscard]] bool isOutputShapeStatic() const;
};

#endif
