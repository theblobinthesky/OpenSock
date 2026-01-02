#ifndef VERSION_FLIPAUGMENTATION_H
#define VERSION_FLIPAUGMENTATION_H

#include "dataAugmenter/augmentation.h"

struct FlipProp {
    bool doesVerticalFlip;
    bool doesHorizontalFlip;
    uint32_t originalHeight;
    uint32_t originalWidth;
};

class FlipAugmentation final : public IDataAugmentation {
public:
    FlipAugmentation(float verticalFlipProbability, float horizontalFlipProbability);

    bool isOutputShapeDetStaticExceptForBatchDim() override;

    bool isOutputShapeDetStaticGivenStaticInputShape() override;

    [[nodiscard]] DataOutputSchema getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) const override;

    void freeItemProp(ItemProp &itemProp) const override;

    [[nodiscard]] Shape getMaxOutputShapeAxesIfSupported(const Shape &inputShape) const override;

    bool isAugmentWithPointsSkipped(
        const Shape &shape,
        DType dtype, ItemProp &itemProp
    ) override;

    void augmentWithPoints(
        const Shape &shape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        ItemProp &itemProp
    ) override;

    bool isAugmentWithRasterSkipped(
        const Shape &inputShape,
        const Shape &outputShape,
        DType dtype, ItemProp &itemProp
    ) override;

    void augmentWithRaster(
        const Shape &inputShape,
        const Shape &outputShape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        ItemProp &itemProp
    ) override;

private:
    float verticalFlipProbability;
    float horizontalFlipProbability;
};

#endif
