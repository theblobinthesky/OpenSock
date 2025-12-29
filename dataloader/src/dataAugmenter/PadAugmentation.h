#ifndef VERSION_PAD_H
#define VERSION_PAD_H

#include "dataAugmenter/augmentation.h"

enum class PadSettings {
    PAD_TOP_LEFT,
    PAD_TOP_RIGHT,
    PAD_BOTTOM_LEFT,
    PAD_BOTTOM_RIGHT
};

class PadAugmentation final : public IDataAugmentation {
public:
    PadAugmentation(
        uint32_t padHeight, uint32_t padWidth,
        PadSettings padSettings
    );

    bool isOutputShapeDetStaticExceptForBatchDim() override;

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
    uint32_t padHeight, padWidth;
    PadSettings padSettings;
};

#endif
