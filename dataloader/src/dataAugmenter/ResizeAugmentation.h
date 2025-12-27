#ifndef VERSION_RESIZEDATAAUGMENTER_H
#define VERSION_RESIZEDATAAUGMENTER_H

#include "dataAugmenter/augmentation.h"

struct ResizeProp {
    uint32_t originalHeight;
    uint32_t originalWidth;
    bool skip;
};

class ResizeAugmentation final : public IDataAugmentation {
public:
    ResizeAugmentation(uint32_t height, uint32_t width);

    bool isOutputShapeDetStaticExceptForBatchDim() override;

    [[nodiscard]] DataOutputSchema getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) const override;

    void freeItemProp(ItemProp &itemProp) const override;

    Shape getMaxOutputShapeAxesIfSupported(const Shape &inputShape) const override;

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
    uint32_t height, width;
};

#endif
