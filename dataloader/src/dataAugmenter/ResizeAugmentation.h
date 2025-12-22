#ifndef VERSION_RESIZEDATAAUGMENTER_H
#define VERSION_RESIZEDATAAUGMENTER_H

#include "dataAugmenter/augmentation.h"

struct ResizeSettings {
    uint32_t originalHeight;
    uint32_t originalWidth;
    bool skip;
};

class ResizeAugmentation final : public IDataAugmentation {
public:
    ResizeAugmentation(uint32_t height, uint32_t width);

    bool isOutputShapeDetStaticExceptForBatchDim() override;

    [[nodiscard]] DataOutputSchema getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) const override;

    void freeItemSettings(void *itemSettings) const override;

    std::vector<uint32_t> getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) const override;

    bool isAugmentWithPointsSkipped(
        const std::vector<uint32_t> &shape,
        DType dtype, void *itemSettings
    ) override;

    void augmentWithPoints(
        const std::vector<uint32_t> &shape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    ) override;

    bool isAugmentWithRasterSkipped(
        const std::vector<uint32_t> &inputShape,
        const std::vector<uint32_t> &outputShape,
        DType dtype, void *itemSettings
    ) override;

    void augmentWithRaster(
        const std::vector<uint32_t> &inputShape,
        const std::vector<uint32_t> &outputShape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    ) override;

private:
    uint32_t height, width;
};

#endif
