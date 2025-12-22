#ifndef VERSION_RANDOMRESIZEDCROP_H
#define VERSION_RANDOMRESIZEDCROP_H

#include "dataAugmenter/augmentation.h"

struct RandomCropSettings {
    size_t top;
    size_t left;
    uint32_t height;
    uint32_t width;
    bool skip;
};

class RandomCropAugmentation final : public IDataAugmentation {
public:
    RandomCropAugmentation(
        uint32_t minCropHeight, uint32_t minCropWidth,
        uint32_t maxCropHeight, uint32_t maxCropWidth
    );

    bool isOutputShapeDetStaticExceptForBatchDim() override;

    [[nodiscard]] DataOutputSchema getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) const override;

    void freeItemSettings(void *itemSettings) const override;

    [[nodiscard]] std::vector<uint32_t> getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) const override;

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
    uint32_t minCropHeight, minCropWidth;
    uint32_t maxCropHeight, maxCropWidth;
};

#endif
