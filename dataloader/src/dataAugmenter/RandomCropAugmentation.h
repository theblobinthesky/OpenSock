#ifndef VERSION_RANDOMRESIZEDCROP_H
#define VERSION_RANDOMRESIZEDCROP_H

#include "dataAugmenter/augmentation.h"

struct RandomCropSettings {
    size_t left;
    size_t top;
    uint32_t height;
    uint32_t width;
};

class RandomCropAugmentation final : public IDataAugmentation {
public:
    RandomCropAugmentation(
        uint32_t minCropHeight, uint32_t minCropWidth,
        uint32_t maxCropHeight, uint32_t maxCropWidth
    );

    bool isOutputShapeStaticExceptForBatch() override;

    DataOutputSchema getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) const override;

    void freeItemSettings(void *itemSettings) const override;

    std::vector<uint32_t> getMaxOutputShapeAxesIfSupported(const std::vector<uint32_t> &inputShape) override;

    bool augmentWithPoints(
        const std::vector<uint32_t> &shape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    ) override;

    bool augmentWithRaster(
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
