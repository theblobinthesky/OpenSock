#ifndef VERSION_FLIPAUGMENTATION_H
#define VERSION_FLIPAUGMENTATION_H

#include "dataAugmenter/augmentation.h"

struct FlipItemSettings {
    bool doesHorizontalFlip;
    bool doesVerticalFlip;
};

class FlipAugmentation final : public IDataAugmentation {
public:
    FlipAugmentation(
        bool flipHorizontal, float horizontalFlipProbability,
        bool flipVertical, float verticalFlipProbability
    );

    bool isOutputShapeStaticExceptForBatch() override;

    [[nodiscard]] DataOutputSchema getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) override;

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
    bool flipHorizontal;
    float horizontalFlipProbability;
    bool flipVertical;
    float verticalFlipProbability;
};

#endif
