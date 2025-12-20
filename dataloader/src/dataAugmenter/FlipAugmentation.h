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

    std::vector<size_t> getOutputShapeIfSupported(const std::vector<size_t> &inputShape) override;

    [[nodiscard]] void *getItemSettings(const std::vector<size_t> &inputShape, uint64_t itemSeed) const override;

    void freeItemSettings(void *itemSettings) const override;

    bool augmentWithPoints(
        const std::vector<size_t> &shape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    ) override;

    bool augmentWithRaster(
        const std::vector<size_t> &inputShape,
        const std::vector<size_t> &outputShape,
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
