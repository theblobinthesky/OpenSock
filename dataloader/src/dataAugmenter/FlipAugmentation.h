#ifndef VERSION_FLIPAUGMENTATION_H
#define VERSION_FLIPAUGMENTATION_H

#include "dataAugmenter/DataAugmentation.h"

struct FlipItemSettings {
    bool doesHorizontalFlip;
    bool doesVerticalFlip;
};

class FlipAugmentation final : public IDataTransformAugmentation {
public:
    FlipAugmentation(
        bool flipHorizontal, float horizontalFlipProbability,
        bool flipVertical, float verticalFlipProbability
    );

    bool isOutputShapeStatic() override;

    std::vector<size_t> getOutputShapeIfSupported(const std::vector<size_t> &inputShape) override;

    [[nodiscard]] void *getItemSettings(uint64_t itemSeed) const override;

    void freeItemSettings(void *itemSettings) const override;

    void augmentWithChannelFirst(
        const std::vector<size_t> &inputShape,
        const std::vector<size_t> &outputShape,
        ItemFormat format,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    ) override;

    void augmentWithChannelLast(
        const std::vector<size_t> &inputShape,
        const std::vector<size_t> &outputShape,
        ItemFormat format,
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
