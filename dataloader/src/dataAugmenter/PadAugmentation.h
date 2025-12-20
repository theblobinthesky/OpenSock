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
        size_t padHeight, size_t padWidth,
        PadSettings padSettings
    );

    bool isOutputShapeStaticExceptForBatch() override;

    std::vector<size_t> getOutputShapeIfSupported(const std::vector<size_t> &inputShape) override;

    [[nodiscard]] void *getItemSettings(const std::vector<size_t> &inputShape, uint64_t itemSeed) const override;

    void freeItemSettings(void *itemSettings) const override;

    bool augmentWithPoints(
        const std::vector<size_t> &inputShape,
        const std::vector<size_t> &outputShape,
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
    size_t padHeight, padWidth;
    PadSettings padSettings;
};

#endif
