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
    uint32_t padHeight, padWidth;
    PadSettings padSettings;
};

#endif
