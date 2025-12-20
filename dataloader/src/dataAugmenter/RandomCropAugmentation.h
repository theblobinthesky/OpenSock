#ifndef VERSION_RANDOMRESIZEDCROP_H
#define VERSION_RANDOMRESIZEDCROP_H

#include "dataAugmenter/augmentation.h"

struct RandomCropSettings {
    size_t left;
    size_t top;
};

class RandomCropAugmentation final : public IDataAugmentation {
public:
    RandomCropAugmentation(
        size_t cropHeight, size_t cropWidth
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
    size_t cropHeight, cropWidth;
};

#endif
