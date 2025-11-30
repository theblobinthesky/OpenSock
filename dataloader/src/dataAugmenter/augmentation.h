#ifndef VERSION_DATAAUGMENTER_H
#define VERSION_DATAAUGMENTER_H
#include <vector>

#include "utils.h"

bool isInputShapeBHWN(const std::vector<size_t> &inputShape, size_t N);

class IDataAugmentation {
public:
    virtual ~IDataAugmentation() = default;

    virtual bool isOutputShapeStatic() = 0;

    virtual std::vector<size_t> getOutputShapeIfSupported(const std::vector<size_t> &inputShape) = 0;

    [[nodiscard]] virtual void *getItemSettings(uint64_t itemSeed) const = 0;

    virtual void freeItemSettings(void *itemSettings) const = 0;

    virtual void augmentWithChannelFirst(
        const std::vector<size_t> &inputShape,
        const std::vector<size_t> &outputShape,
        ItemFormat format,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    ) = 0;

    virtual void augmentWithChannelLast(
        const std::vector<size_t> &inputShape,
        const std::vector<size_t> &outputShape,
        ItemFormat format,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    ) = 0;
};

class DataAugmentationPipe {
public:
    explicit DataAugmentationPipe(std::vector<IDataAugmentation *> dataAugmentations);

    [[nodiscard]] bool isOutputShapeStatic() const;

    [[nodiscard]] std::vector<size_t> getOutputShapeIfSupported(const std::vector<size_t> &inputShape) const;

    [[nodiscard]] void *getItemSettings(uint64_t itemSeed) const;

    void freeItemSettings(void *itemSettings) const;

    void augmentWithChannelFirst(
        const std::vector<size_t> &inputShape,
        const std::vector<size_t> &outputShape,
        ItemFormat format,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    );

    void augmentWithChannelLast(
        const std::vector<size_t> &inputShape,
        const std::vector<size_t> &outputShape,
        ItemFormat format,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    );

private:
    std::vector<IDataAugmentation *> dataAugmentations;
};

#endif
