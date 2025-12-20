#ifndef VERSION_DATAAUGMENTER_H
#define VERSION_DATAAUGMENTER_H

#include <vector>
#include "utils.h"

bool isInputShapeBHWN(const std::vector<uint32_t> &inputShape, size_t N);

struct DataOutputSchema {
    std::vector<uint32_t> outputShape; // Empty, if input shape is not supported.
    void *itemSettings;
};

class IDataAugmentation {
public:
    virtual ~IDataAugmentation() = default;

    virtual bool isOutputShapeStaticExceptForBatch() = 0;

    virtual DataOutputSchema getDataOutputSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) = 0;

    // TODO: I don't love the manual free api here, but we'll go with it for now.
    virtual void freeItemSettings(void *itemSettings) const = 0;

    virtual std::vector<uint32_t> getMaximumOutputShapeGivenInputShapeIfSupported(const std::vector<uint32_t> &inputShape) = 0;

    // TODO: Make sure we use int rather than uint for the points....
    // Returns false if computation was skipped.
    virtual bool augmentWithPoints(
        const std::vector<uint32_t> &shape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void * itemSettings
    ) = 0;

    // Returns false if computation was skipped.
    virtual bool augmentWithRaster(
        const std::vector<uint32_t> &inputShape,
        const std::vector<uint32_t> &outputShape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        void *itemSettings
    ) = 0;
};

struct DataProcessingSchema {
    std::vector<uint32_t> outputShape;
    std::vector<void *> itemSettingsLists;
};

// Usage: Initialize, ask for maximium required buffer size, pass buffers, use.
class DataAugmentationPipe {
public:
    explicit DataAugmentationPipe(
        std::vector<IDataAugmentation *> dataAugmentations,
        const std::vector<uint32_t>& maxInputShape,
        uint32_t maxBytesPerElement
    );

    [[nodiscard]] size_t getMaximumRequiredBufferSize() const;

    void setBuffer(uint8_t *_buffer1, uint8_t *_buffer2);

    [[nodiscard]] DataProcessingSchema getProcessingSchema(const std::vector<uint32_t> &inputShape, uint64_t itemSeed) const;

    // TODO: I don't love the manual free api here, but we'll go with it for now.
    void freeProcessingSchema(const DataProcessingSchema &processingSchema) const;

    void augmentWithPoints(
        const std::vector<uint32_t> &shape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        const std::vector<void *> &itemSettingsList
    ) const;

    void augmentWithRaster(
        const std::vector<uint32_t> &inputShape,
        const std::vector<uint32_t> &outputShape,
        DType dtype,
        const uint8_t *__restrict__ inputData, uint8_t *__restrict__ outputData,
        const std::vector<void *> &itemSettingsList
    ) const;

private:
    std::vector<IDataAugmentation *> dataAugmentations;
    uint8_t *buffer1, *buffer2;
    uint32_t maximumRequiredBufferSize;

    [[nodiscard]] bool isOutputShapeStatic() const;
};

#endif
