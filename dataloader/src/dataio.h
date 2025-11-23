#ifndef DATAIO_H
#define DATAIO_H
#include <atomic>
#include <vector>
#include <filesystem>
#include <format>
#include <pybind11/stl.h>

#include "utils.h"

struct CpuAllocation {
    union {
        uint8_t *uint8;
        float *float32;
    } batchBuffer;
};

enum class SpatialHint : uint8_t {
    XXX,
    C_XXX,
    XXX_C
};

enum class ItemFormat {
    UINT,
    FLOAT
};

struct ProbeResult {
    ItemFormat format;
    uint32_t bytesPerItem;
    std::vector<uint32_t> shape;
    std::string extension;

    [[nodiscard]] uint32_t getShapeSize() const {
        uint32_t size = 1;
        for (const uint32_t dim: shape) size *= dim;
        return size;
    }

    [[nodiscard]] uint32_t getBufferSize() const {
        return bytesPerItem * getShapeSize();
    }
};

struct ItemKey {
    std::string keyName;
    SpatialHint spatialHint;
    ProbeResult probeResult;
};

// These methods are not thread-safe.
class IDataSource {
public:
    virtual ~IDataSource() = default;

    virtual std::vector<ItemKey> getItemKeys() = 0;

    virtual std::vector<std::vector<std::string> > getEntries() = 0;

    virtual CpuAllocation loadItemSliceIntoContigousBatch(BumpAllocator<uint8_t *> alloc,
                                                      const std::vector<std::vector<std::string> > &batchPaths,
                                                      size_t itemKeysIdx) = 0;

    virtual bool preInitDataset(bool forceInvalidation) = 0;

    virtual void initDataset() = 0;

    virtual IDataSource *splitIntoTwoDatasetsAB(size_t aNumEntries) = 0;
};

class IDataDecoder {
public:
    virtual ~IDataDecoder() = default;

    virtual ProbeResult probeFromMemory(uint8_t *inputData, size_t inputSize) = 0;

    virtual uint8_t *loadFromMemory(const ProbeResult &settings,
                                    uint8_t *inputData, size_t inputSize, BumpAllocator<uint8_t *> &output) = 0;

    virtual std::string getExtension() = 0;
};

template<size_t D>
class IDataTransformAugmentation {
public:
    virtual ~IDataTransformAugmentation() = default;

    // Returns true if the input shape is supported by this augmentation.
    virtual bool augment(const std::vector<size_t> &inputShape,
                         std::vector<size_t> &outputShape,
                         double affine[D][D + 1]) = 0;
};

class IoResources {
    std::unordered_map<std::string, IDataSource> sources;
    std::unordered_map<std::string, IDataDecoder> decoders;
    std::unordered_map<std::string, IDataTransformAugmentation<2> > augmenters;
};

struct DatasetBatch {
    int32_t startingOffset;
    std::vector<std::vector<std::string> > batchPaths;
};

// TODO (acktschually necessary or true [lol]?): The dataset is threadsafe by-default and tracks in-flight batches.
class Dataset {
public:
    Dataset(IDataSource *_dataSource,
            std::vector<IDataTransformAugmentation<2> *> _dataAugmentations,

            const pybind11::function &createDatasetFunction,
            bool isVirtualDataset
    );

    Dataset(IDataSource *_dataSource, std::vector<IDataTransformAugmentation<2> *> _dataAugmentations);

    Dataset(const Dataset &other) = default;

    [[nodiscard]] std::tuple<Dataset, Dataset, Dataset> splitTrainValidationTest(float trainPercentage, float validPercentage) const;

    [[nodiscard]] IDataSource *getDataSource() const;

    [[nodiscard]] std::vector<IDataTransformAugmentation<2> *> getDataTransformAugmentations() const;

private:
    IDataSource *dataSource;
    std::vector<IDataTransformAugmentation<2> *> dataAugmentations;
};

class BatchedDataset {
public:
    BatchedDataset(const Dataset &dataset, size_t batchSize);

    BatchedDataset(const Dataset &&dataset, size_t batchSize);

    [[nodiscard]] DatasetBatch getNextInFlightBatch();

    [[nodiscard]] std::vector<std::vector<std::string> > getNextBatch();

    void markBatchWaiting(int32_t batch);

    void popWaitingBatch(int32_t batch);

    void forgetInFlightBatches();

    [[nodiscard]] const Dataset &getDataset() const noexcept;

    [[nodiscard]] const std::atomic_int32_t &getLastWaitingBatch() const;

    [[nodiscard]] size_t getNumberOfBatches() const;

private:
    Dataset dataset;
    size_t batchSize;
    std::mutex mutex;
    std::unordered_set<int32_t> inFlightBatches;
    std::atomic_int32_t currInFlightBatch;
    std::atomic_int32_t lastWaitingBatch;
};

#endif
