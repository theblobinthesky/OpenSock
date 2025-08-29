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
    NONE,
    RASTER,
    POINTS
};

struct ItemKey {
    std::string keyName;
    SpatialHint spatialHint;
};

struct Sample {
    std::string sampleName;
};

class IDataSource {
public:
    virtual ~IDataSource() = default;

    virtual std::vector<ItemKey> getItemKeys() = 0;

    virtual std::vector<Sample> getSamples() = 0;

    virtual void loadFile(uint8_t *&data, size_t &size) = 0;
};

enum class ItemFormat {
    UINT,
    FLOAT
};

struct ItemSettings {
    ItemFormat format;
    uint32_t numBytes;
    std::vector<uint32_t> shape;

    [[nodiscard]] uint32_t getShapeSize() const {
        uint32_t size = 1;
        for (const uint32_t dim: shape) size *= dim;
        return size;
    }

    // TODO: dito. size_t batchBufferSize;
};

class IDataDecoder {
public:
    virtual ~IDataDecoder() = default;

    virtual ItemSettings probeFromMemory(uint8_t *inputData, size_t inputSize) = 0;

    virtual uint8_t *loadFromMemory(const ItemSettings &settings,
                                    uint8_t *inputData, size_t inputSize, BumpAllocator<uint8_t *> &output) = 0;

    virtual std::string getExtension() = 0;
};

template<size_t D>
class IDataTransformAugmentation {
public:
    virtual ~IDataTransformAugmentation() = default;

    // Returns if the input shape is supported by this augmenter.
    virtual bool augment(const std::vector<size_t> &inputShape,
                         std::vector<size_t> &outputShape,
                         double affine[D][D + 1]) = 0;
};

class IoResources {
    std::unordered_map<std::string, IDataSource> sources;
    std::unordered_map<std::string, IDataDecoder> decoders;
    std::unordered_map<std::string, IDataTransformAugmentation> augmenters;
};


struct DatasetBatch {
    int32_t startingOffset;
    std::vector<std::vector<std::string> > batchPaths;
};

// TODO (acktschually necessary or true [lol]?): The dataset is threadsafe by-default and tracks in-flight batches.
class Dataset {
public:
    Dataset(std::string _rootDir, std::vector<Head> _heads,
            std::vector<std::string> _subDirs,
            const pybind11::function &createDatasetFunction,
            bool isVirtualDataset
    );

    Dataset(std::string _rootDir, std::vector<Head> _heads,
            std::vector<std::vector<std::string> > _entries
    );

    Dataset(const Dataset &other) = default;

    std::tuple<Dataset, Dataset, Dataset> splitTrainValidationTest(float trainPercentage, float validPercentage);

    [[nodiscard]] const std::string &getRootDir() const;

    [[nodiscard]] const std::vector<Head> &getHeads() const;

    [[nodiscard]] const std::vector<std::vector<std::string> > &getEntries() const;

private:
    void init();

    std::string rootDir;
    std::vector<Head> heads;
    std::vector<std::string> subDirs;
    std::vector<std::vector<std::string> > entries;
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
