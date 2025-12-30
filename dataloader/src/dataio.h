#ifndef DATAIO_H
#define DATAIO_H
#include <atomic>
#include <vector>
#include <filesystem>
#include <format>
#include "pybind11_includes.h"

#include "utils.h"
#include "dataAugmenter/augmentation.h"

struct CpuAllocation {
    union {
        uint8_t *uint8;
        float *float32;
    } batchBuffer;

    Shapes shapes;
};

struct ProbeResult {
    DType dtype;
    Shape shape;
    std::string extension;
};

enum class ItemType : uint8_t {
    NONE,
    RASTER,
    POINTS
};

constexpr auto PointsLengthsTensorDType = DType::INT32;

struct ItemKey {
    std::string keyName;
    ItemType type;
    ProbeResult probeResult;
};

// These methods are not thread-safe.
class IDataSource {
public:
    virtual ~IDataSource() = default;

    // An item key with type POINTS will always refer back to the last raster.
    virtual std::vector<ItemKey> getItemKeys() = 0;

    virtual std::vector<std::vector<std::string> > getEntries() = 0;

    virtual CpuAllocation loadItemSliceIntoContigousBatch(BumpAllocator<uint8_t *> &alloc,
                                                          const std::vector<std::vector<std::string> > &batchPaths,
                                                          size_t itemKeysIdx, uint32_t bufferSize) = 0;

    virtual bool preInitDataset(bool forceInvalidation) = 0;

    virtual void initDataset() = 0;

    // Do *NOT* pass the data source itself as a parameter.
    virtual void splitIntoTwoDataSources(size_t aNumEntries, std::shared_ptr<IDataSource> &dataSourceA,
                                         std::shared_ptr<IDataSource> &dataSourceB) = 0;
};

struct DecodingResult {
    uint8_t *data;
    Shape shape;
};

class IDataDecoder {
public:
    virtual ~IDataDecoder() = default;

    virtual ProbeResult probeFromMemory(uint8_t *inputData, size_t inputSize) = 0;

    virtual DecodingResult loadFromMemory(uint32_t bufferSize, uint8_t *inputData, size_t inputSize,
                                          BumpAllocator<uint8_t *> &output) = 0;

    virtual std::string getExtension() = 0;
};

class IoResources {
    std::unordered_map<std::string, IDataSource> sources;
    std::unordered_map<std::string, IDataDecoder> decoders;
    std::unordered_map<std::string, IDataAugmentation> augmenters;
};

struct DatasetBatch {
    int32_t startingOffset;
    std::vector<std::vector<std::string> > batchPaths;
};

// TODO (acktschually necessary or true [lol]?): The dataset is threadsafe by-default and tracks in-flight batches.
class Dataset {
public:
    Dataset(std::shared_ptr<IDataSource> _dataSource,
            std::vector<std::shared_ptr<IDataAugmentation> > _dataAugmentations,
            const pybind11::function &createDatasetFunction, bool isVirtualDataset
    );

    Dataset(std::shared_ptr<IDataSource> _dataSource,
            std::vector<std::shared_ptr<IDataAugmentation> > _dataAugmentations);

    Dataset(const Dataset &other) = default;

    [[nodiscard]] std::tuple<std::shared_ptr<Dataset>, std::shared_ptr<Dataset>, std::shared_ptr<Dataset> >
    splitTrainValidationTest(float trainPercentage, float validPercentage) const;

    [[nodiscard]] std::shared_ptr<IDataSource> getDataSource() const;

    [[nodiscard]] std::vector<std::shared_ptr<IDataAugmentation> > getDataAugmentations() const;

private:
    std::shared_ptr<IDataSource> dataSource;
    std::vector<std::shared_ptr<IDataAugmentation> > dataAugmentations;
};

class BatchedDataset {
public:
    BatchedDataset(const std::shared_ptr<Dataset> &dataset, size_t batchSize);

    [[nodiscard]] DatasetBatch getNextInFlightBatch();

    [[nodiscard]] std::vector<std::vector<std::string> > getNextBatch();

    void markBatchWaiting(int32_t batch);

    void popWaitingBatch(int32_t batch);

    void forgetInFlightBatches();

    [[nodiscard]] const std::shared_ptr<Dataset> getDataset() const noexcept;

    [[nodiscard]] const std::atomic_int32_t &getLastWaitingBatch() const;

    [[nodiscard]] size_t getNumberOfBatches() const;

private:
    std::shared_ptr<Dataset> dataset;
    size_t batchSize;
    std::mutex mutex;
    std::unordered_set<int32_t> inFlightBatches;
    std::atomic_int32_t currInFlightBatch;
    std::atomic_int32_t lastWaitingBatch;
};

#endif
