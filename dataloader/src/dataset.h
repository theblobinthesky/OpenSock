#ifndef DATASET_H
#define DATASET_H
#include <atomic>
#include <vector>
#include <filesystem>
#include <format>
#include <pybind11/stl.h>

enum class DirectoryType {
    UNPACKED_IN_FILES,
    PACKED_INTO_SHARDS
};

enum class FileType {
    JPG,
    PNG,
    EXR,
    NPY,
    COMPRESSED
};

enum class ItemFormat {
    UINT,
    FLOAT
};

class Head {
public:
    Head(FileType _filesType, std::string _dictName, std::vector<int> _shape);

    [[nodiscard]] std::string getExt() const;

    [[nodiscard]] std::string getDictName() const;

    [[nodiscard]] const std::vector<int> &getShape() const;

    [[nodiscard]] size_t getShapeSize() const;

    [[nodiscard]] FileType getFilesType() const;

    [[nodiscard]] ItemFormat getItemFormat() const;

    [[nodiscard]] int32_t getBytesPerItem() const;

private:
    DirectoryType directoryType;
    FileType filesType;
    std::string dictName;
    std::vector<int> shape;
};

#define IMAGE_HEIGHT(subDir) static_cast<size_t>(subDir.getShape()[0])
#define IMAGE_WIDTH(subDir) static_cast<size_t>(subDir.getShape()[1])

struct DatasetBatch {
    int32_t startingOffset;
    uint32_t genIdx;
    std::vector<std::vector<std::string> > batchPaths;
};

// The dataset is threadsafe by-default and tracks in-flight batches.
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

    [[nodiscard]] const std::atomic_uint32_t &getGenIdx() const;

private:
    Dataset dataset;
    size_t batchSize;
    std::mutex mutex;
    std::unordered_set<int32_t> inFlightBatches;
    std::atomic_int32_t currInFlightBatch;
    std::atomic_int32_t lastWaitingBatch;
    std::atomic_uint32_t genIdx;
};

#endif //DATASET_H
