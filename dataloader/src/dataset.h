#ifndef DATASET_H
#define DATASET_H
#include <atomic>
#include <vector>
#include <filesystem>
#include <format>
#include <pybind11/stl.h>

enum class FileType {
    JPG,
    EXR,
    NPY
};

class Head {
public:
    Head(FileType _filesType, std::string _dictName, std::vector<int> _shape);

    [[nodiscard]] std::string getExt() const;

    [[nodiscard]] std::string getDictName() const;

    [[nodiscard]] const std::vector<int> &getShape() const;

    [[nodiscard]] size_t getShapeSize() const;

    [[nodiscard]] FileType getFilesType() const;

private:
    FileType filesType;
    std::string dictName;
    std::vector<int> shape;
};

#define IMAGE_HEIGHT(subDir) static_cast<size_t>(subDir.getShape()[0])
#define IMAGE_WIDTH(subDir) static_cast<size_t>(subDir.getShape()[1])

struct DatasetBatch {
    int32_t startingOffset;
    uint32_t genIdx;
    std::vector<std::vector<std::string>> batchPaths;
};

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

    Dataset(const Dataset &other);

    std::tuple<Dataset, Dataset, Dataset> splitTrainValidationTest(float trainPercentage, float validPercentage);

    [[nodiscard]] DatasetBatch getNextBatchI(size_t batchSize);
    [[nodiscard]] std::vector<std::vector<std::string>> getNextBatch(size_t batchSize);

    void resetByNumBatches(size_t numBatches, size_t batchSize);

    [[nodiscard]] std::string getRootDir() const;

    [[nodiscard]] std::vector<Head> getHeads() const;

    [[nodiscard]] std::vector<std::vector<std::string> > getEntries() const;

    [[nodiscard]] int32_t getOffset() const;
    [[nodiscard]] const std::atomic_uint32_t &getGenIdx() const;

private:
    void init();

    std::string rootDir;
    std::vector<Head> heads;
    std::vector<std::string> subDirs;
    std::vector<std::vector<std::string> > entries;
    std::atomic_int32_t offset;
    std::atomic_uint32_t genIdx;
};

#endif //DATASET_H
