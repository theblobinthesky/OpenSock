#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <filesystem>
#include <format>
#include <random>
#include <pybind11/stl.h>
#include "utils.h"

constexpr const char *INVALID_DS_ENV_VAR = "INVALIDATE_DATASET";

enum class FileType {
    JPG,
    EXR,
    NPY
};

class Subdirectory {
public:
    Subdirectory(std::string _subdir, FileType _filesType,
                 std::string _dictName, std::vector<int> _shape);

    [[nodiscard]] std::string getPath(const std::string &root) const;

    [[nodiscard]] std::string getExt() const;

    [[nodiscard]] std::string getSubdir() const;

    [[nodiscard]] std::string getDictName() const;

    [[nodiscard]] std::vector<int> getShape() const;

    [[nodiscard]] size_t getShapeSize() const;

    [[nodiscard]] FileType getFilesType() const;

private:
    std::string subdir;
    FileType filesType;
    std::string dictName;
    std::vector<int> shape;
};

#define IMAGE_HEIGHT(subDir) static_cast<size_t>(subDir.getShape()[0])
#define IMAGE_WIDTH(subDir) static_cast<size_t>(subDir.getShape()[1])

class Dataset {
public:
    Dataset(std::string _rootDir, std::vector<Subdirectory> _subDirs);

    void init();

    void splitTrainValidationTest();

    [[nodiscard]] std::vector<std::vector<std::string> > getNextBatch(
        const size_t batchSize) const;

    [[nodiscard]] std::vector<std::vector<std::string> > getDataset() const;

    [[nodiscard]] std::string getRootDir() const;

    [[nodiscard]] std::vector<Subdirectory> getSubDirs() const;

private:
    std::string rootDir;
    std::vector<Subdirectory> subDirs;
    std::vector<std::vector<std::string> > dataset;
    size_t offset;
};

#endif //DATASET_H
