#include "dataset.h"
#include "io.h"
#include <vector>
#include <filesystem>
#include <format>
#include <jpeglib.h>
#include <random>

namespace py = pybind11;
namespace fs = std::filesystem;

Subdirectory::Subdirectory(
    std::string _subdir,
    const FileType _filesType,
    std::string _dictName,
    std::vector<int> _shape
) : subdir(std::move(_subdir)), filesType(_filesType),
    dictName(std::move(_dictName)), shape(std::move(_shape)) {
    for (const int dim: shape) {
        if (dim <= 0) {
            throw std::invalid_argument(
                "Dimensions need to be strictly positive.");
        }
    }

    if (filesType == FileType::JPG) {
        if (shape.size() != 3) {
            throw std::invalid_argument("Jpeg images have shape (h, w, 3).");
        }

        if (shape[2] != 3) {
            throw std::invalid_argument("Jpeg images must have RGB channels.");
        }
    } else if (filesType == FileType::NPY) {
        if (shape.empty()) {
            throw std::invalid_argument(
                "Jpeg images have at least 1 dimension.");
        }
    } else {
        // throw std::runtime_error(
        //     "File types other than jpg and npy are not supported.");
    }
}

std::string Subdirectory::getPath(const std::string &root) const {
    return std::format("{}/{}", root, subdir);
}

std::string Subdirectory::getExt() const {
    switch (filesType) {
        case FileType::EXR: return "exr";
        case FileType::JPG: return "jpg";
        case FileType::NPY: return "npy";
        default: return "";
    }
}

std::string Subdirectory::getSubdir() const {
    return subdir;
}

std::string Subdirectory::getDictName() const {
    return dictName;
}

std::vector<int> Subdirectory::getShape() const {
    return shape;
}

[[nodiscard]] size_t Subdirectory::getShapeSize() const {
    size_t totalSize = 1;
    for (const int dim: shape) {
        totalSize *= dim;
    }

    return totalSize;
}

[[nodiscard]] FileType Subdirectory::getFilesType() const {
    return filesType;}

std::string replaceAll(std::string str, const std::string &from,
                       const std::string &to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str = str.replace(pos, from.size(), to);
    }
    return str;
}

Dataset::Dataset(
    std::string _rootDir,
    std::vector<Subdirectory> _subDirs
) : rootDir(std::move(_rootDir)), subDirs(std::move(_subDirs)), offset{} {
    if (rootDir.empty()) {
        throw std::runtime_error(
            "Cannot instantiate a dataset with an empty root directory.");
    }
    if (subDirs.empty()) {
        throw std::runtime_error(
            "Cannot instantiate a dataset with not subdirectories.");
    }
}

void Dataset::init() {
    auto files = listAllFiles(subDirs[0].getPath(rootDir));
    for (const auto &file: files) {
        auto &s0 = subDirs[0];
        std::vector paths = {file};
        bool erroneousEntry = false;

        if (!file.ends_with(s0.getExt())) {
            debugLog(
                "Got erroneous dataset with anchor path '%s' that does not end on '%s'!\n",
                file.c_str(), s0.getExt().c_str());
            continue;
        }

        for (size_t s = 1; s < subDirs.size(); s++) {
            auto &sS = subDirs[s];
            std::string newFile(file);
            newFile = replaceAll(newFile, s0.getSubdir(), sS.getSubdir());
            newFile = replaceAll(newFile, s0.getExt(), sS.getExt());

            if (!fs::exists(newFile)) {
                debugLog("Could not find '%s'\n", newFile.c_str());
                erroneousEntry = true;
                break;
            }

            paths.push_back(std::move(newFile));
        }

        if (erroneousEntry) {
            debugLog("Got erroneous dataset with anchor path '%s'!\n",
                     file.c_str());
        } else {
            dataset.push_back(std::move(paths));
        }
    }

    auto rnd = std::default_random_engine{0};
    std::ranges::shuffle(files, rnd);
}

void Dataset::splitTrainValidationTest() {
}

std::vector<std::vector<std::string> > Dataset::getNextBatch(
    const size_t batchSize) const {
    std::vector<std::vector<std::string> > batch;
    for (size_t i = offset; i < offset + batchSize; i++) {
        batch.push_back(dataset[i % dataset.size()]);
    }

    return batch;
}

std::vector<std::vector<std::string> > Dataset::getDataset() const {
    return dataset;
}

std::string Dataset::getRootDir() const {
    return rootDir;
}

std::vector<Subdirectory> Dataset::getSubDirs() const {
    return subDirs;
}
