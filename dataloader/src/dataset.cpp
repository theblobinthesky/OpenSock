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
    int _imageHeight,
    int _imageWidth
) : subdir(std::move(_subdir)), filesType(_filesType),
    dictName(std::move(_dictName)), imageHeight(_imageHeight),
    imageWidth(_imageWidth) {
    if (imageHeight <= 0 || imageWidth <= 0) {
        throw std::runtime_error(
            "Image dimensions need to be strictly positive.");
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
