#include "dataset.h"
#include "io.h"
#include <utility>
#include <vector>
#include <filesystem>
#include <format>
#include <random>
#include <utils.h>

namespace py = pybind11;
namespace fs = std::filesystem;

Head::Head(
    const FileType _filesType,
    std::string _dictName,
    std::vector<int> _shape
) : filesType(_filesType),
    dictName(std::move(_dictName)), shape(std::move(_shape)) {
    for (const int dim: shape) {
        if (dim <= 0) {
            throw std::invalid_argument(
                "Dimensions need to be strictly positive.");
        }
    }

    if (shape.empty()) {
        throw std::invalid_argument(
            "Tensors have at least 1 dimension.");
    }

    switch (filesType) {
        case FileType::JPG: {
            if (shape.size() != 3) {
                throw std::invalid_argument(
                    "Jpeg images have shape (h, w, 3).");
            }

            if (shape[2] != 3) {
                throw std::invalid_argument(
                    "Jpeg images must have RGB channels.");
            }
        }
        break;
        default: break; // TODO
        // throw std::runtime_error(
        //     "File types other than jpg and npy are not supported.");
    }
}

std::string Head::getExt() const {
    switch (filesType) {
        case FileType::EXR: return "exr";
        case FileType::JPG: return "jpg";
        case FileType::NPY: return "npy";
        default: return "";
    }
}

std::string Head::getDictName() const {
    return dictName;
}

const std::vector<int> &Head::getShape() const {
    return shape;
}

[[nodiscard]] size_t Head::getShapeSize() const {
    size_t totalSize = 1;
    for (const int dim: shape) {
        totalSize *= dim;
    }

    return totalSize;
}

[[nodiscard]] FileType Head::getFilesType() const {
    return filesType;
}

std::string replaceAll(std::string str, const std::string &from,
                       const std::string &to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str = str.replace(pos, from.size(), to);
    }
    return str;
}

void initRootDir(std::string &rootDir) {
    if (rootDir.empty()) {
        throw std::runtime_error("Cannot instantiate a dataset with an empty root directory.");
    }

    if (rootDir.ends_with('/')) {
        rootDir.erase(rootDir.end() - 1);
    }
}

Dataset::Dataset(std::string _rootDir, std::vector<Head> _heads,
                 std::vector<std::string> _subDirs,
                 const pybind11::function &createDatasetFunction,
                 const bool isVirtualDataset
) : rootDir(std::move(_rootDir)), heads(std::move(_heads)), subDirs(std::move(_subDirs)), entries({}), offset(0) {
    initRootDir(rootDir);

    if (subDirs.empty()) {
        throw std::runtime_error(
            "Cannot instantiate a dataset with not subdirectories.");
    }

    if (!fs::exists(rootDir) || (existsEnvVar(INVALID_DS_ENV_VAR) && isVirtualDataset)) {
        fs::remove_all(rootDir);
        createDatasetFunction();
    }

    auto files = listAllFiles(
        std::format("{}/{}", rootDir, subDirs[0]) // TODO: Convention is no / in concat
    );

    for (const auto &file: files) {
        auto &h0 = heads[0];
        auto &s0 = subDirs[0];

        std::vector paths = {file};
        bool erroneousEntry = false;

        if (!file.ends_with(h0.getExt())) {
            debugLog(
                "Got erroneous dataset with anchor path '%s' that does not end on '%s'!\n",
                file.c_str(), s0.getExt().c_str());
            continue;
        }

        for (size_t s = 1; s < subDirs.size(); s++) {
            auto &hS = heads[s];
            auto &sS = subDirs[s];

            std::string newFile(file);
            newFile = replaceAll(newFile, s0, sS);
            newFile = replaceAll(newFile, h0.getExt(), hS.getExt());

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
            entries.push_back(std::move(paths));
        }
    }

    init();
}

Dataset::Dataset(std::string _rootDir, std::vector<Head> _heads,
                 std::vector<std::vector<std::string> > _entries
) : rootDir(std::move(_rootDir)), heads(std::move(_heads)), subDirs({}), entries(std::move(_entries)), offset(0) {
    initRootDir(rootDir);

    if (entries.empty()) {
        throw std::runtime_error(
            "Cannot instantiate a dataset with an empty list of entries.");
    }

    const size_t lastSize = entries[0].size();
    for (size_t i = 1; i < entries.size(); i++) {
        if (entries[i].size() != lastSize) {
            throw std::runtime_error("Entries are not of consistent size.");
        }
    }

    init();
}

Dataset::Dataset(const Dataset &other)
    : rootDir(other.rootDir), heads(other.heads),
      subDirs(other.subDirs), entries(other.entries),
      offset(other.offset.load()) {
}

void Dataset::init() {
    // Remove root directory, if necessary.
    for (auto &item: entries) {
        for (auto &subPath: item) {
            if (subPath.size() >= rootDir.size()) {
                if (std::memcmp(subPath.data(), rootDir.data(), rootDir.size()) == 0) {
                    subPath.erase(0, rootDir.size());
                }

                if (const std::string path = std::format("{}{}", rootDir, subPath); !fs::exists(path)) {
                    throw std::runtime_error(
                        std::format("Path does not exist: '{}'.", path));
                }
            }
        }
    }

    auto rnd = std::default_random_engine{0};
    std::ranges::shuffle(entries, rnd);
}

std::tuple<Dataset, Dataset, Dataset> Dataset::splitTrainValidationTest(
    const float trainPercentage, const float validPercentage) {
    if (trainPercentage <= 0.0f || validPercentage <= 0.0f) {
        throw std::runtime_error(
            "Train and validation set must contain more than 0% of elements.");
    }

    const int numTrain = static_cast<int>(std::round(trainPercentage * entries.size()));
    const int numValid = static_cast<int>(std::round(validPercentage * entries.size()));

    if (numTrain + numValid > static_cast<int>(entries.size())) {
        throw std::runtime_error(
            "Violated #train examples + #validation examples <= #all examples.");
    }

    const std::vector trainEntries(entries.begin(), entries.begin() + numTrain);
    const std::vector validEntries(entries.begin() + numTrain, entries.begin() + numTrain + numValid);
    const std::vector testEntries(entries.begin() + numTrain + numValid, entries.end());

    return std::make_tuple<>(
        Dataset(rootDir + "", std::vector(heads), trainEntries),
        Dataset(rootDir + "", std::vector(heads), validEntries),
        Dataset(rootDir + "", std::vector(heads), testEntries)
    );
}

DatasetBatch Dataset::getNextBatchI(const size_t batchSize) {
    std::vector<std::vector<std::string> > batchPaths;

    const uint32_t lastOffset = offset.fetch_add(static_cast<int>(batchSize));
    for (size_t i = lastOffset; i < lastOffset + batchSize; i++) {
        std::vector<std::string> entry;

        for (auto subPath: entries[i % entries.size()]) {
            entry.push_back(std::format("{}{}", rootDir, subPath));
        }

        batchPaths.push_back(std::move(entry));
    }

    return {
        .startingOffset = static_cast<int32_t>(lastOffset),
        .genIdx = genIdx.load(),
        .batchPaths = std::move(batchPaths)
    };
}

std::vector<std::vector<std::string> > Dataset::getNextBatch(const size_t batchSize) {
    return getNextBatchI(batchSize).batchPaths;
}

void Dataset::resetByNumBatches(const size_t numBatches, const size_t batchSize) {
    offset -= static_cast<int>(numBatches * batchSize);
    if (offset < 0) {
        throw std::runtime_error("Offset cannot be negative.");
    }
    genIdx += 1;
}

std::string Dataset::getRootDir() const {
    return rootDir;
}

std::vector<Head> Dataset::getHeads() const {
    return heads;
}

std::vector<std::vector<std::string> > Dataset::getEntries() const {
    return entries;
}

int32_t Dataset::getOffset() const {
    return offset.load();
}

const std::atomic_uint32_t &Dataset::getGenIdx() const {
    return genIdx;
}
