#include "FlatDataSource.h"
#include <algorithm>
#include <random>
#include <algorithm>
#include <ranges>

#include "dataDecoders/DecoderRegister.h"

namespace fs = std::filesystem;

static std::vector<std::string> listAllFiles(const std::string &directoryPath) {
    std::vector<std::string> paths;

    for (const std::filesystem::directory_entry &entry:
         std::filesystem::recursive_directory_iterator(
             directoryPath)) {
        paths.push_back(entry.path());
    }

    return paths;
}

// TODO: There should be a global buffer, with a certain number of prefetched items.
// TODO: We will add this once the flatdatasource is switched to overlapped io.
static uint8_t *loadFileStoopid(const std::string &path, size_t &inputSize) {
    inputSize = 0;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return nullptr;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return nullptr;
    }
    inputSize = ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return nullptr;
    }

    void *buf = malloc(inputSize);
    if (!buf) {
        fclose(f);
        return nullptr;
    }

    if (fread(buf, 1, inputSize, f) != inputSize) {
        free(buf);
        fclose(f);
        return nullptr;
    }

    fclose(f);
    return static_cast<uint8_t *>(buf);
}

static std::vector<ProbeResult> probeAllSubDirs(const std::string &rootDirectory,
                                                const std::vector<std::string> &subDirs) {
    std::vector<ProbeResult> probeResults;
    DecoderRegister &dReg = DecoderRegister::getInstance();

    for (const std::string &subDir: subDirs) {
        bool foundFile = false;
        for (const fs::directory_entry &entry: fs::directory_iterator(std::format("{}/{}", rootDirectory, subDir))) {
            if (entry.is_regular_file()) {
                foundFile = true;
                const std::string ext = entry.path().extension().string().substr(1);
                IDataDecoder *dataDecoder = dReg.getDataDecoderByExtension(ext);
                if (dataDecoder == nullptr) {
                    throw std::runtime_error(std::format("No data decoder registered for extension {}.", ext));
                }

                size_t inputSize;
                uint8_t *inputData = loadFileStoopid(entry.path().string(), inputSize);
                std::printf("path: %s, inputSize: %zu, inputData: %hhu\n", entry.path().c_str(), inputSize, inputData[0]);
                probeResults.push_back(dataDecoder->probeFromMemory(inputData, inputSize));
                free(inputData); // TODO: Delete this once the switch to overlappedio is complete.
                break;
            }
        }

        if (!foundFile) {
            throw std::invalid_argument(std::format("Subdirectory {} is empty.", subDir));
        }
    }

    return probeResults;
}

std::string replaceAll(std::string str, const std::string &from,
                       const std::string &to) {
    if (from == to) {
        return str;
    }

    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str = str.replace(pos, from.size(), to);
    }
    return str;
}

std::string validateRootDirectory(std::string rootDirectory) {
    if (rootDirectory.empty()) {
        throw std::runtime_error("Cannot instantiate a dataset with an empty root directory name.");
    }

    if (rootDirectory.ends_with('/')) {
        rootDirectory.erase(rootDirectory.end() - 1);
    }

    return rootDirectory;
}

SubdirToDictname::SubdirToDictname(std::string subdir, std::string dictname)
    : subdir(std::move(subdir)), dictname(std::move(dictname)) {
}

FlatDataSource::FlatDataSource(std::string _rootDirectory,
                               std::vector<SubdirToDictname> _subdirsToDictNames)
    : rootDirectory(std::move(_rootDirectory)), subdirsToDictNames(std::move(_subdirsToDictNames)),
      initRequired(false) {
    rootDirectory = validateRootDirectory(rootDirectory);
    initRequired = !fs::exists(rootDirectory);
}

FlatDataSource::FlatDataSource(std::string _rootDirectory, std::vector<ItemKey> _itemKeys,
                               std::vector<std::vector<std::string> > _entries)
    : rootDirectory(std::move(_rootDirectory)), itemKeys(std::move(_itemKeys)), entries(std::move(_entries)),
      initRequired(false) {
    rootDirectory = validateRootDirectory(rootDirectory);
}

std::vector<ItemKey> FlatDataSource::getItemKeys() {
    return itemKeys;
}

std::vector<std::vector<std::string> > FlatDataSource::getEntries() {
    return entries;
}

CpuAllocation FlatDataSource::loadItemSliceIntoContigousBatch(BumpAllocator<uint8_t *> alloc,
                                                              const std::vector<std::vector<std::string> > &batchPaths,
                                                              const size_t itemKeysIdx, const uint32_t bufferSize) {
    const auto &itemKey = itemKeys[itemKeysIdx];
    IDataDecoder *decoder = DecoderRegister::getInstance().getDataDecoderByExtension(itemKey.probeResult.extension);
    uint8_t *startOfBuffer = alloc.getCurrent();
    std::vector<Shape> shapes;

    for (const auto &batchPath: batchPaths) {
        const std::string &path = batchPath[itemKeysIdx];

        size_t inputSize;
        uint8_t *inputData = loadFileStoopid(path, inputSize);
        auto [_, shape] = decoder->loadFromMemory(bufferSize, inputData, inputSize, alloc);
        shapes.push_back(shape);
        free(inputData); // TODO: Delete this once the switch to overlappedio is complete.
    }

    return {
        .batchBuffer = {
            .uint8 = startOfBuffer
        },
        .shapes = std::move(shapes)
    };
}

bool FlatDataSource::preInitDataset(const bool forceInvalidation) {
    const bool ret = forceInvalidation || initRequired;
    if (ret) {
        fs::remove_all(rootDirectory);
    }

    return ret;
}

void FlatDataSource::initDataset() {
    if (entries.empty()) {
        initDatasetFromRootDirectory();
    }
    verifyDatasetIsConsistent();
}

void FlatDataSource::splitIntoTwoDataSources(const size_t aNumEntries, std::shared_ptr<IDataSource> &dataSourceA,
                                             std::shared_ptr<IDataSource> &dataSourceB) {
    if (aNumEntries > entries.size()) {
        throw std::runtime_error("Data source can only be split into two smaller sources.");
    }

    const std::vector aEntries(entries.begin(), entries.begin() + static_cast<int>(aNumEntries));
    const std::vector bEntries(entries.begin() + static_cast<int>(aNumEntries), entries.end());

    dataSourceA = std::make_shared<FlatDataSource>(rootDirectory, itemKeys, aEntries);
    dataSourceB = std::make_shared<FlatDataSource>(rootDirectory, itemKeys, bEntries);
}

void FlatDataSource::initDatasetFromRootDirectory() {
    std::vector<std::string> subDirs;
    subDirs.reserve(subdirsToDictNames.size());
    for (const auto &[subdir, _]: subdirsToDictNames) {
        subDirs.push_back(subdir);
    }

    if (subDirs.empty()) {
        throw std::runtime_error(
            "Cannot instantiate a dataset with not subdirectories.");
    }

    const auto files = listAllFiles(
        std::format("{}/{}", rootDirectory, subDirs[0]) // TODO: Convention is no / in concat
    );
    const auto probeResults = probeAllSubDirs(rootDirectory, subDirs);

    for (const auto &file: files) {
        auto &e0 = probeResults[0].extension;
        auto &s0 = subDirs[0];

        std::vector paths = {file};
        bool erroneousEntry = false;

        if (!file.ends_with(e0)) {
            LOG_DEBUG(
                "Got erroneous dataset with anchor path '{}' that does not end on '{}'!",
                file.c_str(), e0.c_str());
            continue;
        }

        for (size_t s = 1; s < subDirs.size(); s++) {
            const auto &eS = probeResults[s].extension;
            const auto &sS = subDirs[s];

            std::string newFile(file);
            newFile = replaceAll(newFile, s0, sS);
            newFile = replaceAll(newFile, e0, eS);

            if (!fs::exists(newFile)) {
                LOG_DEBUG("Could not find '{}'", newFile.c_str());
                erroneousEntry = true;
                break;
            }

            paths.push_back(std::move(newFile));
        }

        if (erroneousEntry) {
            LOG_DEBUG("Got erroneous dataset with anchor path '{}'!", file.c_str());
        } else {
            entries.push_back(std::move(paths));
        }
    }

    auto rnd = std::default_random_engine{0};
    std::ranges::shuffle(entries, rnd);

    for (size_t i = 0; i < subDirs.size(); i++) {
        itemKeys.push_back({
            .keyName = subdirsToDictNames[i].dictname,
            .type = ItemType::NONE,
            .probeResult = probeResults[i]
        });
    }
}

void FlatDataSource::verifyDatasetIsConsistent() {
    if (entries.empty()) {
        throw std::runtime_error("Cannot instantiate an empty dataset.");
    }

    for (auto &item: entries) {
        if (item.size() != itemKeys.size()) {
            throw std::runtime_error("All batch items have to have exactly as many entries as the item keys.");
        }

        for (auto &subPath: item) {
            if (subPath.size() >= rootDirectory.size()) {
                if (std::memcmp(subPath.data(), rootDirectory.data(), rootDirectory.size()) == 0) {
                    subPath.erase(0, rootDirectory.size());
                }

                if (const std::string path = std::format("{}{}", rootDirectory, subPath); !fs::exists(path)) {
                    throw std::runtime_error(std::format("Path does not exist: '{}'.", path));
                }
            }
        }
    }
}
