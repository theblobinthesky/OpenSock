#include "FlatDataSource.h"

namespace fs = std::filesystem;

/*static std::vector<std::string> listAllDirectories(const std::string &directoryPath) {
    std::vector<std::string> paths;

    for (const fs::directory_entry &entry: fs::directory_iterator(directoryPath)) {
        if (entry.is_directory()) {
            paths.push_back(entry.path());
        }
    }

    return paths;
}*/

FlatDataSource::FlatDataSource(std::string _rootDirectory)
    : rootDirectory(std::move(_rootDirectory)) {
    for (const std::string &key: keys) {
        itemKeys.push_back({
            .keyName = key,
            .spatialHint = SpatialHint::POINTS
        });
    }

    std::vector<Sample> samples;
}

std::vector<ItemKey> FlatDataSource::getItemKeys() {
    return itemKeys;
}

std::vector<Sample> FlatDataSource::getSamples() {
}

void FlatDataSource::loadFile(uint8_t *&data, size_t &size) {
}
