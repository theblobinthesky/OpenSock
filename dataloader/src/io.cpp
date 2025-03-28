#include "io.h"
#include <filesystem>

std::vector<std::string> listAllFiles(const std::string &directoryPath) {
    std::vector<std::string> paths;

    for (const std::filesystem::directory_entry &entry:
         std::filesystem::recursive_directory_iterator(
             directoryPath)) {
        paths.push_back(entry.path());
    }

    return paths;
}

/*void create_bulk_request_for_files(const std::vector<std::string> paths) {
}*/
