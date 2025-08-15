#ifndef JPEG_UTILS_H
#define JPEG_UTILS_H
#include "resource.h"
#include "dataset.h"
#include <vector>
#include <string>

Allocation loadJpgFiles(ContiguousAllocation &allocations,
                        const std::vector<std::vector<std::string> > &batchPaths,
                        const std::vector<Head> &heads, size_t headIdx);

Allocation loadNpyFiles(ContiguousAllocation &allocations,
                        const std::vector<std::vector<std::string> > &batchPaths,
                        const std::vector<Head> &heads, size_t headIdx);

#endif //JPEG_UTILS_H
