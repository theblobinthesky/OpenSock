#ifndef JPEG_UTILS_H
#define JPEG_UTILS_H
#include "resource.h"
#include "datasource.h"
#include <vector>
#include <string>

struct CpuAllocation {
    size_t shapeSize;
    size_t batchBufferSize;

    union {
        uint8_t *uint8;
        float *float32;
    } batchBuffer;
};

CpuAllocation loadFilesFromHeadIntoContigousBatch(BumpAllocator<uint8_t *> &cpuAllocator,
                   const std::vector<std::vector<std::string> > &batchPaths,
                   const std::vector<Head> &heads, size_t headIdx);

#endif //JPEG_UTILS_H
