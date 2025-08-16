#include "compression.h"

struct CompressionHeader {
    uint64_t magic = 0x6E61746976656461;
    uint16_t version = 1;
    CompressionOptions options;
};
