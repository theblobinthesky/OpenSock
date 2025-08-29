#ifndef VERSION_JPGDATADECODER_H
#define VERSION_JPGDATADECODER_H

#include "dataio.h"

class JpgDataDecoder final : public IDataDecoder {
public:
    ItemSettings probeFromMemory(uint8_t *inputData, size_t inputSize) override;

    uint8_t *loadFromMemory(const ItemSettings &settings,
                            uint8_t *inputData, size_t inputSize, BumpAllocator<uint8_t *> &output) override;

    std::string getExtension() override;
};


#endif
