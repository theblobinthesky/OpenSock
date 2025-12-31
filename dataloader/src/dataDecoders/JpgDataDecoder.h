#ifndef VERSION_JPGDATADECODER_H
#define VERSION_JPGDATADECODER_H

#include "dataio.h"

class JpgDataDecoder final : public IDataDecoder {
public:
    ProbeResult probeFromMemory(uint8_t *inputData, size_t inputSize) override;

    DecodingResult loadFromMemory(
        uint32_t bufferSize, uint8_t *inputData, size_t inputSize,
        BumpAllocator<uint8_t *> &output,
        uint8_t *__restrict__ scratch1, uint8_t *__restrict__ scratch2
    ) override;

    std::string getExtension() override;
};


#endif
