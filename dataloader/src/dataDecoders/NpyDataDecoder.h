#ifndef VERSION_NPYDATADECODER_H
#define VERSION_NPYDATADECODER_H

#include "dataio.h"

class NpyDataDecoder final : public IDataDecoder {
public:
    ProbeResult probeFromMemory(uint8_t *inputData, size_t inputSize) override;

    DecodingResult loadFromMemory(uint32_t bufferSize, uint8_t *inputData, size_t inputSize,
                                  BumpAllocator<uint8_t *> &output) override;

    std::string getExtension() override;
};

#endif
