#ifndef VERSION_EXRDATADECODER_H
#define VERSION_EXRDATADECODER_H

#include "dataio.h"

class ExrDataDecoder final : public IDataDecoder {
public:
    ProbeResult probeFromMemory(uint8_t *inputData, size_t inputSize) override;

    uint8_t *loadFromMemory(const ProbeResult &settings,
                            uint8_t *inputData, size_t inputSize, BumpAllocator<uint8_t *> &output) override;

    std::string getExtension() override;
};


#endif
