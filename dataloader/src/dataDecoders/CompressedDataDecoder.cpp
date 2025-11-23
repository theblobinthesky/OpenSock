#include "CompressedDataDecoder.h"

ProbeResult CompressedDataDecoder::probeFromMemory(uint8_t *inputData, size_t inputSize) {
    return {};
}

uint8_t *CompressedDataDecoder::loadFromMemory(const ProbeResult &settings,
                        uint8_t *inputData, size_t inputSize, BumpAllocator<uint8_t *> &output) {
    return nullptr;
}

std::string CompressedDataDecoder::getExtension() {
    return "compressed";
}