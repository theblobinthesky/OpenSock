#include "CompressedDataDecoder.h"

ProbeResult CompressedDataDecoder::probeFromMemory(uint8_t *inputData, size_t inputSize) {
    return {};
}

DecodingResult CompressedDataDecoder::loadFromMemory(uint32_t bufferSize, uint8_t *inputData, size_t inputSize,
                              BumpAllocator<uint8_t *> &output) {
    return {};
}

std::string CompressedDataDecoder::getExtension() {
    return "compressed";
}