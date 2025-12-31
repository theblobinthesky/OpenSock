#include "CompressedDataDecoder.h"
#include "compression.h"

ProbeResult CompressedDataDecoder::probeFromMemory(uint8_t *inputData, const size_t inputSize) {
    const CompressorSettings settings = Decompressor::probeArray(inputData, inputSize);
    const DType dtype = (settings.magic & static_cast<uint64_t>(CompressorFlags::CAST_TO_FP16))
                     ? DType::FLOAT16
                     : DType::FLOAT32;
    return {
        .dtype = dtype,
        .shape = settings.getShape(),
        .extension = "compressed"
    };
}

DecodingResult CompressedDataDecoder::loadFromMemory(
    const uint32_t bufferSize, uint8_t *inputData, const size_t inputSize,
    BumpAllocator<uint8_t *> &output,
    uint8_t *__restrict__ scratch1, uint8_t *__restrict__ scratch2
) {
    uint8_t *outputData = output.allocate(bufferSize);
    const CompressorSettings settings = Decompressor::probeArray(inputData, inputSize);
    Decompressor::decompressArray(
        inputData, scratch1, scratch2, outputData,
        settings
    );

    return {
        .data = outputData,
        .shape = settings.getShape()
    };
}

size_t CompressedDataDecoder::getRequiredRawFileBufferSize(const Shape &maxInputShape) const {
    return Decompressor::getMaximumRequiredRawFileBufferSize(maxInputShape);
}

size_t CompressedDataDecoder::getRequiredScratchBufferSize(const Shape &maxInputShape) const {
    return Decompressor::getMaximumRequiredScratchBufferSize(maxInputShape);
}

std::string CompressedDataDecoder::getExtension() {
    return "compressed";
}
