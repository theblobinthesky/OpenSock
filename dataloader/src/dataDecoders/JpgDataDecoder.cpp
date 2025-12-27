#include "JpgDataDecoder.h"
#include <jpeglib.h>

void readJpg(const uint8_t *inputData, const size_t inputSize,
             jpeg_decompress_struct &comprInfo, uint8_t *outputData) {
    jpeg_error_mgr err{};

    comprInfo = {};
    comprInfo.err = jpeg_std_error(&err);
    comprInfo.out_color_space = JCS_EXT_RGB;

    jpeg_create_decompress(&comprInfo);
    jpeg_mem_src(&comprInfo, inputData, inputSize);
    if (jpeg_read_header(&comprInfo, TRUE) != JPEG_HEADER_OK) {
        throw std::runtime_error("Jpg file header is corrupted.");
    }

    if (comprInfo.data_precision == 12) {
        jpeg_destroy_decompress(&comprInfo);
        throw std::runtime_error("Jpg library is not compiled with 12-bit depth support.");
    }

    if (outputData == nullptr) {
        return;
    }

    if (!jpeg_start_decompress(&comprInfo)) {
        jpeg_destroy_decompress(&comprInfo);
        throw std::runtime_error("Jpg file could not start decompression.");
    }

    size_t row_stride = static_cast<size_t>(comprInfo.output_width) * comprInfo.output_components;

    // Allocate a one-row-high sample array.
    while (comprInfo.output_scanline < comprInfo.output_height) {
        JSAMPROW rowPtr = outputData + row_stride * comprInfo.output_scanline;
        jpeg_read_scanlines(&comprInfo, &rowPtr, 1);
    }

    jpeg_finish_decompress(&comprInfo);
    jpeg_destroy_decompress(&comprInfo);
}

ProbeResult JpgDataDecoder::probeFromMemory(uint8_t *inputData, const size_t inputSize) {
    jpeg_decompress_struct comprInfo = {};
    readJpg(inputData, inputSize, comprInfo, nullptr);

    if (comprInfo.image_height == 0 || comprInfo.image_width == 0) {
        throw std::runtime_error("Jpg output dimensions have to be nonzero.");
    }

    return {
        .dtype = DType::UINT8,
        .shape = std::vector<uint32_t>{comprInfo.image_height, comprInfo.image_width, 3},
        .extension = "jpg"
    };
}

DecodingResult JpgDataDecoder::loadFromMemory(const uint32_t bufferSize, uint8_t *inputData, const size_t inputSize,
                                              BumpAllocator<uint8_t *> &output) {
    uint8_t *outputData = output.allocate(bufferSize);
    jpeg_decompress_struct comprInfo = {};
    readJpg(inputData, inputSize, comprInfo, outputData);

    return {
        .data = outputData,
        .shape = {comprInfo.output_height, comprInfo.output_width, 3}
    };
}

std::string JpgDataDecoder::getExtension() {
    return "jpg";
}
