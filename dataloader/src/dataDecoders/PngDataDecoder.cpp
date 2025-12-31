#include "PngDataDecoder.h"
#include <png.h>
#include <stdexcept>
#include <cstring>
#include <vector>

struct PngMemReader {
    const uint8_t *data;
    size_t size;
    size_t offset;
};

void pngReadFromMemory(png_structp pngPtr, png_bytep outBytes, png_size_t bytesToRead) {
    auto *reader = static_cast<PngMemReader *>(png_get_io_ptr(pngPtr));
    if (!reader || reader->offset + bytesToRead > reader->size) {
        png_error(pngPtr, "PNG read beyond buffer");
        return;
    }
    std::memcpy(outBytes, reader->data + reader->offset, bytesToRead);
    reader->offset += bytesToRead;
}

ProbeResult PngDataDecoder::probeFromMemory(uint8_t *inputData, const size_t inputSize) {
    if (inputSize < 8 || png_sig_cmp(inputData, 0, 8)) {
        throw std::runtime_error("Invalid PNG signature");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) throw std::runtime_error("png_create_read_struct failed");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        throw std::runtime_error("PNG probe error");
    }

    PngMemReader reader{inputData, inputSize, 0};
    png_set_read_fn(png, &reader, pngReadFromMemory);

    // libpng expects us to indicate we've not read the signature via png_set_sig_bytes(png, 0)
    // which is default, so we proceed to read info.
    png_read_info(png, info);

    png_uint_32 width, height;
    int bit_depth, color_type;
    png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);

    // Normalize to 8-bit, RGB (no alpha) to match loader output contract.
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (color_type == PNG_COLOR_TYPE_GRAY) png_set_gray_to_rgb(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);

    png_read_update_info(png, info);

    png_destroy_read_struct(&png, &info, nullptr);

    return ProbeResult{
        .dtype = DType::UINT8,
        .shape = std::vector<uint32_t>{height, width, 3},
        .extension = "png"
    };
}

DecodingResult PngDataDecoder::loadFromMemory(
    const uint32_t bufferSize, uint8_t *inputData, const size_t inputSize,
    BumpAllocator<uint8_t *> &output,
    uint8_t *__restrict__ scratch1, uint8_t *__restrict__) {
    if (inputSize < 8 || png_sig_cmp(inputData, 0, 8)) {
        throw std::runtime_error("Invalid PNG signature");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) throw std::runtime_error("png_create_read_struct failed");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        throw std::runtime_error("PNG decode error");
    }

    PngMemReader reader{inputData, inputSize, 0};
    png_set_read_fn(png, &reader, pngReadFromMemory);

    png_read_info(png, info);

    png_uint_32 width, height;
    int bit_depth, color_type;
    png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);

    // Normalize to 8-bit RGB (no alpha)
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (color_type == PNG_COLOR_TYPE_GRAY) png_set_gray_to_rgb(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);

    png_read_update_info(png, info);


    // Set up row pointers directly into output buffer to avoid extra copies
    if (!scratch1) {
        png_destroy_read_struct(&png, &info, nullptr);
        throw std::runtime_error("PNG decoder requires scratch buffer.");
    }

    uint8_t *out = output.allocate(bufferSize);
    auto *row_ptrs = reinterpret_cast<png_bytep *>(scratch1);
    const size_t rowStride = static_cast<size_t>(width) * 3; // RGB8
    for (png_uint_32 y = 0; y < height; ++y) {
        row_ptrs[y] = out + static_cast<size_t>(y) * rowStride;
    }

    png_read_image(png, row_ptrs);
    png_read_end(png, nullptr);
    png_destroy_read_struct(&png, &info, nullptr);

    return {
        .data = out,
        .shape = {height, width, 3}
    };
}

static uint32_t getMaxRasterHeightFromMaxInputShape(const Shape &maxInputShape) {
    if (maxInputShape.size() >= 4) {
        return maxInputShape[1];
    }
    if (maxInputShape.size() >= 1) {
        return maxInputShape[0];
    }
    return 0;
}

size_t PngDataDecoder::getRequiredScratchBufferSize(const Shape &maxInputShape) const {
    const uint32_t maxHeight = getMaxRasterHeightFromMaxInputShape(maxInputShape);
    return static_cast<size_t>(maxHeight) * sizeof(png_bytep);
}

std::string PngDataDecoder::getExtension() {
    return "png";
}
