#include "Resize.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

struct ImageData {
    std::vector<uint8_t> data;
    int width;
    int height;
};

void resizeImageFloatRGB(const float *input, int inW, int inH,
                         float *output, int outW, int outH) {
    // Linear (no gamma) resize for float data
    stbir_resize_float_linear(
        input, inW, inH, inW * 3 * static_cast<int>(sizeof(float)),
        output, outW, outH, outW * 3 * static_cast<int>(sizeof(float)),
        STBIR_RGB);
}

void resizeImage(const ImageData &image, unsigned char *outputBuffer,
                 const size_t outputWidth,
                 const size_t outputHeight) {
    stbir_resize_uint8_srgb(image.data.data(), image.width, image.height,
                            image.width * 3, outputBuffer,
                            static_cast<int>(outputWidth),
                            static_cast<int>(outputHeight),
                            static_cast<int>(outputWidth) * 3, STBIR_RGB);
}
