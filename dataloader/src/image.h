#ifndef JPEG_UTILS_H
#define JPEG_UTILS_H
#include <vector>
#include <string>

struct ImageData {
    std::vector<unsigned char> data;
    int width;
    int height;
};

ImageData readJpegFile(const std::string &path);

void resizeImage(const ImageData &image, unsigned char *outputBuffer,
                 size_t outputWidth,
                 size_t outputHeight);

#endif //JPEG_UTILS_H
