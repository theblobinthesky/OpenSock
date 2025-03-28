#ifndef JPEG_UTILS_H
#define JPEG_UTILS_H
#include <vector>
#include <string>

struct ImageData {
    std::vector<unsigned char> data;
    int width;
    int height;
};

ImageData readJpegFile(const std::string& path);

ImageData resizeImage(const ImageData &image);

#endif //JPEG_UTILS_H
