#include "image.h"
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstring>
#include <jpeglib.h>
#include <setjmp.h>

// Custom error manager structure.
struct my_error_mgr {
    jpeg_error_mgr pub; /* "public" fields */
    jmp_buf setjmp_buffer; /* for return to caller */
};

typedef my_error_mgr *my_error_ptr;

void my_error_exit(j_common_ptr cinfo) {
    my_error_ptr myerr = (my_error_ptr) cinfo->err;
    // Clean up the JPEG object then jump to the setjmp point.
    jpeg_destroy_decompress(reinterpret_cast<jpeg_decompress_struct *>(cinfo));
    longjmp(myerr->setjmp_buffer, 1);
}

ImageData readJpegFile(const std::string &path) {
    my_error_mgr jerr;
    FILE *infile = fopen(path.c_str(), "rb");
    if (!infile) {
        throw std::runtime_error(
            std::string("Cannot open input file: ") + path);
    }

    jpeg_decompress_struct cinfo = {};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        fclose(infile);
        throw std::runtime_error("JPEG decompression error.");
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    (void) jpeg_read_header(&cinfo, TRUE);

    if (cinfo.data_precision == 12) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        throw std::runtime_error(
            "The library is not compiled with 12-bit depth support.");
    }

    (void) jpeg_start_decompress(&cinfo);

    int row_stride = cinfo.output_width * cinfo.output_components;
    size_t output_size = cinfo.output_height * row_stride;

    // Allocate buffer for the image data.
    std::vector<unsigned char> imageBuffer(output_size);

    // Allocate a one-row-high sample array.
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)
    (reinterpret_cast<j_common_ptr>(&cinfo), JPOOL_IMAGE, row_stride,
     1);

    unsigned char *ptr = imageBuffer.data();
    while (cinfo.output_scanline < cinfo.output_height) {
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(ptr, buffer[0], row_stride);
        ptr += row_stride;
    }

    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    ImageData imgData;
    imgData.data = std::move(imageBuffer);
    imgData.width = cinfo.output_width;
    imgData.height = cinfo.output_height;
    return imgData;
}

ImageData rescaleToSize(const ImageData &image) {


    return {};
}

