#include "data.h"
#include "dataset.h"
#include "resource.h"
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstring>
#include <jpeglib.h>
#include <csetjmp>
#include "cnpy.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image_resize2.h"

struct ImageData {
    std::vector<unsigned char> data;
    int width = 0;
    int height = 0;
};

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
    cinfo.out_color_space = JCS_EXT_RGB;
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

void resizeImage(const ImageData &image, unsigned char *outputBuffer,
                 size_t outputWidth,
                 size_t outputHeight) {
    stbir_resize_uint8_srgb(image.data.data(), image.width, image.height,
                            image.width * 3, outputBuffer,
                            static_cast<int>(outputWidth),
                            static_cast<int>(outputHeight),
                            static_cast<int>(outputWidth) * 3, STBIR_RGB);
}


#define IMAGE_HEIGHT(subDir) static_cast<size_t>(subDir.getShape()[0])
#define IMAGE_WIDTH(subDir) static_cast<size_t>(subDir.getShape()[1])

struct BatchAllocation {
    size_t itemSize;
    size_t batchBufferSize;
    Allocation batchAllocation;
    float *batchBuffer;
};

BatchAllocation getBatchAllocation(const Head &head,
                                   BumpAllocator<Allocation> &allocations,
                                   const size_t batchSize) {
    const auto itemSize = head.getShapeSize();
    const auto batchBufferSize = batchSize * itemSize * sizeof(float);
    const auto batchAllocation = allocations.allocate(batchBufferSize);
    const auto batchBuffer = reinterpret_cast<float *>(batchAllocation.host);

    return {
        .itemSize = itemSize,
        .batchBufferSize = batchBufferSize,
        .batchAllocation = batchAllocation,
        .batchBuffer = batchBuffer
    };
}

Allocation loadJpgFiles(BumpAllocator<Allocation> &allocations,
                        const std::vector<std::vector<std::string> > &batchPaths,
                        const std::vector<Head> &heads, const size_t headIdx) {
    const Head &head = heads[headIdx];
    auto [itemSize, batchBufferSize, batchAllocation, batchBuffer]
            = getBatchAllocation(head, allocations, batchPaths.size());

    const auto batchBufferOld = new uint8_t[batchPaths.size() * itemSize];

    for (size_t j = 0; j < batchPaths.size(); j++) {
        ImageData imgData = readJpegFile(batchPaths[j][headIdx]);
        resizeImage(imgData, batchBufferOld + j * itemSize,
                    IMAGE_WIDTH(head), IMAGE_HEIGHT(head));
    }

    for (size_t b = 0; b < batchPaths.size(); b++) {
        for (size_t y = 0; y < IMAGE_HEIGHT(head); y++) {
            for (size_t j = 0; j < IMAGE_WIDTH(head); j++) {
                for (size_t c = 0; c < 3; c++) {
                    const size_t idx = b * itemSize
                                       + y * IMAGE_WIDTH(head) * 3
                                       + j * 3
                                       + c;
                    batchBuffer[idx] = static_cast<float>(batchBufferOld[idx]) / 255.0f;
                }
            }
        }
    }

    delete[] batchBufferOld;
    return batchAllocation;
}

Allocation loadNpyFiles(BumpAllocator<Allocation> &allocations,
                        const std::vector<std::vector<std::string> > &batchPaths,
                        const std::vector<Head> &heads, const size_t headIdx) {
    const Head &head = heads[headIdx];
    auto [itemSize, batchBufferSize, batchAllocation, batchBuffer]
            = getBatchAllocation(head, allocations, batchPaths.size());

    for (size_t j = 0; j < batchPaths.size(); j++) {
        // Load the NPY file from the path
        cnpy::NpyArray arr = cnpy::npy_load(batchPaths[j][headIdx]);

        // Ensure the file contains valid data.
        const std::string &filePath = batchPaths[j][headIdx];
        if (arr.word_size != sizeof(float)) {
            throw std::runtime_error(
                std::format("NPY file {} has word size {} and does not contain float32 data.",
                            filePath, arr.word_size));
        }

        if (arr.fortran_order) {
            throw std::runtime_error(
                std::format("NPY file {} has fortran order, e.g. column-major rather than row-major ordering.",
                            filePath));
        }

        // Compute the total number of elements in the array
        size_t npy_num_elements = 1;
        for (const unsigned long d: arr.shape) {
            npy_num_elements *= d;
        }
        if (npy_num_elements != itemSize) {
            throw std::runtime_error(
                "NPY file " + batchPaths[j][headIdx] +
                " has mismatched size. Expected " +
                std::to_string(itemSize) + ", got " +
                std::to_string(npy_num_elements));
        }
        // Copy the float data into the batch buffer
        const auto *npy_data = arr.data<float>();
        std::memcpy(batchBuffer + j * itemSize, npy_data, itemSize * sizeof(float));
    }

    return batchAllocation;
}

Allocation loadFiles(BumpAllocator<Allocation> &allocations,
                     const std::vector<std::vector<std::string> > &batchPaths,
                     const std::vector<Head> &heads, const size_t headIdx) {
    switch (const Head &head = heads[headIdx]; head.getFilesType()) {
        case FileType::JPG:
            return loadJpgFiles(allocations, batchPaths, heads, headIdx);
        case FileType::NPY:
            return loadNpyFiles(allocations, batchPaths, heads, headIdx);
        default:
            throw std::runtime_error("Cannot load an unsupported file type.");
    }
}
