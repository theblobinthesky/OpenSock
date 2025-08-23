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

#include <png.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <Imath/ImathBox.h>
using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;


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
    const auto myerr = reinterpret_cast<my_error_ptr>(cinfo->err);
    // Clean up the JPEG object then jump to the setjmp point.
    jpeg_destroy_decompress(reinterpret_cast<jpeg_decompress_struct *>(cinfo));
    longjmp(myerr->setjmp_buffer, 1);
}

ImageData readJpegFile(const std::string &path) {
    my_error_mgr jerr{};
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
                 const size_t outputWidth,
                 const size_t outputHeight) {
    stbir_resize_uint8_srgb(image.data.data(), image.width, image.height,
                            image.width * 3, outputBuffer,
                            static_cast<int>(outputWidth),
                            static_cast<int>(outputHeight),
                            static_cast<int>(outputWidth) * 3, STBIR_RGB);
}


#define IMAGE_HEIGHT(subDir) static_cast<size_t>(subDir.getShape()[0])
#define IMAGE_WIDTH(subDir) static_cast<size_t>(subDir.getShape()[1])

CpuAllocation getBatchAllocation(const Head &head,
                                 BumpAllocator<uint8_t *> &cpuAllocator,
                                 const size_t batchSize,
                                 const size_t itemSize) {
    const auto shapeSize = head.getShapeSize();
    const auto batchBufferSize = batchSize * shapeSize * itemSize;
    const auto batchBuffer = cpuAllocator.allocate(batchBufferSize);

    return {
        .shapeSize = shapeSize,
        .batchBufferSize = batchBufferSize,
        .batchBuffer = {batchBuffer}
    };
}

CpuAllocation loadJpgFiles(BumpAllocator<uint8_t *> &cpuAllocator,
                           const std::vector<std::vector<std::string> > &batchPaths,
                           const std::vector<Head> &heads, const size_t headIdx) {
    const Head &head = heads[headIdx];
    const auto batchAllocation = getBatchAllocation(head, cpuAllocator, batchPaths.size(), 1);
    const auto [shapeSize, batchBufferSize, batchBuffer] = batchAllocation;

    for (size_t j = 0; j < batchPaths.size(); j++) {
        ImageData imgData = readJpegFile(batchPaths[j][headIdx]);
        resizeImage(imgData, batchBuffer.uint8 + j * shapeSize,
                    IMAGE_WIDTH(head), IMAGE_HEIGHT(head));
    }

    return batchAllocation;
}

// --- PNG loader: 8-bit RGB (libpng) ---

static ImageData readPngFile(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open PNG file: " + path);

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        throw std::runtime_error("png_create_read_struct failed");
    }
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        throw std::runtime_error("PNG read error");
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    png_uint_32 width, height;
    int bit_depth, color_type;
    png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);

    // Normalize to 8-bit RGB
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (color_type == PNG_COLOR_TYPE_GRAY) png_set_gray_to_rgb(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png); // expand tRNS -> alpha
    // We don't want alpha in the output
    if (color_type & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);

    // Handle interlace if present
    int num_passes = png_set_interlace_handling(png);
    (void) num_passes;

    png_read_update_info(png, info);

    const png_size_t rowbytes = png_get_rowbytes(png, info);
    // Expect 3 channels now
    const int channels = 3;

    if (rowbytes != (png_size_t) width * channels) {
        // We'll still read via row pointers into a tightly packed buffer
    }

    std::vector<unsigned char> pixels((size_t) width * height * channels);
    std::vector<png_bytep> row_ptrs(height);
    for (png_uint_32 y = 0; y < height; ++y) {
        row_ptrs[y] = pixels.data() + (size_t) y * width * channels;
    }

    png_read_image(png, row_ptrs.data());
    png_read_end(png, nullptr);
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    ImageData img;
    img.data = std::move(pixels);
    img.width = (int) width;
    img.height = (int) height;
    return img;
}

CpuAllocation loadPngFiles(BumpAllocator<uint8_t *> &cpuAllocator,
                           const std::vector<std::vector<std::string> > &batchPaths,
                           const std::vector<Head> &heads, const size_t headIdx) {
    const Head &head = heads[headIdx];
    const auto batchAllocation = getBatchAllocation(head, cpuAllocator, batchPaths.size(), 1);
    const auto [shapeSize, batchBufferSize, batchBuffer] = batchAllocation;

    const int outW = static_cast<int>(IMAGE_WIDTH(head));
    const int outH = static_cast<int>(IMAGE_HEIGHT(head));

    for (size_t j = 0; j < batchPaths.size(); ++j) {
        ImageData img = readPngFile(batchPaths[j][headIdx]);

        // Resize into the j-th slot
        unsigned char *out = batchBuffer.uint8 + j * shapeSize;
        stbir_resize_uint8_srgb(
            img.data.data(), img.width, img.height, img.width * 3,
            out, outW, outH, outW * 3, STBIR_RGB);
    }

    return batchAllocation;
}

static void resizeImageFloatRGB(const float *input, int inW, int inH,
                                float *output, int outW, int outH) {
    // Linear (no gamma) resize for float data
    stbir_resize_float_linear(
        input, inW, inH, inW * 3 * static_cast<int>(sizeof(float)),
        output, outW, outH, outW * 3 * static_cast<int>(sizeof(float)),
        STBIR_RGB);
}

static void readExrRGBInterleaved(const std::string &path,
                                  std::vector<float> &outRGB,
                                  int &width, int &height) {
    InputFile file(path.c_str());
    const Header &hdr = file.header();
    const Box2i dw = hdr.dataWindow();

    const int w = dw.max.x - dw.min.x + 1;
    const int h = dw.max.y - dw.min.y + 1;
    width = w;
    height = h;

    outRGB.assign(static_cast<size_t>(w) * h * 3, 0.0f);

    FrameBuffer fb;
    const size_t xStride = sizeof(float) * 3;
    const size_t yStride = xStride * static_cast<size_t>(w);

    // Base pointer adjusted by dataWindow min
    char *base = reinterpret_cast<char *>(outRGB.data());
    const ptrdiff_t xOffset = static_cast<ptrdiff_t>(dw.min.x) * static_cast<ptrdiff_t>(xStride);
    const ptrdiff_t yOffset = static_cast<ptrdiff_t>(dw.min.y) * static_cast<ptrdiff_t>(yStride);

    fb.insert("R", Slice(FLOAT,
                         base - xOffset - yOffset + 0 * sizeof(float),
                         xStride, yStride, 1, 1, 0.0F));
    fb.insert("G", Slice(FLOAT,
                         base - xOffset - yOffset + 1 * sizeof(float),
                         xStride, yStride, 1, 1, 0.0F));
    fb.insert("B", Slice(FLOAT,
                         base - xOffset - yOffset + 2 * sizeof(float),
                         xStride, yStride, 1, 1, 0.0F));

    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);
}

CpuAllocation loadExrFiles(BumpAllocator<uint8_t *> &cpuAllocator,
                           const std::vector<std::vector<std::string> > &batchPaths,
                           const std::vector<Head> &heads, const size_t headIdx) {
    const Head &head = heads[headIdx];
    const auto batchAllocation = getBatchAllocation(head, cpuAllocator, batchPaths.size(), 4);
    const auto [shapeSize, batchBufferSize, batchBuffer] = batchAllocation;

    const int outW = static_cast<int>(IMAGE_WIDTH(head));
    const int outH = static_cast<int>(IMAGE_HEIGHT(head));

    for (size_t j = 0; j < batchPaths.size(); ++j) {
        const std::string &path = batchPaths[j][headIdx];

        int inW;
        int inH;
        std::vector<float> rgb;
        readExrRGBInterleaved(path, rgb, inW, inH);

        float *out = batchBuffer.float32 + j * shapeSize;

        if (inW == outW && inH == outH) {
            // Same size: direct copy
            std::memcpy(out, rgb.data(), static_cast<size_t>(outW) * outH * 3 * sizeof(float));
        } else {
            // Resize in linear space
            resizeImageFloatRGB(rgb.data(), inW, inH, out, outW, outH);
        }
    }

    return batchAllocation;
}

CpuAllocation loadNpyFiles(BumpAllocator<uint8_t *> &cpuAllocator,
                           const std::vector<std::vector<std::string> > &batchPaths,
                           const std::vector<Head> &heads, const size_t headIdx) {
    const Head &head = heads[headIdx];
    const auto batchAllocation = getBatchAllocation(head, cpuAllocator, batchPaths.size(), 4);
    const auto [shapeSize, batchBufferSize, batchBuffer] = batchAllocation;

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
        if (npy_num_elements != shapeSize) {
            throw std::runtime_error(
                "NPY file " + batchPaths[j][headIdx] +
                " has mismatched size. Expected " +
                std::to_string(shapeSize) + ", got " +
                std::to_string(npy_num_elements));
        }
        // Copy the float data into the batch buffer
        const auto *npy_data = arr.data<float>();
        std::memcpy(batchBuffer.float32 + j * shapeSize, npy_data, shapeSize * sizeof(float));
    }

    return batchAllocation;
}

CpuAllocation loadCompressedFiles(BumpAllocator<uint8_t *> &cpuAllocator,
                                  const std::vector<std::vector<std::string> > &batchPaths,
                                  const std::vector<Head> &heads, const size_t headIdx) {
    throw std::runtime_error("loadCompressedFiles is not implemented yet.");
    return {};
}

CpuAllocation loadFilesFromHeadIntoContigousBatch(BumpAllocator<uint8_t *> &cpuAllocator,
                                                  const std::vector<std::vector<std::string> > &batchPaths,
                                                  const std::vector<Head> &heads, const size_t headIdx) {
    switch (const Head &head = heads[headIdx]; head.getFilesType()) {
        case FileType::JPG:
            return loadJpgFiles(cpuAllocator, batchPaths, heads, headIdx);
        case FileType::PNG:
            return loadPngFiles(cpuAllocator, batchPaths, heads, headIdx);
        case FileType::EXR:
            return loadExrFiles(cpuAllocator, batchPaths, heads, headIdx);
        case FileType::NPY:
            return loadNpyFiles(cpuAllocator, batchPaths, heads, headIdx);
        case FileType::COMPRESSED:
            return loadCompressedFiles(cpuAllocator, batchPaths, heads, headIdx);
        default:
            throw std::runtime_error("Cannot load an unsupported file type.");
    }
}
