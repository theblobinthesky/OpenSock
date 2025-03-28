#include "dataloader.h"
#include "image.h"
#include <pybind11/numpy.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <filesystem>

#define CHS 3

namespace py = pybind11;
namespace fs = std::filesystem;

// TODO: Align everything!
MemoryArena::MemoryArena(): data(nullptr), total_size(0), offset(0) {
}

MemoryArena::MemoryArena(const size_t _total_size) : total_size(_total_size),
    offset(0) {
    data = malloc(_total_size);
}

MemoryArena &MemoryArena::operator=(MemoryArena &&arena) noexcept {
    offset = arena.offset;
    total_size = arena.total_size;
    data = arena.data;

    arena.offset = 0;
    arena.total_size = 0;
    arena.data = null;
    return *this;
}

MemoryArena::MemoryArena(MemoryArena &&arena) noexcept : data(arena.data),
    total_size(arena.total_size),
    offset(arena.offset) {
    arena.data = null;
    arena.offset = arena.total_size = 0;
}

MemoryArena::~MemoryArena() {
    free(data);
}

void *MemoryArena::allocate(const size_t size) {
    void *ptr = data;
    offset += size;
    return ptr;
}


pybind11::array_t<float>
tensor_to_numpy(const Eigen::Tensor<float, 4> &tensor) {
    auto dims = tensor.dimensions();
    return py::array_t<float>(
        {dims[0], dims[1], dims[2], dims[3]},
        {
            sizeof(float) * dims[1] * dims[2] * dims[3],
            sizeof(float) * dims[2] * dims[3],
            sizeof(float) * dims[3],
            sizeof(float)
        },
        tensor.data()
    );
}

bool existsEnvVar(const std::string &name) {
    return std::getenv(name.c_str()) != null;
}

DataLoader::DataLoader(
    Dataset _dataset,
    const int _batchSize,
    py::function _createDatasetFunction
) : dataset(std::move(_dataset)),
    batchSize(_batchSize),
    createDatasetFunction(std::move(_createDatasetFunction)) {
    if (batchSize <= 0) {
        throw std::runtime_error(
            "Batch size needs to be strictly positive.");
    }

//    memoryArena = std::move(MemoryArena(
//        _image_height * _image_width * CHS * sizeof(float) * _batchSize));

    if (!fs::exists(dataset.getRootDir()) || existsEnvVar(
            INVALID_DS_ENV_VAR)) {
        fs::remove_all(dataset.getRootDir());
        createDatasetFunction();
    }

    dataset.init();

    numberOfBatches = (dataset.getDataset().size() + batchSize - 1) /
                      batchSize;
}

py::dict DataLoader::getNextBatch() {
    std::vector<std::vector<std::string> > batchPaths = dataset.getNextBatch(
        batchSize);
    py::dict batch;
    const std::vector<Subdirectory> &subDirs = dataset.getSubDirs();
    size_t numImages = batchPaths.size();

    // For each subdirectory, load all images and assemble into one numpy array.
    for (size_t i = 0; i < subDirs.size(); i++) {
        const Subdirectory &subDir = subDirs[i];

        // Load the first image to determine dimensions.
        ImageData firstImg = readJpegFile(batchPaths[0][i]);
        int height = firstImg.height;
        int width = firstImg.width;
        int channels = 3; // assuming RGB output
        size_t imageSize = static_cast<size_t>(height * width * channels);

        // Allocate a contiguous buffer for the entire batch.
        size_t batchBufferSize = numImages * imageSize;
        auto *batchBuffer = new std::vector<unsigned char>(batchBufferSize);

        // Copy first image data into the batch buffer.
        std::copy(firstImg.data.begin(), firstImg.data.end(),
                  batchBuffer->begin());

        // Load the remaining images.
        for (size_t j = 1; j < numImages; j++) {
            ImageData imgData = readJpegFile(batchPaths[j][i].c_str());
            std::printf("height: %d, width: %d, newHeight: %d, newWidth: %d\n",
                        height, width, imgData.height, imgData.width);
            // Ensure dimensions match.
            if (imgData.height != height || imgData.width != width) {
                delete batchBuffer;
                throw std::runtime_error(
                    "Image dimensions do not match across the batch.");
            }
            std::copy(imgData.data.begin(), imgData.data.end(),
                      batchBuffer->begin() + j * imageSize);
        }

        // Create a capsule that will free the batchBuffer when the numpy array is garbage collected.
        py::capsule free_when_done(batchBuffer, [](void *f) {
            delete reinterpret_cast<std::vector<unsigned char> *>(f);
        });

        // Define the shape and strides for the numpy array.
        std::vector<ssize_t> shape = {
            static_cast<ssize_t>(numImages), height, width, channels
        };
        std::vector<ssize_t> strides = {
            static_cast<ssize_t>(imageSize * sizeof(unsigned char)),
            static_cast<ssize_t>(width * channels * sizeof(unsigned char)),
            static_cast<ssize_t>(channels * sizeof(unsigned char)),
            static_cast<ssize_t>(sizeof(unsigned char))
        };

        // Create the numpy array from the contiguous buffer.
        py::array np_batch(py::buffer_info(
                               batchBuffer->data(), /* Pointer to buffer */
                               sizeof(unsigned char), /* Size of one scalar */
                               py::format_descriptor<unsigned char>::format(),
                               /* Format descriptor */
                               4, /* Number of dimensions */
                               shape, /* Buffer dimensions */
                               strides /* Strides (in bytes) */
                           ), free_when_done);

        // Store the assembled batch in the dictionary under the subdirectory's dict name.
        batch[subDir.getDictName().c_str()] = np_batch;
    }

    return batch;
}

size_t DataLoader::getNumberOfBatches() const {
    return numberOfBatches;
}
