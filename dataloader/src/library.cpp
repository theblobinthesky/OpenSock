#include "dataio.h"
#include "dataloader.h"
#include "compression.h"
#include "resource.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include "dataAugmenter/FlipAugmentation.h"
#include "dataAugmenter/PadAugmentation.h"
#include "dataAugmenter/RandomCropAugmentation.h"
#include "dataAugmenter/ResizeAugmentation.h"
#include "dataSources/FlatDataSource.h"
#include "spdlog/fmt/bundled/xchar.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

void bindDatasetRelated(const py::module &m) {
    py::enum_<ItemFormat>(m, "ItemFormat")
            .value("FLOAT", ItemFormat::FLOAT)
            .value("UINT", ItemFormat::UINT)
            .export_values();

    py::class_<Dataset>(m, "Dataset")
            .def(py::init<std::shared_ptr<IDataSource>,
                std::vector<IDataAugmentation *>,
                const pybind11::function &,
                bool>())
            .def("getEntries", [](const Dataset &self) {
                return self.getDataSource()->getEntries();
            })
            .def("splitTrainValidationTest", &Dataset::splitTrainValidationTest);

    py::class_<BatchedDataset>(m, "BatchedDataset")
            .def(py::init<const Dataset &, size_t>())
            .def("getNextBatch", &BatchedDataset::getNextBatch)
            .def("__len__", &BatchedDataset::getNumberOfBatches);
}

void bindDataloaderRelated(const py::module &m) {
    py::class_<DLWrapper>(m, "DLWrapper")
            .def("__dlpack__", &DLWrapper::getDLpackCapsule, py::arg("stream") = py::none())
            .def("__dlpack_device__", &DLWrapper::getDLpackDevice);

    // TODO Comment(getNextBatch): Important convention is that memory of the last batch gets invalid when you call getNextBatch!
    py::class_<DataLoader, std::shared_ptr<DataLoader> >(m, "DataLoader")
            .def(py::init<Dataset &, int, int, int>())
            .def("getNextBatch", &DataLoader::getNextBatch)
            .def("__len__", [](const DataLoader &self) -> size_t {
                return self.batchedDataset.getNumberOfBatches();
            });
}

void bindCompressionRelated(const py::module &m) {
    py::enum_<Codec>(m, "Codec")
            .value("ZSTD_LEVEL_3", Codec::ZSTD_LEVEL_3)
            .value("ZSTD_LEVEL_7", Codec::ZSTD_LEVEL_7)
            .value("ZSTD_LEVEL_22", Codec::ZSTD_LEVEL_22)
            .export_values();

    py::class_<CompressorOptions>(m, "CompressorOptions")
            .def(py::init<const size_t,
                std::string,
                std::string,
                std::vector<uint32_t>,
                const bool,
                std::vector<std::vector<int> >,
                const bool,
                const std::vector<Codec> &,
                const float>());

    py::class_<Compressor>(m, "Compressor")
            .def(py::init<CompressorOptions>())
            .def("start", &Compressor::start);

    py::class_<Decompressor>(m, "Decompressor")
            .def(py::init<std::vector<uint32_t> >())
            .def("decompress", [](const Decompressor &self, const std::string &path) -> py::array {
                // ReSharper disable once CppDFAMemoryLeak
                const auto data = new std::vector<uint8_t>();
                std::vector<size_t> shape;
                int bytesPerItem;
                self.decompress(path, *data, shape, bytesPerItem);

                std::string dtypeString;
                if (bytesPerItem == 2) {
                    dtypeString = "float16";
                } else if (bytesPerItem == 4) {
                    dtypeString = "float32";
                } else {
                    throw std::runtime_error("Encountered unsupported dtype.");
                }
                const py::dtype dtype(dtypeString);

                const py::capsule capsule(data, [](void *p) {
                    delete static_cast<std::vector<uint8_t> *>(p);
                });
                return {dtype, shape, data->data(), capsule};
            });
}

void bindDataSources(const py::module &m) {
    py::class_<FlatDataSource, IDataSource, std::shared_ptr<FlatDataSource> >(m, "FlatDataSource")
            .def(py::init<std::string, std::unordered_map<std::string, std::string> >(),
                 py::arg("root_directory"),
                 py::arg("subdir_to_dict") = std::unordered_map<std::string, std::string>{});
}

DType getDType(const py::array &array) {
    if (py::isinstance<py::array_t<float> >(array)) {
        return DType::FLOAT32;
    }
    if (py::isinstance<py::array_t<uint8_t> >(array)) {
        return DType::UINT8;
    }
    if (py::isinstance<py::array_t<int32_t> >(array)) {
        return DType::INT32;
    }
    throw std::runtime_error("py::array has unknown DType.");
}

DType getAndAssertEqualityOfDType(const py::array &array1, const py::array &array2) {
    const DType dtype1 = getDType(array1);
    if (dtype1 != getDType(array2)) {
        throw std::runtime_error("DTypes of arrays don't match, when they should.");
    }
    return dtype1;
}

void bindAugmentations(const py::module &m) {
    py::class_<IDataAugmentation>(m, "IDataAugmentation");

    py::class_<FlipAugmentation, IDataAugmentation>(m, "FlipAugmentation")
            .def(py::init<float, float>())
            .def("get_item_settings", [](const FlipAugmentation &self, const std::vector<uint32_t> &inputShape,
                                         const uint64_t itemSeed) {
                const py::dict pyBatch;
                const auto settings = static_cast<FlipItemSettings *>(self.getDataOutputSchema(inputShape, itemSeed).
                    itemSettings);
                pyBatch["does_horizontal_flip"] = settings->doesHorizontalFlip;
                pyBatch["does_vertical_flip"] = settings->doesVerticalFlip;
                return pyBatch;
            });

    py::enum_<PadSettings>(m, "PadSettings")
            .value("PAD_TOP_LEFT", PadSettings::PAD_TOP_LEFT)
            .value("PAD_TOP_RIGHT", PadSettings::PAD_TOP_RIGHT)
            .value("PAD_BOTTOM_LEFT", PadSettings::PAD_BOTTOM_LEFT)
            .value("PAD_BOTTOM_RIGHT", PadSettings::PAD_BOTTOM_RIGHT)
            .export_values();

    py::class_<PadAugmentation, IDataAugmentation>(m, "PadAugmentation")
            .def(py::init<size_t, size_t, PadSettings>());

    py::class_<RandomCropAugmentation, IDataAugmentation>(m, "RandomCropAugmentation")
            .def(py::init<size_t, size_t, size_t, size_t>())
            .def("get_item_settings", [](const RandomCropAugmentation &self, const std::vector<uint32_t> &inputShape,
                                         const uint64_t itemSeed) {
                const py::dict pyBatch;
                const auto settings = static_cast<RandomCropSettings *>(self.getDataOutputSchema(inputShape, itemSeed).
                    itemSettings);
                pyBatch["left"] = settings->left;
                pyBatch["top"] = settings->top;
                pyBatch["height"] = settings->height;
                pyBatch["width"] = settings->width;
                return pyBatch;
            });

    py::class_<ResizeAugmentation, IDataAugmentation>(m, "ResizeAugmentation")
            .def(py::init<uint32_t, uint32_t>());
}

auto toUint32Shape(const py::buffer_info &info) {
    return std::vector<uint32_t>(info.shape.begin(), info.shape.end());
}

void verifyArraysIntegrity(py::array array, Shape &&shape, const Shape &expShape) {
    if (shape != expShape) {
        throw std::runtime_error(std::format("Shape {} must match the expected shape {}",
                                             formatVector(shape), formatVector(expShape)));
    }

    if (!(array.flags() & py::array::c_style)) {
        throw std::runtime_error("Array must be C-style contiguous.");
    }

    if (!array.ptr()) {
        throw std::runtime_error("Array must be non-null.");
    }
}

void bindDataProcessingPipe(const py::module &m) {
    py::class_<DataAugmentationPipe>(m, "DataAugmentationPipe")
            .def(py::init([](std::vector<IDataAugmentation *> augs, std::vector<uint32_t> maxIn, uint32_t maxBytes,
                             uint32_t maxNumPoints) {
                return new DataAugmentationPipe(std::move(augs), maxIn, maxNumPoints, maxBytes);
            }), py::keep_alive<1, 2>())
            .def("get_processing_schema",
                 [](const DataAugmentationPipe &self, const std::vector<uint32_t> &inShape, uint64_t seed) {
                     auto *schema = new DataProcessingSchema(self.getProcessingSchema(inShape, seed));
                     // Return (OutputShape, SchemaCapsule)
                     return py::make_tuple(
                         schema->outputShape,
                         py::capsule(schema, "DataProcessingSchema", [](void *p) {
                             delete static_cast<DataProcessingSchema *>(p);
                         })
                     );
                 })
            .def("augment_raster",
                 [](DataAugmentationPipe &self, py::array input, py::array output, py::capsule schemaCap) {
                     const auto inInfo = input.request();
                     const auto outInfo = output.request();
                     const auto *schema = static_cast<DataProcessingSchema *>(schemaCap.get_pointer());

                     // Alloc temporary buffers
                     const size_t bufSize = self.getMaximumRequiredBufferSize();
                     std::vector<uint8_t> b1(bufSize), b2(bufSize);
                     self.setBuffer(b1.data(), b2.data());

                     verifyArraysIntegrity(input, toUint32Shape(inInfo), schema->dataAugInputShapes[0]);
                     verifyArraysIntegrity(output, toUint32Shape(outInfo),
                                           schema->dataAugOutputShapes[schema->dataAugOutputShapes.size() - 1]);

                     self.augmentWithRaster(
                         getAndAssertEqualityOfDType(input, output),
                         static_cast<const uint8_t *>(inInfo.ptr),
                         static_cast<uint8_t *>(outInfo.ptr),
                         *schema
                     );
                 })
            .def("augment_points",
                 [](DataAugmentationPipe &self, py::array input, py::array output, py::capsule schemaCap) {
                     const auto inInfo = input.request();
                     const auto outInfo = output.request();
                     const auto *schema = static_cast<DataProcessingSchema *>(schemaCap.get_pointer());

                     // Alloc temporary buffers
                     const size_t bufSize = self.getMaximumRequiredBufferSize();
                     std::vector<uint8_t> b1(bufSize), b2(bufSize);
                     self.setBuffer(b1.data(), b2.data());

                     verifyArraysIntegrity(input, toUint32Shape(inInfo), toUint32Shape(outInfo));

                     self.augmentWithPoints(
                         toUint32Shape(inInfo),
                         getAndAssertEqualityOfDType(input, output),
                         static_cast<const uint8_t *>(inInfo.ptr),
                         static_cast<uint8_t *>(outInfo.ptr),
                         *schema
                     );
                 });
}

PYBIND11_MODULE(_core, m) {
#if defined(ENABLE_DEBUG_PRINT)
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%l]%$ [%s:%# %!] %v");
#else
    spdlog::set_level(spdlog::level::off);
#endif

    m.doc() = R"pbdoc(
        Native Dataloader that outputs tensors in the DLPack format.
        Compatible with all major deep learning frameworks including jax, tensorflow and pytorch.
        -----------------------

        .. currentmodule:: native_dataloader

        .. autosummary::
           :toctree: _generate

        DataSet
    )pbdoc";

    bindDatasetRelated(m);
    bindDataloaderRelated(m);
    bindCompressionRelated(m);

    py::class_<IDataSource, std::shared_ptr<IDataSource> >(m, "IDataSource");
    bindDataSources(m);

    // Expose augmentations and pipe for testing:
    bindAugmentations(m);
    bindDataProcessingPipe(m);

    // Expose explicit shutdown for the global resource pool.
    m.def("shutdown_resource_pool", [] { ResourcePool::get().shutdown(); });

    // Ensure automatic shutdown on interpreter exit.
    try {
        const py::module_ atexit = py::module_::import("atexit");
        atexit.attr("register")(py::cpp_function([]() {
            LOG_DEBUG_FUN("atexit", "atexit has been called");
            ResourcePool::get().shutdown();
        }));
    } catch (const std::exception &e) {
        // Swallow errors to avoid import-time failures if atexit is unavailable.
        LOG_WARNING("Failed to register atexit shutdown: {}", e.what());
    }

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
