#include "dataio.h"
#include "dataloader.h"
#include "compression.h"
#include "resource.h"
#include "pybind11_includes.h"

#include "dataAugmenter/FlipAugmentation.h"
#include "dataAugmenter/PadAugmentation.h"
#include "dataAugmenter/RandomCropAugmentation.h"
#include "dataAugmenter/ResizeAugmentation.h"
#include "dataSources/FlatDataSource.h"
#include "spdlog/fmt/bundled/xchar.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

static void bindDatasetRelated(const py::module &m) {
    py::enum_<DType>(m, "DType")
            .value("UINT8", DType::UINT8)
            .value("INT32", DType::INT32)
            .value("FLOAT32", DType::FLOAT32)
            .export_values();

    py::class_<Dataset, std::shared_ptr<Dataset>>(m, "Dataset")
            .def(py::init<std::shared_ptr<IDataSource>,
                std::vector<std::shared_ptr<IDataAugmentation> >,
                const pybind11::function &,
                bool>())
            .def("getEntries", [](const Dataset &self) {
                return self.getDataSource()->getEntries();
            })
            .def("splitTrainValidationTest", &Dataset::splitTrainValidationTest);

    py::class_<BatchedDataset>(m, "BatchedDataset")
            .def(py::init<const std::shared_ptr<Dataset> &, size_t>())
            .def("getNextBatch", &BatchedDataset::getNextBatch)
            .def("__len__", &BatchedDataset::getNumberOfBatches);
}

static void bindDataloaderRelated(const py::module &m) {
    py::class_<DLWrapper>(m, "DLWrapper")
            .def("__dlpack__", &DLWrapper::getDLpackCapsule, py::arg("stream") = py::none())
            .def("__dlpack_device__", &DLWrapper::getDLpackDevice);

    // TODO Comment(getNextBatch): Important convention is that memory of the last batch gets invalid when you call getNextBatch!
    py::class_<DataLoader, std::shared_ptr<DataLoader> >(m, "DataLoader")
            .def(py::init<const std::shared_ptr<Dataset> &, int, int, int, const std::shared_ptr<DataAugmentationPipe> &>())
            .def("getNextBatch", [](DataLoader &self) {
                std::pair<py::dict, py::dict> result = self.getNextBatch();
                return py::make_tuple(result.first, result.second);
            })
            .def("__len__", [](const DataLoader &self) -> size_t {
                return self.batchedDataset.getNumberOfBatches();
            });
}

static void bindCompressionRelated(const py::module &m) {
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

static void bindDataSources(const py::module &m) {
    py::enum_<ItemType>(m, "ItemType")
            .value("NONE", ItemType::NONE)
            .value("RASTER", ItemType::RASTER)
            .value("POINTS", ItemType::POINTS)
            .export_values();

    py::class_<SubdirToDictname>(m, "SubdirToDictname")
            .def(py::init<std::string, std::string, ItemType>());

    py::class_<IDataSource, std::shared_ptr<IDataSource> >(m, "IDataSource"); // NOLINT(bugprone-unused-raii)

    py::class_<FlatDataSource, IDataSource, std::shared_ptr<FlatDataSource> >(m, "FlatDataSource")
            .def(py::init<std::string, std::vector<SubdirToDictname> >(),
                 py::arg("root_directory"),
                 py::arg("subdir_to_dict") = std::vector<SubdirToDictname>{});
}

static DType getDType(const py::array &array) {
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

static DType getAndAssertEqualityOfDType(const py::array &array1, const py::array &array2) {
    const DType dtype1 = getDType(array1);
    if (dtype1 != getDType(array2)) {
        throw std::runtime_error("DTypes of arrays don't match, when they should.");
    }
    return dtype1;
}

static void bindAugmentations(const py::module &m) {
    py::class_<IDataAugmentation, std::shared_ptr<IDataAugmentation> >(m, "IDataAugmentation"); // NOLINT(bugprone-unused-raii)

    py::class_<FlipAugmentation, IDataAugmentation, std::shared_ptr<FlipAugmentation> >(m, "FlipAugmentation")
            .def(py::init<float, float>())
            .def("get_item_settings", [](const FlipAugmentation &self, const std::vector<uint32_t> &inputShape,
                                         const uint64_t itemSeed) {
                const py::dict pyBatch;
                const void *propPtr = self.getDataOutputSchema(inputShape, itemSeed).itemProp;
                const auto prop = static_cast<const FlipProp *>(propPtr);
                pyBatch["does_horizontal_flip"] = prop->doesHorizontalFlip;
                pyBatch["does_vertical_flip"] = prop->doesVerticalFlip;
                return pyBatch;
            });

    py::enum_<PadSettings>(m, "PadSettings")
            .value("PAD_TOP_LEFT", PadSettings::PAD_TOP_LEFT)
            .value("PAD_TOP_RIGHT", PadSettings::PAD_TOP_RIGHT)
            .value("PAD_BOTTOM_LEFT", PadSettings::PAD_BOTTOM_LEFT)
            .value("PAD_BOTTOM_RIGHT", PadSettings::PAD_BOTTOM_RIGHT)
            .export_values();

    py::class_<PadAugmentation, IDataAugmentation, std::shared_ptr<PadAugmentation> >(m, "PadAugmentation")
            .def(py::init<size_t, size_t, PadSettings>());

    py::class_<RandomCropAugmentation, IDataAugmentation, std::shared_ptr<RandomCropAugmentation> >(
                m, "RandomCropAugmentation")
            .def(py::init<size_t, size_t, size_t, size_t>())
            .def("get_item_settings", [](const RandomCropAugmentation &self, const std::vector<uint32_t> &inputShape,
                                         const uint64_t itemSeed) {
                const py::dict pyBatch;
                const void *propPtr = self.getDataOutputSchema(inputShape, itemSeed).itemProp;
                const auto prop = static_cast<const RandomCropProp *>(propPtr);
                pyBatch["left"] = prop->left;
                pyBatch["top"] = prop->top;
                pyBatch["height"] = prop->height;
                pyBatch["width"] = prop->width;
                return pyBatch;
            });

    py::class_<ResizeAugmentation, IDataAugmentation, std::shared_ptr<ResizeAugmentation> >(m, "ResizeAugmentation")
            .def(py::init<uint32_t, uint32_t>());
}

static auto toUint32Shape(const py::buffer_info &info) {
    return std::vector<uint32_t>(info.shape.begin(), info.shape.end());
}

static void verifyArraysIntegrity(const py::array &array, const Shape &shape, const Shape &expShape) {
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

static void verifyArraysIntegrity(const py::array &array, const Shape &shape, const Shapes &expShapes) {
    if (std::vector(shape.begin() + 1, shape.end()) != expShapes[0] || shape[0] != expShapes.size()) {
        throw std::runtime_error(std::format("Shape {} must match the expected shape {}, {}",
                                             formatVector(shape), expShapes.size(), formatVector(expShapes[0])));
    }

    if (!(array.flags() & py::array::c_style)) {
        throw std::runtime_error("Array must be C-style contiguous.");
    }

    if (!array.ptr()) {
        throw std::runtime_error("Array must be non-null.");
    }
}

static void bindDataProcessingPipe(const py::module &m) {
    py::class_<DataAugmentationPipe, std::shared_ptr<DataAugmentationPipe> >(m, "DataAugmentationPipe")
            .def(py::init([](std::vector<std::shared_ptr<IDataAugmentation> > augs, const std::vector<uint32_t> &maxIn,
                             const uint32_t maxNumPoints, const uint32_t maxBytes) {
                return new DataAugmentationPipe(std::move(augs), maxIn, maxNumPoints, maxBytes);
            }), py::keep_alive<1, 2>())
            .def("get_processing_schema",
                 [](const DataAugmentationPipe &self, const Shapes &inShapes, const uint64_t seed) {
                     auto *schema = new DataProcessingSchema(self.getProcessingSchema(inShapes, seed));
                     // Return (OutputShape, SchemaCapsule)
                     return py::make_tuple(
                         schema->outputShape,
                         py::capsule(schema, "DataProcessingSchema", [](void *p) {
                             delete static_cast<DataProcessingSchema *>(p);
                         })
                     );
                 })
            .def("augment_raster",
                 [](const DataAugmentationPipe &self, const py::array &input, const py::array &output,
                    const py::capsule &schemaCap) {
                     const auto inInfo = input.request();
                     const auto outInfo = output.request();
                     const auto *schema = static_cast<DataProcessingSchema *>(schemaCap.get_pointer());

                     // Alloc temporary buffers
                     const size_t bufSize = self.getMaximumRequiredBufferSize();
                     std::vector<uint8_t> b1(bufSize), b2(bufSize);

                     const auto inputShape = toUint32Shape(inInfo);
                     verifyArraysIntegrity(input, inputShape, schema->inputShapesPerAug[0]);
                     verifyArraysIntegrity(output, toUint32Shape(outInfo), schema->outputShapesPerAug.back());

                     const auto inpShapeWithoutBatch = std::span(inputShape).subspan(1);
                     const DType dtype = getAndAssertEqualityOfDType(input, output);
                     const size_t maxBytesOfInput = getShapeSize(inpShapeWithoutBatch) * getWidthOfDType(dtype);

                     self.augmentWithRaster(
                         dtype,
                         static_cast<const uint8_t *>(inInfo.ptr),
                         static_cast<uint8_t *>(outInfo.ptr),
                         b1.data(), b2.data(),
                         maxBytesOfInput,
                         *schema
                     );
                 })
            .def("augment_points",
                 [](const DataAugmentationPipe &self, const py::array &input, const py::array &output,
                    const py::capsule &schemaCap) {
                     const auto inInfo = input.request();
                     const auto outInfo = output.request();
                     const auto *schema = static_cast<DataProcessingSchema *>(schemaCap.get_pointer());

                     // Alloc temporary buffers
                     const size_t bufSize = self.getMaximumRequiredBufferSize();
                     std::vector<uint8_t> b1(bufSize), b2(bufSize);

                     verifyArraysIntegrity(input, toUint32Shape(inInfo), toUint32Shape(outInfo));

                     const auto inShape = toUint32Shape(inInfo);
                     const auto subShape = std::vector(inShape.begin() + 1, inShape.end());
                     Shapes inputShapes;
                     for (size_t b = 0; b < inShape[0]; b++) {
                         inputShapes.push_back(subShape);
                     }

                     self.augmentWithPoints(
                         inputShapes,
                         getAndAssertEqualityOfDType(input, output),
                         static_cast<const uint8_t *>(inInfo.ptr),
                         static_cast<uint8_t *>(outInfo.ptr),
                         b1.data(), b2.data(),
                         *schema
                     );
                 });
}

PYBIND11_MODULE(_core, m) {
#if defined(ENABLE_DEBUG)
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
    bindDataSources(m);

    // Expose augmentations and pipe for testing:
    bindAugmentations(m);
    bindDataProcessingPipe(m);

    // By default, we run with a Cuda device interface. You can change it to custom implementations though.
    m.def("set_device_for_resource_pool", [](const std::shared_ptr<HostAndGpuDeviceInterface> &device) {
        ResourcePool::setDeviceForNewResourcePool(device);
    });

    // Expose explicit shutdown for the global resource pool.
    m.def("shutdown_resource_pool", [] {
        ResourcePool::shutdownLazily();
    });

    // Ensure automatic shutdown on interpreter exit.
    try {
        const py::module_ atexit = py::module_::import("atexit");
        atexit.attr("register")(py::cpp_function([] {
            LOG_DEBUG_FUN("atexit", "atexit has been called");
            ResourcePool::shutdownLazily();
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
