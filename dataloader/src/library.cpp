#include "dataset.h"
#include "dataloader.h"
#include "compression.h"
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
#if defined(ENABLE_DEBUG_PRINT)
    spdlog::set_level(spdlog::level::trace);
#else
    spdlog::set_level(spdlog::level::off);
#endif


    m.doc() = R"pbdoc(
        Native Dataloader compatible with jax.
        -----------------------

        .. currentmodule:: native_dataloader

        .. autosummary::
           :toctree: _generate

        DataSet
    )pbdoc";

    py::enum_<FileType>(m, "FileType")
            .value("JPG", FileType::JPG)
            .value("PNG", FileType::PNG)
            .value("EXR", FileType::EXR)
            .value("NPY", FileType::NPY)
            .value("COMPRESSED", FileType::COMPRESSED)
            .export_values();

    py::enum_<ItemFormat>(m, "ItemFormat")
            .value("FLOAT", ItemFormat::FLOAT)
            .value("UINT", ItemFormat::UINT)
            .export_values();

    py::class_<Head>(m, "Head")
            .def(py::init<const FileType, std::string, std::vector<int> >())
            .def("getExt", &Head::getExt)
            .def("getDictName", &Head::getDictName)
            .def("getShape", &Head::getShape)
            .def("getShapeSize", &Head::getShapeSize)
            .def("getFilesType", &Head::getFilesType)
            .def("getItemFormat", &Head::getItemFormat)
            .def("getBytesPerItem", &Head::getBytesPerItem);

    py::class_<Dataset>(m, "Dataset")
            .def(py::init<std::string, std::vector<Head>,
                std::vector<std::string>,
                const pybind11::function &,
                bool>())

            .def(py::init<std::string, std::vector<Head>,
                std::vector<std::vector<std::string> >
            >())

            .def("splitTrainValidationTest", &Dataset::splitTrainValidationTest)
            .def("getRootDir", &Dataset::getRootDir)
            .def("getHeads", &Dataset::getHeads)
            .def("getEntries", &Dataset::getEntries);

    py::class_<BatchedDataset>(m, "BatchedDataset")
            .def(py::init<const Dataset &, size_t>())
            .def("getNextBatch", &BatchedDataset::getNextBatch);

    // TODO Comment(getNextBatch): Important convention is that memory of the last batch gets invalid when you call getNextBatch!
    py::class_<DataLoader>(m, "DataLoader")
            .def(py::init<Dataset &, int, int, int>())
            .def("getNextBatch", &DataLoader::getNextBatch)
            .def("__len__", &DataLoader::getNumberOfBatches);

    py::enum_<Codec>(m, "Codec")
            .value("ZSTD_LEVEL_3", Codec::ZSTD_LEVEL_3)
            .value("ZSTD_LEVEL_7", Codec::ZSTD_LEVEL_7)
            .value("ZSTD_LEVEL_22", Codec::ZSTD_LEVEL_22)
            .export_values();

    py::class_<CompressorOptions>(m, "CompressorOptions")
            .def(py::init<const size_t,
                std::string,
                std::string,
                std::vector<int>,
                const bool,
                std::vector<std::vector<int> >,
                const bool,
                const std::vector<Codec> &,
                const float>());

    py::class_<Compressor>(m, "Compressor")
            .def(py::init<CompressorOptions>())
            .def("start", &Compressor::start);

    py::class_<Decompressor>(m, "Decompressor")
            .def(py::init<std::vector<int> >())
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

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
