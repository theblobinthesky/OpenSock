#include "dataset.h"
#include "dataloader.h"
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Native Dataloader compatible with jax.
        -----------------------

        .. currentmodule:: native_dataloader

        .. autosummary::
           :toctree: _generate

        DataSet
    )pbdoc";

    py::enum_<FileType>(m, "FileType")
            .value("EXR", FileType::EXR)
            .value("JPG", FileType::JPG)
            .value("NPY", FileType::NPY)
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

    py::class_<DataLoader>(m, "DataLoader")
            .def(py::init<Dataset &, int, int, int>())
            .def("getNextBatch", &DataLoader::getNextBatch)
            .def("__len__", &DataLoader::getNumberOfBatches);
    // TODO Comment(getNextBatch): Important convention is that memory of the last batch gets invalid when you call getNextBatch!

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
