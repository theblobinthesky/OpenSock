#include "dataset.h"
#include "dataloader.h"
#include <pybind11/eigen.h>
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

    py::class_<Subdirectory>(m, "Subdirectory")
            .def(py::init<std::string, const FileType, std::string, int, int>());

    py::class_<Dataset>(m, "Dataset")
            .def(py::init<std::string, std::vector<Subdirectory> >())
            .def("init", &Dataset::init)
            .def("getDataset", &Dataset::getDataset);

    py::class_<DataLoader>(m, "DataLoader")
            .def(py::init<Dataset, int, py::function, int, int>())
            .def("getNextBatch", &DataLoader::getNextBatch)
            .def("__len__", &DataLoader::getNumberOfBatches);
    // TODO Comment(getNextBatch): Important convention is that memory of the last batch gets invalid when you call getNextBatch!

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
