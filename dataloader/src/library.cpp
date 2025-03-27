#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>
#include <filesystem>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;
namespace fs = std::filesystem;

#define null nullptr

constexpr std::string INVALID_DS_ENV_VAR = "INVALIDATE_DATASET";

enum class FileType {
    JPG,
    EXR,
    NPY
};

class Subdirectory {
public:
    Subdirectory(
        std::string subdir,
        const FileType filesType,
        std::string dictName
    ) : subdir(std::move(subdir)), filesType(filesType),
        dictName(std::move(dictName)) {
    }

private:
    std::string subdir;
    FileType filesType;
    std::string dictName;
};

bool existsEnvVar(const std::string& name) {
    return std::getenv(name.c_str()) != null;
}

class Dataset {
public:
    Dataset(
        std::string rootDir,
        std::vector<Subdirectory> subDirs,
        py::function createDatasetFunction
    ) : rootDir(std::move(rootDir)), subDirs(std::move(subDirs)),
        createDatasetFunction(std::move(createDatasetFunction)) {
    }

    void init() const {
        if (!fs::exists(rootDir) || existsEnvVar(INVALID_DS_ENV_VAR)) {
            fs::remove_all(rootDir);
            createDatasetFunction();
        }
    }

private:
    std::string rootDir;
    std::vector<Subdirectory> subDirs;
    py::function createDatasetFunction;
};

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
            .def(py::init<std::string, const FileType, std::string>());

    py::class_<Dataset>(m, "Dataset")
            .def(py::init<std::string, std::vector<Subdirectory>,
                py::function>())
            .def("init", &Dataset::init);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
