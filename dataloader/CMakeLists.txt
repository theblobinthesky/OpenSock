cmake_minimum_required(VERSION 3.30)
project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 26)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(_core MODULE src/library.cpp WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)

# Pass parameters to the program:
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION native_dataloader)