cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -mavx2")

project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX
)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(URING REQUIRED liburing)
find_package(JPEG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(dlpack CONFIG REQUIRED)

message(CMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}")
message(URING_LIBRARIES="${URING_LIBRARIES}")

python_add_library(_core MODULE
        src/async_io_linux.cpp
        src/dataloader.cpp
        src/dataset.cpp
        src/io.cpp
        src/data.cpp
        src/resource.cpp
        src/utils.cpp
        src/library.cpp
        src/cnpy.cpp
        WITH_SOABI
)

target_include_directories(_core PRIVATE src/)
target_include_directories(_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(_core PRIVATE pybind11::headers)
target_link_libraries(_core PRIVATE ${URING_LIBRARIES})
target_link_libraries(_core PRIVATE JPEG::JPEG)
target_link_libraries(_core PRIVATE CUDA::cudart)
target_link_libraries(_core PRIVATE dlpack::dlpack)
target_compile_options(_core PRIVATE -fvisibility=hidden -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)

# Pass parameters to the program:
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_definitions(_core PRIVATE PLATFORM_LINUX)
target_compile_definitions(_core PRIVATE $<$<CONFIG:Debug>: ENABLE_DEBUG_PRINT=1>)

install(TARGETS _core DESTINATION native_dataloader)