cmake_minimum_required(VERSION 3.28.0)
option(ENABLE_ASAN "Enable address sanitizer." OFF)
set(VECTOR_EXTENSIONS "AVX2" CACHE STRING "Vector instruction set to use")

set(CMAKE_GENERATOR Ninja)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer -mavx2 -mf16c")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -mavx2 -mf16c")

project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES C CXX
)

set(Python_FIND_UNVERSIONED_NAMES FIRST)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(dlpack CONFIG REQUIRED)
find_package(Imath 3.0 REQUIRED CONFIG) # for OpenEXR
find_package(OpenEXR 3.0 REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

if (DEFINED OpenEXR_VERSION AND OpenEXR_VERSION VERSION_LESS 3.0)
    message(FATAL_ERROR "Found OpenEXR ${OpenEXR_VERSION}, but >= 3.0 is required.")
endif ()
if (DEFINED Imath_VERSION AND Imath_VERSION VERSION_LESS 3.0)
    message(FATAL_ERROR "Found Imath ${Imath_VERSION}, but >= 3.0 is required.")
endif ()

message(CMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}")


# Build the bitshuffle library.
set(BITSHUFFLE_CORE ${CMAKE_SOURCE_DIR}/submodules/bitshuffle)
add_library(bitshuffle_core STATIC
        ${BITSHUFFLE_CORE}/src/bitshuffle_core.c
        ${BITSHUFFLE_CORE}/src/iochain.c
)
target_include_directories(bitshuffle_core PUBLIC ${BITSHUFFLE_CORE}/src)
target_compile_options(bitshuffle_core PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter -Wno-unused-but-set-variable -mavx2)
set_target_properties(bitshuffle_core PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED YES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
)

# Build the cnpy library.
set(CNPY_CORE ${CMAKE_SOURCE_DIR}/submodules/cnpy)
add_library(cnpy_core STATIC ${CNPY_CORE}/cnpy.cpp)
target_include_directories(cnpy_core PUBLIC ${CNPY_CORE})
target_compile_options(cnpy_core PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter -Wno-unused-but-set-variable -mavx2)
set_target_properties(cnpy_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Build the logging library.
set(SPDLOG_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/submodules/spdlog")
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
add_subdirectory("${SPDLOG_SUBMODULE_DIR}" EXCLUDE_FROM_ALL) # EXCLUDE_FROM_ALL means it doesn't compile unless the lib is linked into the program.

# Build the python library.
python_add_library(_core MODULE
        src/dataloader.cpp
        src/resource.cpp
        src/utils.cpp
        src/library.cpp
        src/dataio.cpp
        src/compression.cpp
        src/vectorisation/NoVectorisation.cpp
        src/vectorisation/SSEVectorisation.cpp
        src/vectorisation/AVX2Vectorisation.cpp
        src/dataSources/FlatDataSource.cpp
        src/dataSources/WebDatasetDataSource.cpp
        src/dataDecoders/DecoderRegister.cpp
        src/dataDecoders/JpgDataDecoder.cpp
        src/dataDecoders/PngDataDecoder.cpp
        src/dataDecoders/ExrDataDecoder.cpp
        src/dataDecoders/NpyDataDecoder.cpp
        src/dataDecoders/CompressedDataDecoder.cpp
        src/dataAugmenter/augmentation.cpp
        src/dataAugmenter/ResizeAugmentation.cpp
        src/dataAugmenter/RandomCropAugmentation.cpp
        src/dataAugmenter/PadAugmentation.cpp
        src/dataAugmenter/FlipAugmentation.cpp
        WITH_SOABI
)

# Includes etc.
target_include_directories(_core PRIVATE src/)
target_include_directories(_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(_core PRIVATE ${ZSTD_INCLUDE_DIRS})
target_include_directories(_core PUBLIC ${CMAKE_SOURCE_DIR}/submodules/stb)
target_link_libraries(_core PRIVATE pybind11::headers)
target_link_libraries(_core PRIVATE JPEG::JPEG)
target_link_libraries(_core PRIVATE PNG::PNG)
target_link_libraries(_core PRIVATE OpenEXR::OpenEXR Imath::Imath)
target_link_libraries(_core PRIVATE CUDA::cudart)
target_link_libraries(_core PRIVATE dlpack::dlpack)
target_link_libraries(_core PRIVATE ${ZSTD_LIBRARIES})
target_link_libraries(_core PRIVATE bitshuffle_core cnpy_core)
target_link_libraries(_core PRIVATE spdlog::spdlog)
target_compile_options(_core PRIVATE -fvisibility=hidden -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-but-set-variable)

# Pass parameters to the program:
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_definitions(_core PRIVATE PLATFORM_LINUX)
target_compile_definitions(_core PRIVATE $<$<CONFIG:Debug>: ENABLE_DEBUG_PRINT=1>)
target_compile_definitions(_core PRIVATE SIMD_SSE)
target_compile_definitions(_core PRIVATE SIMD_AVX2)

# Set logging level based on build type.
target_compile_definitions(_core PRIVATE
        $<$<CONFIG:Debug>: SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>
        $<$<CONFIG:Release>: SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF>
)

target_link_options(_core PRIVATE -fuse-ld=mold)

# Address sanitizer if enabled.
if (ENABLE_ASAN)
    target_compile_options(_core PRIVATE -fsanitize=address)
    target_link_options(_core PRIVATE -fsanitize=address)
endif()

# Vectorisation:
include(CheckCXXSourceCompiles)
set(SSE41_FLAG "-mavx2")    # Hardcoded for clang.
set(AVX2_FLAG "-sse4.1")    # Hardcoded for clang.

set(CMAKE_REQUIRED_FLAGS "${SSE41_FLAG}")
check_cxx_source_compiles("
    #include <smmintrin.h>
    int main() {
        __m128i a = _mm_setzero_si128();
        return 0;
    }" CAN_COMPILE_SSE41)

set(CMAKE_REQUIRED_FLAGS "${AVX2_FLAG}")
check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m256i a = _mm256_setzero_si256();
        return 0;
    }" CAN_COMPILE_AVX2)

if (CAN_COMPILE_SSE41 AND NOT CAN_COMPILE_AVX2 AND VECTOR_EXTENSIONS STREQUAL "SSE41")
    MESSAGE(STATUS "Enabling SSE4.1.")
    target_compile_options(_core PRIVATE -msse4.1)
    target_compile_definitions(_core PRIVATE HAS_SSE)
elseif (CAN_COMPILE_AVX2 AND VECTOR_EXTENSIONS STREQUAL "AVX2")
    MESSAGE(STATUS "Enabling AVX2.")
    target_compile_options(_core PRIVATE -mavx2)
    target_compile_definitions(_core PRIVATE HAS_AVX2)
else ()
    MESSAGE(STATUS "Disabling vectorisation.")
    target_compile_definitions(_core PRIVATE NO_VECTORISATION)
endif ()

MESSAGE(STATUS "Destination: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS _core DESTINATION native_dataloader)
