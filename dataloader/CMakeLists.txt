cmake_minimum_required(VERSION 3.30)
project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(URING REQUIRED liburing)

message(CMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}")
message(URING_LIBRARIES="${URING_LIBRARIES}")

python_add_library(_core MODULE src/library.cpp WITH_SOABI)
target_include_directories(_core PRIVATE src/)
target_link_libraries(_core PRIVATE pybind11::headers)
target_link_libraries(_core PRIVATE ${URING_LIBRARIES})
target_compile_options(_core PRIVATE -fvisibility=hidden)

# Pass parameters to the program:
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_definitions(_core PRIVATE PLATFORM_LINUX)

install(TARGETS _core DESTINATION native_dataloader)