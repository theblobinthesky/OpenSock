#pragma once

// When compiling with the undefined sanitizer,
// the pybind11.h header trips up value range propagation somehow.
// So disable that particular warning as a workaround.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#pragma GCC diagnostic pop
