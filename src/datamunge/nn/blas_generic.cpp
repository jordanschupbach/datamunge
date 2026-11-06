/// @file blas_generic.cpp
/// @brief In-house BLAS kernels for the baseline instruction set (SSE2 on x86-64,
///        NEON on AArch64, scalar elsewhere); see blas_kernels.hpp.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#if defined(__clang__)
#pragma clang fp contract(fast)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=fast")
#endif
#include "blas_kernels.hpp"
DATAMUNGE_BLAS_DEFINE_ISA(blas_generic, 16)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
