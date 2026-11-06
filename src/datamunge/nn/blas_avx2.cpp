/// @file blas_avx2.cpp
/// @brief In-house BLAS kernels compiled for AVX2 + FMA; see blas_kernels.hpp.
///
/// Only ever called after blas.cpp has checked those CPU features. The
/// instruction set is selected here with target pragmas (CMakeLists.txt also
/// passes -mavx2 -mfma), so the Python and R source builds, which compile
/// every file with the same flags, get these kernels too. The standard
/// headers come first so that only the kernels, which have internal linkage,
/// are compiled for the target.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#pragma clang fp contract(fast)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#pragma GCC optimize("fp-contract=fast")
#endif
#include "blas_kernels.hpp"
DATAMUNGE_BLAS_DEFINE_ISA(blas_avx2, 32)
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif
