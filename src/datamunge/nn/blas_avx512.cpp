/// @file blas_avx512.cpp
/// @brief In-house BLAS kernels compiled for AVX-512 (F/VL/DQ/BW + FMA); see blas_kernels.hpp.
///
/// Only ever called after blas.cpp has checked those CPU features. The
/// instruction set is selected here with target pragmas (CMakeLists.txt also
/// passes -mavx512f -mavx512vl -mavx512dq -mavx512bw -mfma), so the Python and R source builds, which compile
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
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,fma"))), apply_to = function)
#pragma clang fp contract(fast)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512vl,avx512dq,avx512bw,fma")
#pragma GCC optimize("fp-contract=fast")
#endif
#include "blas_kernels.hpp"
DATAMUNGE_BLAS_DEFINE_ISA(blas_avx512, 64)
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif
