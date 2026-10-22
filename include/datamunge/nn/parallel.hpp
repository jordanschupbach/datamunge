#pragma once

/// @file parallel.hpp
/// @brief Intra-op parallelism for nn kernels (GEMM, convolution, large
///        elementwise ops) and which GEMM backend is in use.

#include <cstdint>
#include <functional>
#include <string>

namespace datamunge::nn {

/// Number of threads nn kernels may use. Defaults to the environment variable
/// DATAMUNGE_NUM_THREADS, else the hardware concurrency.
int get_num_threads();
/// Sets the thread count (1 runs everything on the calling thread). Values
/// below 1 restore the default.
void set_num_threads(int threads);

/// Calls fn(lo, hi) over disjoint chunks covering [begin, end), in parallel
/// when the range has more than `grain` elements. Calls from inside a
/// parallel region run serially. Exceptions thrown by fn are rethrown.
void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t grain,
                  const std::function<void(std::int64_t, std::int64_t)>& fn);

/// "accelerate", "openblas", "cblas" or "builtin": the GEMM used for float matmul and convolution.
std::string gemm_backend();

}  // namespace datamunge::nn
