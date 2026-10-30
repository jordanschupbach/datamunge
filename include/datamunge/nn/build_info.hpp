#pragma once

/// @file build_info.hpp
/// @brief What this build of datamunge::nn was compiled with and what it can use at run time:
///        dtypes, GEMM and LAPACK backends, threads, compiler and CPU features.
///
/// The counterpart of TensorFlow's "hello" device listing. nn runs on the CPU only, so its one
/// device is the host processor with nn's thread pool.

#include "datamunge/nn/dtype.hpp"

#include <string>
#include <vector>

namespace datamunge::nn {

/// @brief A snapshot of build and run-time facts, as returned by build_info().
struct BuildInfo {
  /// Library version string, or "unknown" when the build did not generate a version header.
  std::string version;
  /// Supported tensor dtypes, in DType order.
  std::vector<DType> dtypes;
  /// The GEMM used for float matmul and convolution: "accelerate", "openblas", "cblas" or "builtin".
  std::string gemm_backend;
  /// Whether dense linear algebra (cholesky, eigh, svd, ...) uses LAPACK.
  bool lapack{false};
  /// Threads nn kernels may use (get_num_threads()).
  int threads{1};
  /// Hardware threads reported by the C++ runtime (0 if unknown).
  unsigned hardware_threads{0};
  /// CPU architecture the library was compiled for ("x86_64", "aarch64", ...).
  std::string architecture;
  /// Compiler name and version.
  std::string compiler;
  /// SIMD and arithmetic features the running CPU reports (e.g. "AVX2", "FMA", "NEON").
  std::vector<std::string> cpu_features;

  /// @brief A two-column Markdown table of the fields, one row each.
  /// @return The table, ending in a newline.
  [[nodiscard]] std::string to_markdown() const;
};

/// @brief Collects the build information (cheap; probes CPU features once per call).
/// @return The snapshot.
[[nodiscard]] BuildInfo build_info();

/// @brief Floating-point operations of a matrix product \f$[m,k]\times[k,n]\f$: \f$2mkn\f$ (one multiply and one add per term).
/// @param m  Rows of the left factor.
/// @param k  Inner dimension.
/// @param n  Columns of the right factor.
/// @return The count, as a double to avoid overflow.
[[nodiscard]] double matmul_flops(std::int64_t m, std::int64_t k, std::int64_t n);

}  // namespace datamunge::nn
