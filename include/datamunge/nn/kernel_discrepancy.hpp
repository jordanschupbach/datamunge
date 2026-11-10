#pragma once
/// @file kernel_discrepancy.hpp
/// @brief Differentiable inverse-multiquadric kernels and signed unbiased two-sample MMD.
#include <limits>
#include <algorithm>
#include <utility>

#include "datamunge/nn/gaussian_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Shared shape and allocation-product checks for explicit pairwise kernels.
namespace mmd_detail {
/// @brief Validate a representable pairwise difference tensor without allocating it.
/// @param n Positive first sample count.
/// @param m Positive second sample count.
/// @param width Positive feature count.
/// @throws std::invalid_argument For nonpositive dimensions or int64 product overflow.
inline void Products(std::int64_t n, std::int64_t m, std::int64_t width) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (n <= 0 || m <= 0 || width <= 0 || n > limit / m || n * m > limit / width)
    throw std::invalid_argument("IMQ pairwise dimensions must have representable products");
}
/// @brief Validate finite same-width floating samples and minimum row counts.
/// @param x First finite Float32/64 matrix[N,D].
/// @param y Second matching-dtype finite matrix[M,D].
/// @param minimum Minimum count required independently in both samples.
/// @throws std::invalid_argument For invalid matrices/counts or pairwise product overflow.
inline void Inputs(const Tensor& x, const Tensor& y, std::int64_t minimum) {
  if (!x.defined() || x.dim() != 2)
    throw std::invalid_argument("IMQ samples must be floating matrices");
  aae_detail::Matrix(x, x.size(1), x.dtype(), false);
  aae_detail::Matrix(y, x.size(1), x.dtype(), false);
  if (x.size(0) < minimum || y.size(0) < minimum)
    throw std::invalid_argument("IMQ sample counts are too small");
  Products(x.size(0), y.size(0), x.size(1));
}
} // namespace mmd_detail

/// @brief Evaluate the equally averaged IMQ kernels with scales1,4,16 for every sample pair.
///
/// Returns mean_c c/(c+sum_d(x_i[d]-y_j[d])^2), c in{1,4,16}. Finite nonempty
/// same-width/dtype Float32/64 matrices may have unequal row counts and strides.
/// All x/y derivatives, including second derivatives, remain live. Self-pair
/// diagonals are exactly one for finite computed differences. No bandwidth fit,
/// normalization by dimension, RNG or clipping is hidden. Inputs unchanged;
/// result owns lifetime, no shared state. Work/storage O(N*M*D), materializing
/// pairwise differences; int64 product overflow is rejected before allocation.
/// Squared-distance overflow is not rescaled and can invalidate gradients.
/// @param x Finite first sample[N,D], N,D positive.
/// @param y Finite matching second sample[M,D], M positive.
/// @return Owning differentiable kernel matrix[N,M].
/// @throws std::invalid_argument For invalid shape/dtype/values or dimension products.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto k=n::models::ImqKernel(n::zeros({2,1}),n::ones({3,1}));
/// // Every entry is(1/2+4/5+16/17)/3, approximately0.7470588.
/// @endcode
/// @par References
/// - Zhao, Song and Ermon (2017/2018), InfoVAE: Information Maximizing Variational
///   Autoencoders, https://arxiv.org/abs/1706.02262 . Uses a small fixed three-scale
///   IMQ adaptation from the migrated teaching experiment, not a fitted bandwidth.
/// @see ImqMmd(), MmdOutput
inline Tensor ImqKernel(const Tensor& x, const Tensor& y) {
  mmd_detail::Inputs(x, y, 1);
  const auto distance = sum(square(expand_dims(x, 1) - expand_dims(y, 0)), {2});
  return (1.0 / (distance + 1.0) + 4.0 / (distance + 4.0) + 16.0 / (distance + 16.0)) / 3.0;
}

/// @brief Own all kernel matrices and scalar terms of the signed unbiased IMQ MMD estimate.
///
/// Every field owns lifetime and retains its continuous input paths. No fitting
/// state or random draws are stored. The finite-sample estimate may be negative;
/// it is MMD squared, not its square root or a calibrated hypothesis-test p-value.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto out=n::models::ImqMmd(n::zeros({2,1}),n::zeros({3,1}));
/// // xx[2,2], yy[3,3], xy[2,3] are all one; value is exactly zero.
/// @endcode
/// @see ImqMmd(), ImqKernel(), InfoVariationalAutoencoder()
struct MmdOutput {
  Tensor xx;       ///< Owning within-first kernel matrix[N,N], including its unit diagonal.
  Tensor yy;       ///< Owning within-second kernel matrix[M,M], including its unit diagonal.
  Tensor xy;       ///< Owning cross kernel matrix[N,M], with every pair included.
  Tensor within_x; ///< Owning(sum(xx)-N)/(N*(N-1)), excluding self-pairs only.
  Tensor within_y; ///< Owning(sum(yy)-M)/(M*(M-1)), excluding self-pairs only.
  Tensor cross;    ///< Owning mean(xy), including corresponding-index pairs when N=M.
  Tensor value;    ///< Owning signed scalar within_x+within_y-2*cross, unbiased MMD squared.
};

/// @brief Compute an unclipped unbiased two-sample MMD squared using fixed averaged IMQ kernels.
///
/// Uses independent samples x[N,D], y[M,D], with N,M>=2 and arbitrary unequal
/// counts. Excludes only within-sample diagonals; cross pairs are all included.
/// Self-comparison of the same nonconstant array can therefore be negative:
/// unbiasedness refers to independent sample sets, not a nonnegative finite
/// V-statistic. No sqrt, clipping or test calibration is applied. Coefficients
/// and reduction parenthesization preserve the original Float32 calculation.
///
/// Finite matching Float32/64 widths/dtypes, signed or zero values and strides
/// are accepted. First/second derivatives through both samples stay live. Inputs
/// unchanged; every result owns lifetime. Work/storage O((N^2+M^2+N*M)*D);
/// products are checked before kernels, but arithmetic overflow is not repaired.
/// @param x Finite first sample[N,D], N>=2,D>0.
/// @param y Finite matching second sample[M,D], M>=2.
/// @return Owning seven-field kernel/within/cross/signed-value result.
/// @throws std::invalid_argument For invalid samples or pairwise dimension products.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto x=n::tensor({0.,1.},{2,1});
/// const auto out=n::models::ImqMmd(x,x);
/// // Negative value: excluding within diagonals differs from the full cross mean.
/// @endcode
/// @par References
/// - Gretton et al. (2012), A Kernel Two-Sample Test, JMLR13,723-773,
///   https://www.jmlr.org/papers/v13/gretton12a.html . Uses the unbiased quadratic estimator.
/// - Zhao, Song and Ermon (2017/2018), https://arxiv.org/abs/1706.02262 .
/// @see MmdOutput, ImqKernel(), InfoVariationalAutoencoder()
inline MmdOutput ImqMmd(const Tensor& x, const Tensor& y) {
  mmd_detail::Inputs(x, y, 2);
  mmd_detail::Products(x.size(0), x.size(0), x.size(1));
  mmd_detail::Products(y.size(0), y.size(0), y.size(1));
  const double n = double(x.size(0)), m = double(y.size(0));
  MmdOutput    out;
  out.xx       = ImqKernel(x, x);
  out.yy       = ImqKernel(y, y);
  out.xy       = ImqKernel(x, y);
  out.within_x = (sum(out.xx) - n) / (n * (n - 1));
  out.within_y = (sum(out.yy) - m) / (m * (m - 1));
  out.cross    = mean(out.xy);
  out.value    = out.within_x + out.within_y - 2.0 * out.cross;
  return out;
}
} // namespace datamunge::nn::models
