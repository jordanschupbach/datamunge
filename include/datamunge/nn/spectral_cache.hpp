#pragma once
/// @file spectral_cache.hpp
/// @brief Explicit column-vector spectral power iteration, cached normalization and state updates.
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/params.hpp"

namespace datamunge::nn {
/// @brief Internal validation for finite cached spectral kernels and column-vector geometry.
namespace spectral_cache_detail {
/// @brief Require positive finite Float32/64 CPU tensors, allowing scalar intermediates.
/// @param value Defined tensor whose axes are positive; scalars are allowed.
/// @throws std::invalid_argument For undefined, empty, nonfloating or nonfinite values.
/// @throws std::logic_error For non-CPU storage.
inline void Finite(const Tensor& value) {
  if (!value.defined() || value.numel() < 1 ||
      (value.dtype() != DType::Float32 && value.dtype() != DType::Float64))
    throw std::invalid_argument("Cached spectral values must be nonempty Float32/64 tensors");
  for (auto axis : value.shape())
    if (axis <= 0)
      throw std::invalid_argument("Cached spectral axes must be positive");
  for (double x : value.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("Cached spectral values must be finite");
}
/// @brief Flatten all kernel axes except the final output axis, preserving its live graph.
/// @param weight Finite floating CPU kernel of rank at least two, strides allowed.
/// @return Owning live matrix[product(leading axes),last axis].
/// @throws std::invalid_argument For invalid weight or overflowing leading products.
/// @throws std::logic_error For non-CPU storage.
inline Tensor Matrix(const Tensor& weight) {
  Finite(weight);
  if (weight.dim() < 2)
    throw std::invalid_argument("Cached spectral kernels need rank >= 2");
  std::int64_t rows = 1;
  for (std::size_t i = 0; i + 1 < weight.shape().size(); ++i) {
    if (rows > std::numeric_limits<std::int64_t>::max() / weight.size(i))
      throw std::invalid_argument("Cached spectral leading product overflow");
    rows *= weight.size(i);
  }
  return reshape(weight, {rows, weight.shape().back()});
}
/// @brief Require a matching finite column vector.
/// @param value Finite floating CPU[rows,1] vector.
/// @param rows Required positive length.
/// @param dtype Required computation dtype.
/// @throws std::invalid_argument For incompatible geometry, dtype or values.
/// @throws std::logic_error For non-CPU storage.
inline void Column(const Tensor& value, std::int64_t rows, DType dtype) {
  Finite(value);
  if (value.shape() != Shape({rows, 1}) || value.dtype() != dtype)
    throw std::invalid_argument("Cached spectral vector needs matching dtype and column geometry");
}
/// @brief Require a finite positive scalar that remains finite and positive in the tensor dtype.
/// @param value Requested additive squared-norm epsilon or denominator floor.
/// @param dtype Float32/64 computation dtype.
/// @throws std::invalid_argument For nonpositive, nonfinite or unrepresentable values.
inline void Positive(double value, DType dtype) {
  const double rounded =
      dtype == DType::Float32 ? static_cast<double>(static_cast<float>(value)) : value;
  if (!std::isfinite(value) || value <= 0 || !std::isfinite(rounded) || rounded <= 0)
    throw std::invalid_argument("Cached spectral guard must be positive and finite in its dtype");
}
} // namespace spectral_cache_detail

/// @brief Own detached column-vector estimates of a kernel's right and left singular directions.
///
/// u[out,1] and v[flattened input,1] retain storage after the inputs disappear.
/// An additive squared-norm epsilon means their lengths need not equal one;
/// zero kernels produce zero vectors. No store or fitting counter is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto pair=n::cached_spectral_power_iteration(n::tensor({3.,0.,0.,2.},{2,2}),n::ones({2,1}));
/// // pair.u and pair.v are detached columns; no parameter has changed.
/// @endcode
/// @par References
/// - Miyato et al. (2018), Spectral Normalization for Generative Adversarial Networks,
///   https://arxiv.org/abs/1802.05957 . Column-vector convention with explicit epsilon here.
/// @see cached_spectral_power_iteration(), cached_spectral_kernel()
struct SpectralPowerPair {
  Tensor u; ///< Owning detached right column[out,1], from the final W-transpose update.
  Tensor v; ///< Owning detached left column[flattened input,1], from the final W update.
};

/// @brief Advance detached spectral columns without normalizing the supplied initial direction.
///
/// Flatten W to M[in,out]. Each ordered iteration is v=M*u/sqrt(sum((M*u)^2)+epsilon),
/// then u=M^T*v/sqrt(sum((M^T*v)^2)+epsilon). Uses division and sqrt, not rsqrt;
/// epsilon is added to the squared norm. Default epsilon is double(1e-12f),
/// preserving the Float32-rounded guard even for Float64 inputs. No initial
/// normalization, sigma evaluation, store writes, RNG or graph is created.
/// W/u0 finite matching Float32/64 CPU, positive axes, rank(W)>=2, u0[out,1].
/// Strides allowed; results own storage and have no first/higher gradients.
/// O(iterations*in*out) work and O(in+out) temporary storage plus host validation.
/// Inputs unchanged; independent calls are safe without concurrent input writes.
/// @param weight Finite floating kernel, flattened over all but the final axis.
/// @param u0 Matching finite right column[out,1]; may be zero or non-unit.
/// @param iterations Positive ordered iteration count, default1.
/// @param epsilon Positive representable squared-norm guard, default double(1e-12f).
/// @return Owning detached SpectralPowerPair; zero kernels give two zero columns.
/// @throws std::invalid_argument For invalid inputs/options/geometry or nonfinite intermediate
/// values.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto pair=n::cached_spectral_power_iteration(n::zeros({2,3}),n::ones({3,1}),2);
/// // pair.u[3,1] and pair.v[2,1] are finite zeros, detached from the inputs.
/// @endcode
/// @par References
/// - Miyato et al. (2018), https://arxiv.org/abs/1802.05957 , Algorithm 1.
///   This explicit cached-column protocol adds a squared-norm guard and does not normalize u0
///   first.
/// @see SpectralPowerPair, cached_spectral_kernel(), advance_cached_spectral_state()
inline SpectralPowerPair cached_spectral_power_iteration(
    const Tensor& weight,
    const Tensor& u0,
    std::int64_t  iterations = 1,
    double        epsilon    = static_cast<double>(1e-12f)) {
  NoGradGuard guard;
  const auto  matrix = spectral_cache_detail::Matrix(weight.detach());
  spectral_cache_detail::Column(u0, matrix.size(1), weight.dtype());
  spectral_cache_detail::Positive(epsilon, weight.dtype());
  if (iterations <= 0)
    throw std::invalid_argument("Cached spectral iteration count must be positive");
  Tensor u = u0.detach(), v;
  for (std::int64_t i = 0; i < iterations; ++i) {
    v         = matmul(matrix, u);
    auto norm = sqrt(sum(square(v)) + epsilon);
    spectral_cache_detail::Finite(norm);
    v    = v / norm;
    u    = matmul(matrix, v, true);
    norm = sqrt(sum(square(u)) + epsilon);
    spectral_cache_detail::Finite(norm);
    u = u / norm;
    spectral_cache_detail::Finite(u);
    spectral_cache_detail::Finite(v);
  }
  return {u.detach(), v.detach()};
}

/// @brief Own a live normalized kernel and its signed bilinear denominator estimate.
///
/// Handles retain storage and applicable W gradients after inputs disappear.
/// sigma is raw sum(v*(W*u)), before the positive floor; u/v paths are stopped.
/// Finite power iteration does not certify a unit operator norm.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto result=n::cached_spectral_kernel(n::full({1,1},2.),n::ones({1,1}),n::ones({1,1}));
/// // result.sigma==2 and result.weight[0,0]==1.
/// @endcode
/// @par References
/// - Miyato et al. (2018), https://arxiv.org/abs/1802.05957 .
/// @see cached_spectral_kernel(), SpectralPowerPair
struct CachedSpectralKernel {
  Tensor weight; ///< Owning live W/max(sigma,floor), retaining W's original shape.
  Tensor sigma;  ///< Owning live scalar sum(detach(v)*(flatten(W)*detach(u))); signed, without abs.
};

/// @brief Normalize a kernel using fixed cached columns and a differentiable signed sigma.
///
/// M=flatten(W); sigma=sum(detach(v)*(M*detach(u))); weight=W/max(sigma,floor).
/// No abs, power iteration, cache write, RNG or implicit normalization of u/v.
/// Default floor=double(1e-6f). A negative/zero sigma therefore uses the positive
/// floor. Max uses the backend boundary subgradient. First/higher W paths remain
/// live including the sigma denominator; u/v paths are always stopped.
/// Finite matching Float32/64 CPU W of rank>=2 and columns u[out,1],v[in,1]
/// with positive axes; strides supported. Inputs unchanged; owning outputs retain
/// lifetime. O(in*out) work/storage plus host validation. Independent calls are
/// safe without concurrent input writes. This cached estimate is not an SVD bound.
/// @param weight Finite floating kernel; all leading axes are flattened as input.
/// @param u Matching finite right column[out,1], held constant in backpropagation.
/// @param v Matching finite left column[in,1], held constant in backpropagation.
/// @param floor Positive representable denominator floor, default double(1e-6f).
/// @return Owning CachedSpectralKernel with live effective weight and raw signed sigma.
/// @throws std::invalid_argument For invalid inputs/geometry/guard or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto w=n::full({1,1},2.).requires_grad_();
/// auto result=n::cached_spectral_kernel(w,n::ones({1,1}),n::ones({1,1}));
/// auto derivative=n::grad(n::sum(result.weight),{w});
/// // derivative is zero: the differentiable denominator cancels this scalar scale.
/// @endcode
/// @par References
/// - Miyato et al. (2018), https://arxiv.org/abs/1802.05957 , equations 8 and 10.
///   The positive signed-sigma floor is this library's explicit source protocol.
/// @see CachedSpectralKernel, cached_spectral_power_iteration(), spectral_norm_dense()
inline CachedSpectralKernel cached_spectral_kernel(
    const Tensor& weight,
    const Tensor& u,
    const Tensor& v,
    double        floor = static_cast<double>(1e-6f)) {
  const auto matrix = spectral_cache_detail::Matrix(weight);
  spectral_cache_detail::Column(u, matrix.size(1), weight.dtype());
  spectral_cache_detail::Column(v, matrix.size(0), weight.dtype());
  spectral_cache_detail::Positive(floor, weight.dtype());
  auto sigma = sum(v.detach() * matmul(matrix, u.detach()));
  spectral_cache_detail::Finite(sigma);
  auto normalized = weight / maximum(sigma, scalar_like(floor, sigma));
  spectral_cache_detail::Finite(normalized);
  return {normalized, sigma};
}

/// @brief Explicitly advance every cached /W kernel pair, committing only after all computations
/// pass.
///
/// For each existing kernel "prefix/W", reads "prefix/u" and "prefix/v" in
/// state, validates both, then computes detached ordered iterations from u.
/// Biases/other names are ignored. All candidates are computed before any write,
/// so invalid/missing state or nonfinite computation leaves both stores unchanged.
/// No initialization, optimizer, gradient, RNG, weight write or mode switch is hidden.
/// Existing state Tensor handles see copied values; their requires_grad flags are
/// unchanged, but cache math always detaches them. Exclude state from optimizers.
/// Empty kernel sets are a no-op. Distinct matching Float32/64 stores are required.
/// Work O(iterations*sum(kernel sizes)); candidate storage O(sum(in+out)) plus
/// validation. Synchronize calls sharing either store; this is not a concurrent transaction.
/// @param parameters Mutable parameter owner, read only by this operation.
/// @param state Distinct matching floating store of existing column pairs; only those pairs are
/// written.
/// @param iterations Positive iterations per kernel, default1.
/// @param epsilon Positive representable squared-norm guard, default double(1e-12f).
/// @throws std::invalid_argument For aliased/mismatched stores, invalid options, geometry or
/// nonfinite computation.
/// @throws std::out_of_range For a missing cached u or v; no writes occur.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1),s(0);
/// p.get("layer/W",{2,3},n::init::ones());
/// s.get("layer/u",{3,1},n::init::ones());s.get("layer/v",{2,1},n::init::ones());
/// n::advance_cached_spectral_state(p,s); // only the two state arrays change.
/// @endcode
/// @par References
/// - Miyato et al. (2018), https://arxiv.org/abs/1802.05957 , Algorithm 1.
/// @see cached_spectral_power_iteration(), cached_spectral_kernel()
inline void advance_cached_spectral_state(
    ParameterStore& parameters,
    ParameterStore& state,
    std::int64_t    iterations = 1,
    double          epsilon    = static_cast<double>(1e-12f)) {
  if (&parameters == &state || parameters.dtype() != state.dtype() || iterations <= 0)
    throw std::invalid_argument(
        "Cached spectral advance needs distinct matching stores and positive iterations");
  spectral_cache_detail::Positive(epsilon, parameters.dtype());
  NoGradGuard                                 guard;
  std::vector<std::pair<std::string, Tensor>> next;
  for (const auto& key : parameters.names()) {
    if (key.size() < 2 || key.substr(key.size() - 2) != "/W")
      continue;
    const auto name   = key.substr(0, key.size() - 2);
    const auto weight = parameters.at(key);
    const auto matrix = spectral_cache_detail::Matrix(weight);
    auto       u = state.at(name + "/u"), v = state.at(name + "/v");
    spectral_cache_detail::Column(u, matrix.size(1), parameters.dtype());
    spectral_cache_detail::Column(v, matrix.size(0), parameters.dtype());
    auto pair = cached_spectral_power_iteration(weight, u, iterations, epsilon);
    next.emplace_back(name + "/u", pair.u);
    next.emplace_back(name + "/v", pair.v);
  }
  for (const auto& entry : next)
    state.set(entry.first, entry.second);
}
} // namespace datamunge::nn
