#pragma once
/// @file captured_batch_norm.hpp
/// @brief Explicit batch normalization captures and detached moving-moment arithmetic, without
/// state writes.
#include <cmath>
#include <limits>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/params.hpp"

namespace datamunge::nn {
/// @brief Internal finite floating, channel-vector and population-variance validation.
namespace captured_bn_detail {
/// @brief Require defined finite nonempty Float32/64 CPU values with positive axes.
/// @param value Tensor to validate, scalar intermediates allowed.
/// @throws std::invalid_argument For invalid dtype/axes or nonfinite values.
/// @throws std::logic_error For non-CPU storage.
inline void Finite(const Tensor& value) {
  if (!value.defined() || value.numel() < 1 ||
      (value.dtype() != DType::Float32 && value.dtype() != DType::Float64))
    throw std::invalid_argument("Captured BN needs nonempty Float32/64 tensors");
  for (auto n : value.shape())
    if (n <= 0)
      throw std::invalid_argument("Captured BN axes must be positive");
  for (double x : value.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("Captured BN values must be finite");
}
/// @brief Validate matching channel mean/variance vectors and nonnegative variances.
/// @param mean Finite floating CPU[C] mean, C>0.
/// @param variance Finite matching CPU[C] variance, each value nonnegative.
/// @param channels Required positive channel width.
/// @param dtype Required Float32/64 dtype.
/// @throws std::invalid_argument For incompatible or nonfinite/negative moments.
/// @throws std::logic_error For non-CPU storage.
inline void
Moments(const Tensor& mean, const Tensor& variance, std::int64_t channels, DType dtype) {
  Finite(mean);
  Finite(variance);
  if (mean.shape() != Shape({channels}) || variance.shape() != mean.shape() ||
      mean.dtype() != dtype || variance.dtype() != dtype)
    throw std::invalid_argument("Captured BN moments need matching channel vectors");
  for (double x : variance.contiguous().to_vector<double>())
    if (x < 0)
      throw std::invalid_argument("Captured BN variances must be nonnegative");
}
} // namespace captured_bn_detail

/// @brief Own normalized values and optional live batch mean/unbiased variance captures.
///
/// value has the input shape. Batch-mode mean/unbiased_variance[C] retain their
/// first/higher input graphs; both fields are undefined on the saved-statistics
/// path. Handles own storage after inputs/stores disappear. No store, fitting
/// mode, optimizer or deferred callback is retained; no moving update is hidden.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore stats(0);
/// auto out=n::capture_batch_norm(stats,"bn",n::tensor({1.,3.},{2,1}),true);
/// // out.mean[0]==2; out.unbiased_variance[0]==2; stats remains at0/1.
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), Batch Normalization, https://arxiv.org/abs/1502.03167 .
/// @see capture_batch_norm(), BatchNormMovingValues, batch_norm_moving_values()
struct BatchNormCapture {
  Tensor value; ///< Owning live (input-mean)/sqrt(population_variance+epsilon).
  Tensor mean;  ///< Owning live batch channel means[C], undefined in saved-statistics mode.
  Tensor unbiased_variance; ///< Owning live population variance times the source-rounded sample
                            ///< correction[C].
};

/// @brief Normalize with batch or saved moments while capturing training moments without writing
/// state.
///
/// Finite Float32/64 CPU input[...,C], rank>=2 and positive axes, matches stats
/// dtype. Creates name/moving_mean[C] zeros and name/moving_var[C] ones only
/// when missing. Reads saved moments detached; in batch mode computes mean over
/// every leading axis and population variance=mean(square(input-mean)), then
/// divides by sqrt(var+epsilon). No rsqrt, fused affine reassociation, gamma/beta,
/// EMA write, optimizer or fitting is hidden. Default epsilon=double(1e-5f).
/// Captured unbiased variance uses count=double(float(numel)/float(C)) and
/// correction=double(float(count)/float(count-1)), preserving the source-rounded
/// count/correction even for Float64. Batch count must exceed1 after that rounding.
/// Inference allows one observation and returns undefined capture fields.
///
/// Positive representable epsilon, nonempty name, finite matching saved moments
/// and nonnegative saved variance required. Strides and applicable first/higher
/// input paths remain live; saved moments always stop. All input checks precede
/// leaves; incompatible/nonfinite stored moments or computation may fail later
/// after earlier missing state was created. Inputs unchanged, owning outputs
/// retain lifetime. Work/storage O(input.numel()) plus validation. Synchronize
/// shared state and exclude it from optimizers despite ParameterStore flags.
/// @param stats Mutable floating saved-moment owner, unchanged after initialization.
/// @param name Nonempty prefix of moving_mean/moving_var.
/// @param input Finite signed matching floating CPU[...,C], rank>=2, positive axes.
/// @param batch_statistics Use batch moments/capture when true; saved statistics when
/// false(default).
/// @param epsilon Positive finite representable variance guard, default double(1e-5f).
/// @return Owning BatchNormCapture with normalized values and optional batch moments.
/// @throws std::invalid_argument For invalid inputs/options/counts, incompatible state or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore stats(0);
/// auto out=n::capture_batch_norm(stats,"bn",n::tensor({1.,3.},{2,1}),true);
/// // Apply learned affine terms separately; capture moments before the optimizer write.
/// auto saved=n::capture_batch_norm(stats,"bn",n::ones({1,1}),false);
/// // saved.mean is undefined; no EMA update occurred in either call.
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167 . Explicit
///   division/sqrt and rounded unbiased correction preserve the migrated source protocol.
/// @see BatchNormCapture, batch_norm_moving_values(), batch_norm()
inline BatchNormCapture capture_batch_norm(
    ParameterStore&    stats,
    const std::string& name,
    const Tensor&      input,
    bool               batch_statistics = false,
    double             epsilon          = static_cast<double>(1e-5f)) {
  captured_bn_detail::Finite(input);
  const double rounded = input.dtype() == DType::Float32 ? double(float(epsilon)) : epsilon;
  if (input.dim() < 2 || input.dtype() != stats.dtype() || name.empty() ||
      !std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(rounded) || rounded <= 0)
    throw std::invalid_argument("Invalid captured BN input, prefix or epsilon");
  const auto   c     = input.shape().back();
  const double count = double(float(input.numel()) / float(c));
  if (batch_statistics && count <= 1)
    throw std::invalid_argument("Captured unbiased BN needs more than one observation");
  auto saved_mean = stats.get(name + "/moving_mean", {c}, init::zeros()).detach();
  auto saved_var  = stats.get(name + "/moving_var", {c}, init::ones()).detach();
  captured_bn_detail::Moments(saved_mean, saved_var, c, input.dtype());
  auto             mean_value = saved_mean, variance = saved_var;
  BatchNormCapture result;
  if (batch_statistics) {
    Axes axes;
    for (std::int64_t i = 0; i < input.dim() - 1; ++i)
      axes.push_back(i);
    mean_value               = mean(input, axes);
    variance                 = mean(square(input - mean_value), axes);
    result.mean              = mean_value;
    result.unbiased_variance = variance * double(float(count) / float(count - 1));
    captured_bn_detail::Moments(result.mean, result.unbiased_variance, c, input.dtype());
  }
  result.value = (input - mean_value) / sqrt(variance + epsilon);
  captured_bn_detail::Finite(result.value);
  return result;
}

/// @brief Own detached candidate moving means and variances without mutating any store.
///
/// Fields[C] retain storage after inputs disappear. No store reference, graph,
/// optimizer, update counter or write callback is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto
/// next=n::batch_norm_moving_values(n::zeros({1}),n::ones({1}),n::ones({1}),n::full({1},2.),.5);
/// // next.mean[0]==.5; next.variance[0]==1.5; nothing has been written.
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167 .
/// @see batch_norm_moving_values(), BatchNormCapture
struct BatchNormMovingValues {
  Tensor mean;     ///< Owning detached candidate momentum*old_mean+(1-momentum)*captured_mean[C].
  Tensor variance; ///< Owning detached candidate
                   ///< momentum*old_variance+(1-momentum)*captured_unbiased_variance[C].
};

/// @brief Compute detached moving-moment candidates from one already captured batch.
///
/// next=momentum*old+(1-momentum)*captured, with separate elementwise products
/// followed by addition. Default momentum=double(.9f), so its Double complement
/// is exactly double(1.f-.9f). Finite matching Float32/64 CPU[C] vectors with C>0,
/// nonnegative variances and finite momentum in[0,1] required; strides allowed.
/// All four input paths stop at every derivative order. No store initialization,
/// write, optimizer or inference recomputation is hidden. Caller may validate
/// candidates for several layers before writing any, after the optimizer write
/// that uses moments captured in the same forward. Inputs unchanged, outputs own
/// lifetime. Work/storage O(C) plus validation; safe without concurrent input writes.
/// @param old_mean Finite floating CPU[C] saved channel means.
/// @param old_variance Finite matching nonnegative saved channel variances.
/// @param captured_mean Finite matching channel means from the selected forward.
/// @param captured_unbiased_variance Finite matching nonnegative unbiased channel variances.
/// @param momentum Saved-statistic retention in[0,1], default double(.9f).
/// @return Owning detached BatchNormMovingValues; no store is mutated.
/// @throws std::invalid_argument For invalid geometry/dtype/variance/momentum or nonfinite result.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore stats(0);
/// auto capture=n::capture_batch_norm(stats,"bn",n::tensor({1.,3.},{2,1}),true);
/// auto
/// next=n::batch_norm_moving_values(stats.at("bn/moving_mean"),stats.at("bn/moving_var"),capture.mean,capture.unbiased_variance);
/// // Compute the selected gradient and write weights, then explicitly copy next into stats.
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167 .
/// @see BatchNormMovingValues, capture_batch_norm(), ParameterStore::set()
inline BatchNormMovingValues batch_norm_moving_values(
    const Tensor& old_mean,
    const Tensor& old_variance,
    const Tensor& captured_mean,
    const Tensor& captured_unbiased_variance,
    double        momentum = static_cast<double>(.9f)) {
  NoGradGuard guard;
  if (!old_mean.defined() || old_mean.dim() != 1 || !std::isfinite(momentum) || momentum < 0 ||
      momentum > 1)
    throw std::invalid_argument("Invalid moving BN vectors or momentum");
  const auto c     = old_mean.size(0);
  const auto dtype = old_mean.dtype();
  captured_bn_detail::Moments(old_mean, old_variance, c, dtype);
  captured_bn_detail::Moments(captured_mean, captured_unbiased_variance, c, dtype);
  auto mean_value = momentum * old_mean.detach() + (1. - momentum) * captured_mean.detach();
  auto variance =
      momentum * old_variance.detach() + (1. - momentum) * captured_unbiased_variance.detach();
  captured_bn_detail::Moments(mean_value, variance, c, dtype);
  return {mean_value.detach(), variance.detach()};
}
} // namespace datamunge::nn
