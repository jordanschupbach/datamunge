#pragma once
/// @file captured_affine_batch_norm.hpp
/// @brief Fused affine BN with detached batch captures and caller-owned moving updates.
#include "datamunge/nn/captured_batch_norm.hpp"
#include "datamunge/nn/conv.hpp"

namespace datamunge::nn {
/// @brief Own fused affine normalized values and optional detached batch moments.
///
/// value retains applicable input/gamma/beta graphs; mean and unbiased_variance[C]
/// are detached as returned by batch_norm_train(), undefined in inference mode.
/// Fields retain storage after parent/input/store collection. No store reference,
/// optimizer, moving write or callback is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(0),s(0);
/// auto out=dm::nn::capture_affine_batch_norm(p,s,"bn",dm::nn::ones({2,1}),true);
/// // out.mean[0]==1, unbiased_variance[0]==0; saved statistics remain0/1.
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), Batch Normalization, https://arxiv.org/abs/1502.03167 .
/// @see capture_affine_batch_norm(), batch_norm_train(), batch_norm_moving_values()
struct AffineBatchNormCapture {
  Tensor value; ///< Owning normalized/affine input-shaped output with applicable graphs.
  Tensor mean;  ///< Owning detached channel mean[C], undefined for saved-statistics mode.
  Tensor unbiased_variance; ///< Owning detached sample variance[C], undefined for saved-statistics
                            ///< mode.
};

/// @brief Apply learned affine fused BN and capture moments without updating moving state.
///
/// Finite matching Float32/64 CPU input[...,C], rank>=2, positive axes and nonempty
/// name. Two distinct same-dtype stores are required. Creates gamma[C] ones and
/// beta[C] zeros before moving_mean[C] zeros/moving_var[C] ones, preserving source
/// parameter order. Batch mode delegates exactly to batch_norm_train(): population
/// normalization, detached mean and unbiased sample variance (zero for one
/// observation). Saved mode delegates to batch_norm_infer() with detached saved
/// moments and returns undefined captures. Epsilon is Double .001 by default;
/// this fused path has different Float32 rounding from capture_batch_norm().
///
/// Applicable first/higher input/affine paths stay live, captured/saved moments
/// stop. Strides allowed; no clipping, EMA, optimizer, fitting or hidden state
/// write after initialization. Finite matching saved moments and nonnegative
/// variance required. Input checks precede leaves; stored-leaf/math failures may
/// occur after earlier missing leaves were created. Owning outputs; inputs unchanged.
/// Work/storage O(input.numel()); synchronize shared store/input mutation and
/// exclude moving state from optimizers despite ParameterStore leaf flags.
/// @param parameters Mutable matching affine owner, distinct from moving.
/// @param moving Mutable matching saved-moment owner, unchanged after initialization.
/// @param name Nonempty shared prefix for gamma/beta/moving_mean/moving_var.
/// @param input Finite signed Float32/64 CPU[...,C], rank>=2, positive axes.
/// @param batch_statistics Use fused batch normalization/captures when true, saved statistics when
/// false(default).
/// @param epsilon Positive finite representable variance guard, default .001.
/// @return Owning AffineBatchNormCapture with live output and optional detached moments.
/// @throws std::invalid_argument For invalid inputs/options/stores, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(0),s(0);
/// auto out=n::capture_affine_batch_norm(p,s,"bn",n::tensor({1.,3.},{2,1}),true);
/// auto
/// next=n::batch_norm_moving_values(s.at("bn/moving_mean"),s.at("bn/moving_var"),out.mean,out.unbiased_variance,.9);
/// // Explicit Double .9 retention; next is detached and has not been written.
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167 .
/// - TensorFlow FusedBatchNormV3,
/// https://www.tensorflow.org/api_docs/python/tf/raw_ops/FusedBatchNormV3 , sample variance
/// captures.
/// @see AffineBatchNormCapture, capture_batch_norm(), batch_norm_moving_values()
inline AffineBatchNormCapture capture_affine_batch_norm(
    ParameterStore&    parameters,
    ParameterStore&    moving,
    const std::string& name,
    const Tensor&      input,
    bool               batch_statistics = false,
    double             epsilon          = .001) {
  captured_bn_detail::Finite(input);
  const double rounded = input.dtype() == DType::Float32 ? double(float(epsilon)) : epsilon;
  if (input.dim() < 2 || &parameters == &moving || parameters.dtype() != input.dtype() ||
      moving.dtype() != input.dtype() || name.empty() || !std::isfinite(epsilon) || epsilon <= 0 ||
      !std::isfinite(rounded) || rounded <= 0)
    throw std::invalid_argument("Invalid captured affine BN geometry, stores or epsilon");
  const auto c     = input.shape().back();
  auto       gamma = parameters.get(name + "/gamma", {c}, init::ones());
  auto       beta  = parameters.get(name + "/beta", {c}, init::zeros());
  captured_bn_detail::Finite(gamma);
  captured_bn_detail::Finite(beta);
  auto mean_value = moving.get(name + "/moving_mean", {c}, init::zeros()).detach();
  auto variance   = moving.get(name + "/moving_var", {c}, init::ones()).detach();
  captured_bn_detail::Moments(mean_value, variance, c, input.dtype());
  AffineBatchNormCapture result;
  if (batch_statistics) {
    auto out = batch_norm_train(input, gamma, beta, epsilon);
    result   = {out.y, out.mean.detach(), out.variance.detach()};
    captured_bn_detail::Moments(result.mean, result.unbiased_variance, c, input.dtype());
  } else
    result.value = batch_norm_infer(input, gamma, beta, mean_value, variance, epsilon);
  captured_bn_detail::Finite(result.value);
  return result;
}
} // namespace datamunge::nn
