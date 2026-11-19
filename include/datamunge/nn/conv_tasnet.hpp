#pragma once

/// @file conv_tasnet.hpp
/// @brief Two-source waveform separation and a permutation-invariant objective.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/audio.hpp"
#include "datamunge/nn/tcn_models.hpp"

namespace datamunge::nn {

/// @brief Separate two waveforms with a compact Conv-TasNet model.
/// Uses a learned overlapping encoder, four residual causal mask convolutions
/// and a shared learned overlap-add decoder. Lazily creates model parameters.
/// @param model Parameter store with the mixture dtype.
/// @param mixture Tensor [batch, samples, 1] with positive batch and sample length divisible by 8, at least 16.
/// @param width Encoder and mask network width, default 16.
/// @return Tensor [batch, 2, samples] containing the two estimates.
/// @throws std::invalid_argument For incompatible shapes, dtype, width or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(1);
/// auto estimates = dm::nn::conv_tasnet(model, dm::nn::ones({1, 64, 1}), 4);
/// @endcode
/// @par References
/// - Y. Luo and N. Mesgarani (2019), Conv-TasNet. doi:10.1109/TASLP.2019.2915167
/// @see overlap_and_add(), permutation_invariant_si_snr_loss()
inline Tensor conv_tasnet(ParameterStore& model, const Tensor& mixture, int width = 16) {
  constexpr int window = 16, hop = 8;
  if (!mixture.defined() || mixture.dim() != 3 || mixture.size(0) < 1 ||
      mixture.size(1) < window || mixture.size(1) % hop != 0 || mixture.size(2) != 1 ||
      mixture.dtype() != model.dtype() || width < 1) {
    throw std::invalid_argument("Invalid Conv-TasNet input");
  }
  const auto latent = conv1d(model, "encoder", mixture, width, window,
      {.stride = hop, .padding = Padding::Valid, .activation = Activation::Relu});
  auto state = latent;
  for (int dilation : {1, 2, 4, 8}) {
    const auto residual = models::TcnCausalConv(model, "mask" + std::to_string(dilation),
                                              state, width, 3, dilation);
    state = state + 0.25 * relu(residual);
  }
  const auto mask = sigmoid(conv1d(model, "mask_head", state, 2LL * width, 1));
  std::vector<Tensor> sources;
  for (int source : {0, 1}) {
    const auto masked = latent * slice_axis(mask, 2, 1LL * source * width, (source + 1LL) * width);
    sources.push_back(overlap_and_add(dense(model, "decoder", masked, window), hop));
  }
  return stack(sources, 1);
}

/// @brief Negative mean scale-invariant SNR for the best two-source assignment.
/// Centers each waveform and compares the projection onto its reference to the
/// residual energy, with a 1e-8 floor added to the energy terms. The better
/// assignment is selected independently for each example for backpropagation.
/// @param estimate Inferred waveforms [batch, 2, samples].
/// @param reference Matching reference waveforms in either source order.
/// @return Differentiable scalar loss in negative decibels.
/// @throws std::invalid_argument For incompatible shapes or nonfloating dtypes.
/// @par Example
/// @code
/// auto signal = dm::nn::arange(0, 16, 1, dm::nn::DType::Float32).reshape({1, 2, 8});
/// auto loss = dm::nn::permutation_invariant_si_snr_loss(signal, signal);
/// @endcode
/// @par References
/// - Y. Luo and N. Mesgarani (2019), Conv-TasNet. doi:10.1109/TASLP.2019.2915167
/// @see conv_tasnet()
inline Tensor permutation_invariant_si_snr_loss(const Tensor& estimate, const Tensor& reference) {
  if (!estimate.defined() || !reference.defined() || estimate.dim() != 3 ||
      estimate.shape() != reference.shape() || estimate.size(1) != 2 ||
      estimate.size(0) < 1 || estimate.size(2) < 1 || estimate.dtype() != reference.dtype() ||
      (estimate.dtype() != DType::Float32 && estimate.dtype() != DType::Float64)) {
    throw std::invalid_argument("SI-SNR requires matching floating [batch, 2, samples] tensors");
  }
  const auto y = estimate - mean(estimate, {2}, true);
  const auto x = reference - mean(reference, {2}, true);
  /// Computes per-example decibel scores for one source pairing.
  const auto score = [](const Tensor& prediction, const Tensor& target) {
    constexpr double eps = 1e-8;
    const auto projection = target * (sum(target * prediction, {2}, true) /
                                      (sum(square(target), {2}, true) + eps));
    return (10 / std::log(10.0)) * log((sum(square(projection), {2}) + eps) /
                                      (sum(square(prediction - projection), {2}) + eps));
  };
  const auto y0 = slice_axis(y, 1, 0, 1), y1 = slice_axis(y, 1, 1, 2);
  const auto x0 = slice_axis(x, 1, 0, 1), x1 = slice_axis(x, 1, 1, 2);
  return -0.5 * mean(maximum(score(y0, x0) + score(y1, x1),
                             score(y0, x1) + score(y1, x0)));
}

}  // namespace datamunge::nn
