#pragma once

/// @file informer_forecast.hpp
/// @brief Compact Informer forecasting with ProbSparse query selection and distillation.

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Projected operands, sampled keys, and selected-query contexts from a sparse encoder stage.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::informer_forecast(p, "model", dm::nn::ones({1, 24, 1}), 6);
/// std::println("{}", y.full.selected.size(1)); // 4
/// @endcode
/// @see informer_forecast()
struct InformerSparseTrace {
  Tensor query; ///< Query projections [batch, time, width].
  Tensor key; ///< Key projections [batch, time, width].
  Tensor value; ///< Value projections [batch, time, width].
  Tensor sample; ///< Sampled key indices [time, samples].
  Tensor selected; ///< Active query indices [batch, active].
  Tensor mixed; ///< Context after selected-query replacement [batch, time, width].
};

/// @brief Direct forecast and encoder and decoder diagnostics.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::informer_forecast(p, "model", dm::nn::ones({1, 24, 1}), 6);
/// std::println("{}", y.distilled.size(1)); // 12
/// @endcode
/// @see informer_forecast()
struct InformerForecast {
  Tensor forecast; ///< Future predictions [batch, horizon].
  InformerSparseTrace full; ///< First sparse encoder stage.
  InformerSparseTrace compressed; ///< Second sparse encoder stage after distillation.
  Tensor distill_input; ///< Activated convolution before pooling [batch, lookback, width].
  Tensor distilled; ///< Pairwise maximum [batch, lookback / 2, width].
  Tensor seed; ///< Recent observed values followed by zero placeholders [batch, 2 * horizon, 1].
  Tensor decoder_weights; ///< Causal self-attention weights [batch, 2, 2 * horizon, 2 * horizon].
};

/// @brief Forecast all leads with a distilled sparse encoder and dense causal decoder.
///
/// Each encoder stage uses a fixed per-query sample of keys. Top queries
/// attend to the full sequence and all other queries receive mean context.
/// A convolution and pairwise max pooling halve the encoder resolution.
/// Decoder self-attention is causal. Cross-attention reads the full encoded history.
/// @param store CPU parameter store matching the input dtype.
/// @param name Learned parameter prefix.
/// @param history Floating CPU tensor [batch, even lookback, 1].
/// @param horizon Positive number of leads, no greater than lookback.
/// @param width Positive even hidden width, default 12.
/// @param samples Number of sampled keys per query, default 6.
/// @param active Number of full-attention queries, default 4.
/// @return Differentiable forecast and intermediate tensors.
/// @throws std::invalid_argument Invalid dimensions, dtype, or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::informer_forecast(p, "model", dm::nn::ones({1, 24, 1}), 6);
/// std::println("{}", y.forecast.size(1)); // 6
/// @endcode
/// @par References
/// - Zhou et al., Informer, https://arxiv.org/abs/2012.07436
/// @see scaled_dot_product_attention(), multi_head_attention()
[[nodiscard]] inline InformerForecast informer_forecast(
    ParameterStore& store, const std::string& name, const Tensor& history,
    std::int64_t horizon, std::int64_t width = 12, std::int64_t samples = 6,
    std::int64_t active = 4) {
  if (!history.defined() || history.dim() != 3 || history.size(0) < 1 ||
      history.size(2) != 1 || history.size(1) % 2 != 0 ||
      history.dtype() != store.dtype() || history.device().backend != compute::Backend::CPU ||
      horizon < 1 || horizon > history.size(1) || width < 2 || width % 2 != 0 ||
      width > std::numeric_limits<std::int64_t>::max() / 2 ||
      samples < 1 || samples > history.size(1) / 2 || active < 1 || active > history.size(1) / 2)
    throw std::invalid_argument("informer_forecast: invalid history or architecture");

  /// @brief Embed scalar readings and add learned positional vectors.
  const auto embed = [&](const std::string& prefix, const Tensor& x) {
    const auto projected = dense(store, prefix + "/input", x, width);
    return projected + store.get(prefix + "/position", {x.size(1), width}, init::normal(.02));
  };
  /// @brief Apply the normalized feed-forward residual.
  const auto feedforward = [&](const std::string& prefix, const Tensor& x) {
    const auto hidden = dense(store, prefix + "/expand", x, 2 * width, Activation::Relu);
    return layer_norm(store, prefix + "/norm", x + dense(store, prefix + "/contract", hidden, width));
  };
  /// @brief Replace mean context at the selected query positions.
  const auto sparse = [&](const std::string& prefix, const Tensor& x, InformerSparseTrace& trace) {
    const auto length = x.size(1);
    const auto q = dense(store, prefix + "/query", x, width);
    const auto k = dense(store, prefix + "/key", x, width);
    const auto v = dense(store, prefix + "/value", x, width);
    std::mt19937 rng(43017u + static_cast<unsigned>(length));
    std::vector<std::int64_t> keys(length), picks(length * samples);
    std::iota(keys.begin(), keys.end(), 0);
    for (std::int64_t t = 0; t < length; ++t) {
      std::shuffle(keys.begin(), keys.end(), rng);
      std::copy_n(keys.begin(), samples, picks.begin() + t * samples);
    }
    const auto sample = from_vector(picks, {length, samples});
    const auto estimate = sum(expand_dims(q, 2) * gather(k, sample, 1), {3});
    const auto selected = top_k(max(estimate, {2}) - sum(estimate, {2}) / double(length), active).second;
    const auto mask = one_hot(selected, length, 1., 0., x.dtype());
    const auto attention = scaled_dot_product_attention(matmul(mask, q), k, v).output;
    const auto average = mean(v, {1}, true);
    const auto mixed = average + matmul(mask, attention - average, true);
    trace = {q, k, v, sample, selected, mixed};
    return layer_norm(store, prefix + "/norm", x + dense(store, prefix + "/out", mixed, width));
  };

  InformerForecast result;
  const auto encoded = embed(name + "/encoder", history);
  const auto full = sparse(name + "/full", encoded, result.full);
  ConvLayerOptions options;
  options.activation = Activation::Elu;
  result.distill_input = conv1d(store, name + "/distill", feedforward(name + "/full_ff", full), width, 3, options);
  result.distilled = max(result.distill_input.reshape({history.size(0), history.size(1) / 2, 2, width}), {2});
  const auto compressed = sparse(name + "/short", result.distilled, result.compressed);
  const auto memory = feedforward(name + "/short_ff", compressed);
  const auto tail = slice_axis(history, 1, history.size(1) - horizon, history.size(1));
  result.seed = concat({tail, zeros_like(tail)}, 1);
  const auto decoder = embed(name + "/decoder", result.seed);
  MultiHeadOptions self_options;
  self_options.causal = true;
  self_options.need_weights = true;
  const auto self = multi_head_attention(store, name + "/self", decoder, decoder, 2, self_options);
  result.decoder_weights = self.weights;
  const auto state = layer_norm(store, name + "/self_norm", decoder + self.output);
  const auto cross = multi_head_attention(store, name + "/cross", state, memory, 2);
  const auto combined = layer_norm(store, name + "/cross_norm", state + cross.output);
  const auto prediction = dense(store, name + "/head", feedforward(name + "/decoder_ff", combined), 1);
  result.forecast = squeeze(slice_axis(prediction, 1, horizon, prediction.size(1)), {2});
  return result;
}

}  // namespace datamunge::nn
