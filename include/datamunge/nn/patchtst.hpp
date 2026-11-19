#pragma once

/// @file patchtst.hpp
/// @brief A univariate PatchTST encoder for direct multi-step forecasts.

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/views.hpp>

#include <limits>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Direct forecast and encoder traces.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::patchtst(p, "series", dm::nn::ones({2, 24}), 6);
/// std::println("{}", y.patches.size(1)); // 7
/// @endcode
/// @see patchtst()
struct PatchTSTOutput {
  Tensor forecast; ///< Predicted values [batch, horizon].
  Tensor patches; ///< Observed windows [batch, patches, length].
  Tensor weights; ///< Attention weights [batch, heads, patches, patches].
};

/// @brief Encode overlapping observed patches with one attention block and project all leads.
///
/// Patches are right-aligned so the final patch ends at the newest
/// observation. Each is projected and given a learned position embedding.
/// The encoder uses post-normalized residual attention and a GELU feed-forward
/// layer. The head reads all encoded patches. A parameter prefix is tied to
/// its architecture and patch count.
/// @param store CPU parameter store matching the history dtype.
/// @param name Parameter prefix.
/// @param history CPU Float32 or Float64 matrix [batch, lookback].
/// @param horizon Positive forecast length.
/// @param length Positive patch length at most lookback, default 6.
/// @param stride Positive patch stride, default 3.
/// @param width Positive embedding width, default 16.
/// @param heads Positive attention head count dividing width, default 2.
/// @return Differentiable forecast, patches and attention weights.
/// @throws std::invalid_argument Invalid dimensions, dtype, or stored parameter shapes.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::patchtst(p, "series", dm::nn::ones({2, 24}), 6);
/// std::println("{}", y.forecast.size(1)); // 6
/// @endcode
/// @par References
/// - Nie et al., A Time Series is Worth 64 Words, https://arxiv.org/abs/2211.14730
/// @see sliding_window(), multi_head_attention()
[[nodiscard]] inline PatchTSTOutput patchtst(
    ParameterStore& store, const std::string& name, const Tensor& history, std::int64_t horizon,
    std::int64_t length = 6, std::int64_t stride = 3, std::int64_t width = 16,
    std::int64_t heads = 2) {
  if (!history.defined() || history.dim() != 2 || history.size(0) < 1 ||
      history.device().backend != compute::Backend::CPU || history.dtype() != store.dtype() ||
      length < 1 || length > history.size(1) || stride < 1 || horizon < 1 ||
      heads < 1 || width < 1 || width % heads != 0 ||
      width > std::numeric_limits<std::int64_t>::max() / 2)
    throw std::invalid_argument("patchtst: invalid history or architecture");
  const auto lookback = history.size(1);
  const auto offset = (lookback - length) % stride;
  const auto patches = sliding_window(slice_axis(history, 1, offset, lookback), 1, length, stride);
  const auto count = patches.size(1);
  if (count > std::numeric_limits<std::int64_t>::max() / width)
    throw std::invalid_argument("patchtst: flattened encoder width overflow");
  const auto embedded = dense(store, name + "/patch", patches, width);
  const auto position = store.get(name + "/position", {count, width}, init::normal(.02));
  const auto tokens = embedded + position;
  MultiHeadOptions options;
  options.need_weights = true;
  const auto attended = multi_head_attention(store, name + "/attention", tokens, tokens, heads, options);
  const auto state = layer_norm(store, name + "/attention_norm", tokens + attended.output);
  const auto hidden = dense(store, name + "/expand", state, 2 * width, Activation::GeluTanh);
  const auto projected = dense(store, name + "/contract", hidden, width);
  const auto encoded = layer_norm(store, name + "/output_norm", state + projected);
  return {dense(store, name + "/head", encoded.reshape({history.size(0), count * width}), horizon),
          patches, attended.weights};
}

}  // namespace datamunge::nn
