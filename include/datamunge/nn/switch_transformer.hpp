#pragma once

/// @file switch_transformer.hpp
/// @brief Batch-wide capacity-limited Switch routing and masked character prediction.

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <random>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Differentiable expert contribution and routing diagnostics.
/// @par Example
/// @code
/// dm::nn::ParameterStore parameters(7);
/// auto result = dm::nn::switch_experts(parameters, dm::nn::ones({8, 16}), 4, 2);
/// @endcode
struct SwitchRouting {
  Tensor contribution; ///< Gated expert output [tokens,width], zero on overflow.
  Tensor probabilities; ///< Softmax router probabilities [tokens,experts].
  Tensor route; ///< Int32 highest-scoring expert for each token.
  Tensor slot; ///< Int64 zero-based arrival index at that expert, including overflow.
  Tensor balance; ///< Expert count times the dot product of route frequency and mean probability.
};

/// @brief Route token rows to the highest-scoring expert with a fixed batch-wide capacity.
/// Expert buffers have exactly `capacity` rows. Token order determines which
/// assignments are accepted. Gradients flow through the accepted router
/// weights and through the mean probabilities in the balancing term.
/// @param parameters Trainable CPU parameter store.
/// @param hidden Nonempty floating CPU matrix [tokens,width] matching the parameter dtype.
/// @param experts Positive number of experts.
/// @param capacity Positive number of accepted tokens per expert.
/// @return Expert contribution, probabilities, expert IDs, slots and balance loss.
/// @throws std::invalid_argument For incompatible shapes, dtype, device or capacity.
/// @par Example
/// @code
/// dm::nn::ParameterStore parameters(7);
/// const auto routed = dm::nn::switch_experts(parameters, dm::nn::ones({8, 16}), 4, 2);
/// @endcode
/// @par References
/// - Fedus, Zoph and Shazeer (2022), https://jmlr.org/papers/v23/21-0998.html
inline SwitchRouting switch_experts(ParameterStore& parameters, const Tensor& hidden,
                                    std::int64_t experts, std::int64_t capacity) {
  if (!hidden.defined() || hidden.dim() != 2 || hidden.numel() == 0 ||
      hidden.dtype() != parameters.dtype() || !is_floating(hidden.dtype()) ||
      hidden.device().backend != compute::Backend::CPU || experts < 1 ||
      experts > std::numeric_limits<int>::max() || capacity < 1 ||
      hidden.size(1) > std::numeric_limits<std::int64_t>::max() / 2)
    throw std::invalid_argument("switch_experts: expected a nonempty CPU feature matrix and positive routing sizes");
  const auto tokens = hidden.size(0);
  const auto width = hidden.size(1);
  const auto scores = dense(parameters, "router", hidden, experts);
  const auto probabilities = softmax(scores, -1);
  const auto route = cast(argmax(scores, -1), DType::Int32);
  const auto chosen = route.to_vector<int>();
  std::vector<std::int64_t> counts(experts), slots(tokens);
  std::vector<std::vector<std::int64_t>> accepted(experts);
  for (std::int64_t i = 0; i < tokens; ++i) {
    const auto expert = chosen[i];
    slots[i] = counts[expert]++;
    if (slots[i] < capacity) accepted[expert].push_back(i);
  }
  const auto padded = concat({zeros({1, width}, hidden.dtype()), hidden}, 0);
  auto contribution = zeros_like(hidden);
  for (std::int64_t expert = 0; expert < experts; ++expert) {
    const auto& rows = accepted[expert];
    std::vector<std::int64_t> packing(capacity), unpacking(tokens);
    for (std::size_t slot = 0; slot < rows.size(); ++slot) {
      packing[slot] = rows[slot] + 1;
      unpacking[rows[slot]] = slot + 1;
    }
    const auto packed = gather(padded, from_vector(packing));
    const auto prefix = "expert" + std::to_string(expert);
    const auto activated = dense(parameters, prefix + "/up", packed, 2 * width, Activation::Relu);
    const auto transformed = dense(parameters, prefix + "/down", activated, width);
    const auto scattered = gather(concat({zeros({1, width}, hidden.dtype()), transformed}, 0),
                                 from_vector(unpacking));
    contribution = contribution + scattered * select(probabilities, 1, expert).reshape({tokens, 1});
  }
  const auto frequency = cast(from_vector(counts), hidden.dtype()) / static_cast<double>(tokens);
  const auto balance = sum(frequency * mean(probabilities, {0})) * static_cast<double>(experts);
  return {contribution, probabilities, route, from_vector(slots), balance};
}

/// @brief Masked character model endpoints in row-major token order.
/// @par Example
/// @code
/// const dm::nn::SwitchCharacterOutput output{};
/// @endcode
struct SwitchCharacterOutput {
  Tensor logits; ///< Character scores [batch*time,65].
  Tensor hidden; ///< Inputs to the expert layer [batch*time,24].
  SwitchRouting routing; ///< Expert outputs and routing before residual normalization.
};

/// @brief Encode masked character windows with bidirectional attention and a Switch layer.
/// @param parameters Trainable CPU parameter store.
/// @param input Integral CPU matrix [batch,time] with IDs 0 through 65, including MASK=65.
/// @param capacity Batch-wide capacity per expert.
/// @return Character logits, attention hidden states and routing diagnostics.
/// @throws std::invalid_argument For invalid input geometry, IDs or capacity.
/// @par Example
/// @code
/// dm::nn::ParameterStore parameters(7);
/// const auto result = dm::nn::switch_character_model(parameters, dm::nn::zeros({2, 16}, dm::nn::DType::Int32), 8);
/// @endcode
/// @see switch_experts()
inline SwitchCharacterOutput switch_character_model(ParameterStore& parameters, const Tensor& input,
                                                    std::int64_t capacity) {
  if (!input.defined() || input.dim() != 2 || input.numel() == 0 ||
      !is_integral(input.dtype()) || input.device().backend != compute::Backend::CPU ||
      min(input).item() < 0 || max(input).item() > 65 || capacity < 1)
    throw std::invalid_argument("switch_character_model: expected character IDs in [0,65] and positive capacity");
  const auto batch = input.size(0);
  const auto time = input.size(1);
  const auto word = parameters.get("word", {66, 24}, init::normal(.1));
  const auto position = parameters.get("position", {time, 24}, init::normal(.1));
  const auto embedded = gather(word, input) + position;
  const auto matrix = [&](const std::string& projection) {
    return parameters.get("attention/" + projection, {24, 24}, init::uniform(std::sqrt(3. / 24.)));
  };
  const auto project = [&](const std::string& projection) {
    return matmul(embedded, matrix(projection)).reshape({batch, time, 3, 8}).transpose({0, 2, 1, 3});
  };
  const auto q = project("Wq");
  const auto k = project("Wk");
  const auto v = project("Wv");
  const auto context = scaled_dot_product_attention(q, k, v).output
                           .transpose({0, 2, 1, 3}).reshape({batch, time, 24});
  const auto hidden = layer_norm(parameters, "attention_norm", embedded + matmul(context, matrix("Wo")))
                          .reshape({batch * time, 24});
  auto routing = switch_experts(parameters, hidden, 4, capacity);
  const auto normalized = layer_norm(parameters, "switch_norm", hidden + routing.contribution);
  return {dense(parameters, "head", normalized, 65), hidden, std::move(routing)};
}

/// @brief Replace exactly a fixed number of character IDs per row with MASK=65.
/// @param gold Nonempty integral CPU matrix of character IDs in [0,64].
/// @param rng Random engine advanced by shuffling each row's positions.
/// @param count Number of replacements per row, between zero and the row width.
/// @return Int32 matrix of observed character IDs with the same shape as gold.
/// @throws std::invalid_argument For invalid IDs, shape or count.
/// @par Example
/// @code
/// std::mt19937 rng(7);
/// const auto observed = dm::nn::mask_character_positions(dm::nn::zeros({2,16}, dm::nn::DType::Int32), rng, 3);
/// @endcode
inline Tensor mask_character_positions(const Tensor& gold, std::mt19937& rng, std::int64_t count) {
  if (!gold.defined() || gold.dim() != 2 || gold.numel() == 0 || !is_integral(gold.dtype()) ||
      gold.device().backend != compute::Backend::CPU || count < 0 || count > gold.size(1) ||
      min(gold).item() < 0 || max(gold).item() >= 65)
    throw std::invalid_argument("mask_character_positions: invalid character matrix or mask count");
  auto input = gold.to_vector<int>();
  std::vector<std::int64_t> positions(gold.size(1));
  for (std::int64_t row = 0; row < gold.size(0); ++row) {
    std::iota(positions.begin(), positions.end(), 0);
    std::shuffle(positions.begin(), positions.end(), rng);
    for (std::int64_t j = 0; j < count; ++j) input[row * gold.size(1) + positions[j]] = 65;
  }
  return from_vector(input, gold.shape());
}

} // namespace datamunge::nn
