#pragma once

/// @file cellular_automata.hpp
/// @brief Differentiable local rollouts for neural cellular automata.

#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/ops.hpp>

#include <stdexcept>

namespace datamunge::nn {

/// @brief Unrolls a shared 3-by-3 perception and pointwise residual update rule.
/// Each step computes tanh(perception(state)), applies the pointwise update,
/// and adds the result gated by that step's mask. Convolutions use zero
/// padding and unit stride. Float32 and Float64 CPU tensors are supported.
/// @param seed Nonempty state tensor [batch, height, width, channels].
/// @param perception Weights [3, 3, channels, hidden].
/// @param update Weights [1, 1, hidden, channels].
/// @param masks Finite update multipliers [steps, batch, height, width, 1]; values
/// zero and one freeze and update a cell, respectively.
/// @return Final state with the seed shape and a differentiable rollout.
/// @throws std::invalid_argument Incompatible sizes, dtypes or devices.
/// @par Example
/// @code
/// const auto cells = dm::nn::zeros({1, 4, 4, 2});
/// const auto state = dm::nn::cellular_automaton_rollout(cells, dm::nn::ones({3, 3, 2, 4}),
///     dm::nn::ones({1, 1, 4, 2}), dm::nn::ones({1, 1, 4, 4, 1}));
/// std::print("{}\n", state.size(3)); // 2
/// @endcode
/// @par References
/// - Mordvintsev et al. (2020), https://distill.pub/2020/growing-ca/. This variant uses learned perception and no alive mask.
/// @see conv2d()
inline Tensor cellular_automaton_rollout(const Tensor& seed, const Tensor& perception,
                                         const Tensor& update, const Tensor& masks) {
  if (!seed.defined() || seed.dim() != 4 || seed.numel() == 0 ||
      (seed.dtype() != DType::Float32 && seed.dtype() != DType::Float64) ||
      seed.device().backend != compute::Backend::CPU ||
      !perception.defined() || perception.dim() != 4 || perception.size(3) == 0 ||
      perception.shape() != Shape{3, 3, seed.size(3), perception.size(3)} ||
      !update.defined() || update.shape() != Shape{1, 1, perception.size(3), seed.size(3)} ||
      !masks.defined() || masks.dim() != 5 || masks.size(0) == 0 ||
      masks.shape() != Shape{masks.size(0), seed.size(0), seed.size(1), seed.size(2), 1} ||
      perception.dtype() != seed.dtype() || update.dtype() != seed.dtype() ||
      masks.dtype() != seed.dtype() || perception.device().backend != compute::Backend::CPU ||
      update.device().backend != compute::Backend::CPU || masks.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("cellular_automaton_rollout: incompatible CPU tensors");
  auto state = seed;
  for (std::int64_t step = 0; step < masks.size(0); ++step) {
    const auto hidden = tanh(conv2d(state, perception));
    state = state + select(masks, 0, step) * conv2d(hidden, update);
  }
  return state;
}

} // namespace datamunge::nn
