#pragma once

/// @file program_induction.hpp
/// @brief Differentiable arithmetic programs and convolutional tape updates.

#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Evaluates a left-to-right arithmetic program with shared opcode weights.
///
/// Each instruction combines the accumulator and the next operand. The weights
/// multiply the add, subtract, and multiply results in that order. Pass a
/// row-wise softmax of logits to learn a soft program or one-hot weights to
/// execute a decoded program. Inputs use a common floating dtype and device
/// supported by the underlying tensor operations.
/// @param operands Nonempty tensor with operands on the last axis, at least two.
/// @param weights Tensor [number of operands minus one, 3].
/// @return Tensor with the operand axis removed. Gradients flow to both inputs.
/// @throws std::invalid_argument Incompatible shapes, dtypes, or devices.
///
/// @par Example
/// @code
/// const auto x = dm::nn::tensor({2., 3., 4.}, {1, 3});
/// const auto code = dm::nn::tensor({{1., 0., 0.}, {0., 0., 1.}});
/// std::println("{}", dm::nn::arithmetic_program(x, code).item<double>()); // 20
/// @endcode
/// @see softmax(), one_hot()
inline Tensor arithmetic_program(const Tensor& operands, const Tensor& weights) {
  if (!operands.defined() || operands.dim() == 0 || operands.numel() == 0 ||
      operands.shape().back() < 2 || !weights.defined() ||
      weights.shape() != Shape{operands.shape().back() - 1, 3} ||
      weights.dtype() != operands.dtype() || weights.device().backend != operands.device().backend ||
      weights.device().index != operands.device().index ||
      (operands.dtype() != DType::Float32 && operands.dtype() != DType::Float64))
    throw std::invalid_argument("arithmetic_program: incompatible operands and weights");

  const auto axis = operands.dim() - 1;
  auto state = select(operands, axis, 0);
  for (std::int64_t slot = 0; slot < weights.size(0); ++slot) {
    auto rhs = select(operands, axis, slot + 1);
    auto row = select(weights, 0, slot);
    state = select(row, 0, 0) * (state + rhs) +
            select(row, 0, 1) * (state - rhs) +
            select(row, 0, 2) * (state * rhs);
  }
  return state;
}

/// @brief Unrolls a shared gated width-three convolution on a batch of tapes.
///
/// Each step interpolates the previous state and a tanh convolution using a
/// sigmoid gate. The width dimension uses SAME zero padding. All tensors use
/// the same Float32 or Float64 dtype and CPU device.
/// @param seed Nonempty state [batch, 1, width, channels].
/// @param proposal Convolution weights [1, 3, channels, channels].
/// @param gate Convolution weights with the proposal shape.
/// @param steps Number of recurrent updates, including zero.
/// @param gate_bias Scalar added to gate logits, default 0.6.
/// @return Final state with the seed shape. The rollout is differentiable.
/// @throws std::invalid_argument Incompatible tensors or invalid steps or bias.
///
/// @par Example
/// @code
/// const auto state = dm::nn::ones({1, 1, 4, 2});
/// const auto kernels = dm::nn::zeros({1, 3, 2, 2});
/// const auto updated = dm::nn::gated_tape_rollout(state, kernels, kernels, 1, 0.);
/// std::println("{}", dm::nn::mean(updated).item<double>()); // 0.5
/// @endcode
/// @par References
/// - Kaiser and Sutskever (2016), Neural GPUs Learn Algorithms, https://arxiv.org/abs/1511.08228.
///   This variant uses one interpolation gate and no reset gate.
/// @see conv2d(), cellular_automaton_rollout()
inline Tensor gated_tape_rollout(const Tensor& seed, const Tensor& proposal,
                                 const Tensor& gate, std::int64_t steps,
                                 double gate_bias = 0.6) {
  if (!seed.defined() || seed.dim() != 4 || seed.numel() == 0 || seed.size(1) != 1 ||
      !proposal.defined() || !gate.defined() ||
      proposal.shape() != Shape{1, 3, seed.size(3), seed.size(3)} ||
      gate.shape() != proposal.shape() || proposal.dtype() != seed.dtype() ||
      gate.dtype() != seed.dtype() || proposal.device().backend != compute::Backend::CPU ||
      gate.device().backend != compute::Backend::CPU || seed.device().backend != compute::Backend::CPU ||
      proposal.device().index != seed.device().index || gate.device().index != seed.device().index ||
      (seed.dtype() != DType::Float32 && seed.dtype() != DType::Float64) ||
      steps < 0 || !std::isfinite(gate_bias))
    throw std::invalid_argument("gated_tape_rollout: incompatible tensors or invalid update settings");

  auto state = seed;
  for (std::int64_t step = 0; step < steps; ++step) {
    const auto fraction = sigmoid(conv2d(state, gate) + gate_bias);
    state = (1. - fraction) * state + fraction * tanh(conv2d(state, proposal));
  }
  return state;
}

} // namespace datamunge::nn
