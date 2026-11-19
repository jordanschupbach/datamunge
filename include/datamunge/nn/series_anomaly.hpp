#pragma once

/// @file series_anomaly.hpp
/// @brief Recurrent reconstruction and next-step scores for univariate windows.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Reconstruct each history and predict its next sample from one LSTM encoding.
///
/// A second LSTM decodes the encoder's final hidden state with a zero initial
/// cell, using the same code as input at every step. A separate linear head
/// directly predicts the next observation. Both heads support backpropagation.
/// @param store CPU parameter store of the same dtype as history.
/// @param name Prefix for the recurrent layers and heads.
/// @param history Floating CPU tensor [batch, time, 1] with positive batch and time.
/// @param width Positive hidden width, default 12.
/// @return [batch, time + 1], with reconstructed history followed by the next prediction.
/// @throws std::invalid_argument Invalid history, width, or existing parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::series_anomaly_output(p, "detector", dm::nn::zeros({2, 5, 1}));
/// std::println("{}", y.size(1)); // 6
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory, https://doi.org/10.1162/neco.1997.9.8.1735
/// @see lstm_step(), series_anomaly_scores()
[[nodiscard]] inline Tensor series_anomaly_output(
    ParameterStore& store, const std::string& name, const Tensor& history,
    std::int64_t width = 12) {
  if (!history.defined() || history.dim() != 3 || history.size(0) < 1 ||
      history.size(1) < 1 || history.size(2) != 1 || width < 1 ||
      history.device().backend != compute::Backend::CPU || history.dtype() != store.dtype())
    throw std::invalid_argument("series_anomaly_output: invalid history or hidden width");
  const auto zero = lstm_zero_state(history.size(0), width, history.dtype());
  auto encoded = zero;
  const auto encoder = name + "/encoder";
  for (const auto& step : time_steps(history))
    encoded = lstm_step(store, encoder, step, encoded);
  LstmState decoded{encoded.h, zero.c};
  const auto decoder = name + "/decoder";
  const auto reconstruction = name + "/reconstruction";
  std::vector<Tensor> output;
  output.reserve(history.size(1) + 1);
  for (std::int64_t time = 0; time < history.size(1); ++time) {
    decoded = lstm_step(store, decoder, encoded.h, decoded);
    output.push_back(dense(store, reconstruction, decoded.h, 1));
  }
  output.push_back(dense(store, name + "/prediction", encoded.h, 1));
  return concat(output, 1);
}

/// @brief Score reconstructed history and the next prediction separately.
/// @param output Floating CPU matrix [batch, time + 1], with time at least one.
/// @param target Aligned floating CPU observations with the same shape.
/// @return Float64 [batch, 2] with history mean squared error and next-step squared error.
/// @throws std::invalid_argument Undefined, incompatible, or nonfloating matrices.
/// @par Example
/// @code
/// const auto scores = dm::nn::series_anomaly_scores(dm::nn::zeros({1, 3}), dm::nn::ones({1, 3}));
/// std::println("{}", scores.to_vector<double>()[1]); // 1
/// @endcode
/// @see series_anomaly_output()
[[nodiscard]] inline Tensor series_anomaly_scores(const Tensor& output, const Tensor& target) {
  if (!output.defined() || !target.defined() || output.dim() != 2 ||
      output.size(0) < 1 || output.size(1) < 2 || target.shape() != output.shape() ||
      output.device().backend != compute::Backend::CPU || target.device().backend != compute::Backend::CPU ||
      (output.dtype() != DType::Float32 && output.dtype() != DType::Float64) ||
      (target.dtype() != DType::Float32 && target.dtype() != DType::Float64))
    throw std::invalid_argument("series_anomaly_scores: expected aligned CPU float matrices");
  const auto error = square(output.to(DType::Float64) - target.to(DType::Float64));
  const auto rows = output.size(0), time = output.size(1) - 1;
  return concat({mean(slice(error, {0, 0}, {rows, time}), {1}, true),
                 slice(error, {0, time}, {rows, 1})}, 1);
}

}  // namespace datamunge::nn
