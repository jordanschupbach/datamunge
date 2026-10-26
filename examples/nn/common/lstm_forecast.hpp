// Direct three-horizon forecast from twelve past observations (port of
// tflow's common/nn/lstm_forecast.h): an 8-unit LSTM "lstm" over the
// normalized series from a zero state and a Glorot dense "head" 8 -> 3 on
// the last hidden state; 347 parameters in 5 tensors.
#pragma once

#include <datamunge/nn/nn.hpp>

namespace forecast {

namespace nn = datamunge::nn;
using nn::Tensor;

/// x [B, 12, 1] (already normalized) -> [B, 3].
inline Tensor LstmForecast(nn::ParameterStore& p, const Tensor& x) {
  nn::LstmState s = nn::lstm_zero_state(x.shape()[0], 8, x.dtype());
  for (const auto& step : nn::time_steps(x)) s = nn::lstm_step(p, "lstm", step, s);
  return nn::dense(p, "head", s.h, 3);
}

/// Prediction in observation units: the net works on (x - mean) / sd.
inline Tensor Predict(nn::ParameterStore& p, const Tensor& x, double mean, double sd) {
  return nn::add(nn::mul(LstmForecast(p, nn::div(nn::sub(x, mean), sd)), sd), mean);
}

/// Mean squared error in normalized units.
inline Tensor NormalizedMse(const Tensor& prediction, const Tensor& target, double sd) {
  return nn::mean(nn::square(nn::div(nn::sub(prediction, target), sd)));
}

}  // namespace forecast
