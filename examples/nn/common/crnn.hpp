// Shi/Bai/Yao CRNN: seven convolutions, asymmetric pooling, two BLSTMs
// (port of tflow's common/nn/crnn.h). Input [N, 32, 100, 1] in [0, 1];
// output time-major logits [26, N, 37] for CTC with blank = 36.
#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

namespace crnn {

namespace nn = datamunge::nn;
using nn::Tensor;

struct CrnnConfig {
  int divisor = 1, hidden = 256;
};

struct CrnnOutput {
  Tensor              logits;  // [26, N, 37]
  std::vector<Tensor> stages;  // after each convolution's ReLU
};

/// Bidirectional LSTM over x [N, T, features] (zero initial state, forget
/// bias 0) with a shared projection of [forward, backward] to `output` units.
inline Tensor CrnnBiLstm(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t hidden,
                         std::int64_t output) {
  const auto          xs = nn::time_steps(x);
  const std::size_t   T  = xs.size();
  const auto          N  = x.shape()[0];
  nn::LstmState       fw = nn::lstm_zero_state(N, hidden), bw = nn::lstm_zero_state(N, hidden);
  std::vector<Tensor> left(T), right(T), both(T);
  for (std::size_t t = 0; t < T; ++t) {
    fw      = nn::lstm_step(p, name + "/forward", xs[t], fw, 0.0);
    left[t] = fw.h;
    const std::size_t r = T - 1 - t;
    bw       = nn::lstm_step(p, name + "/backward", xs[r], bw, 0.0);
    right[r] = bw.h;
  }
  for (std::size_t t = 0; t < T; ++t) both[t] = nn::concat({left[t], right[t]}, 1);
  nn::DenseOptions projection;
  projection.init = nn::init::glorot_uniform();
  return nn::dense(p, name + "/projection", nn::stack_time(both), output, projection);
}

/// `update_stats` = false runs training-mode batch norm without touching the
/// moving averages (TF evaluates gradients without running the update op).
inline CrnnOutput Crnn(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& pixels,
                       const CrnnConfig& config = {}, bool training = true, bool update_stats = true) {
  if (config.divisor < 1 || 64 % config.divisor || config.hidden < 1)
    throw std::invalid_argument("CRNN width configuration");
  CrnnOutput result;
  Tensor     x       = nn::div(nn::sub(nn::mul(pixels, 255.0), 128.0), 128.0);
  const int  widths[] = {64, 128, 256, 256, 512, 512, 512};
  for (int layer = 0; layer < 7; ++layer) {
    const int            out = widths[layer] / config.divisor;
    nn::ConvLayerOptions opt;
    opt.padding = layer == 6 ? nn::Padding::Valid : nn::Padding::Same;
    x           = nn::conv2d(p, "conv" + std::to_string(layer + 1), x, out, layer == 6 ? 2 : 3, opt);
    if (layer == 2 || layer == 4 || layer == 6) {
      nn::BatchNormOptions bn;
      bn.momentum     = .9;
      bn.eps          = 1e-5;
      bn.update_stats = update_stats;
      x               = nn::batch_norm(p, state, "bn" + std::to_string(layer + 1), x, training, bn);
    }
    x = nn::relu(x);
    result.stages.push_back(x);
    if (layer == 0 || layer == 1) x = nn::max_pool2d(x, 2, 2);
    if (layer == 3 || layer == 5) {
      // Torch padding is one column on BOTH sides; TensorFlow SAME differs.
      x = nn::pad(x, {{0, 0}, {0, 0}, {1, 1}, {0, 0}}, nn::PadMode::Constant, -std::numeric_limits<double>::infinity());
      x = nn::max_pool2d(x, {2, 2}, {2, 1});
    }
  }
  const Tensor sequence = nn::squeeze(x, {1});  // [N, 26, C]
  const Tensor first  = CrnnBiLstm(p, "blstm1", sequence, config.hidden, config.hidden);
  const Tensor second = CrnnBiLstm(p, "blstm2", first, config.hidden, 37);
  result.logits       = nn::transpose(second, {1, 0, 2});
  return result;
}

}  // namespace crnn
