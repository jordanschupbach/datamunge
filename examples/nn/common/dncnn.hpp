// DnCNN: predict additive noise with Conv/ReLU, 15 Conv/BN/ReLU, linear Conv
// (a port of tflow's common/nn/dncnn.h). Kernels are orthogonal (columns when
// fan-in 9*in >= out, rows otherwise; nn::init::orthogonal over [9 in, out]);
// only the first convolution has a bias. BN keeps .05 of the old moving state
// (Torch momentum .95 weights the new batch), eps 1e-4.
// TF -> datamunge: TF draws each kernel from RandomNormal(seed 410 + layer)
// + QR; here the store's generator feeds the same QR construction.
#pragma once

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace denoise {

namespace nn = datamunge::nn;

struct DnCNNResult {
  nn::Tensor              noise, clean;
  std::vector<nn::Tensor> stages;
};

inline nn::Tensor DnSubtract(const nn::Tensor& noisy, const nn::Tensor& noise) { return noisy - noise; }

/// BN with the .05 old-state retention; `update` refreshes the moving statistics (training only).
inline nn::Tensor DnNorm(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& x,
                         bool training, bool update = true) {
  nn::BatchNormOptions o;
  o.momentum     = .05;
  o.eps          = 1e-4;
  o.update_stats = update;
  return nn::batch_norm(p, state, name, x, training, o);
}

inline DnCNNResult DnCNN(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& noisy, int width = 64,
                         bool training = false, bool update = true) {
  if (width < 1) throw std::invalid_argument("positive DnCNN width");
  DnCNNResult result;
  auto convolution = [&](const std::string& name, const nn::Tensor& x, int out, bool bias) {
    nn::ConvLayerOptions o;
    o.bias = bias;
    o.init = nn::init::orthogonal();
    return nn::conv2d(p, name, x, out, 3, o);
  };
  nn::Tensor x = nn::relu(convolution("conv1", noisy, width, true));
  result.stages.push_back(x);
  for (int layer = 2; layer <= 16; ++layer) {
    const auto name = "conv" + std::to_string(layer);
    x = convolution(name, x, width, false);
    x = nn::relu(DnNorm(p, state, name + "/bn", x, training, update));
    result.stages.push_back(x);
  }
  result.noise = convolution("conv17", x, 1, false);
  result.stages.push_back(result.noise);
  result.clean = DnSubtract(noisy, result.noise);
  return result;
}

/// Paper residual objective, averaged over pixels as well: .5 mean((r - v)^2).
inline nn::Tensor DnResidualLoss(const nn::Tensor& predicted_noise, const nn::Tensor& true_noise) {
  return .5 * nn::mean(nn::squared_difference(predicted_noise, true_noise));
}

}  // namespace denoise
