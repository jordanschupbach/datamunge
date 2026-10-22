// Small MNIST CNN (port of tflow's common/nn/cnn_baselines.h). Parameter
// names match tflow's cnn_mnist checkpoint: conv1, conv2, dense, output.
// Intermediate activations are exposed for interpretation examples.
#pragma once

#include <datamunge/nn/nn.hpp>

namespace cnn_baselines {

namespace nn = datamunge::nn;

struct MnistCnnOutputs {
  nn::Tensor image, conv1, pool1, conv2, pool2, hidden, logits;
};

/// 3x3 Conv(16)-ReLU-pool, 3x3 Conv(32)-ReLU-pool, Dense(64)-ReLU, 10 logits.
/// `input` is [N, 784] or [N, 28, 28, 1].
inline MnistCnnOutputs MnistCnn(nn::ParameterStore& p, const nn::Tensor& input) {
  MnistCnnOutputs        out;
  nn::ConvLayerOptions   options;
  options.activation = nn::Activation::Relu;
  out.image  = nn::reshape(input, {-1, 28, 28, 1});
  out.conv1  = nn::conv2d(p, "conv1", out.image, 16, 3, options);
  out.pool1  = nn::max_pool2d(out.conv1, 2, 2);
  out.conv2  = nn::conv2d(p, "conv2", out.pool1, 32, 3, options);
  out.pool2  = nn::max_pool2d(out.conv2, 2, 2);
  out.hidden = nn::dense(p, "dense", nn::reshape(out.pool2, {-1, 7 * 7 * 32}), 64, nn::Activation::Relu);
  out.logits = nn::dense(p, "output", out.hidden, 10);
  return out;
}

}  // namespace cnn_baselines
