// Highway layers with coupled transform/carry gates (port of tflow's
// common/nn/highway.h): y = H(x)*T(x) + x*(1-T(x)), T(x) = sigmoid(W_T*x+b_T).
// Dense and convolutional forms preserve dimensions; use a separate plain
// projection when changing width or spatial resolution. Negative gate biases
// initially favor carrying inputs. They do not guarantee identity gradients.
// https://arxiv.org/abs/1505.00387 (Equations 3-5 and Section 2.1)
//
// Parameters: "<name>/transform/W|b" (He-uniform, ReLU by default) and
// "<name>/gate/W" (Glorot-uniform) with "<name>/gate/b" initialized to the
// gate bias; convolutions use SAME padding.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>

namespace highway {

namespace nn = datamunge::nn;

struct HighwayOutputs {
  nn::Tensor transformed, gate, output;
};
inline nn::Tensor HighwayMix(const nn::Tensor& x, const nn::Tensor& transformed, const nn::Tensor& gate) {
  return transformed * gate + x * (1.0 - gate);
}
inline HighwayOutputs HighwayDense(nn::ParameterStore& p, const std::string& name, const nn::Tensor& input,
                                   double gate_bias = -2, nn::Activation activation = nn::Activation::Relu) {
  const std::int64_t width = input.shape().back();
  if (width < 1 || !std::isfinite(gate_bias)) throw std::invalid_argument("invalid highway width/gate bias");
  nn::DenseOptions transform;
  transform.init       = nn::init::he_uniform();
  transform.activation = activation;
  HighwayOutputs out;
  out.transformed = nn::dense(p, name + "/transform", input, width, transform);
  const nn::Tensor weights = p.get(name + "/gate/W", {width, width}, nn::init::glorot_uniform());
  const nn::Tensor bias    = p.get(name + "/gate/b", {width}, nn::init::constant(gate_bias));
  out.gate   = nn::sigmoid(nn::matmul(input, weights) + bias);
  out.output = HighwayMix(input, out.transformed, out.gate);
  return out;
}
inline HighwayOutputs HighwayConv2D(nn::ParameterStore& p, const std::string& name, const nn::Tensor& input,
                                    std::int64_t kernel = 3, double gate_bias = -2) {
  const std::int64_t channels = input.shape().back();
  if (channels < 1 || kernel < 1 || kernel % 2 == 0 || !std::isfinite(gate_bias))
    throw std::invalid_argument("invalid highway convolution dimensions/gate bias");
  nn::ConvLayerOptions transform;
  transform.init       = nn::init::he_uniform();
  transform.activation = nn::Activation::Relu;
  HighwayOutputs out;
  out.transformed = nn::conv2d(p, name + "/transform", input, channels, kernel, transform);
  const nn::Tensor weights = p.get(name + "/gate/W", {kernel, kernel, channels, channels}, nn::init::glorot_uniform());
  const nn::Tensor bias    = p.get(name + "/gate/b", {channels}, nn::init::constant(gate_bias));
  out.gate   = nn::sigmoid(nn::conv2d(input, weights, nn::conv_options(1, nn::Padding::Same)) + bias);
  out.output = HighwayMix(input, out.transformed, out.gate);
  return out;
}

}  // namespace highway
