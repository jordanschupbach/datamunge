// Port of tflow's common/{nn,ml}/stacked_autoencoder.h: symmetric all-sigmoid dense
// autoencoders for greedy layer-wise pretraining with explicit parameter transfer.
// Training steps (forward, MSE, nn::grad / optimizer) are written out in the example.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

/// Dense sigmoid layers "layer<i>" through widths (input, ..., code, ..., output); returns
/// every layer output. Widths must be odd in number, mirrored and strictly decreasing to the code.
inline std::vector<nn::Tensor> SigmoidAutoencoder(nn::ParameterStore& p, const nn::Tensor& image,
                                                  const std::vector<int>& widths) {
  check::Require(widths.size() >= 3 && widths.size() % 2 == 1, "odd symmetric width list includes input and output");
  for (std::size_t i = 0; i < widths.size(); ++i) {
    check::Require(widths[i] > 0 && widths[i] == widths[widths.size() - 1 - i],
                   "positive mirrored encoder/decoder dimensions");
    if (i < widths.size() / 2) check::Require(widths[i] > widths[i + 1], "strictly decreasing encoder widths");
  }
  std::vector<nn::Tensor> outputs;
  auto                    value = image;
  for (std::size_t i = 1; i < widths.size(); ++i) {
    value = nn::dense(p, "layer" + std::to_string(i - 1), value, widths[i], nn::Activation::Sigmoid);
    outputs.push_back(value);
  }
  return outputs;
}

class StackedAutoencoderModel {
 public:
  nn::ParameterStore             parameters;
  std::unique_ptr<nn::Optimizer> optimizer;
  std::vector<int>               widths;
  int                            features, code_index;

  StackedAutoencoderModel(std::vector<int> widths_, std::uint64_t seed, bool training = true)
      : parameters(seed), widths(std::move(widths_)), features(widths.empty() ? 0 : widths.front()),
        code_index(int(widths.size() / 2) - 1) {
    {
      nn::NoGradGuard guard;
      SigmoidAutoencoder(parameters, nn::zeros({1, std::max(features, 1)}), widths);
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  std::vector<nn::Tensor> Net(const nn::Tensor& x) {
    check::Require(x.dim() == 2 && x.size(1) == features, "stage input dimensionality");
    auto layers = SigmoidAutoencoder(parameters, x, widths);
    layers.push_back(ReconstructionMse(layers.back(), x));
    return layers;
  }
  /// Every layer output, then the MSE.
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& x) {
    nn::NoGradGuard guard;
    auto            out = Net(x);
    for (auto& t : out) t = t.contiguous();
    return out;
  }
  /// The layers after the code layer, from an explicit code.
  nn::Tensor Decode(const nn::Tensor& code) {
    nn::NoGradGuard guard;
    auto            value = code;
    for (std::size_t i = std::size_t(code_index) + 2; i < widths.size(); ++i)
      value = nn::dense(parameters, "layer" + std::to_string(i - 1), value, widths[i], nn::Activation::Sigmoid);
    return value.contiguous();
  }
  /// Explicit whole-model weight transfer (optimizer state unchanged).
  void Load(const nn::NamedTensors& arrays) {
    check::Require(arrays.size() == parameters.size(), "transfer all model tensors");
    for (const auto& name : parameters.names()) {
      auto value = Find(arrays, name);
      check::Require(value.dtype() == nn::DType::Float32 && value.shape() == parameters.at(name).shape(),
                     "exact transfer shape and dtype");
      parameters.set(name, value);
    }
  }
};

}  // namespace ae
