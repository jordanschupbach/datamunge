// Six temporal convolutions and three dense layers on raw one-hot characters
// (port of tflow's common/nn/character_cnn.h; Zhang, Zhao & LeCun 2015).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/kim_cnn.hpp"

namespace charcnn {

namespace nn = datamunge::nn;
using nn::Tensor;

/// The authors' 69-slot alphabet (the hyphen appears twice; lookups use the last one).
inline const std::string CharacterAlphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}";

/// Reversed text, lowercased, truncated from the end; unknown characters and padding are -1.
inline std::vector<int> CharacterEncode(const std::string& text, int time) {
  std::vector<int> out(time, -1);
  for (int i = 0; i < std::min<int>(time, static_cast<int>(text.size())); ++i) {
    unsigned char c = text[text.size() - 1 - i];
    if (c >= 'A' && c <= 'Z') c += 32;
    const auto position = CharacterAlphabet.rfind(static_cast<char>(c));
    if (position != std::string::npos) out[i] = static_cast<int>(position);
  }
  return out;
}

struct CharacterConfig {
  int time = 1014, channels = 256, dense = 1024, alphabet = 69;
};

/// Length after the six VALID convolutions (7, 7, 3, 3, 3, 3) and three pools of 3.
inline int CharacterFinalLength(int length) {
  for (int layer = 0; layer < 6; ++layer) {
    length -= layer < 2 ? 6 : 2;
    if (length < 1) throw std::invalid_argument("character CNN input too short for six convolutions");
    if (layer == 0 || layer == 1 || layer == 5) {
      if (length < 3) throw std::invalid_argument("character CNN input too short for pool");
      length = (length - 3) / 3 + 1;
    }
    if (length < 1) throw std::invalid_argument("character CNN input too short for pool");
  }
  return length;
}

struct CharacterOutput {
  Tensor              logits, quantized, flat;
  std::vector<Tensor> stages;
};

/// ids [N, T] int (-1 = zero vector), lengths [N]. Dropout (keep) after both dense layers while training.
inline CharacterOutput CharacterCnn(nn::ParameterStore& p, const Tensor& ids, const Tensor& lengths, const CharacterConfig& config = {},
                                    bool training = false, double keep = 1.0, nn::Generator* gen = nullptr) {
  const int final_length = CharacterFinalLength(config.time);
  if (config.channels < 1 || config.dense < 1 || config.alphabet < 1) throw std::invalid_argument("character CNN dimensions");
  CharacterOutput out;
  const Tensor    onehot = nn::one_hot(ids, config.alphabet, 1.0, 0.0, p.dtype());
  out.quantized          = nn::mul(onehot, nn::expand_dims(sentence::LengthMask(lengths, config.time, p.dtype()), 2));
  Tensor x               = out.quantized;
  for (int layer = 0; layer < 6; ++layer) {
    nn::ConvLayerOptions opt;
    opt.padding    = nn::Padding::Valid;
    opt.activation = nn::Activation::Relu;
    opt.init       = nn::init::he_uniform();
    x              = nn::conv1d(p, "conv" + std::to_string(layer + 1), x, config.channels, layer < 2 ? 7 : 3, opt);
    out.stages.push_back(x);
    if (layer == 0 || layer == 1 || layer == 5) {
      x = nn::max_pool1d(x, 3, 3, nn::Padding::Valid);
      out.stages.push_back(x);
    }
  }
  out.flat = nn::reshape(x, {-1, static_cast<std::int64_t>(final_length) * config.channels});
  x        = nn::dense(p, "dense1", out.flat, config.dense, nn::Activation::Relu);
  if (training && keep < 1) x = nn::dropout(x, 1 - keep, true, *gen);
  x = nn::dense(p, "dense2", x, config.dense, nn::Activation::Relu);
  if (training && keep < 1) x = nn::dropout(x, 1 - keep, true, *gen);
  out.logits = nn::dense(p, "head", x, 2);
  return out;
}

}  // namespace charcnn
