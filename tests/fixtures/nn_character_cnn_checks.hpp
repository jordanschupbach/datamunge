#pragma once
// Independent original host formula and complete-graph regression checks.
#include "nn_sequence_check_support.hpp"
namespace {
std::array<double, 2> Host(const Weights& weights) {
  int                              channels = 3;
  std::vector<std::vector<double>> sequence(123, std::vector<double>(3));
  for (int t = 0; t < 121; ++t)
    sequence[t][(t * 17 + (t / 7) * 5 + (t * t / 11)) % 3] = 1;
  for (int layer = 0; layer < 6; ++layer) {
    const int                        kernel = layer < 2 ? 7 : 3;
    std::vector<std::vector<double>> next(sequence.size() - kernel + 1, std::vector<double>(1));
    for (std::size_t t = 0; t < next.size(); ++t) {
      double value = weights[2 * layer + 1][0];
      for (int k = 0; k < kernel; ++k)
        for (int c = 0; c < channels; ++c)
          value += sequence[t + k][c] * weights[2 * layer][k * channels + c];
      next[t][0] = std::max(0., value);
    }
    channels = 1;
    sequence = next;
    if (layer == 0 || layer == 1 || layer == 5) {
      std::vector<std::vector<double>> pooled(
          (sequence.size() - 3) / 3 + 1, std::vector<double>(1));
      for (std::size_t t = 0; t < pooled.size(); ++t)
        pooled[t][0] =
            std::max({sequence[3 * t][0], sequence[3 * t + 1][0], sequence[3 * t + 2][0]});
      sequence = pooled;
    }
  }
  std::array<double, 2> first{}, second{}, result{};
  for (int c = 0; c < 2; ++c)
    first[c] = std::max(0., weights[13][c] + sequence[0][0] * weights[12][c]);
  for (int c = 0; c < 2; ++c) {
    second[c] = weights[15][c];
    for (int j = 0; j < 2; ++j)
      second[c] += first[j] * weights[14][2 * j + c];
    second[c] = std::max(0., second[c]);
  }
  for (int c = 0; c < 2; ++c) {
    result[c] = weights[17][c];
    for (int j = 0; j < 2; ++j)
      result[c] += second[j] * weights[16][2 * j + c];
  }
  return result;
}
double Objective(const Weights& weights) {
  const auto out = Host(weights);
  return .3 * out[0] - .2 * out[1];
}

void EncodingChecks() {
  using charcnn::CharacterAlphabet;
  const auto a = charcnn::CharacterEncode("Ab -?", 7);
  check::Require(
      a[0] == int(CharacterAlphabet.rfind('?')) && a[1] == int(CharacterAlphabet.rfind('-')) &&
          a[2] == -1 && a[3] == 1 && a[4] == 0 && a[5] == -1 && a[6] == -1,
      "raw reverse/lowercase/space/PAD/hyphen encoding");
  check::Require(
      charcnn::CharacterEncode("abcde", 3) == std::vector<int>({4, 3, 2}),
      "truncate from ending characters before right padding");
  check::Require(
      CharacterAlphabet.size() == 69 && CharacterAlphabet.find('-') != CharacterAlphabet.rfind('-'),
      "exact author69-slot alphabet including repeated hyphen");
  check::Require(
      charcnn::CharacterFinalLength(1014) == 34 && charcnn::CharacterFinalLength(256) == 5 &&
          charcnn::CharacterFinalLength(123) == 1,
      "independent convolution/pool lengths");
  bool rejected = false;
  try {
    charcnn::CharacterFinalLength(122);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected, "reject insufficient convolution/pool length");
  nn::ParameterStore p(389);
  std::vector<int>   ids(123, -1);
  ids[0]         = 0;
  ids[1]         = 68;
  ids[2]         = -1;
  ids[5]         = 1;
  const auto net = charcnn::CharacterCnn(
      p,
      nn::from_vector(ids, {1, 123}),
      nn::from_vector(std::vector<int>{3}, {1}),
      {123, 1, 2, 69});
  const auto q = net.quantized.to_vector<float>();
  for (int t = 0; t < 123; ++t)
    for (int c = 0; c < 69; ++c)
      check::Near(
          q[t * 69 + c],
          float((t == 0 && c == 0) || (t == 1 && c == 68)),
          0,
          "known onehot,unknown zero,legal trailing replacement masked");
  std::cout << " raw "
               "reverse/lowercase/truncation/unknown/duplicate-hyphen/onehot/padding/"
               "minimum-length fixtures passed\n";
}

void HostChecks() {
  nn::ParameterStore p(397);
  std::vector<int>   tokens(123);
  for (int t = 0; t < 123; ++t)
    tokens[t] = (t * 17 + (t / 7) * 5 + (t * t / 11)) % 3;
  const Tensor ids     = nn::from_vector(tokens, {1, 123}),
               lengths = nn::from_vector(std::vector<int>{121}, {1});
  const charcnn::CharacterConfig config{123, 1, 2, 3};
  charcnn::CharacterCnn(p, ids, lengths, config);
  const auto weights =
      sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape& shape) {
        return shape.size() == 1 ? .07f + .02f * static_cast<float>(std::sin(i + v))
                                 : .25f + .1f * static_cast<float>(std::sin(.7 * i + .9 * v + .3));
      });
  const auto net      = charcnn::CharacterCnn(p, ids, lengths, config);
  const auto gradient = nn::grad(
      nn::sum(nn::mul(net.logits, nn::from_vector(std::vector<float>{.3f, -.2f}, {1, 2}))),
      p.parameters());
  const auto expected    = Host(weights);
  const auto z           = net.logits.to_vector<double>();
  double     value_error = 0;
  for (int c = 0; c < 2; ++c)
    value_error = std::max(value_error, std::abs(expected[c] - z[c]));
  int          coordinates = 0;
  const double gradient_error =
      sentence::HostGradientError(weights, gradient, Objective, coordinates);
  check::Require(
      coordinates == 62 && p.size() == 18, "all62 tiny parameter coordinates across18 tensors");
  check::Near(
      value_error, 0, 2e-6, "independent six-conv/ReLU/three-pool/flatten/two-dense/head forward");
  check::Near(gradient_error, 0, 2e-5, "EVERY six-conv and dense parameter host finite difference");
  std::cout << " independent six-conv character forward error=" << value_error
            << ",ALL62 host parameter-coordinate gradient error=" << gradient_error << std::endl;
}

void FullChecks() {
  nn::ParameterStore p(401);
  std::vector<int>   input(1014);
  for (int t = 0; t < 1014; ++t)
    input[t] = (t * 17 + (t / 7) * 5) % 69;
  const auto model = charcnn::CharacterCnn(
      p, nn::from_vector(input, {1, 1014}), nn::from_vector(std::vector<int>{1014}, {1}));
  const auto norms =
      sentence::SquaredNorms(nn::grad(nn::sum(nn::square(model.logits)), p.parameters()));
  const int lengths[] = {1008, 336, 330, 110, 108, 106, 104, 102, 34};
  for (int i = 0; i < 9; ++i)
    check::Require(
        model.stages[i].shape() == nn::Shape{1, lengths[i], 256},
        "original six-conv and three-pool lengths");
  check::Require(
      model.flat.shape() == nn::Shape{1, 8704} && model.logits.shape() == nn::Shape{1, 2},
      "8704flatten and binary adapted head");
  check::Require(
      p.count() == 11335938 && p.size() == 18,
      "independent full original-small binary parameter count");
  for (double n : norms)
    check::Require(std::isfinite(n) && n > 0, "full18 learned gradient tensors finite/nonzero");
  std::cout << " full original-small character model11335938 parameters/18 "
               "tensors,T1014,C256,dense1024,flatten8704;all stage shapes/all learned "
               "gradients passed\n";
}

} // namespace
