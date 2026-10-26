// Elman word-level language model with shared full-sequence and one-step
// parameters (port of tflow's common/nn/word_rnn_lm.h and the model of
// common/ml/word_rnn_lm.h): one-hot words [.., 10000] x "embedding/W"
// [10000, 8] ~ N(0, .1^2), an 8-unit tanh RNN "recurrent" (Wx ~ N(0, .1^2),
// Wh ~ N(0, .15^2), b = 0) from a zero state, a 10000-way "output" dense
// (W ~ N(0, .1^2), b = 0); 170136 parameters in 6 tensors. Teacher sequences
// are 16 tokens.
#pragma once

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/checks.hpp"

namespace word_lm {

namespace nn = datamunge::nn;
using nn::Tensor;

constexpr int kVocab = 10000, kUnits = 8, kSteps = 16;

inline void Parameters(nn::ParameterStore& p) {
  p.get("embedding/W", {kVocab, kUnits}, nn::init::normal(.1));
  p.get("recurrent/Wx", {kUnits, kUnits}, nn::init::normal(.1));
  p.get("recurrent/Wh", {kUnits, kUnits}, nn::init::normal(.15));
  p.get("recurrent/b", {kUnits}, nn::init::zeros());
  p.get("output/W", {kUnits, kVocab}, nn::init::normal(.1));
  p.get("output/b", {kVocab}, nn::init::zeros());
}

struct StepPath {
  Tensor hidden, logits;
};

/// One word: one-hot [B, 10000] and state [B, 8] -> next state, logits [B, 10000].
inline StepPath Step(nn::ParameterStore& p, const Tensor& onehot, const Tensor& state) {
  const Tensor h = nn::rnn_step(p, "recurrent", nn::matmul(onehot, p.at("embedding/W")), state);
  return {h, nn::dense(p, "output", h, kVocab)};
}

struct Path {
  Tensor logits, hidden;  // [B, 16, 10000], [B, 16, 8] (states after each word)
};

/// Full zero-state unroll over one-hot words [B, 16, 10000].
inline Path Sequence(nn::ParameterStore& p, const Tensor& encoded) {
  Parameters(p);
  Tensor              h = nn::zeros({encoded.shape()[0], kUnits}, encoded.dtype());
  std::vector<Tensor> states, scores;
  for (const auto& x : nn::time_steps(encoded)) {
    const auto step = Step(p, x, h);
    h               = step.hidden;
    states.push_back(h);
    scores.push_back(step.logits);
  }
  return {nn::stack_time(scores), nn::stack_time(states)};
}

/// Mean next-word NLL over every position.
inline Tensor Loss(const Tensor& logits, const Tensor& target) {
  const Tensor labels = nn::one_hot(nn::reshape(target, {-1}), kVocab);
  return nn::neg(nn::mean(nn::sum(nn::mul(labels, nn::log_softmax(nn::reshape(logits, {-1, kVocab}), -1)), {1})));
}

}  // namespace word_lm
