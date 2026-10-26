// Elman character language model: port of tflow's common/nn/character_rnn_lm.h
// and common/ml/character_rnn_lm.h. A learned 65x8 character embedding, tanh
// hidden 16 (h' = tanh(e Wx + h Wh + b)), 65-class head; the full 64-step BPTT
// graph and the one-step streaming graph share the same 6 tensors (2025
// parameters). Init as tflow: embedding/Wx/output W ~ N(0, .1), Wh ~ N(0, .15),
// biases 0; store seed 8903. Adam .003.
#pragma once

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/tf_replay.hpp"

namespace charlm {

namespace nn = datamunge::nn;
using nn::Tensor;
using Map = replay::Map;

class CharacterRnnLM {
 public:
  nn::ParameterStore             store{8903};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit CharacterRnnLM(bool training = true) {
    store.get("embedding/W", {65, 8}, nn::init::normal(.1));
    store.get("recurrent/Wx", {8, 16}, nn::init::normal(.1));
    store.get("recurrent/Wh", {16, 16}, nn::init::normal(.15));
    store.get("recurrent/b", {16}, nn::init::zeros());
    store.get("output/W", {16, 65}, nn::init::normal(.1));
    store.get("output/b", {65}, nn::init::zeros());
    if (training) Train();
  }
  void Train() { optimizer = std::make_unique<nn::Optimizer>(store, .003, replay::Adam()); }

  /// One step: one-hot [N, 65], state [N, 16] -> {hidden, logits}.
  std::pair<Tensor, Tensor> StepGraph(const Tensor& onehot, const Tensor& state) const {
    const Tensor e = nn::matmul(onehot, store.at("embedding/W"));
    const Tensor h = nn::tanh(nn::bias_add(nn::add(nn::matmul(e, store.at("recurrent/Wx")), nn::matmul(state, store.at("recurrent/Wh"))),
                                           store.at("recurrent/b")));
    return {h, nn::bias_add(nn::matmul(h, store.at("output/W")), store.at("output/b"))};
  }
  /// encoded [N, 64, 65] -> {logits [N, 64, 65], hidden [N, 64, 16]} from a zero state.
  std::pair<Tensor, Tensor> Sequence(const Tensor& encoded) const {
    // tflow: h0 = encoded[:, 0, 0:16] * 0 (a zero state with the batch size)
    Tensor              h = nn::mul(nn::reshape(nn::slice_axis(nn::slice_axis(encoded, 1, 0, 1), 2, 0, 16), {-1, 16}), 0.0);
    std::vector<Tensor> states, scores;
    for (int t = 0; t < 64; ++t) {
      const auto step = StepGraph(nn::reshape(nn::slice_axis(encoded, 1, t, t + 1), {-1, 65}), h);
      h               = step.first;
      states.push_back(nn::expand_dims(h, 1));
      scores.push_back(nn::expand_dims(step.second, 1));
    }
    return {nn::concat(scores, 1), nn::concat(states, 1)};
  }
  static Tensor Loss(const Tensor& logits, const Tensor& y) {
    return nn::neg(nn::mean(nn::sum(nn::mul(nn::one_hot(nn::reshape(y, {-1}), 65), nn::log_softmax(nn::reshape(logits, {-1, 65}), 1)), {1})));
  }

  Map Evaluate(const Tensor& x, const Tensor& y) {
    nn::NoGradGuard guard;
    const auto      s = Sequence(nn::one_hot(x, 65));
    Map             out{{"logits", s.first.contiguous()}, {"hidden", s.second.contiguous()}, {"loss", Loss(s.first, y)}};
    for (const auto& [k, v] : out)
      for (float z : v.to_vector<float>()) check::Require(std::isfinite(z), "finite character LM fields");
    return out;
  }
  /// One observed character per row and an explicit previous state -> {logits [N, 65], state [N, 16]}.
  std::pair<Tensor, Tensor> Step(const Tensor& token, const Tensor& state) {
    check::Require(token.dim() == 1 && state.shape() == nn::Shape({token.size(0), 16}), "valid single-character state batch");
    for (int c : nn::cast(token, nn::DType::Int32).to_vector<int>()) check::Require(c >= 0 && c < 65, "valid character ID");
    nn::NoGradGuard guard;
    const auto      s = StepGraph(nn::one_hot(token, 65), state);
    return {s.second.contiguous(), s.first.contiguous()};
  }
  /// d sum(logits[:, 32, 0]) / d one-hot input.
  Tensor InputGradient(const Tensor& x) {
    Tensor encoded = nn::one_hot(x, 65);
    encoded.requires_grad_(true);
    const Tensor logits = Sequence(encoded).first;
    return nn::grad(nn::sum(nn::slice_axis(nn::slice_axis(logits, 1, 32, 33), 2, 0, 1)), {encoded})[0].detach().contiguous();
  }
  std::vector<Tensor> RawGradients(const Tensor& x, const Tensor& y) {
    nn::GradOptions go;
    go.allow_unused = true;
    auto       g  = nn::grad(Loss(Sequence(nn::one_hot(x, 65)).first, y), store.parameters(), go);
    const auto ps = store.parameters();
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i].defined() ? g[i].detach() : nn::zeros_like(ps[i]);
    return g;
  }
  Map  Gradient(const Tensor& x, const Tensor& y) { return replay::Named(store, RawGradients(x, y)); }
  void Update(const Tensor& x, const Tensor& y) { optimizer->step(RawGradients(x, y)); }
  Map  Arrays() const { return replay::Arrays(store); }

  /// tflow's Sample: stream the prompt through the one-step graph, then draw
  /// `count` characters from softmax(z / temperature) (u = (mt19937() + .5) / 2^32),
  /// feeding each SAMPLED character back.
  Map Sample(const std::vector<int>& prompt, int count, std::uint32_t seed, double temperature) {
    check::Require(!prompt.empty() && count > 0 && temperature > 0 && std::isfinite(temperature), "valid character prompt/count/temperature");
    Tensor                    state = nn::zeros({1, 16});
    std::vector<std::int32_t> codes(prompt.begin(), prompt.end());
    std::vector<double>       uniform;
    std::vector<float>        scores, states;
    std::pair<Tensor, Tensor> output;
    for (int p : prompt) {
      output = Step(nn::from_vector(std::vector<std::int32_t>{p}, {1}), state);
      state  = output.second;
    }
    std::mt19937 rng(seed);
    for (int t = 0; t < count; ++t) {
      const double u = (double(rng()) + .5) / 4294967296.;
      uniform.push_back(u);
      const auto z = output.first.to_vector<float>();
      for (float v : z) scores.push_back(v);
      for (float v : state.to_vector<float>()) states.push_back(v);
      double maximum = -INFINITY, mass = 0, cumulative = 0;
      for (float v : z) maximum = std::max(maximum, double(v) / temperature);
      for (float v : z) mass += std::exp(double(v) / temperature - maximum);
      int chosen = 64;
      for (int c = 0; c < 65; ++c) {
        cumulative += std::exp(double(z[c]) / temperature - maximum) / mass;
        if (u < cumulative) {
          chosen = c;
          break;
        }
      }
      codes.push_back(chosen);
      output = Step(nn::from_vector(std::vector<std::int32_t>{chosen}, {1}), state);
      state  = output.second;
    }
    return {{"codes", nn::from_vector(codes, {std::int64_t(codes.size())})},
            {"uniform", nn::from_vector(uniform, {count})},
            {"logits", nn::from_vector(scores, {count, 65})},
            {"states", nn::from_vector(states, {count, 16})}};
  }
};

}  // namespace charlm
