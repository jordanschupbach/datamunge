// Compact post-norm bidirectional BERT encoder with a tied masked-language-model
// head, and its selected-position-only MLM likelihood: port of tflow's
// common/nn/bert_mlm.h and common/ml/bert_mlm.h.
//
// Encoder (T 32, D 8, 2 heads of 4, FF 16, vocabulary 70): token one-hot @
// token/W + position/W + segment/W[segments] -> LN; one block of biased Q/K/V
// self-attention restricted to valid keys and queries (attn::AttentionAllowed),
// residual + LN, GELU-tanh feed-forward, residual + LN. Every LN output is
// multiplied by the length mask, so padded positions are exactly zero. MLM
// head: LN(gelu(mlm(hidden))) @ token/W^T + output/b (tied input embedding),
// masked. 1606 parameters in 26 tensors with tflow's names and initialisers
// (N(0, .02) weights, zero biases).
#pragma once

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/attention.hpp"
#include "common/checks.hpp"

namespace bert {

namespace nn = datamunge::nn;
using nn::Tensor;
using Batch = std::map<std::string, Tensor>;

inline Batch Rows(const Batch& b, const std::vector<std::int64_t>& rows) {
  Batch out;
  for (const auto& [k, v] : b) out[k] = nn::take_rows(v, rows);
  return out;
}

struct EncoderPath {
  Tensor hidden, attention, embedded;
};
struct MlmPath {
  Tensor logits, hidden, attention, embedded, transformed;
};

/// Float mask [N, T, 1]: 1 where t < lengths[n].
inline Tensor LengthMask(const Tensor& lengths, std::int64_t T) {
  const auto         l = nn::cast(lengths, nn::DType::Int64).to_vector<std::int64_t>();
  std::vector<float> m;
  for (auto n : l)
    for (std::int64_t t = 0; t < T; ++t) m.push_back(t < n ? 1.f : 0.f);
  return nn::from_vector(m, {static_cast<std::int64_t>(l.size()), T, 1});
}

inline Tensor Linear(nn::ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t out) {
  nn::DenseOptions o;
  o.init = nn::init::normal(.02);
  return nn::dense(store, name, x, out, o);
}

inline Tensor Heads(const Tensor& x, std::int64_t heads) {
  return nn::transpose(nn::reshape(x, {x.size(0), x.size(1), heads, x.size(2) / heads}), {0, 2, 1, 3});
}

inline EncoderPath BertEncoder(nn::ParameterStore& store, const Tensor& encoded, const Tensor& segments,
                               const Tensor& lengths, std::int64_t T = 32) {
  constexpr std::int64_t V = 70, D = 8, F = 16;
  if (T < 1) throw std::invalid_argument("positive BERT context required");
  const Tensor table     = store.get("token/W", {V, D}, nn::init::normal(.02));
  const Tensor positions = store.get("position/W", {T, D}, nn::init::normal(.02));
  const Tensor types     = store.get("segment/W", {2, D}, nn::init::normal(.02));
  const Tensor mask      = LengthMask(lengths, T);
  auto norm = [&](const Tensor& x, const std::string& name) { return nn::mul(nn::layer_norm(store, name, x, 1e-5), mask); };
  const Tensor tokens   = nn::matmul(encoded, table);
  const Tensor embedded = norm(nn::add(nn::add(tokens, positions), nn::gather(types, segments, 0)), "embedding_ln");
  const Tensor allowed  = nn::expand_dims(attn::AttentionAllowed(lengths, lengths, T, T), 1);
  const Tensor q = Heads(Linear(store, "attention/q", embedded, D), 2), k = Heads(Linear(store, "attention/k", embedded, D), 2),
               v = Heads(Linear(store, "attention/v", embedded, D), 2);
  const auto   a      = attn::ScaledDotProductAttention(q, k, v, 4, allowed);
  const Tensor mixed  = nn::reshape(nn::transpose(a.output, {0, 2, 1, 3}), {encoded.size(0), T, D});
  Tensor       hidden = norm(nn::add(embedded, Linear(store, "attention/out", mixed, D)), "attention_ln");
  const Tensor ff     = nn::gelu(Linear(store, "ff1", hidden, F), true);
  hidden              = norm(nn::add(hidden, Linear(store, "ff2", ff, D)), "ff_ln");
  return {hidden, a.weights, embedded};
}

inline MlmPath BertMlm(nn::ParameterStore& store, const Tensor& encoded, const Tensor& segments, const Tensor& lengths) {
  constexpr std::int64_t T = 32, D = 8, V = 70;
  const auto   core  = BertEncoder(store, encoded, segments, lengths, T);
  const Tensor table = store.at("token/W");
  const Tensor mask  = LengthMask(lengths, T);
  const Tensor transformed =
      nn::mul(nn::layer_norm(store, "mlm_ln", nn::gelu(Linear(store, "mlm", core.hidden, D), true), 1e-5), mask);
  const Tensor bias   = store.get("output/b", {V}, nn::init::zeros());
  const Tensor logits = nn::mul(nn::bias_add(nn::matmul(transformed, table, false, true), bias), mask);
  return {logits, core.hidden, core.attention, core.embedded, transformed};
}

/// tflow's ml::BertMlmModel: CE averaged over the selected positions only
/// (max(count, 1) in the denominator), Adam .003.
class BertMlmModel {
 public:
  nn::ParameterStore             store{46403};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit BertMlmModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      const Batch     b{{"input", nn::zeros({1, 32}, nn::DType::Int32)},
                        {"segments", nn::zeros({1, 32}, nn::DType::Int32)},
                        {"lengths", nn::full({1}, 32, nn::DType::Int32)}};
      BertMlm(store, Encoded(b), b.at("segments"), b.at("lengths"));
    }
    if (training) Train();
  }
  void Train() {
    nn::OptimizerConfig c;
    c.kind    = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(store, .003, c);
  }

  static Tensor Encoded(const Batch& b) { return nn::one_hot(b.at("input"), 70); }

  Tensor Loss(const MlmPath& p, const Batch& b) const {
    const Tensor ce       = nn::neg(nn::sum(nn::mul(nn::log_softmax(p.logits, -1), nn::one_hot(b.at("target"), 70)), {2}));
    const Tensor selected = b.at("selected");
    return nn::div(nn::sum(nn::mul(ce, selected)), nn::maximum(nn::sum(selected), nn::scalar_like(1.0, selected)));
  }

  /// Forward fields, loss and d loss / d one-hot input.
  Batch Evaluate(const Batch& b) {
    Tensor encoded = Encoded(b);
    encoded.requires_grad_(true);
    const auto   p    = BertMlm(store, encoded, b.at("segments"), b.at("lengths"));
    const Tensor loss = Loss(p, b);
    const Tensor g    = nn::grad(loss, {encoded})[0];
    return {{"logits", p.logits.detach()},   {"hidden", p.hidden.detach()},
            {"attention", p.attention.detach()}, {"embedded", p.embedded.detach()},
            {"transformed", p.transformed.detach()}, {"loss", loss.detach()},
            {"input_gradient", g.detach()}};
  }

  float LossValue(const Batch& b) {
    nn::NoGradGuard guard;
    return Loss(BertMlm(store, Encoded(b), b.at("segments"), b.at("lengths")), b).item<float>();
  }

  /// Input-only mask predictions (no target/selected needed).
  Tensor Predict(const Batch& b) {
    nn::NoGradGuard guard;
    return BertMlm(store, Encoded(b), b.at("segments"), b.at("lengths")).logits;
  }

  std::vector<Tensor> RawGradients(const Batch& b) {
    const auto      p = BertMlm(store, Encoded(b), b.at("segments"), b.at("lengths"));
    nn::GradOptions go;
    go.allow_unused = true;
    auto       g    = nn::grad(Loss(p, b), store.parameters(), go);
    const auto ps   = store.parameters();
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i].defined() ? g[i].detach() : nn::zeros_like(ps[i]);
    return g;
  }

  /// Parameters (or, with `b`, their loss gradients on that batch), by name.
  nn::NamedTensors Arrays(const Batch* b = nullptr) {
    nn::NamedTensors out;
    if (b) {
      const auto g = RawGradients(*b);
      for (std::size_t i = 0; i < g.size(); ++i) out.emplace_back(store.names()[i], g[i]);
    } else {
      for (const auto& name : store.names()) out.emplace_back(name, store.at(name).detach().clone());
    }
    return out;
  }

  void Update(const Batch& b) { optimizer->step(RawGradients(b)); }
};

}  // namespace bert
