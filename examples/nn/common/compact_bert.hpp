// Factorized embeddings, actual cross-layer parameter sharing, and frozen-target
// two-layer-to-one-layer MLM distillation; port of tflow common/{nn,ml}/compact_bert.h.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/tf_replay.hpp"
namespace compact_bert {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = bert::Batch;
struct Config {
  int  embedding = 8, layers = 2;
  bool shared = false, segments = true;
};
struct Path {
  Tensor              logits, hidden;
  std::vector<Tensor> states, attention;
};
inline Path Forward(
    nn::ParameterStore& p,
    const Tensor&       encoded,
    const Tensor&       segments,
    const Tensor&       lengths,
    Config              c = {}) {
  constexpr int T = 32, D = 8, V = 70;
  int           E = c.embedding;
  check::Require(E > 0 && c.layers > 0, "positive compact embedding/depth");
  auto mask   = bert::LengthMask(lengths, T);
  auto linear = [&](const Tensor& x, const std::string& name, int out) {
    return bert::Linear(p, name, x, out);
  };
  auto norm = [&](const Tensor& x, const std::string& name) {
    return nn::layer_norm(p, name, x, 1e-5) * mask;
  };
  auto table    = p.get("token/W", {V, E}, nn::init::normal(.02)),
       position = p.get("position/W", {T, E}, nn::init::normal(.02));
  auto x        = nn::matmul(encoded, table) + position;
  if (c.segments)
    x = x + nn::gather(p.get("segment/W", {2, E}, nn::init::normal(.02)), segments, 0);
  x = norm(x, "embedding_ln");
  if (E != D)
    x = linear(x, "embedding_projection", D) * mask;
  Path path;
  path.states.push_back(x);
  auto allowed = nn::expand_dims(attn::AttentionAllowed(lengths, lengths, T, T), 1);
  for (int l = 0; l < c.layers; ++l) {
    auto name  = c.shared ? std::string("shared") : "block" + std::to_string(l);
    auto q     = bert::Heads(linear(x, name + "/q", D), 2),
         k     = bert::Heads(linear(x, name + "/k", D), 2),
         v     = bert::Heads(linear(x, name + "/v", D), 2);
    auto a     = attn::ScaledDotProductAttention(q, k, v, 4, allowed);
    auto mixed = nn::reshape(nn::transpose(a.output, {0, 2, 1, 3}), {encoded.size(0), T, D});
    x          = norm(x + linear(mixed, name + "/out", D), name + "/attention_ln");
    x          = norm(
        x + linear(nn::gelu(linear(x, name + "/ff1", 16), true), name + "/ff2", D),
        name + "/ff_ln");
    path.states.push_back(x);
    path.attention.push_back(a.weights);
  }
  path.hidden      = x;
  auto transformed = norm(nn::gelu(linear(x, "mlm", E), true), "mlm_ln");
  path.logits =
      (nn::matmul(transformed, table, false, true) + p.get("output/b", {V}, nn::init::zeros())) *
      mask;
  return path;
}
class Model {
 public:
  Config                         config;
  bool                           distilled;
  nn::ParameterStore             parameters{46603};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(Config c = {}, bool distill = false, bool training = true)
      : config(c), distilled(distill) {
    {
      nn::NoGradGuard guard;
      Forward(
          parameters,
          nn::zeros({1, 32, 70}),
          nn::zeros({1, 32}, nn::DType::Int32),
          nn::full({1}, 32, nn::DType::Int32),
          config);
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, opt);
    }
  }
  A Losses(const Path& path, const A& b) const {
    auto mean = [&](const Tensor& values, const Tensor& weights) {
      return nn::sum(values * weights) / nn::maximum(nn::sum(weights), nn::scalar(1));
    };
    auto target = nn::one_hot(b.at("target"), 70);
    auto mlm    = mean(-nn::sum(nn::log_softmax(path.logits, -1) * target, {2}), b.at("selected"));
    auto kd = nn::scalar(0), cos = nn::scalar(0);
    if (distilled) {
      auto valid = nn::squeeze(bert::LengthMask(b.at("lengths"), 32), {2});
      auto tlp   = nn::log_softmax(b.at("teacher_logits").detach() / 2., -1),
           slp   = nn::log_softmax(path.logits / 2., -1);
      kd         = mean(nn::sum(nn::exp(tlp) * (tlp - slp), {2}), valid) * 4.;
      auto th    = b.at("teacher_hidden").detach();
      auto dot   = nn::sum(th * path.hidden, {2});
      auto tn    = nn::sqrt(nn::maximum(nn::sum(nn::square(th), {2}), nn::scalar(1e-12))),
           sn    = nn::sqrt(nn::maximum(nn::sum(nn::square(path.hidden), {2}), nn::scalar(1e-12)));
      cos        = mean(1. - dot / (tn * sn), valid);
    }
    return {{"mlm_loss", mlm}, {"kd_loss", kd}, {"cos_loss", cos}, {"loss", mlm + kd + cos}};
  }
  A Run(const A& b, bool gradients = false, const A& overrides = {}) {
    A saved;
    for (const auto& [k, v] : overrides) {
      saved[k] = parameters.at(k).detach().clone();
      parameters.set(k, v);
    }
    struct Restore {
      nn::ParameterStore& p;
      A&                  saved;
      ~Restore() {
        for (const auto& [k, v] : saved)
          p.set(k, v);
      }
    } restore{parameters, saved};
    auto encoded = nn::one_hot(b.at("input"), 70).requires_grad_();
    auto path    = Forward(parameters, encoded, b.at("segments"), b.at("lengths"), config);
    auto out     = Losses(path, b);
    std::vector<Tensor> inputs{encoded};
    if (gradients) {
      auto p = parameters.parameters();
      inputs.insert(inputs.end(), p.begin(), p.end());
    }
    auto g                = nn::grad(out.at("loss"), inputs);
    out["logits"]         = path.logits;
    out["hidden"]         = path.hidden;
    out["input_gradient"] = g[0];
    for (size_t i = 0; i < path.states.size(); ++i)
      out["state" + std::to_string(i)] = path.states[i];
    for (size_t i = 0; i < path.attention.size(); ++i)
      out["attention" + std::to_string(i)] = path.attention[i];
    if (gradients)
      for (size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 1];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  Tensor Predict(const A& b) {
    nn::NoGradGuard guard;
    return Forward(
               parameters,
               nn::one_hot(b.at("input"), 70),
               b.at("segments"),
               b.at("lengths"),
               config)
        .logits;
  }
  void Update(const A& b) {
    auto path = Forward(
        parameters, nn::one_hot(b.at("input"), 70), b.at("segments"), b.at("lengths"), config);
    optimizer->minimize(Losses(path, b).at("loss"));
  }
  A    Arrays() const { return replay::Arrays(parameters); }
  void Import(const A& a) {
    for (const auto& name : parameters.names()) {
      check::Require(
          parameters.at(name).shape() == a.at(name).shape(), "exact compact initialization shape");
      parameters.set(name, a.at(name));
    }
  }
};
} // namespace compact_bert
