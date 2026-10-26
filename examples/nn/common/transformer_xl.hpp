// Transformer-XL: detached pre-layer segment states and four-term relative
// attention. Port of tflow common/{nn,ml}/transformer_xl.h.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/tf_replay.hpp"
namespace xl {
namespace nn = datamunge::nn;
using nn::Tensor;
using A      = bert::Batch;
using Memory = std::vector<Tensor>;
struct Path {
  Tensor              logits;
  std::vector<Tensor> hids, attention;
};
inline Path Forward(nn::ParameterStore& p, const Tensor& encoded, const Memory& memory) {
  int  b = encoded.size(0), qt = encoded.size(1), mt = memory[0].size(1), kt = mt + qt;
  auto linear = [&](const Tensor& x, const std::string& name, int out) {
    return nn::matmul(x, p.get(name, {x.size(2), out}, nn::init::normal(.15)));
  };
  std::vector<float> rel(qt * kt * 8);
  std::vector<bool>  permitted(qt * kt);
  for (int i = 0; i < qt; ++i)
    for (int j = 0; j < kt; ++j) {
      permitted[i * kt + j] = j <= mt + i;
      for (int d = 0; d < 4; ++d) {
        float angle                   = float(mt + i - j) * std::pow(10000.f, -2.f * d / 8);
        rel[(i * kt + j) * 8 + d]     = std::sin(angle);
        rel[(i * kt + j) * 8 + d + 4] = std::cos(angle);
      }
    }
  auto relative = nn::from_vector(rel, {qt, kt, 8}), allowed = nn::from_vector(permitted, {qt, kt});
  auto u = p.get("relative/u", {2, 1, 4}, nn::init::normal(.02)),
       v = p.get("relative/v", {2, 1, 4}, nn::init::normal(.02));
  auto x = linear(encoded, "token/W", 8) * std::sqrt(8.f);
  Path result;
  for (int l = 0; l < 2; ++l) {
    auto name = "layer" + std::to_string(l);
    result.hids.push_back(x);
    auto cat   = nn::concat({memory[l].detach(), x}, 1);
    auto heads = [&](const Tensor& z, int t) {
      return nn::transpose(nn::reshape(z, {b, t, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(linear(x, name + "/Wq", 8), qt), k = heads(linear(cat, name + "/Wk", 8), kt),
         value = heads(linear(cat, name + "/Wv", 8), kt);
    auto r =
        nn::transpose(nn::reshape(linear(relative, name + "/Wr", 8), {qt, kt, 2, 4}), {2, 0, 1, 3});
    auto content  = nn::matmul(q + u, k, false, true),
         position = nn::sum(nn::expand_dims(q + v, 3) * r, {-1});
    auto scores   = (content + position) * .5;
    auto weights  = nn::softmax(nn::where(allowed, scores, nn::full(scores.shape(), -1e9)), -1);
    result.attention.push_back(weights);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(weights, value), {0, 2, 1, 3}), {b, qt, 8});
    x          = nn::layer_norm(p, name + "/ln1", x + linear(mixed, name + "/Wo", 8), 1e-5);
    auto w1 = linear(x, name + "/W1", 16), b1 = p.get(name + "/b1", {16}, nn::init::zeros());
    auto ff = linear(nn::relu(w1 + b1), name + "/W2", 8),
         b2 = p.get(name + "/b2", {8}, nn::init::zeros());
    x       = nn::layer_norm(p, name + "/ln2", x + ff + b2, 1e-5);
  }
  auto bias     = p.get("head/b", {65}, nn::init::zeros());
  result.logits = linear(x, "head/W", 65) + bias;
  return result;
}
class Model {
 public:
  nn::ParameterStore             parameters{47101};
  std::unique_ptr<nn::Optimizer> optimizer;
  static Memory Empty(int batch) { return {nn::zeros({batch, 0, 8}), nn::zeros({batch, 0, 8})}; }
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, nn::zeros({1, 1, 65}), Empty(1));
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, opt);
    }
  }
  static void Validate(const Tensor& x, const Memory& m) {
    check::Require(
        x.dtype() == nn::DType::Int32 && x.dim() == 2 && x.size(1) > 0 && m.size() == 2,
        "XL input/memory ranks");
    for (int i : x.to_vector<int>())
      check::Require(i >= 0 && i < 65, "XL character ID");
    for (const auto& t : m)
      check::Require(
          t.dim() == 3 && t.dtype() == nn::DType::Float32 && t.size(0) == x.size(0) &&
              t.size(1) == m[0].size(1) && t.size(2) == 8,
          "XL independent batch memory");
  }
  static Tensor Loss(const Tensor& logits, const Tensor& y) {
    return -nn::mean(nn::sum(nn::log_softmax(logits, -1) * nn::one_hot(y, 65), {-1}));
  }
  A Run(
      const Tensor& x,
      const Memory& memory,
      const Tensor* y         = nullptr,
      bool          gradients = false,
      const A&      overrides = {}) {
    Validate(x, memory);
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
    auto   encoded = nn::one_hot(x, 65).requires_grad_();
    Memory m;
    for (const auto& t : memory)
      m.push_back(t.detach().clone().requires_grad_());
    auto path = Forward(parameters, encoded, m);
    A    out{
           {"logits", path.logits},
           {"hid0", path.hids[0]},
           {"hid1", path.hids[1]},
           {"attention0", path.attention[0]},
           {"attention1", path.attention[1]}};
    if (y) {
      auto                loss = Loss(path.logits, *y) + nn::sum(m[0]) * 0. + nn::sum(m[1]) * 0.;
      std::vector<Tensor> inputs{encoded, m[0], m[1]};
      if (gradients) {
        auto p = parameters.parameters();
        inputs.insert(inputs.end(), p.begin(), p.end());
      }
      auto g                  = nn::grad(loss, inputs);
      out["loss"]             = loss;
      out["input_gradient"]   = g[0];
      out["memory_gradient0"] = g[1];
      out["memory_gradient1"] = g[2];
      if (gradients)
        for (size_t i = 0; i < parameters.size(); ++i)
          out[parameters.names()[i]] = g[i + 3];
    }
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  static Memory Advance(const Memory& m, const A& out, int capacity = 16) {
    check::Require(capacity >= 0, "nonnegative memory capacity");
    Memory result;
    for (int l = 0; l < 2; ++l) {
      auto cat  = nn::concat({m[l], out.at("hid" + std::to_string(l))}, 1);
      auto keep = std::min<std::int64_t>(capacity, cat.size(1));
      result.push_back(
          nn::slice(cat, {0, cat.size(1) - keep, 0}, {cat.size(0), keep, 8}).detach().clone());
    }
    return result;
  }
  void Update(const Tensor& x, const Tensor& y, const Memory& m) {
    Validate(x, m);
    auto path = Forward(parameters, nn::one_hot(x, 65), m);
    optimizer->minimize(Loss(path.logits, y));
  }
  A Arrays() const { return replay::Arrays(parameters); }
};
} // namespace xl
