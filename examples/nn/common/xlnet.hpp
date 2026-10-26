// XLNet permutation factorization: shared content/query streams, relative
// attention and detached segment memory. Port of tflow common/{nn,ml}/xlnet.h.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/tf_replay.hpp"
namespace xlnet {
namespace nn = datamunge::nn;
using nn::Tensor;
using A      = bert::Batch;
using Memory = std::vector<Tensor>;
struct Path {
  Tensor              logits;
  Tensor              content, query;
  std::vector<Tensor> hids, content_attention, query_attention;
};
inline Path
Forward(nn::ParameterStore& p, const Tensor& encoded, const Tensor& rank, const Memory& memory) {
  int  b = encoded.size(0), qt = encoded.size(1), mt = memory[0].size(1), kt = mt + qt;
  auto linear = [&](const Tensor& x, const std::string& name, int out) {
    return nn::matmul(x, p.get(name, {x.size(2), out}, nn::init::normal(.15)));
  };
  std::vector<float> rel(qt * kt * 8);
  for (int i = 0; i < qt; ++i)
    for (int j = 0; j < kt; ++j) {
      for (int d = 0; d < 4; ++d) {
        float angle                   = float(mt + i - j) * std::pow(10000.f, -2.f * d / 8);
        rel[(i * kt + j) * 8 + d]     = std::sin(angle);
        rel[(i * kt + j) * 8 + d + 4] = std::cos(angle);
      }
    }
  auto relative = nn::from_vector(rel, {qt, kt, 8});
  auto u        = p.get("relative/u", {2, 1, 4}, nn::init::normal(.02)),
       v        = p.get("relative/v", {2, 1, 4}, nn::init::normal(.02));
  auto table    = p.get("token/W", {65, 8}, nn::init::normal(.2));
  auto h        = nn::matmul(encoded, table);
  auto g        = nn::zeros_like(h) + p.get("query/seed", {8}, nn::init::normal(.2));
  // Both streams share the same token table, relative biases and layer weights.
  Path              result;
  std::vector<bool> cm(b * 16 * kt), qm(cm.size());
  auto              ranks = rank.to_vector<int>();
  for (int row = 0; row < b; ++row)
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < kt; ++j) {
        int index = (row * 16 + i) * kt + j;
        cm[index] = j < mt || ranks[row * 16 + j - mt] <= ranks[row * 16 + i];
        qm[index] = j < mt || ranks[row * 16 + j - mt] < ranks[row * 16 + i];
      }
  auto content_allowed = nn::from_vector(cm, {b, 1, 16, kt}),
       query_allowed   = nn::from_vector(qm, {b, 1, 16, kt});
  for (int l = 0; l < 2; ++l) {
    auto name = "layer" + std::to_string(l);
    result.hids.push_back(h);
    auto cat   = nn::concat({memory[l].detach(), h}, 1);
    auto heads = [&](const Tensor& z, int t) {
      return nn::transpose(nn::reshape(z, {b, t, 2, 4}), {0, 2, 1, 3});
    };
    auto k     = heads(linear(cat, name + "/Wk", 8), kt),
         value = heads(linear(cat, name + "/Wv", 8), kt);
    auto r =
        nn::transpose(nn::reshape(linear(relative, name + "/Wr", 8), {16, kt, 2, 4}), {2, 0, 1, 3});
    auto stream = [&](const Tensor& x, const Tensor& mask, std::vector<Tensor>& trace) {
      auto q        = heads(linear(x, name + "/Wq", 8), 16);
      auto content  = nn::matmul(q + u, k, false, true),
           position = nn::sum(nn::expand_dims(q + v, 3) * r, {-1});
      auto scores   = (content + position) * .5;
      auto raw      = nn::softmax(nn::where(mask, scores, nn::full(scores.shape(), -1e9)), -1) *
                 nn::cast(mask, nn::DType::Float32);
      auto a = raw / nn::maximum(nn::sum(raw, {-1}, true), nn::scalar(1));
      trace.push_back(a);
      auto mixed = nn::reshape(nn::transpose(nn::matmul(a, value), {0, 2, 1, 3}), {b, 16, 8});
      auto z     = nn::layer_norm(p, name + "/ln1", x + linear(mixed, name + "/Wo", 8), 1e-5);
      auto b1    = p.get(name + "/b1", {16}, nn::init::zeros());
      auto ff    = linear(nn::relu(linear(z, name + "/W1", 16) + b1), name + "/W2", 8);
      auto b2    = p.get(name + "/b2", {8}, nn::init::zeros());
      return nn::layer_norm(p, name + "/ln2", z + ff + b2, 1e-5);
    };
    auto next_h = stream(h, content_allowed, result.content_attention);
    auto next_g = stream(g, query_allowed, result.query_attention);
    h           = next_h;
    g           = next_g;
  }
  result.logits  = nn::matmul(g, table, false, true) + p.get("head/b", {65}, nn::init::zeros());
  result.content = h;
  result.query   = g;
  return result;
}

class Model {
 public:
  nn::ParameterStore             parameters{47301};
  std::unique_ptr<nn::Optimizer> optimizer;
  static Memory Empty(int batch) { return {nn::zeros({batch, 0, 8}), nn::zeros({batch, 0, 8})}; }
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(
          parameters,
          nn::zeros({1, 16, 65}),
          nn::reshape(nn::arange(0, 16, 1, nn::DType::Int32), {1, 16}),
          Empty(1));
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
  static Tensor Loss(const Tensor& logits, const Tensor& y, const Tensor& rank) {
    auto selected = nn::cast(nn::greater_equal(rank, 8), nn::DType::Float32);
    auto ce       = -nn::sum(nn::log_softmax(logits, -1) * nn::one_hot(y, 65), {-1});
    return nn::sum(ce * selected) / nn::sum(selected);
  }
  A Run(
      const Tensor& x,
      const Memory& memory,
      const Tensor& rank,
      const Tensor* y         = nullptr,
      bool          gradients = false,
      const A&      overrides = {}) {
    Validate(x, memory);
    check::Require(
        x.size(1) == 16 && rank.dtype() == nn::DType::Int32 && rank.shape() == x.shape(),
        "XLNet rank matrix");
    auto ranks = rank.to_vector<int>();
    for (int b = 0; b < x.size(0); ++b) {
      auto r = std::vector<int>(ranks.begin() + b * 16, ranks.begin() + (b + 1) * 16);
      std::sort(r.begin(), r.end());
      for (int i = 0; i < 16; ++i)
        check::Require(r[i] == i, "each rank row permutation0..15");
    }
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
    auto path = Forward(parameters, encoded, rank, m);
    A    out{
           {"logits", path.logits},
           {"hid0", path.hids[0]},
           {"hid1", path.hids[1]},
           {"content", path.content},
           {"query", path.query},
           {"content_attention0", path.content_attention[0]},
           {"content_attention1", path.content_attention[1]},
           {"query_attention0", path.query_attention[0]},
           {"query_attention1", path.query_attention[1]},
           {"selected", nn::cast(nn::greater_equal(rank, 8), nn::DType::Float32)}};
    nn::GradOptions retained;
    retained.retain_graph = true;
    out["query_gradient"] = nn::grad(
        nn::sum(nn::slice(path.logits, {0, 7, 0}, {x.size(0), 1, 1})), {encoded}, retained)[0];
    if (y) {
      auto loss = Loss(path.logits, *y, rank) + nn::sum(m[0]) * 0. + nn::sum(m[1]) * 0.;
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
  void Update(const Tensor& x, const Tensor& y, const Memory& m, const Tensor& rank) {
    Validate(x, m);
    auto path = Forward(parameters, nn::one_hot(x, 65), rank, m);
    optimizer->minimize(Loss(path.logits, y, rank));
  }
  A Arrays() const { return replay::Arrays(parameters); }
};
} // namespace xlnet
