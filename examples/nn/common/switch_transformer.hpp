// Top1 Switch routing with real accepted-token gathers, per-sequence capacity,
// gate weighting and differentiable balancing. Port of tflow common/nn/switch_moe.h
// and common/{nn,ml}/switch_transformer.h.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/tf_replay.hpp"
namespace switch_lm {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = bert::Batch;
struct Moe {
  A                   fields;
  std::vector<Tensor> indices, dispatched, expert_outputs;
};
inline Moe
Switch(nn::ParameterStore& p, const Tensor& input, const Tensor& valid, int capacity = -1) {
  int  n = input.size(0), t = input.size(1), total = n * t;
  auto flat          = nn::reshape(input, {total, 8}),
       rw            = p.get("switch/router/W", {8, 4}, nn::init::normal(.15));
  auto mask          = nn::cast(valid, nn::DType::Float32);
  auto probabilities = nn::softmax(nn::matmul(input, rw), -1) * nn::expand_dims(mask, 2);
  auto chosen        = nn::cast(nn::argmax(probabilities, 2), nn::DType::Int32);
  auto choice        = chosen.to_vector<int>();
  auto mv            = mask.to_vector<float>();
  std::vector<float> assigned(total * 4), accepted(total * 4), slots(total * 4), counts(n * 4),
      used(n * 4), length(n), active(n);
  std::vector<int> caps(n);
  for (int b = 0; b < n; ++b) {
    for (int j = 0; j < t; ++j)
      length[b] += mv[b * t + j];
    active[b] = length[b] > 0;
    caps[b]   = capacity < 0 ? int(std::ceil(length[b] * 1.25f / 4)) : capacity;
    for (int j = 0; j < t; ++j) {
      int i = b * t + j;
      for (int e = 0; e < 4; ++e)
        slots[i * 4 + e] = counts[b * 4 + e];
      if (mv[i]) {
        int e               = choice[i];
        assigned[i * 4 + e] = 1;
        accepted[i * 4 + e] = counts[b * 4 + e] < caps[b];
        counts[b * 4 + e] += 1;
        used[b * 4 + e] += accepted[i * 4 + e];
      }
    }
  }
  auto accept = nn::from_vector(accepted, {n, t, 4}), gate = nn::sum(probabilities * accept, {2}),
       count = nn::from_vector(counts, {n, 4});
  auto denom = nn::expand_dims(nn::maximum(nn::from_vector(length, {n}), nn::scalar(1)), 1);
  auto f = count / denom, prob = nn::sum(probabilities, {1}) / denom;
  auto activity  = nn::from_vector(active, {n});
  auto auxiliary = nn::sum(nn::sum(f * prob, {1}) * activity) /
                   nn::maximum(nn::sum(activity), nn::scalar(1)) * 4.;
  Moe    result;
  Tensor combined = nn::zeros({total, 8});
  for (int e = 0; e < 4; ++e) {
    std::vector<std::int64_t> ids;
    std::vector<int>          inverse(total);
    for (int i = 0; i < total; ++i)
      if (accepted[i * 4 + e]) {
        ids.push_back(i);
        inverse[i] = ids.size();
      }
    auto list  = nn::from_vector(ids, {static_cast<std::int64_t>(ids.size())});
    auto batch = nn::gather(flat, list, 0);
    auto name  = "switch/expert" + std::to_string(e);
    auto w1    = p.get(name + "/W1", {8, 16}, nn::init::normal(.15)),
         b1    = p.get(name + "/b1", {16}, nn::init::zeros()),
         w2    = p.get(name + "/W2", {16, 8}, nn::init::normal(.15)),
         b2    = p.get(name + "/b2", {8}, nn::init::zeros());
    auto out   = nn::matmul(nn::relu(nn::matmul(batch, w1) + b1), w2) + b2;
    result.indices.push_back(list);
    result.dispatched.push_back(batch);
    result.expert_outputs.push_back(out);
    combined =
        combined +
        nn::gather(nn::concat({nn::zeros({1, 8}), out}, 0), nn::from_vector(inverse, {total}), 0);
  }
  result.fields = {
      {"output", nn::reshape(combined, {n, t, 8}) * nn::expand_dims(gate, 2)},
      {"probabilities", probabilities},
      {"chosen", chosen},
      {"assigned", nn::from_vector(assigned, {n, t, 4})},
      {"accepted", accept},
      {"slot", nn::from_vector(slots, {n, t, 4})},
      {"gate", gate},
      {"capacity", nn::from_vector(caps, {n})},
      {"counts", count},
      {"used", nn::from_vector(used, {n, 4})},
      {"auxiliary", auxiliary}};
  return result;
}
struct Path {
  Tensor logits, hidden, attention, embedded, moe_input;
  Moe    moe;
};
inline Path Forward(nn::ParameterStore& p, const Tensor& encoded, const Tensor& lengths) {
  auto mask = bert::LengthMask(lengths, 16), valid = nn::greater(nn::squeeze(mask, {2}), 0);
  auto table     = p.get("token/W", {65, 8}, nn::init::normal(.15)),
       positions = p.get("position/W", {16, 8}, nn::init::normal(.02));
  auto x         = (nn::matmul(encoded, table) + positions) * mask;
  auto norm      = [&](const Tensor& z, const std::string& name) {
    return nn::layer_norm(p, name, z, 1e-5) * mask;
  };
  auto first = norm(x, "ln1");
  auto a     = attn::MultiHeadAttention(
      p,
      "attention",
      first,
      first,
      8,
      2,
      nn::expand_dims(attn::AttentionAllowed(lengths, lengths, 16, 16, true), 1));
  auto h      = (x + a.output) * mask;
  auto mi     = norm(h, "ln2");
  auto moe    = Switch(p, mi, valid, 5);
  h           = norm(h + moe.fields.at("output"), "ln_final");
  auto w      = p.get("head/W", {8, 65}, nn::init::normal(.15)),
       b      = p.get("head/b", {65}, nn::init::zeros());
  auto logits = (nn::matmul(h, w) + b) * mask;
  return {logits, h, a.weights, x, mi, moe};
}
class Model {
 public:
  nn::ParameterStore             parameters{47701};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, nn::zeros({1, 16, 65}), nn::full({1}, 16, nn::DType::Int32));
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, opt);
    }
  }
  static Tensor CE(const Tensor& logits, const Tensor& y, const Tensor& lens) {
    auto valid = nn::squeeze(bert::LengthMask(lens, 16), {2});
    auto ce    = -nn::sum(nn::log_softmax(logits, -1) * nn::one_hot(y, 65), {2});
    return nn::sum(ce * valid) / nn::maximum(nn::sum(valid), nn::scalar(1));
  }
  A Run(
      const Tensor& x,
      const Tensor& lens,
      const Tensor* y         = nullptr,
      bool          gradients = false,
      const A&      overrides = {}) {
    check::Require(
        x.dtype() == nn::DType::Int32 && x.dim() == 2 && x.size(0) > 0 && x.size(1) == 16 &&
            lens.dtype() == nn::DType::Int32 && lens.numel() == x.size(0),
        "Switch tokens/lengths");
    for (int i : x.to_vector<int>())
      check::Require(i >= 0 && i < 65, "Switch token range");
    for (int i : lens.to_vector<int>())
      check::Require(i >= 0 && i <= 16, "Switch valid length");
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
    auto encoded = nn::one_hot(x, 65).requires_grad_();
    auto path    = Forward(parameters, encoded, lens);
    A    out     = path.moe.fields;
    out.erase("output");
    out["moe_output"] = path.moe.fields.at("output");
    out["moe_input"]  = path.moe_input;
    out["logits"]     = path.logits;
    out["hidden"]     = path.hidden;
    out["attention"]  = path.attention;
    out["embedded"]   = path.embedded;
    for (int e = 0; e < 4; ++e) {
      auto name                 = "expert" + std::to_string(e);
      out[name + "_indices"]    = path.moe.indices[e];
      out[name + "_dispatched"] = path.moe.dispatched[e];
      out[name + "_outputs"]    = path.moe.expert_outputs[e];
    }
    nn::GradOptions retained;
    retained.retain_graph  = true;
    out["prefix_gradient"] = nn::grad(
        nn::sum(nn::slice(path.logits, {0, 8, 0}, {x.size(0), 1, 1})), {encoded}, retained)[0];
    if (y) {
      auto ce = CE(path.logits, *y, lens), loss = ce + path.moe.fields.at("auxiliary") * .01;
      auto router                 = parameters.at("switch/router/W");
      out["router_main_gradient"] = nn::grad(ce, {router}, retained)[0];
      out["router_balance_gradient"] =
          nn::grad(path.moe.fields.at("auxiliary"), {router}, retained)[0];
      std::vector<Tensor> inputs{encoded};
      if (gradients) {
        auto p = parameters.parameters();
        inputs.insert(inputs.end(), p.begin(), p.end());
      }
      auto g                = nn::grad(loss, inputs);
      out["loss"]           = loss;
      out["ce"]             = ce;
      out["input_gradient"] = g[0];
      if (gradients)
        for (size_t i = 0; i < parameters.size(); ++i)
          out[parameters.names()[i]] = g[i + 1];
    }
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  void Update(const Tensor& x, const Tensor& y, const Tensor& lens) {
    auto path = Forward(parameters, nn::one_hot(x, 65), lens);
    optimizer->minimize(CE(path.logits, y, lens) + path.moe.fields.at("auxiliary") * .01);
  }
  A Arrays() const { return replay::Arrays(parameters); }
};
} // namespace switch_lm
