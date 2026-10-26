// Frozen BERT plus rank2/alpha4 Q,V LoRA or two residual bottleneck adapters.
// Port of tflow common/{nn,ml}/peft_bert.h. Only task storage is an Adam target.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/tf_replay.hpp"
namespace peft {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = bert::Batch;
struct Path {
  Tensor              logits, hidden, attention, embedded;
  std::vector<Tensor> updates;
};
inline Path Forward(
    nn::ParameterStore& base,
    nn::ParameterStore& task,
    const Tensor&       encoded,
    const Tensor&       lengths,
    const std::string&  mode) {
  check::Require(mode == "lora" || mode == "adapter" || mode == "head", "lora,adapter,head modes");
  constexpr int T = 128, D = 8;
  auto          mask = bert::LengthMask(lengths, T);
  Path          result;
  auto          linear = [&](const Tensor& x, const std::string& name, int out) {
    auto value = bert::Linear(base, name, x, out);
    if (mode == "lora" && (name == "attention/q" || name == "attention/v")) {
      auto prefix = "lora/" + name;
      auto a      = task.get(prefix + "/A", {x.size(2), 2}, nn::init::normal(.02)),
           b      = task.get(prefix + "/B", {2, out}, nn::init::zeros());
      auto update = nn::matmul(nn::matmul(x, a), b) * 2.;
      result.updates.push_back(update);
      value = value + update;
    }
    return value;
  };
  auto norm = [&](const Tensor& x, const std::string& name) {
    return nn::layer_norm(base, name, x, 1e-5) * mask;
  };
  auto adapter = [&](const Tensor& x, const std::string& name) {
    if (mode != "adapter")
      return x;
    auto prefix = "adapter/" + name;
    auto down   = task.get(prefix + "/down/W", {D, 2}, nn::init::normal(.02)),
         db     = task.get(prefix + "/down/b", {2}, nn::init::zeros()),
         up     = task.get(prefix + "/up/W", {2, D}, nn::init::zeros()),
         ub     = task.get(prefix + "/up/b", {D}, nn::init::zeros());
    auto update = nn::matmul(nn::gelu(nn::matmul(x, down) + db, true), up) + ub;
    result.updates.push_back(update);
    return x + update;
  };
  auto token    = base.get("token/W", {70, D}, nn::init::normal(.02)),
       position = base.get("position/W", {T, D}, nn::init::normal(.02)),
       segment  = base.get("segment/W", {2, D}, nn::init::normal(.02));
  auto embedded = norm(
      nn::matmul(encoded, token) + position + nn::squeeze(nn::slice(segment, {0, 0}, {1, D}), {0}),
      "embedding_ln");
  auto q        = bert::Heads(linear(embedded, "attention/q", D), 2),
       k        = bert::Heads(linear(embedded, "attention/k", D), 2),
       v        = bert::Heads(linear(embedded, "attention/v", D), 2);
  auto attended = attn::ScaledDotProductAttention(
      q, k, v, 4, nn::expand_dims(attn::AttentionAllowed(lengths, lengths, T, T), 1));
  auto mixed = nn::reshape(nn::transpose(attended.output, {0, 2, 1, 3}), {encoded.size(0), T, D});
  auto h = norm(embedded + adapter(linear(mixed, "attention/out", D), "attention"), "attention_ln");
  auto ff          = linear(nn::gelu(linear(h, "ff1", 16), true), "ff2", D);
  h                = norm(h + adapter(ff, "ff"), "ff_ln");
  auto cls         = nn::squeeze(nn::slice(h, {0, 0, 0}, {encoded.size(0), 1, D}), {1});
  result.logits    = bert::Linear(task, "head", cls, 2);
  result.hidden    = h;
  result.attention = attended.weights;
  result.embedded  = embedded;
  return result;
}
class Model {
 public:
  std::string                    mode;
  nn::ParameterStore             base{47903}, task{47907};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(std::string kind, bool training = true) : mode(std::move(kind)) {
    task.get("head/W", {8, 2}, nn::init::normal(.02));
    task.get("head/b", {2}, nn::init::zeros());
    {
      nn::NoGradGuard guard;
      Forward(base, task, nn::zeros({1, 128, 70}), nn::full({1}, 128, nn::DType::Int32), mode);
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(task, .003, opt);
    }
  }
  static Tensor Loss(const Tensor& logits, const A& b) {
    return -nn::mean(nn::sum(nn::one_hot(b.at("labels"), 2) * nn::log_softmax(logits, -1), {1}));
  }
  A Run(const A& b, bool grad = false, const A& overrides = {}) {
    A saved;
    for (const auto& [k, v] : overrides) {
      saved[k] = task.at(k).detach().clone();
      task.set(k, v);
    }
    struct Restore {
      nn::ParameterStore& p;
      A&                  saved;
      ~Restore() {
        for (const auto& [k, v] : saved)
          p.set(k, v);
      }
    } restore{task, saved};
    auto                encoded = nn::one_hot(b.at("input"), 70).requires_grad_();
    auto                path    = Forward(base, task, encoded, b.at("lengths"), mode);
    auto                loss    = Loss(path.logits, b);
    std::vector<Tensor> inputs{encoded};
    if (grad) {
      auto t = task.parameters(), f = base.parameters();
      inputs.insert(inputs.end(), t.begin(), t.end());
      inputs.insert(inputs.end(), f.begin(), f.end());
    }
    auto g = nn::grad(loss, inputs);
    A    out{
           {"logits", path.logits},
           {"hidden", path.hidden},
           {"attention", path.attention},
           {"embedded", path.embedded},
           {"loss", loss},
           {"input_gradient", g[0]}};
    for (size_t i = 0; i < path.updates.size(); ++i)
      out["update" + std::to_string(i)] = path.updates[i];
    if (grad) {
      for (size_t i = 0; i < task.size(); ++i)
        out[task.names()[i]] = g[i + 1];
      for (size_t i = 0; i < base.size(); ++i)
        out["base_gradient/" + base.names()[i]] = g[task.size() + i + 1];
    }
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  Tensor Predict(const A& b) {
    nn::NoGradGuard guard;
    return Forward(base, task, nn::one_hot(b.at("input"), 70), b.at("lengths"), mode).logits;
  }
  A    Arrays(bool frozen = false) const { return replay::Arrays(frozen ? base : task); }
  void Import(const A& source, bool frozen = true) {
    auto& p = frozen ? base : task;
    for (auto name : p.names()) {
      auto old = source.at(name), value = p.at(name);
      if (frozen && name == "position/W" && old.size(0) == 32) {
        auto x = value.to_vector<float>(), y = old.to_vector<float>();
        std::copy(y.begin(), y.end(), x.begin());
        p.set(name, nn::from_vector(x, value.shape()));
      } else {
        check::Require(value.shape() == old.shape(), "PEFT import shape");
        p.set(name, old);
      }
    }
  }
  void Update(const A& b) {
    auto path = Forward(base, task, nn::one_hot(b.at("input"), 70), b.at("lengths"), mode);
    optimizer->minimize(Loss(path.logits, b));
  }
  void Save(const std::filesystem::path& root) const {
    base.save((root / "base.npz").string());
    task.save((root / "delta.npz").string());
  }
  void Load(const std::filesystem::path& root) {
    base.load((root / "base.npz").string());
    task.load((root / "delta.npz").string());
  }
};
} // namespace peft
