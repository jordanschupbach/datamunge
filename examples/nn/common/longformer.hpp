// Sliding-window gathers and separate global QKV attention for bidirectional ViT.
// Port of tflow's common/{nn,ml}/longformer.h.
#pragma once
#include "common/vision_transformer.hpp"
namespace longformer {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
using vit::Loss;
struct ContextPath {
  Tensor context, attention, global_attention;
};
inline ContextPath Context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gq,
    const Tensor& gk,
    const Tensor& gv,
    int           radius = 2) {
  const auto n = q.size(2);
  check::Require(n > 1 && radius >= 0, "valid Longformer window");
  const int          width = 2 * radius + 2;
  std::vector<int>   ids(n * width);
  std::vector<float> bias(n * width);
  for (int i = 0; i < n; ++i)
    for (int slot = 0; slot < width; ++slot) {
      int  j                 = slot == 0 ? 0 : i + slot - 1 - radius;
      bool valid             = slot == 0 || (j > 0 && j < n);
      ids[i * width + slot]  = valid ? j : 0;
      bias[i * width + slot] = valid ? 0.f : -1e9f;
    }
  auto indices = nn::from_vector(ids, {n, width});
  auto gather  = [&](const Tensor& x) {
    return nn::transpose(nn::gather(nn::transpose(x, {2, 0, 1, 3}), indices, 0), {2, 3, 0, 1, 4});
  };
  auto        keys = gather(k), values = gather(v);
  ContextPath out;
  out.attention = nn::softmax(
      nn::sum(nn::expand_dims(q, 3) * keys, {4}) * .5 + nn::from_vector(bias, {n, width}), -1);
  auto local           = nn::sum(nn::expand_dims(out.attention, 4) * values, {3});
  auto first           = nn::slice(gq, {0, 0, 0, 0}, {gq.size(0), gq.size(1), 1, gq.size(3)});
  out.global_attention = nn::softmax(nn::matmul(first, gk, false, true) * .5, -1);
  auto global          = nn::matmul(out.global_attention, gv);
  out.context          = nn::concat(
      {global, nn::slice(local, {0, 0, 1, 0}, {q.size(0), q.size(1), n - 1, q.size(3)})}, 2);
  return out;
}
struct Path {
  Tensor              logits, patches, embedded, hidden;
  std::vector<Tensor> attention, states, queries, keys, values, global_queries, global_keys,
      global_values, global_attention;
};
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto gs = nn::split(linear(x, name + "/global_qkv", 3 * D), 3, 2);
    auto gq = heads(gs[0]), gk = heads(gs[1]), gv = heads(gs[2]);
    auto attended = Context(q, k, v, gq, gk, gv);
    out.queries.push_back(q);
    out.keys.push_back(k);
    out.values.push_back(v);
    out.global_queries.push_back(gq);
    out.global_keys.push_back(gk);
    out.global_values.push_back(gv);
    out.attention.push_back(attended.attention);
    out.global_attention.push_back(attended.global_attention);
    auto mixed = nn::reshape(nn::transpose(attended.context, {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden    = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.logits    = nn::dense(p, "head", features, 10, dense);
  return out;
}
class Model {
 public:
  nn::ParameterStore             parameters{50701};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, nn::zeros({1, 32, 32, 3}));
    }
    if (training) {
      nn::OptimizerConfig c;
      c.kind    = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, c);
    }
  }
  static void Validate(const A& b) {
    const auto& x = b.at("images");
    check::Require(
        x.dtype() == nn::DType::Float32 && x.dim() == 4 && x.size(0) > 0 && x.size(1) == 32 &&
            x.size(2) == 32 && x.size(3) == 3,
        "ViT32x32RGB images");
    for (float v : x.to_vector<float>())
      check::Require(std::isfinite(v), "finite ViT pixels");
  }
  A Run(const A& b, bool gradients = false, const A& overrides = {}) {
    Validate(b);
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
    auto                x    = b.at("images").detach().requires_grad_();
    auto                path = Forward(parameters, x);
    auto                loss = Loss(path.logits, b.at("labels"));
    std::vector<Tensor> inputs{x};
    if (gradients) {
      auto values = parameters.parameters();
      inputs.insert(inputs.end(), values.begin(), values.end());
    }
    auto g = nn::grad(loss, inputs);
    A    out{
           {"logits", path.logits},
           {"patches", path.patches},
           {"embedded", path.embedded},
           {"hidden", path.hidden},
           {"attention0", path.attention[0]},
           {"attention1", path.attention[1]},
           {"state0", path.states[0]},
           {"state1", path.states[1]},
           {"loss", loss},
           {"input_gradient", g[0]}};
    for (int i = 0; i < 2; ++i) {
      auto suffix                      = std::to_string(i);
      out["q" + suffix]                = path.queries[i];
      out["k" + suffix]                = path.keys[i];
      out["v" + suffix]                = path.values[i];
      out["gq" + suffix]               = path.global_queries[i];
      out["gk" + suffix]               = path.global_keys[i];
      out["gv" + suffix]               = path.global_values[i];
      out["global_attention" + suffix] = path.global_attention[i];
    }
    if (gradients)
      for (std::size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 1];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  Tensor Predict(const A& b) {
    Validate(b);
    nn::NoGradGuard guard;
    return Forward(parameters, b.at("images")).logits;
  }
  void Update(const A& b) {
    Validate(b);
    auto path = Forward(parameters, b.at("images"));
    optimizer->minimize(Loss(path.logits, b.at("labels")));
  }
  A Arrays() const { return replay::Arrays(parameters); }
};
} // namespace longformer
