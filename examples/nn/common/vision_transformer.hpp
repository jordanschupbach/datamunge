// Nonoverlapping RGB patches, learned CLS/positions, two pre-LN ViT blocks.
// Port of tflow's common/{nn,ml}/vision_transformer.h.
#pragma once
#include "common/tf_replay.hpp"
namespace vit {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = replay::Map;
struct Path {
  Tensor              logits, patches, embedded, hidden;
  std::vector<Tensor> attention, states;
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
    auto attention = nn::softmax(nn::matmul(q, k, false, true) * .5, -1);
    out.attention.push_back(attention);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n, T, D});
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
inline Tensor Loss(const Tensor& logits, const Tensor& labels) {
  return -nn::mean(
      nn::sum(
          nn::one_hot(labels, 10, 1, 0, nn::DType::Float32) * nn::log_softmax(logits, -1), {1}));
}
class Model {
 public:
  nn::ParameterStore             parameters{48301};
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
inline A Pool(const A& data, const std::string& split) {
  A out;
  for (const auto key : {"images", "labels", "source_ids"})
    out[key] = data.at(split + "_" + key);
  return out;
}
inline Tensor Permute(const Tensor& images) {
  const auto n       = images.size(0);
  auto       patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 8, 8, 3});
  return nn::reshape(
      nn::transpose(nn::reshape(nn::reverse(patches, {1}), {n, 4, 4, 8, 8, 3}), {0, 1, 3, 2, 4, 5}),
      images.shape());
}
} // namespace vit
