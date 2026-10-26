// Asymmetric MAE: visible-only encoder and full-grid decoder.
// Port of tflow's common/{nn,ml}/mae_pretraining.h.
#pragma once
#include <numeric>
#include <random>

#include "common/vision_transformer.hpp"
namespace mae {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
struct Path {
  Tensor              patches, visible, latent, restored, prediction;
  std::vector<Tensor> attention, states;
};
inline Tensor Position(int d) {
  std::vector<float> pos(17 * d);
  for (int patch = 0; patch < 16; ++patch)
    for (int axis = 0; axis < 2; ++axis)
      for (int j = 0; j < d / 4; ++j) {
        double omega                            = std::pow(10000., -double(j) / (d / 4)),
               x                                = (axis == 0 ? patch % 4 : patch / 4) * omega;
        pos[(patch + 1) * d + axis * d / 2 + j] = std::sin(x);
        pos[(patch + 1) * d + axis * d / 2 + d / 4 + j] = std::cos(x);
      }
  return nn::from_vector(pos, {1, 17, d});
}
inline Tensor Linear(nn::ParameterStore& p, const Tensor& x, const std::string& name, int width) {
  nn::DenseOptions o;
  o.init = nn::init::normal(.1);
  return nn::dense(p, name, x, width, o);
}
inline Tensor Block(
    nn::ParameterStore& p,
    Tensor              h,
    const std::string&  name,
    int                 d,
    int                 t,
    int                 heads,
    Path&               out) {
  const auto n     = h.size(0);
  auto       z     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
  auto       parts = nn::split(Linear(p, z, name + "/qkv", 3 * d), 3, 2);
  auto       split = [&](const Tensor& x) {
    return nn::transpose(nn::reshape(x, {n, t, heads, d / heads}), {0, 2, 1, 3});
  };
  auto q = split(parts[0]), k = split(parts[1]), v = split(parts[2]);
  auto att =
      nn::softmax(nn::matmul(q, k, false, true) * float(1 / std::sqrt(double(d / heads))), -1);
  out.attention.push_back(att);
  auto mixed = nn::reshape(nn::transpose(nn::matmul(att, v), {0, 2, 1, 3}), {n, t, d});
  h          = h + Linear(p, mixed, name + "/out", d);
  auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
  h          = h + Linear(p, nn::gelu(Linear(p, ff, name + "/ff1", 2 * d), true), name + "/ff2", d);
  out.states.push_back(h);
  return h;
}
inline Path
Forward(nn::ParameterStore& p, const Tensor& images, const Tensor& keep, const Tensor& restore) {
  Path       out;
  const auto n = images.size(0);
  out.patches  = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  auto projected = Linear(p, out.patches, "patch", 8),
       position  = nn::slice(Position(8), {0, 1, 0}, {1, 16, 8});
  out.visible    = nn::matmul(keep, projected + position);
  auto cls       = p.get("cls", {1, 1, 8}, nn::init::normal(.02));
  auto h         = nn::concat({nn::tile(cls, {n, 1, 1}), out.visible}, 1);
  for (int i = 0; i < 2; ++i)
    h = Block(p, h, "encoder/block" + std::to_string(i), 8, 5, 2, out);
  out.latent     = nn::layer_norm(p, "encoder/ln", h, 1e-6);
  h              = Linear(p, out.latent, "decoder/embed", 4);
  auto mask      = p.get("mask_token", {1, 1, 4}, nn::init::normal(.02));
  auto shuffled  = nn::concat({nn::slice(h, {0, 1, 0}, {n, 4, 4}), nn::tile(mask, {n, 12, 1})}, 1);
  auto restored  = nn::matmul(restore, shuffled);
  out.restored   = nn::concat({nn::slice(h, {0, 0, 0}, {n, 1, 4}), restored}, 1) + Position(4);
  h              = Block(p, out.restored, "decoder/block0", 4, 17, 1, out);
  h              = nn::layer_norm(p, "decoder/ln", h, 1e-6);
  h              = Linear(p, h, "decoder/pred", 192);
  out.prediction = nn::slice(h, {0, 1, 0}, {n, 16, 192});
  return out;
}
inline A Masked(A b, unsigned seed) {
  const auto                n = b.at("images").size(0);
  std::vector<float>        keep(n * 4 * 16), restore(n * 16 * 16), mask(n * 16, 1);
  std::vector<std::int32_t> shuffle(n * 16);
  std::mt19937              rng(seed);
  for (int row = 0; row < n; ++row) {
    std::vector<int> ids(16);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int j = 0; j < 16; ++j) {
      int original                            = ids[j];
      shuffle[row * 16 + j]                   = original;
      restore[(row * 16 + original) * 16 + j] = 1;
      if (j < 4) {
        keep[(row * 4 + j) * 16 + original] = 1;
        mask[row * 16 + original]           = 0;
      }
    }
  }
  b.emplace("keep", nn::from_vector(keep, {n, 4, 16}));
  b.emplace("restore", nn::from_vector(restore, {n, 16, 16}));
  b.emplace("mask", nn::from_vector(mask, {n, 16}));
  b.emplace("shuffle", nn::from_vector(shuffle, {n, 16}));
  return b;
}
inline Tensor Loss(const Path& path, const Tensor& mask) {
  auto error = nn::mean(nn::square(path.prediction - path.patches), {2});
  return nn::sum(error * mask) / nn::sum(mask);
}
class Model {
 public:
  nn::ParameterStore             parameters{49101};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, nn::zeros({1, 32, 32, 3}), nn::zeros({1, 4, 16}), nn::zeros({1, 16, 16}));
    }
    if (training) {
      nn::OptimizerConfig c;
      c.kind    = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, c);
    }
  }
  A Run(const A& b, bool gradients = false, const A& overrides = {}) {
    vit::Model::Validate(b);
    A saved;
    for (const auto& [k, v] : overrides) {
      saved[k] = parameters.at(k).detach().clone();
      parameters.set(k, v);
    }
    struct Restore {
      nn::ParameterStore& p;
      A&                  a;
      ~Restore() {
        for (const auto& [k, v] : a)
          p.set(k, v);
      }
    } restore{parameters, saved};
    auto                x    = b.at("images").detach().requires_grad_();
    auto                path = Forward(parameters, x, b.at("keep"), b.at("restore"));
    auto                loss = Loss(path, b.at("mask"));
    std::vector<Tensor> inputs{x};
    if (gradients) {
      auto p = parameters.parameters();
      inputs.insert(inputs.end(), p.begin(), p.end());
    }
    auto g = nn::grad(loss, inputs);
    A    out{
           {"prediction", path.prediction},
           {"patches", path.patches},
           {"visible", path.visible},
           {"latent", path.latent},
           {"restored", path.restored},
           {"loss", loss},
           {"input_gradient", g[0]}};
    for (int i = 0; i < 3; ++i) {
      auto suffix               = std::to_string(i);
      out["attention" + suffix] = path.attention[i];
      out["state" + suffix]     = path.states[i];
    }
    if (gradients)
      for (std::size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 1];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  Tensor Predict(const A& b) {
    vit::Model::Validate(b);
    nn::NoGradGuard guard;
    return Forward(parameters, b.at("images"), b.at("keep"), b.at("restore")).prediction;
  }
  void Update(const A& b) {
    vit::Model::Validate(b);
    auto path = Forward(parameters, b.at("images"), b.at("keep"), b.at("restore"));
    optimizer->minimize(Loss(path, b.at("mask")));
  }
  A Arrays() const { return replay::Arrays(parameters); }
};
} // namespace mae
