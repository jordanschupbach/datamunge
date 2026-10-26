// Four-stage Pyramid Vision Transformer with spatial-reduction attention.
// Port of tflow's common/{nn,ml}/pyramid_vision_transformer.h.
#pragma once
#include "common/vision_transformer.hpp"
namespace pvt {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
using vit::Loss;
struct Path {
  Tensor              logits, patches, hidden;
  std::vector<Tensor> embedded, reduced, attention, states;
};
inline Tensor Patches(const Tensor& x, int grid, int patch, int channels) {
  const auto n = x.size(0);
  return nn::reshape(
      nn::transpose(
          nn::reshape(x, {n, grid / patch, patch, grid / patch, patch, channels}),
          {0, 1, 3, 2, 4, 5}),
      {n, (grid / patch) * (grid / patch), patch * patch * channels});
}
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  Path             out;
  auto             h             = images;
  const auto       n             = images.size(0);
  int              previous_grid = 32, previous_d = 3;
  const int        widths[] = {4, 8, 12, 16}, heads[] = {1, 2, 3, 4}, ratios[] = {4, 2, 2, 1};
  nn::DenseOptions o;
  o.init      = nn::init::normal(.1);
  auto linear = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, o);
  };
  for (int stage = 0; stage < 4; ++stage) {
    const int patch = stage ? 2 : 4, grid = previous_grid / patch, d = widths[stage],
              spatial = grid * grid, t = spatial + (stage == 3), nh = heads[stage],
              r = ratios[stage], keys = stage == 3 ? t : (grid / r) * (grid / r);
    const auto name    = "stage" + std::to_string(stage);
    auto       patches = Patches(h, previous_grid, patch, previous_d);
    if (stage == 0)
      out.patches = patches;
    h = linear(patches, name + "/patch", d);
    h = nn::layer_norm(p, name + "/patch_ln", h, 1e-6);
    if (stage == 3) {
      auto cls = p.get("cls", {1, 1, d}, nn::init::normal(.02));
      h        = nn::concat({nn::tile(cls, {n, 1, 1}), h}, 1);
    }
    h = h + p.get(name + "/position", {1, t, d}, nn::init::normal(.02));
    out.embedded.push_back(h);
    auto normalized = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto q = linear(normalized, name + "/q", d), reduced = normalized;
    if (r > 1) {
      auto patches = Patches(nn::reshape(normalized, {n, grid, grid, d}), grid, r, d);
      reduced      = linear(patches, name + "/sr", d);
      reduced      = nn::layer_norm(p, name + "/sr_ln", reduced, 1e-6);
    }
    out.reduced.push_back(reduced);
    auto parts   = nn::split(linear(reduced, name + "/kv", 2 * d), 2, 2);
    auto toheads = [&](const Tensor& x, int count) {
      return nn::transpose(nn::reshape(x, {n, count, nh, d / nh}), {0, 2, 1, 3});
    };
    auto qh = toheads(q, t), k = toheads(parts[0], keys), v = toheads(parts[1], keys);
    auto attention = nn::softmax(nn::matmul(qh, k, false, true) * .5, -1);
    out.attention.push_back(attention);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n, t, d});
    h          = h + linear(mixed, name + "/out", d);
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    h          = h + linear(nn::gelu(linear(ff, name + "/ff1", 2 * d), true), name + "/ff2", d);
    out.states.push_back(h);
    if (stage < 3)
      h = nn::reshape(h, {n, grid, grid, d});
    previous_grid = grid;
    previous_d    = d;
  }
  out.hidden = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto cls   = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, 16}), {1});
  out.logits = linear(cls, "head", 10);
  return out;
}
class Model {
 public:
  nn::ParameterStore             parameters{48901};
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
           {"hidden", path.hidden},
           {"loss", loss},
           {"input_gradient", g[0]}};
    for (int i = 0; i < 4; ++i) {
      const auto suffix         = std::to_string(i);
      out["attention" + suffix] = path.attention[i];
      out["state" + suffix]     = path.states[i];
      out["embedded" + suffix]  = path.embedded[i];
      out["reduced" + suffix]   = path.reduced[i];
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
} // namespace pvt
