// Hierarchical Swin Transformer: shifted 2x2 windows and patch merging.
// Eager port of tflow's common/{nn,ml}/swin_transformer.h.
#pragma once
#include "common/vision_transformer.hpp"
namespace swin {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
using vit::Loss;
struct Path {
  Tensor              logits, patches, embedded, merged, hidden;
  std::vector<Tensor> attention, states, masks;
};
inline Tensor Shift(const Tensor& x, int grid, int shift) {
  if (!shift)
    return x;
  std::vector<std::int32_t> ids(grid);
  for (int i = 0; i < grid; ++i)
    ids[i] = (i + shift + grid) % grid;
  auto index = nn::from_vector(ids, {grid});
  return nn::gather(nn::gather(x, index, 1), index, 2);
}
inline Tensor Windows(const Tensor& x, int grid, int d) {
  return nn::reshape(
      nn::transpose(nn::reshape(x, {x.size(0), grid / 2, 2, grid / 2, 2, d}), {0, 1, 3, 2, 4, 5}),
      {x.size(0) * grid * grid / 4, 4, d});
}
inline Tensor Reverse(const Tensor& x, int grid, int d) {
  const auto n = x.size(0) / (grid * grid / 4);
  return nn::reshape(
      nn::transpose(nn::reshape(x, {n, grid / 2, grid / 2, 2, 2, d}), {0, 1, 3, 2, 4, 5}),
      {n, grid, grid, d});
}
inline Path Forward(nn::ParameterStore& p, const Tensor& images, const A& controls = {}) {
  Path       out;
  const auto n = images.size(0);
  out.patches  = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions o;
  o.init      = nn::init::normal(.1);
  auto linear = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, o);
  };
  auto h       = nn::layer_norm(p, "patch_ln", linear(out.patches, "patch", 8), 1e-6);
  out.embedded = h;
  for (int stage = 0; stage < 2; ++stage) {
    const int grid = stage == 0 ? 4 : 2, d = stage == 0 ? 8 : 16, t = grid * grid, nw = t / 4;
    for (int block = 0; block < 2; ++block) {
      const int  shift      = stage == 0 && block == 1 ? 1 : 0;
      const auto name       = "stage" + std::to_string(stage) + "/block" + std::to_string(block);
      auto       normalized = nn::layer_norm(p, name + "/ln1", h, 1e-6);
      auto       windows =
          Windows(Shift(nn::reshape(normalized, {n, grid, grid, d}), grid, shift), grid, d);
      auto parts = nn::split(linear(windows, name + "/qkv", 3 * d), 3, 2);
      auto heads = [&](const Tensor& x) {
        return nn::transpose(nn::reshape(x, {n * nw, 4, 2, d / 2}), {0, 2, 1, 3});
      };
      auto q = heads(parts[0]), k = heads(parts[1]), v = heads(parts[2]);
      auto table = p.get(name + "/relative_bias", {9, 2}, nn::init::normal(.02));
      std::vector<std::int32_t> ids;
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          ids.push_back((i / 2 - j / 2 + 1) * 3 + i % 2 - j % 2 + 1);
      auto bias   = nn::transpose(nn::gather(table, nn::from_vector(ids, {4, 4}), 0), {2, 0, 1});
      auto scores = nn::matmul(q, k, false, true) * float(1 / std::sqrt(d / 2.)) + bias;
      std::vector<float> mask(nw * 16);
      if (shift) {
        auto region = [](int y, int x) {
          auto id = [](int a) { return a < 2 ? 0 : (a < 3 ? 1 : 2); };
          return 3 * id(y) + id(x);
        };
        for (int w = 0; w < nw; ++w)
          for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
              int yi = w / (grid / 2) * 2 + i / 2, xi = w % (grid / 2) * 2 + i % 2,
                  yj = w / (grid / 2) * 2 + j / 2, xj = w % (grid / 2) * 2 + j % 2;
              mask[w * 16 + i * 4 + j] = region(yi, xi) == region(yj, xj) ? 0.f : -100.f;
            }
      }
      const auto key = "mask" + std::to_string(stage * 2 + block);
      out.masks.push_back(
          controls.count(key) ? controls.at(key) : nn::from_vector(mask, {1, nw, 1, 4, 4}));
      auto masked =
          nn::reshape(nn::reshape(scores, {n, nw, 2, 4, 4}) + out.masks.back(), {n * nw, 2, 4, 4});
      auto attention = nn::softmax(masked, -1);
      out.attention.push_back(attention);
      auto mixed =
          nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n * nw, 4, d});
      auto projected = linear(mixed, name + "/out", d),
           reversed  = Shift(Reverse(projected, grid, d), grid, -shift);
      h              = h + nn::reshape(reversed, {n, t, d});
      auto ff        = nn::layer_norm(p, name + "/ln2", h, 1e-6);
      h = h + linear(nn::gelu(linear(ff, name + "/ff1", 2 * d), true), name + "/ff2", d);
      out.states.push_back(h);
    }
    if (stage == 0) {
      auto gridh = nn::reshape(h, {n, 4, 4, 8});
      auto sub   = [&](int y, int x) {
        return nn::gather(
            nn::gather(gridh, nn::from_vector(std::vector<std::int32_t>{y, y + 2}, {2}), 1),
            nn::from_vector(std::vector<std::int32_t>{x, x + 2}, {2}),
            2);
      };
      auto joined =
          nn::reshape(nn::concat({sub(0, 0), sub(1, 0), sub(0, 1), sub(1, 1)}, 3), {n, 4, 32});
      auto norm  = nn::layer_norm(p, "merge/ln", joined, 1e-6),
           w     = p.get("merge/W", {32, 16}, nn::init::normal(.1));
      h          = nn::matmul(norm, w);
      out.merged = h;
    }
  }
  out.hidden = nn::layer_norm(p, "final_ln", h, 1e-6);
  out.logits = linear(nn::mean(out.hidden, {1}), "head", 10);
  return out;
}
class Model {
 public:
  nn::ParameterStore             parameters{48701};
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
    A saved, controls;
    for (const auto& [k, v] : overrides) {
      if (k.rfind("mask", 0) == 0) {
        controls[k] = v;
        continue;
      }
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
    auto                path = Forward(parameters, x, controls);
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
           {"merged", path.merged},
           {"hidden", path.hidden},
           {"loss", loss},
           {"input_gradient", g[0]}};
    for (int i = 0; i < 4; ++i) {
      auto suffix               = std::to_string(i);
      out["attention" + suffix] = path.attention[i];
      out["state" + suffix]     = path.states[i];
      out["mask" + suffix]      = path.masks[i];
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
} // namespace swin
