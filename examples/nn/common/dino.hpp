// DINO: normalized projection head, detached teacher, cross-crop loss and EMA.
// Port of tflow's common/{nn,ml}/dino.h.
#pragma once
#include <random>

#include "common/vision_transformer.hpp"
namespace dino {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
struct Path {
  Tensor              logits, patches, embedded, hidden, features, bottleneck;
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
  out.hidden          = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features       = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.features        = features;
  auto projected_head = nn::gelu(nn::dense(p, "projection/0", features, 16, dense), true);
  projected_head      = nn::gelu(nn::dense(p, "projection/1", projected_head, 16, dense), true);
  projected_head      = nn::dense(p, "projection/2", projected_head, 4, dense);
  out.bottleneck =
      projected_head *
      nn::rsqrt(nn::maximum(nn::sum(nn::square(projected_head), {1}, true), nn::scalar(1e-12)));
  auto direction = p.get("projection/last_v", {4, 16}, nn::init::normal(.1));
  auto normalized =
      direction *
      nn::rsqrt(nn::maximum(nn::sum(nn::square(direction), {0}, true), nn::scalar(1e-12)));
  out.logits = nn::matmul(out.bottleneck, normalized);
  return out;
}
inline A Views(A b, unsigned seed, bool original = false) {
  const auto                            n      = b.at("images").size(0);
  const auto                            images = b.at("images").to_vector<float>();
  std::vector<float>                    views(4 * n * 3072), gains(4 * n);
  std::vector<std::int32_t>             metadata(4 * n * 4);
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> uniform(.8f, 1.2f);
  for (int v = 0; v < 4; ++v)
    for (int row = 0; row < n; ++row) {
      int size = original ? 32 : (v < 2 ? 24 : 12) + rng() % 9,
          y = original ? 0 : rng() % (33 - size), x = original ? 0 : rng() % (33 - size),
          flip           = original ? 0 : rng() % 2;
      float gain         = original ? 1.f : uniform(rng);
      int   m            = (v * n + row) * 4;
      metadata[m]        = y;
      metadata[m + 1]    = x;
      metadata[m + 2]    = size;
      metadata[m + 3]    = flip;
      gains[v * n + row] = gain;
      for (int oy = 0; oy < 32; ++oy)
        for (int ox = 0; ox < 32; ++ox)
          for (int c = 0; c < 3; ++c) {
            int sy = y + oy * size / 32,
                sx = x + (flip ? size - 1 - ox * size / 32 : ox * size / 32);
            views[((v * n + row) * 32 * 32 + oy * 32 + ox) * 3 + c] =
                std::min(1.f, std::max(0.f, gain * images[(row * 32 * 32 + sy * 32 + sx) * 3 + c]));
          }
    }
  b.emplace("views", nn::from_vector(views, {4, n, 32, 32, 3}));
  b.emplace("crop_metadata", nn::from_vector(metadata, {4, n, 4}));
  b.emplace("crop_gains", nn::from_vector(gains, {4, n}));
  return b;
}
class Model {
 public:
  nn::ParameterStore             student{50101}, teacher{50103}, state{50105};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(student, nn::zeros({1, 32, 32, 3}));
      Forward(teacher, nn::zeros({1, 32, 32, 3}));
      state.get("center", {1, 16}, nn::init::zeros());
      teacher.copy_from(student);
    }
    if (training) {
      nn::OptimizerConfig c;
      c.kind    = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(student, .0003, c);
    }
  }
  A Arrays() const {
    A out;
    for (const auto& [k, v] : replay::Arrays(student))
      out["student/" + k] = v;
    for (const auto& [k, v] : replay::Arrays(teacher))
      out["teacher/" + k] = v;
    out["state/center"] = state.at("center").detach().clone();
    return out;
  }
  void Assign(const A& arrays) {
    for (const auto& [k, v] : arrays) {
      if (k.rfind("student/", 0) == 0) {
        check::Require(student.contains(k.substr(8)), "DINO student key");
        student.set(k.substr(8), v);
      } else if (k.rfind("teacher/", 0) == 0) {
        check::Require(teacher.contains(k.substr(8)), "DINO teacher key");
        teacher.set(k.substr(8), v);
      } else {
        check::Require(k == "state/center", "DINO center key");
        state.set("center", v);
      }
    }
  }
  A Fields(const Tensor& views) {
    check::Require(
        views.dim() == 5 && views.size(0) == 4 && views.size(1) > 0 && views.size(2) == 32 &&
            views.size(3) == 32 && views.size(4) == 3,
        "four DINO RGB views");
    std::vector<Tensor> sl, tl, sf, tf;
    auto                crops = nn::unstack(views, 0);
    A                   out;
    for (int v = 0; v < 4; ++v) {
      auto path = Forward(student, crops[v]);
      sl.push_back(path.logits);
      sf.push_back(path.features);
      if (v == 0) {
        out["student_attention"]  = path.attention.back();
        out["student_bottleneck"] = path.bottleneck;
      }
      if (v < 2) {
        auto p = Forward(teacher, crops[v]);
        tl.push_back(p.logits);
        tf.push_back(p.features);
      }
    }
    auto student_logits = nn::stack(sl, 0), teacher_logits = nn::stack(tl, 0);
    out["student_logits"]   = student_logits;
    out["teacher_logits"]   = teacher_logits;
    out["student_features"] = nn::stack(sf, 0);
    out["teacher_features"] = nn::stack(tf, 0);
    auto targets = nn::stop_gradient(nn::softmax((teacher_logits - state.at("center")) / .04, -1));
    out["targets"]                   = targets;
    auto                target_views = nn::unstack(targets, 0);
    std::vector<Tensor> terms;
    for (int q = 0; q < 2; ++q)
      for (int v = 0; v < 4; ++v)
        if (q != v)
          terms.push_back(
              -nn::mean(nn::sum(target_views[q] * nn::log_softmax(sl[v] / .1, -1), {1})));
    out["terms"] = nn::stack(terms, 0);
    out["loss"]  = nn::mean(out.at("terms"));
    return out;
  }
  A Run(const A& b, bool gradients = false) {
    auto                x   = b.at("views").detach().requires_grad_();
    auto                out = Fields(x);
    std::vector<Tensor> inputs{x};
    if (gradients) {
      auto s = student.parameters(), t = teacher.parameters();
      inputs.insert(inputs.end(), s.begin(), s.end());
      inputs.insert(inputs.end(), t.begin(), t.end());
    }
    nn::GradOptions options;
    options.allow_unused  = true;
    auto g                = nn::grad(out.at("loss"), inputs, options);
    out["input_gradient"] = g[0];
    if (gradients) {
      for (std::size_t i = 0; i < student.size(); ++i) {
        check::Require(g[i + 1].defined(), "every DINO student parameter connected");
        out["student/" + student.names()[i]] = g[i + 1];
      }
      for (std::size_t i = 0; i < teacher.size(); ++i) {
        check::Require(
            !g[1 + student.size() + i].defined(),
            "every teacher parameter disconnected by StopGradient");
        out["teacher/" + teacher.names()[i]] = nn::zeros_like(teacher.at(teacher.names()[i]));
      }
    }
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  void Update(const A& b, int step, int total, bool frozen) {
    check::Require(total > 0 && step >= 0 && step < total, "DINO EMA step range");
    auto                before = Arrays();
    auto                out    = Run(b, true);
    std::vector<Tensor> gradients;
    for (const auto& key : student.names())
      gradients.push_back(
          frozen && key == "projection/last_v" ? nn::zeros_like(student.at(key))
                                               : out.at("student/" + key));
    optimizer->step(gradients);
    if (frozen)
      check::Require(
          replay::SameBits(student.at("projection/last_v"), before.at("student/projection/last_v")),
          "last projection frozen in first epoch");
    const double momentum =
        1. - (1. - .996) * (.5 * (1. + std::cos(3.141592653589793 * step / total)));
    teacher.copy_from(student, 1. - momentum);
    const auto raw    = out.at("teacher_logits").to_vector<float>();
    const int  n      = out.at("teacher_logits").size(1);
    auto       center = before.at("state/center").to_vector<float>();
    for (int k = 0; k < 16; ++k) {
      double mean = 0;
      for (int view = 0; view < 2; ++view)
        for (int row = 0; row < n; ++row)
          mean += raw[(view * n + row) * 16 + k];
      center[k] = .9f * center[k] + .1f * float(mean / (2 * n));
    }
    state.set("center", nn::from_vector(center, {1, 16}));
  }
};
} // namespace dino
