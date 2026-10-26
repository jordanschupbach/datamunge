// Port of tflow's common/nn/progressive_gan.h and common/ml/progressive_gan.h:
// progressive growing 4x4 -> 8x8 with genuine old/new RGB paths, a faded
// discriminator and eight independently optimised parameter groups:
// 0 g_core (6401), 1 g_old_rgb (6407), 2 g_new_block (6413), 3 g_new_rgb (6421),
// 4 d_core (6427), 5 d_old_rgb (6449), 6 d_new_block (6451), 7 d_new_rgb (6469).
// Each phase captures the gradients of ALL its active groups before any write.
#pragma once

#include "common/pix2pix.hpp"

namespace gan {

inline Tensor ProgressivePixelNorm(const Tensor& x) {
  return x / nn::sqrt(nn::mean(nn::square(x), {3}, true) + double(1e-8f));
}
inline Tensor ProgressiveUp(const Tensor& x) { return nn::upsample_nearest(x, 2); }
inline Tensor ProgressiveDown(const Tensor& x) { return nn::avg_pool2d(x, 2, 2); }
inline Tensor ProgressiveConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, bool normalize) {
  const Tensor h = nn::leaky_relu(PixConv(p, name, x, 4, 4, 3, 1), .2);
  return normalize ? ProgressivePixelNorm(h) : h;
}

struct ProgressiveGenerated {
  Tensor old4, new8, image;
};

class ProgressiveGanModel : public GroupedGan {
 public:
  explicit ProgressiveGanModel(bool training = true,
                               std::vector<std::uint64_t> seeds = {6401, 6407, 6413, 6421, 6427, 6449, 6451, 6469})
      : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      const Tensor    alpha = nn::scalar_like(0.0, nn::zeros({1}));
      Discriminate(Generate(nn::zeros({1, 16}), alpha).image, alpha);
    }
    if (training) MakeAdam(.0002, .5);
  }

  /// Active groups: stable4 {0, 1}, fade {0..3}, stable8 {0, 2, 3}; +4 for D.
  static std::vector<int> Active(int stage, bool generator) {
    std::vector<int> g = stage == 0 ? std::vector<int>{0, 1} : stage == 1 ? std::vector<int>{0, 1, 2, 3} : std::vector<int>{0, 2, 3};
    if (!generator)
      for (auto& x : g) x += 4;
    return g;
  }
  static Tensor Alpha(float value) {
    check::Require(std::isfinite(value) && value >= 0 && value <= 1, "fade alpha 0..1");
    return nn::from_vector(std::vector<float>{value}, {});
  }

  /// tflow ProgressiveGenerate: z -> 64 tanh = 4x4x4 pixel-normed -> conv -> old 1x1 rgb;
  /// up -> conv1, conv2 -> new rgb; image = alpha new8 + (1 - alpha) up(old4).
  ProgressiveGenerated Generate(const Tensor& z, const Tensor& alpha) {
    Tensor h = ProgressivePixelNorm(nn::reshape(nn::dense(Group(0), "stem", z, 64, nn::Activation::Tanh), {-1, 4, 4, 4}));
    h        = ProgressiveConv(Group(0), "conv", h, true);
    const Tensor old = PixConv(Group(1), "rgb", h, 4, 1, 1, 1, nn::Activation::Sigmoid);
    Tensor       high = ProgressiveConv(Group(2), "conv1", ProgressiveUp(h), true);
    high              = ProgressiveConv(Group(2), "conv2", high, true);
    const Tensor generated = PixConv(Group(3), "rgb", high, 4, 1, 1, 1, nn::Activation::Sigmoid);
    return {old, generated, alpha * generated + (1.0 - alpha) * ProgressiveUp(old)};
  }
  /// tflow ProgressiveDiscriminate: faded old (avg-pooled) / new 8x8 paths -> core conv -> score.
  Tensor Discriminate(const Tensor& image, const Tensor& alpha) {
    const Tensor old  = nn::leaky_relu(PixConv(Group(5), "rgb", ProgressiveDown(image), 1, 4, 1, 1), .2);
    Tensor       high = nn::leaky_relu(PixConv(Group(7), "rgb", image, 1, 4, 1, 1), .2);
    high              = ProgressiveConv(Group(6), "conv1", high, false);
    high              = ProgressiveConv(Group(6), "conv2", high, false);
    const Tensor mixed = alpha * ProgressiveDown(high) + (1.0 - alpha) * old;
    const Tensor base  = ProgressiveConv(Group(4), "conv", mixed, false);
    return nn::dense(Group(4), "score", nn::reshape(base, {-1, 64}), 1);
  }

  Arrays Forward(const Tensor& x, const Tensor& z, float value) {
    const Tensor alpha = Alpha(value);
    const auto   g     = Generate(z, alpha);
    const Tensor real_score = Discriminate(x, alpha), fake_score = Discriminate(g.image, alpha);
    return {{"fake", g.image},
            {"old4", g.old4},
            {"new8", g.new8},
            {"real_score", real_score},
            {"fake_score", fake_score},
            {"d_loss", .5 * (nn::mean(nn::square(real_score - 1.0)) + nn::mean(nn::square(fake_score)))},
            {"g_loss", nn::mean(nn::square(fake_score - 1.0))}};
  }
  Arrays Evaluate(const Tensor& x, const Tensor& z, float value) {
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, z, value);
    RequireFinite(out, "finite progressive fields");
    return out;
  }
  /// Noise + fade alpha ONLY generation.
  Tensor Sample(const Tensor& z, float value) {
    const Tensor    alpha = Alpha(value);
    nn::NoGradGuard guard;
    return Generate(z, alpha).image;
  }
  /// Gradients of every requested group's objective (G groups: g_loss, D groups:
  /// d_loss) from ONE forward, before any write; keyed "<k>/<name>".
  Arrays Gradient(const std::vector<int>& requested, const Tensor& x, const Tensor& z, float value) {
    const auto fields = Forward(x, z, value);
    Arrays     out;
    for (int k : requested) {
      nn::GradOptions o;
      o.retain_graph = true;
      o.allow_unused = true;
      auto&      store = Group(k);
      const auto g     = nn::grad(fields.at(k < 4 ? "g_loss" : "d_loss"), store.parameters(), o);
      for (std::size_t i = 0; i < g.size(); ++i)
        out.emplace(std::to_string(k) + "/" + store.names()[i],
                    g[i].defined() ? g[i] : nn::zeros(store.at(store.names()[i]).shape(), store.dtype()));
    }
    return out;
  }
};

}  // namespace gan
