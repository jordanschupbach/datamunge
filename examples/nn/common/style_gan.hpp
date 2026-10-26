// Port of tflow's common/nn/style_gan.h and common/ml/style_gan.h: tiny
// StyleGAN (AdaIN after each styled conv) and StyleGAN2 (per-example
// modulated / demodulated convolutions) generators with a mapping network,
// a learned 4x4x4 constant, explicit per-layer spatial noise with learned
// strengths, and a small LSGAN critic. Groups: 0 = G (6607), 1 = D (6619).
#pragma once

#include "common/pix2pix.hpp"

namespace gan {

/// tflow StyleMap: pixel-normalized z [n, 8] -> 8 leaky -> 8 leaky = w.
inline Tensor StyleMap(nn::ParameterStore& p, const Tensor& z) {
  const Tensor normalized = z / nn::sqrt(nn::mean(nn::square(z), {1}, true) + double(1e-8f));
  return nn::dense(p, "mapping2", nn::dense(p, "mapping1", normalized, 8, nn::Activation::LeakyRelu), 8,
                   nn::Activation::LeakyRelu);
}
inline Tensor StyleInstanceNorm(const Tensor& x) {
  const Tensor centered = x - nn::mean(x, {1, 2}, true);
  return centered / nn::sqrt(nn::mean(nn::square(centered), {1, 2}, true) + double(1e-8f));
}

struct StyleModulation {
  Tensor image, style, demodulator;
};
/// tflow StyleModConv: W [k, k, 4, out] ~ N(0, .02); style = dense(w, 4) + 1 (or `style_override`);
/// demod = rsqrt(sum (W style)^2 + 1e-8) per example/output (1 when !demodulate); SAME
/// per-example convolution as a sum over kernel taps of batched [size^2, 4] x [4, out] products.
inline StyleModulation StyleModConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, const Tensor& w,
                                    std::int64_t size, std::int64_t kernel, std::int64_t output_channels, bool demodulate,
                                    const Tensor* style_override = nullptr) {
  const Tensor weights = p.get(name + "/W", {kernel, kernel, 4, output_channels}, nn::init::normal(.02));
  Tensor       style   = nn::dense(p, name + "/style", w, 4) + 1.0;
  if (style_override) style = *style_override;
  const std::int64_t n         = x.size(0);
  const Tensor       modulated = nn::expand_dims(weights, 0) * nn::reshape(style, {-1, 1, 1, 4, 1});
  Tensor             demod     = nn::rsqrt(nn::sum(nn::square(modulated), {1, 2, 3}) + double(1e-8f));
  if (!demodulate) demod = demod * 0.0 + 1.0;
  const std::int64_t r      = kernel / 2;
  const Tensor       padded = nn::pad(x, {{0, 0}, {r, r}, {r, r}, {0, 0}});
  Tensor             sum;
  for (std::int64_t dy = 0; dy < kernel; ++dy)
    for (std::int64_t dx = 0; dx < kernel; ++dx) {
      const Tensor patch = nn::reshape(nn::slice(padded, {0, dy, dx, 0}, {n, size, size, 4}), {-1, size * size, 4});
      const Tensor base  = nn::reshape(nn::slice(weights, {dy, dx, 0, 0}, {1, 1, 4, output_channels}), {1, 4, output_channels});
      const Tensor filter = (base * nn::reshape(style, {-1, 4, 1})) * nn::reshape(demod, {-1, 1, output_channels});
      const Tensor term   = nn::matmul(patch, filter);
      sum                 = sum.defined() ? sum + term : term;
    }
  return {nn::reshape(sum, {-1, size, size, output_channels}), style, demod};
}

struct StyledLayer {
  Tensor image, style, demodulator;
};
/// tflow StyledConv: (modulated | plain 3x3) conv + strength * noise + b, leaky .2, then AdaIN for StyleGAN1.
inline StyledLayer StyledConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, const Tensor& w,
                              const Tensor& noise, std::int64_t size, bool style2, const Tensor* style_override = nullptr) {
  Tensor pre, style, demod;
  if (style2) {
    const auto layer = StyleModConv(p, name, x, w, size, 3, 4, true, style_override);
    pre = layer.image, style = layer.style, demod = layer.demodulator;
  } else {
    const Tensor weights = p.get(name + "/W", {3, 3, 4, 4}, nn::init::normal(.02));
    pre   = nn::conv2d(x, weights);
    style = nn::dense(p, name + "/style", w, 8);
    demod = nn::scalar_like(1.0, pre);
  }
  const Tensor strength = p.get(name + "/noise_strength", {4}, nn::init::zeros());
  const Tensor bias     = p.get(name + "/b", {4}, nn::init::zeros());
  Tensor       h        = nn::leaky_relu((pre + noise * strength) + bias, .2);
  if (!style2) {
    const std::int64_t n     = style.size(0);
    const Tensor       scale = nn::slice(style, {0, 0}, {n, 4}) + 1.0, shift = nn::slice(style, {0, 4}, {n, 4});
    h = StyleInstanceNorm(h) * nn::reshape(scale, {-1, 1, 1, 4}) + nn::reshape(shift, {-1, 1, 1, 4});
  }
  return {h, style, demod};
}

struct StyledGenerated {
  Tensor image, first_style, second_style, first_demodulator, second_demodulator;
};
/// tflow StyleSynthesize: constant 4x4x4 -> layer4 (w0, noise4) -> nearest x2 -> layer8 (w1, noise8) -> rgb sigmoid.
inline StyledGenerated StyleSynthesize(nn::ParameterStore& p, const Tensor& w0, const Tensor& w1, const Tensor& noise4,
                                       const Tensor& noise8, bool style2, const Tensor* first_style = nullptr) {
  const Tensor constant = p.get("constant", {1, 4, 4, 4}, nn::init::normal(.02));
  const Tensor tiled    = nn::tile(constant, {w0.size(0), 1, 1, 1});
  const auto   first    = StyledConv(p, "layer4", tiled, w0, noise4, 4, style2, first_style);
  const Tensor up       = nn::upsample_nearest(first.image, 2);
  const auto   second   = StyledConv(p, "layer8", up, w1, noise8, 8, style2);
  Tensor       image;
  if (style2) {
    const auto rgb = StyleModConv(p, "rgb", second.image, w1, 8, 1, 1, false);
    image          = nn::sigmoid(rgb.image + p.get("rgb/b", {1}, nn::init::zeros()));
  } else
    image = PixConv(p, "rgb", second.image, 4, 1, 1, 1, nn::Activation::Sigmoid);
  return {image, first.style, second.style, first.demodulator, second.demodulator};
}

/// tflow StyledDiscriminate: 4x4/2 conv 4 leaky -> 3x3 conv 8 leaky -> 128 -> raw score.
inline Tensor StyledDiscriminate(nn::ParameterStore& p, const Tensor& image) {
  Tensor h = nn::leaky_relu(PixConv(p, "down4", image, 1, 4, 4, 2), .2);
  h        = nn::leaky_relu(PixConv(p, "conv", h, 4, 8, 3, 1), .2);
  return nn::dense(p, "score", nn::reshape(h, {-1, 128}), 1);
}

class StyleGanModel : public GroupedGan {
 public:
  bool style2;
  StyleGanModel(bool version2, bool training = true, std::vector<std::uint64_t> seeds = {6607, 6619})
      : GroupedGan(seeds), style2(version2) {
    {
      nn::NoGradGuard guard;
      const Tensor    w = StyleMap(Group(0), nn::zeros({1, 8}));
      StyledDiscriminate(Group(1), StyleSynthesize(Group(0), w, w, nn::zeros({1, 4, 4, 1}), nn::zeros({1, 8, 8, 1}), style2).image);
    }
    if (training) MakeAdam(.0002, .5);
  }
  Arrays Forward(const Tensor& x, const Tensor& z, const Tensor& n4, const Tensor& n8) {
    const Tensor mapped = StyleMap(Group(0), z);
    const auto   g      = StyleSynthesize(Group(0), mapped, mapped, n4, n8, style2);
    const Tensor real_score = StyledDiscriminate(Group(1), x), fake_score = StyledDiscriminate(Group(1), g.image);
    return {{"fake", g.image},
            {"mapped", mapped},
            {"first_style", g.first_style},
            {"second_style", g.second_style},
            {"first_demodulator", g.first_demodulator},
            {"second_demodulator", g.second_demodulator},
            {"real_score", real_score},
            {"fake_score", fake_score},
            {"d_loss", .5 * (nn::mean(nn::square(real_score - 1.0)) + nn::mean(nn::square(fake_score)))},
            {"g_loss", nn::mean(nn::square(fake_score - 1.0))}};
  }
  Arrays Evaluate(const Tensor& x, const Tensor& z, const Tensor& n4, const Tensor& n8) {
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, z, n4, n8);
    RequireFinite(out, "finite styled GAN fields");
    return out;
  }
  /// Z ONLY latent mapping.
  Tensor Map(const Tensor& z) {
    nn::NoGradGuard guard;
    return StyleMap(Group(0), z);
  }
  /// Z + spatial noise ONLY generation.
  Tensor GenerateZ(const Tensor& z, const Tensor& n4, const Tensor& n8) {
    nn::NoGradGuard guard;
    const Tensor    w = StyleMap(Group(0), z);
    return StyleSynthesize(Group(0), w, w, n4, n8, style2).image;
  }
  /// Per-layer W styles + noise ONLY synthesis.
  Tensor GenerateW(const Tensor& w0, const Tensor& w1, const Tensor& n4, const Tensor& n8) {
    nn::NoGradGuard guard;
    return StyleSynthesize(Group(0), w0, w1, n4, n8, style2).image;
  }
  /// StyleGAN2 only: synthesis with the first layer's style replaced.
  Arrays FirstStyleControl(const Tensor& z, const Tensor& n4, const Tensor& n8, const Tensor& style) {
    check::Require(style2, "modulation control only StyleGAN2");
    nn::NoGradGuard guard;
    const Tensor    w = StyleMap(Group(0), z);
    const auto      g = StyleSynthesize(Group(0), w, w, n4, n8, style2, &style);
    return {{"fake", g.image}, {"demodulator", g.first_demodulator}};
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& z, const Tensor& n4, const Tensor& n8) {
    return GradientOf(group, Forward(x, z, n4, n8).at(group == 0 ? "g_loss" : "d_loss"));
  }
  void Update(int group, const Tensor& x, const Tensor& z, const Tensor& n4, const Tensor& n8) {
    Apply(group, Gradient(group, x, z, n4, n8));
  }
};

}  // namespace gan
