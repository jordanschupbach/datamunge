// Port of tflow's common/nn/pix2pix.h: the two-level 28x28 U-Net generator
// with learned up-convolutions, the conditional PatchGAN discriminator and the
// logistic PatchGAN objectives (also used by cycle_gan). Kernels "<name>/W"
// are N(0, .02), biases "<name>/b" zero; SAME padding throughout.
#pragma once

#include "common/gan.hpp"

namespace gan {

/// tflow PixConv: k x k conv (or transposed conv, W [k, k, out, in]) with bias and activation.
inline Tensor PixConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t in,
                      std::int64_t out, std::int64_t k, std::int64_t stride,
                      nn::Activation act = nn::Activation::None, bool transpose = false) {
  p.get(name + "/W", transpose ? nn::Shape{k, k, out, in} : nn::Shape{k, k, in, out}, nn::init::normal(.02));
  nn::ConvLayerOptions o;
  o.stride     = stride;
  o.activation = act;
  return transpose ? nn::conv2d_transpose(p, name, x, out, k, o) : nn::conv2d(p, name, x, out, k, o);
}

/// tflow PixGenerate: source [n, 28, 28, 1] -> down14 4 ReLU -> down7 8 ReLU ->
/// up14 4 ReLU (+ skip down14) -> up28 2 ReLU (+ skip source) -> 3x3 sigmoid 1.
inline Tensor PixGenerate(nn::ParameterStore& p, const Tensor& source) {
  const Tensor down14 = PixConv(p, "down14", source, 1, 4, 4, 2, nn::Activation::Relu);
  const Tensor down7  = PixConv(p, "down7", down14, 4, 8, 4, 2, nn::Activation::Relu);
  const Tensor up14   = PixConv(p, "up14", down7, 8, 4, 4, 2, nn::Activation::Relu, true);
  const Tensor up28   = PixConv(p, "up28", nn::concat({up14, down14}, 3), 8, 2, 4, 2, nn::Activation::Relu, true);
  return PixConv(p, "output", nn::concat({up28, source}, 3), 3, 1, 3, 1, nn::Activation::Sigmoid);
}

/// tflow PixDiscriminate: concat(source, image) -> 4x4/2 conv 4 leaky .2 -> 4x4/2 conv 8 leaky .2 -> 3x3 conv 1.
inline Tensor PixDiscriminate(nn::ParameterStore& p, const Tensor& source, const Tensor& image) {
  const Tensor down14 = nn::leaky_relu(PixConv(p, "down14", nn::concat({source, image}, 3), 2, 4, 4, 2), .2);
  const Tensor down7  = nn::leaky_relu(PixConv(p, "down7", down14, 4, 8, 4, 2), .2);
  return PixConv(p, "output", down7, 8, 1, 3, 1);
}

/// Logistic PatchGAN objectives averaged over every patch logit.
inline LogisticGanLoss PixObjectives(const Tensor& real, const Tensor& fake) { return LogisticGanObjectives(real, fake); }

/// tflow ml::Pix2PixModel: group 0 = U-Net generator (seed 6101), group 1 =
/// conditional PatchGAN (6113), separate Adam (.0002, beta1 .5). G objective =
/// non-saturating patch CE + 100 mean L1 to the PAIRED target.
class Pix2PixModel : public GroupedGan {
 public:
  explicit Pix2PixModel(bool training = true, std::vector<std::uint64_t> seeds = {6101, 6113}) : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      const Tensor    zero = nn::zeros({1, 28, 28, 1});
      PixDiscriminate(Group(1), zero, PixGenerate(Group(0), zero));
    }
    if (training) MakeAdam(.0002, .5);
  }
  /// Every tflow Evaluate field for targets x and sources (conditions) c, [n, 28, 28, 1].
  Arrays Forward(const Tensor& x, const Tensor& c) {
    const Tensor fake = PixGenerate(Group(0), c);
    const Tensor real_logit = PixDiscriminate(Group(1), c, x), fake_logit = PixDiscriminate(Group(1), c, fake);
    const auto   loss = PixObjectives(real_logit, fake_logit);
    const Tensor l1   = nn::mean(nn::abs(fake - x));
    return {{"fake", fake},
            {"real_logit", real_logit},
            {"fake_logit", fake_logit},
            {"g_adversarial", loss.generator},
            {"l1", l1},
            {"discriminator_loss", loss.discriminator},
            {"generator_loss", loss.generator + 100.0 * l1}};
  }
  Arrays Evaluate(const Tensor& x, const Tensor& c) {
    check::Require(x.size(0) >= 2 && c.size(0) >= 2, "pixel MMD diagnostic needs >= 2 cases EACH");
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, c);
    out.emplace("pixel_mmd", ImqMmd(GanCoarsePixels(Flat784(x)), GanCoarsePixels(Flat784(out.at("fake")))));
    RequireFinite(out, "all GAN fields finite");
    return out;
  }
  /// Paired translation from the SOURCE ONLY (no target/label/noise).
  Tensor Generate(const Tensor& c) {
    nn::NoGradGuard guard;
    return PixGenerate(Group(0), c);
  }
  /// Conditional PatchGAN logits of a (source, target) pair only.
  Tensor ScorePair(const Tensor& c, const Tensor& x) {
    nn::NoGradGuard guard;
    return PixDiscriminate(Group(1), c, x);
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& c) {
    return GradientOf(group, Forward(x, c).at(group == 0 ? "generator_loss" : "discriminator_loss"));
  }
  void Update(int group, const Tensor& x, const Tensor& c) { Apply(group, Gradient(group, x, c)); }
};

}  // namespace gan
