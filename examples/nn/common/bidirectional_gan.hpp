// Port of tflow's common/nn/bidirectional_gan.h and common/ml/bidirectional_gan.h:
// BiGAN/ALI on 2-D points. A joint discriminator scores (data, encoded code)
// against (generated data, prior code) pairs; the non-saturating E/G objective
// mean softplus(real) + mean softplus(-fake) trains G and E as separate groups.
// Groups: 0 = G, 1 = E, 2 = joint D ("0/...", "1/...", "2/..."). No
// reconstruction term in training; reconstructions are diagnostics only.
#pragma once

#include "common/gan.hpp"

namespace gan {

/// tflow BiganMap: 2 -> 16 tanh -> 2 linear.
inline Tensor BiganMap(nn::ParameterStore& p, const Tensor& x) {
  return nn::dense(p, "output", nn::dense(p, "hidden", x, 16, nn::Activation::Tanh), 2);
}
/// tflow BiganJointScore: concat(image 2, code 2) -> 16 tanh -> 1 raw logit.
inline Tensor BiganJointScore(nn::ParameterStore& p, const Tensor& image, const Tensor& code) {
  return nn::dense(p, "output", nn::dense(p, "hidden", nn::concat({image, code}, 1), 16, nn::Activation::Tanh), 1);
}

class BidirectionalGanModel : public GroupedGan {
 public:
  explicit BidirectionalGanModel(bool training = true, std::vector<std::uint64_t> seeds = {6007, 6011, 6029})
      : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      const Tensor    zero = nn::zeros({1, 2});
      BiganMap(Group(0), zero);
      BiganMap(Group(1), zero);
      BiganJointScore(Group(2), zero, zero);
    }
    if (training) MakeAdam(.0002, .5);
  }

  Arrays Forward(const Tensor& x, const Tensor& z) {
    const Tensor fake = BiganMap(Group(0), z), encoded = BiganMap(Group(1), x);
    const Tensor real_score = BiganJointScore(Group(2), x, encoded), fake_score = BiganJointScore(Group(2), fake, z);
    const auto   source     = LogisticGanObjectives(real_score, fake_score);
    return {{"fake", fake},
            {"encoded", encoded},
            {"real_score", real_score},
            {"fake_score", fake_score},
            {"discriminator_loss", source.discriminator},
            {"eg_loss", nn::mean(nn::softplus(real_score)) + source.generator},
            {"reconstruction", BiganMap(Group(0), encoded)},
            {"latent_cycle", BiganMap(Group(1), fake)}};
  }
  Arrays Evaluate(const Tensor& x, const Tensor& z) {
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, z);
    RequireFinite(out, "finite BiGAN fields");
    return out;
  }
  /// Prior noise ONLY generation.
  Tensor Generate(const Tensor& z) {
    nn::NoGradGuard guard;
    return BiganMap(Group(0), z);
  }
  /// Data ONLY encoding.
  Tensor Encode(const Tensor& x) {
    nn::NoGradGuard guard;
    return BiganMap(Group(1), x);
  }
  /// Joint pair score from supplied data AND code ONLY.
  Tensor ScorePair(const Tensor& x, const Tensor& code) {
    nn::NoGradGuard guard;
    return BiganJointScore(Group(2), x, code);
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& z) {
    return GradientOf(group, Forward(x, z).at(group == 2 ? "discriminator_loss" : "eg_loss"));
  }
  void Update(int group, const Tensor& x, const Tensor& z) { Apply(group, Gradient(group, x, z)); }
};

}  // namespace gan
