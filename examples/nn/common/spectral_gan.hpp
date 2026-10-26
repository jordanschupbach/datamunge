// Port of tflow's common/nn/spectral_gan.h and common/ml/spectral_gan.h:
// spectral normalization of an MLP discriminator with cached, nontrainable
// singular vectors (u [out, 1], v [in, 1]) and a DIFFERENTIABLE sigma =
// v^T W u (u, v held constant in backpropagation). Groups: 0 = MLP generator,
// 1 = discriminator (Glorot-uniform kernels "<layer>/W", zero biases), 2 =
// spectral state ("<layer>/u", "<layer>/v"); Adam only for 0 and 1.
#pragma once

#include "common/big_gan.hpp"

namespace gan {

/// tflow SpectralDense: x W / max(v^T W u, 1e-6) + b, then activation.
inline Tensor SpectralDense(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const Tensor& x,
                            std::int64_t in, std::int64_t out, nn::Activation act = nn::Activation::None) {
  const Tensor w = p.get(name + "/W", {in, out}, nn::init::uniform(std::sqrt(6.f / float(in + out))));
  const Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  const Tensor u = state.get(name + "/u", {out, 1}, nn::init::constant(1.f / std::sqrt(float(out))));
  const Tensor v = state.get(name + "/v", {in, 1}, nn::init::constant(1.f / std::sqrt(float(in))));
  const Tensor sigma = nn::sum(v.detach() * nn::matmul(w, u.detach()));
  return nn::activate(nn::matmul(x, w / nn::maximum(sigma, nn::scalar_like(double(1e-6f), sigma))) + b, act);
}

/// tflow SpectralDiscriminate: image [n, 784] -> spectral 32 tanh -> spectral 1 raw score.
inline Tensor SpectralDiscriminate(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& x) {
  return SpectralDense(p, state, "output", SpectralDense(p, state, "hidden", x, 784, 32, nn::Activation::Tanh), 32, 1);
}

/// tflow ml::SpectralGanModel: hinge objectives, 16 warm-up power iterations
/// when training, Advance() (one ordered state-only power iteration) before each D step.
class SpectralGanModel : public GroupedGan {
 public:
  explicit SpectralGanModel(bool training = true, std::vector<std::uint64_t> seeds = {5801, 5807, 0})
      : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      SpectralDiscriminate(Group(1), Group(2), MlpGanGenerate(Group(0), nn::zeros({2, 16})));
    }
    if (training) {
      MakeAdam(.0002, .5, .999, {0, 1});
      for (int i = 0; i < 16; ++i) Advance();
    }
  }
  void Advance() { BigAdvance(Group(1), Group(2)); }

  Arrays Forward(const Tensor& x, const Tensor& z) {
    const Tensor fake       = MlpGanGenerate(Group(0), z);
    const Tensor real_score = SpectralDiscriminate(Group(1), Group(2), Flat784(x));
    const Tensor fake_score = SpectralDiscriminate(Group(1), Group(2), fake);
    const auto   loss       = HingeGanObjectives(real_score, fake_score);
    return {{"fake", fake},
            {"real_score", real_score},
            {"fake_score", fake_score},
            {"discriminator_loss", loss.discriminator},
            {"generator_loss", loss.generator}};
  }
  Arrays Evaluate(const Tensor& x, const Tensor& z) {
    check::Require(x.size(0) >= 2 && z.size(0) >= 2, "pixel MMD diagnostic needs >= 2 cases EACH");
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, z);
    out.emplace("pixel_mmd", ImqMmd(GanCoarsePixels(Flat784(x)), GanCoarsePixels(out.at("fake"))));
    RequireFinite(out, "all GAN fields finite");
    return out;
  }
  Tensor Sample(const Tensor& z) {
    nn::NoGradGuard guard;
    return MlpGanGenerate(Group(0), z);
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& z) {
    return GradientOf(group, Forward(x, z).at(group == 0 ? "generator_loss" : "discriminator_loss"));
  }
  void Update(int group, const Tensor& x, const Tensor& z) { Apply(group, Gradient(group, x, z)); }
};

}  // namespace gan
