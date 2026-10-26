// Port of tflow's common/nn/star_gan.h and common/ml/star_gan.h: the StarGAN
// teaching variant with ONE domain-conditioned U-Net generator (source ++ tiled
// one-hot domain code, pix2pix's PixConv layers) and a shared discriminator
// with a 7x7 raw source (LSGAN) head and a pooled 3-way domain classifier.
// Groups: 0 = G (seed 6301), 1 = D (6311); separate Adam (.0002, beta1 .5).
#pragma once

#include "common/pix2pix.hpp"

namespace gan {

inline Tensor StarOneHot(const Tensor& domain) { return nn::one_hot(nn::cast(domain, nn::DType::Int64), 3); }

/// tflow StarGenerate: concat(source, tiled one-hot target) -> two-level U-Net -> 3x3 sigmoid 1.
inline Tensor StarGenerate(nn::ParameterStore& p, const Tensor& source, const Tensor& domain) {
  const Tensor code  = nn::tile(nn::reshape(StarOneHot(domain), {-1, 1, 1, 3}), {1, 28, 28, 1});
  const Tensor input = nn::concat({source, code}, 3);
  const Tensor d14   = PixConv(p, "down14", input, 4, 4, 4, 2, nn::Activation::Relu);
  const Tensor d7    = PixConv(p, "down7", d14, 4, 8, 4, 2, nn::Activation::Relu);
  const Tensor u14   = PixConv(p, "up14", d7, 8, 4, 4, 2, nn::Activation::Relu, true);
  const Tensor u28   = PixConv(p, "up28", nn::concat({u14, d14}, 3), 8, 2, 4, 2, nn::Activation::Relu, true);
  return PixConv(p, "output", nn::concat({u28, input}, 3), 6, 1, 3, 1, nn::Activation::Sigmoid);
}

struct StarDiscriminator {
  Tensor source, domain_logits, domain_probability;
};
/// tflow StarDiscriminate: image only -> 4 -> 8 leaky .2 -> (3x3 raw source map, pooled 3 domain logits).
inline StarDiscriminator StarDiscriminate(nn::ParameterStore& p, const Tensor& image) {
  const Tensor d14    = nn::leaky_relu(PixConv(p, "down14", image, 1, 4, 4, 2), .2);
  const Tensor d7     = nn::leaky_relu(PixConv(p, "down7", d14, 4, 8, 4, 2), .2);
  const Tensor logits = nn::dense(p, "domain", nn::mean(d7, {1, 2}), 3);
  return {PixConv(p, "source", d7, 8, 1, 3, 1), logits, nn::softmax(logits, -1)};
}
/// Mean softmax cross entropy of 3 domain logits against integer targets.
inline Tensor StarDomainLoss(const Tensor& logits, const Tensor& target) {
  return nn::mean(-nn::sum(StarOneHot(target) * nn::log_softmax(logits, -1), {1}));
}

class StarGanModel : public GroupedGan {
 public:
  explicit StarGanModel(bool training = true, std::vector<std::uint64_t> seeds = {6301, 6311}) : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      const Tensor    zero = nn::zeros({1, 28, 28, 1});
      StarDiscriminate(Group(1), StarGenerate(Group(0), zero, nn::zeros({1}, nn::DType::Int32)));
    }
    if (training) MakeAdam(.0002, .5);
  }
  /// One integer domain 0..2 per 28x28 image, checked BEFORE any one-hot.
  static void Validate(const Tensor& x, const Tensor& domain) {
    check::Require(x.dim() == 4 && x.size(1) == 28 && x.size(2) == 28 && x.size(3) == 1 &&
                       domain.dtype() == nn::DType::Int32 && domain.shape() == nn::Shape({x.size(0)}),
                   "one integer domain per 28x28 image");
    for (auto d : domain.contiguous().to_vector<std::int32_t>()) check::Require(d >= 0 && d < 3, "domain ID 0..2 before OneHot");
  }
  Arrays Forward(const Tensor& x, const Tensor& original, const Tensor& target) {
    Validate(x, original);
    Validate(x, target);
    const Tensor fake = StarGenerate(Group(0), x, target), reconstruction = StarGenerate(Group(0), fake, original);
    const auto   real = StarDiscriminate(Group(1), x), generated = StarDiscriminate(Group(1), fake);
    const Tensor source_loss_g = nn::mean(nn::square(generated.source - 1.0));
    const Tensor source_loss_d = .5 * (nn::mean(nn::square(real.source - 1.0)) + nn::mean(nn::square(generated.source)));
    const Tensor real_domain_loss = StarDomainLoss(real.domain_logits, original);
    const Tensor fake_domain_loss = StarDomainLoss(generated.domain_logits, target);
    const Tensor reconstruction_loss = nn::mean(nn::abs(reconstruction - x));
    return {{"fake", fake},
            {"reconstruction", reconstruction},
            {"real_source", real.source},
            {"fake_source", generated.source},
            {"real_domain_logits", real.domain_logits},
            {"fake_domain_logits", generated.domain_logits},
            {"real_domain_probability", real.domain_probability},
            {"fake_domain_probability", generated.domain_probability},
            {"source_loss_g", source_loss_g},
            {"source_loss_d", source_loss_d},
            {"real_domain_loss", real_domain_loss},
            {"fake_domain_loss", fake_domain_loss},
            {"reconstruction_loss", reconstruction_loss},
            {"generator_loss", (source_loss_g + fake_domain_loss) + 10.0 * reconstruction_loss},
            {"discriminator_loss", source_loss_d + real_domain_loss}};
  }
  Arrays Evaluate(const Tensor& x, const Tensor& original, const Tensor& target) {
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, original, target);
    RequireFinite(out, "finite StarGAN fields");
    return out;
  }
  /// SOURCE + TARGET DOMAIN only (no original domain / paired image).
  Tensor Generate(const Tensor& x, const Tensor& target) {
    Validate(x, target);
    nn::NoGradGuard guard;
    return StarGenerate(Group(0), x, target);
  }
  /// IMAGE ONLY source + domain classifier.
  Arrays Classify(const Tensor& x) {
    nn::NoGradGuard guard;
    const auto      d = StarDiscriminate(Group(1), x);
    return {{"source", d.source}, {"domain_probability", d.domain_probability}};
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& original, const Tensor& target) {
    return GradientOf(group, Forward(x, original, target).at(group == 0 ? "generator_loss" : "discriminator_loss"));
  }
  void Update(int group, const Tensor& x, const Tensor& original, const Tensor& target) {
    Apply(group, Gradient(group, x, original, target));
  }
};

}  // namespace gan
