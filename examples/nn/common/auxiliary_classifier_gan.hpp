// Port of tflow's common/nn/auxiliary_classifier_gan.h and
// common/ml/auxiliary_classifier_gan.h: ACGAN with a class-conditional G and an
// IMAGE-ONLY D whose shared tanh features feed a source logit and a 10-class
// auxiliary head. Group 0 = G, group 1 = D (arrays "0/...", "1/...").
#pragma once

#include "common/gan.hpp"

namespace gan {

struct AuxiliaryDiscriminator {
  Tensor source, class_logits, class_probability;
};

/// tflow AuxiliaryGanDiscriminate: 784 -> 32 tanh -> {source 1, class 10}.
inline AuxiliaryDiscriminator AuxiliaryGanDiscriminate(nn::ParameterStore& p, const Tensor& image) {
  const Tensor h      = nn::dense(p, "hidden", image, 32, nn::Activation::Tanh);
  const Tensor logits = nn::dense(p, "class", h, 10);
  return {nn::dense(p, "source", h, 1), logits, nn::softmax(logits)};
}

/// Mean categorical cross entropy of logits against integer labels.
inline Tensor AuxiliaryClassLoss(const Tensor& logits, const Tensor& labels) {
  return nn::mean(-nn::sum(OneHot10(labels) * nn::log_softmax(logits), {1}));
}

/// tflow ml::AuxiliaryClassifierGanModel. D loss = logistic source loss + CE(real)
/// + CE(fake); G loss = non-saturating source loss + CE(fake), both with the
/// requested labels as targets.
class AuxiliaryClassifierGanModel : public GroupedGan {
 public:
  explicit AuxiliaryClassifierGanModel(bool training = true, std::uint64_t g_seed = 5209, std::uint64_t d_seed = 5219)
      : GroupedGan({g_seed, d_seed}) {
    {
      nn::NoGradGuard guard;
      AuxiliaryGanDiscriminate(Group(1), ConditionalGanGenerate(Group(0), nn::zeros({2, 16}), nn::zeros({2}, nn::DType::Int32)));
    }
    if (training) MakeAdam(.0002, .5);
  }

  Arrays Forward(const Tensor& x, const Tensor& z, const Tensor& labels) {
    ValidateCondition(z, labels);
    check::Require(x.size(0) == z.size(0), "matching real/fake conditional batch sizes");
    const Tensor fake = ConditionalGanGenerate(Group(0), z, labels);
    const auto   real = AuxiliaryGanDiscriminate(Group(1), Flat784(x)), generated = AuxiliaryGanDiscriminate(Group(1), fake);
    const auto   source          = LogisticGanObjectives(real.source, generated.source);
    const Tensor real_class_loss = AuxiliaryClassLoss(real.class_logits, labels);
    const Tensor fake_class_loss = AuxiliaryClassLoss(generated.class_logits, labels);
    return {{"fake", fake},
            {"real_logit", real.source},
            {"fake_logit", generated.source},
            {"real_probability", nn::sigmoid(real.source)},
            {"fake_probability", nn::sigmoid(generated.source)},
            {"discriminator_loss", source.discriminator + (real_class_loss + fake_class_loss)},
            {"generator_loss", source.generator + fake_class_loss},
            {"real_class_logits", real.class_logits},
            {"fake_class_logits", generated.class_logits},
            {"real_class_probability", real.class_probability},
            {"fake_class_probability", generated.class_probability},
            {"real_class_loss", real_class_loss},
            {"fake_class_loss", fake_class_loss},
            {"source_discriminator_loss", source.discriminator},
            {"source_generator_loss", source.generator}};
  }

  Arrays Evaluate(const Tensor& x, const Tensor& z, const Tensor& labels) {
    check::Require(x.size(0) >= 2 && z.size(0) >= 2, "pixel MMD diagnostic needs >= 2 cases EACH");
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, z, labels);
    out.emplace("pixel_mmd", ImqMmd(GanCoarsePixels(Flat784(x)), GanCoarsePixels(out.at("fake"))));
    RequireFinite(out, "all GAN fields finite");
    return out;
  }
  Tensor Sample(const Tensor& z, const Tensor& labels) {
    ValidateCondition(z, labels);
    nn::NoGradGuard guard;
    return ConditionalGanGenerate(Group(0), z, labels);
  }
  /// IMAGE-ONLY D: no class condition, noise or targets.
  Arrays Classify(const Tensor& x) {
    nn::NoGradGuard guard;
    const auto      d = AuxiliaryGanDiscriminate(Group(1), Flat784(x));
    return {{"source_logit", d.source}, {"class_probability", d.class_probability}};
  }
  /// Isolated G partials of the source (auxiliary = false) or class term.
  Arrays ComponentGradient(bool auxiliary, const Tensor& x, const Tensor& z, const Tensor& labels) {
    return GradientOf(0, Forward(x, z, labels).at(auxiliary ? "fake_class_loss" : "source_generator_loss"));
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& z, const Tensor& labels) {
    return GradientOf(group, Forward(x, z, labels).at(group == 0 ? "generator_loss" : "discriminator_loss"));
  }
  void Update(int group, const Tensor& x, const Tensor& z, const Tensor& labels) {
    Apply(group, Gradient(group, x, z, labels));
  }
};

}  // namespace gan
