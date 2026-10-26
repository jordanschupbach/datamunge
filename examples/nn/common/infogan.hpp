// Port of tflow's common/nn/infogan.h and common/ml/infogan.h: unsupervised
// InfoGAN with a 10-way categorical and a 2-D continuous generated-prior code.
// Groups: 0 = G, 1 = D, 2 = recognizer Q (arrays "0/...", "1/...", "2/...").
#pragma once

#include <cmath>
#include <random>

#include "common/gan.hpp"

namespace gan {

/// tflow InfoGanGenerate: concat(z 16, onehot(category) 10, continuous 2) -> 32 tanh -> 784 sigmoid.
inline Tensor InfoGanGenerate(nn::ParameterStore& p, const Tensor& z, const Tensor& category, const Tensor& continuous) {
  return MlpGanGenerate(p, nn::concat({z, OneHot10(category), continuous}, 1));
}

struct InfoRecognition {
  Tensor logits, probability, mean;
};
/// tflow InfoGanRecognize: image -> 32 tanh -> {category 10 logits, continuous 2 means}.
inline InfoRecognition InfoGanRecognize(nn::ParameterStore& p, const Tensor& image) {
  const Tensor h      = nn::dense(p, "hidden", image, 32, nn::Activation::Tanh);
  const Tensor logits = nn::dense(p, "category", h, 10);
  return {logits, nn::softmax(logits), nn::dense(p, "continuous", h, 2)};
}

struct InfoRecognitionLoss {
  Tensor categorical, continuous, total, lower_bound;
};
/// Categorical CE + unit-variance Gaussian NLL .5 mean ||u - mean||^2 + log(2 pi);
/// MI lower bound = H(c) + H(u) - total = log 10 + 2 log 2 - total.
inline InfoRecognitionLoss InfoGanRecognitionLoss(const InfoRecognition& q, const Tensor& category,
                                                  const Tensor& continuous) {
  const Tensor cat   = nn::mean(-nn::sum(OneHot10(category) * nn::log_softmax(q.logits), {1}));
  const Tensor cont  = .5 * nn::mean(nn::sum(nn::square(continuous - q.mean), {1})) + std::log(2 * std::acos(-1.));
  const Tensor total = cat + cont;
  return {cat, cont, total, (std::log(10.) + 2 * std::log(2.)) - total};
}

/// tflow's InfoGAN Prior: category = v % 10 from a rejection-sampled mt19937
/// draw (v < 4294967290), then two uniform(-1, 1) codes, row by row.
struct InfoCodes {
  Tensor category, continuous, raw;
};
inline InfoCodes InfoPrior(std::int64_t n, std::uint32_t seed) {
  std::vector<std::int32_t> category(n);
  std::vector<float>        continuous(n * 2);
  std::vector<double>       raw(n * 3);
  std::mt19937              rng(seed);
  for (std::int64_t b = 0; b < n; ++b) {
    std::uint32_t v;
    do v = rng();
    while (v >= 4294967290u);
    raw[b * 3]  = v;
    category[b] = std::int32_t(v % 10);
    for (int j = 0; j < 2; ++j) {
      v                     = rng();
      raw[b * 3 + j + 1]    = v;
      continuous[b * 2 + j] = float(2 * ((double(v) + .5) / 4294967296.) - 1);
    }
  }
  return {nn::from_vector(category, {n}), nn::from_vector(continuous, {n, 2}),
          nn::from_vector(raw, {n, 3})};
}

class InfoGanModel : public GroupedGan {
 public:
  static void Validate(const Tensor& z, const Tensor& c, const Tensor& u) {
    check::Require(z.dtype() == nn::DType::Float32 && z.dim() == 2 && z.size(1) == 16 &&
                       c.dtype() == nn::DType::Int32 && c.shape() == nn::Shape({z.size(0)}) &&
                       u.dtype() == nn::DType::Float32 && u.shape() == nn::Shape({z.size(0), 2}),
                   "prior code/noise shapes");
    for (auto k : c.contiguous().to_vector<std::int32_t>()) check::Require(k >= 0 && k < 10, "categorical prior 0..9");
    for (auto x : u.contiguous().to_vector<float>())
      check::Require(std::isfinite(x) && std::abs(x) <= 1, "continuous prior [-1, 1]");
  }

  explicit InfoGanModel(bool training = true, std::vector<std::uint64_t> seeds = {5303, 5309, 5317})
      : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      const Tensor    fake =
          InfoGanGenerate(Group(0), nn::zeros({2, 16}), nn::zeros({2}, nn::DType::Int32), nn::zeros({2, 2}));
      MlpGanDiscriminate(Group(1), fake);
      InfoGanRecognize(Group(2), fake);
    }
    if (training) MakeAdam(.0002, .5);
  }

  Arrays Forward(const Tensor& x, const Tensor& z, const Tensor& c, const Tensor& u) {
    Validate(z, c, u);
    const Tensor fake       = InfoGanGenerate(Group(0), z, c, u);
    const Tensor real_logit = MlpGanDiscriminate(Group(1), Flat784(x)), fake_logit = MlpGanDiscriminate(Group(1), fake);
    const auto   loss       = LogisticGanObjectives(real_logit, fake_logit);
    const auto   q          = InfoGanRecognize(Group(2), fake);
    const auto   r          = InfoGanRecognitionLoss(q, c, u);
    return {{"fake", fake},
            {"real_logit", real_logit},
            {"fake_logit", fake_logit},
            {"real_probability", nn::sigmoid(real_logit)},
            {"fake_probability", nn::sigmoid(fake_logit)},
            {"discriminator_loss", loss.discriminator},
            {"generator_loss", loss.generator + r.total},
            {"source_generator_loss", loss.generator},
            {"recognition_loss", r.total},
            {"categorical_loss", r.categorical},
            {"continuous_loss", r.continuous},
            {"mi_lower_bound", r.lower_bound},
            {"q_logits", q.logits},
            {"q_probability", q.probability},
            {"q_mean", q.mean}};
  }
  Arrays Evaluate(const Tensor& x, const Tensor& z, const Tensor& c, const Tensor& u) {
    check::Require(x.size(0) >= 2 && z.size(0) >= 2, "pixel MMD diagnostic needs >= 2 cases EACH");
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, z, c, u);
    out.emplace("pixel_mmd", ImqMmd(GanCoarsePixels(Flat784(x)), GanCoarsePixels(out.at("fake"))));
    RequireFinite(out, "all GAN fields finite");
    return out;
  }
  Tensor Sample(const Tensor& z, const Tensor& c, const Tensor& u) {
    Validate(z, c, u);
    nn::NoGradGuard guard;
    return InfoGanGenerate(Group(0), z, c, u);
  }
  /// Q from pixels ONLY (no original image, noise, code or target).
  Arrays Recognize(const Tensor& pixels) {
    nn::NoGradGuard guard;
    const auto      q = InfoGanRecognize(Group(2), Flat784(pixels));
    return {{"q_probability", q.probability}, {"q_mean", q.mean}};
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& z, const Tensor& c, const Tensor& u) {
    static const char* objective[] = {"generator_loss", "discriminator_loss", "recognition_loss"};
    return GradientOf(group, Forward(x, z, c, u).at(objective[group]));
  }
  void Update(int group, const Tensor& x, const Tensor& z, const Tensor& c, const Tensor& u) {
    Apply(group, Gradient(group, x, z, c, u));
  }
};

}  // namespace gan
