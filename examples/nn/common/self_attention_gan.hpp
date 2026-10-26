// Port of tflow's common/nn/self_attention_gan.h and common/ml/self_attention_gan.h:
// the compact patch SAGAN teaching variant. 16 spatial tokens (a 4x4 grid of
// 7x7 patches) with 8 features each; unscaled softmax(q k^T) attention over all
// 16 positions (q, k 2-D, v 4-D, projection 8) and a residual gamma (init 0).
// Every kernel is spectrally normalized (spectral_gan.hpp's SpectralDense).
// Groups: 0 = G, 1 = D, 2 = G spectral state, 3 = D spectral state.
#pragma once

#include "common/spectral_gan.hpp"

namespace gan {

struct SpatialAttention {
  Tensor input, probability, output;
};

/// tflow SagAttention: tokens [n, 16, 8] -> input + gamma * project(softmax(q k^T) v).
inline SpatialAttention SagAttention(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& input) {
  const Tensor flat = nn::reshape(input, {-1, 8});
  const Tensor q = nn::reshape(SpectralDense(p, state, "attention/query", flat, 8, 2), {-1, 16, 2});
  const Tensor k = nn::reshape(SpectralDense(p, state, "attention/key", flat, 8, 2), {-1, 16, 2});
  const Tensor v = nn::reshape(SpectralDense(p, state, "attention/value", flat, 8, 4), {-1, 16, 4});
  const Tensor probability = nn::softmax(nn::matmul(q, k, false, true), -1);
  const Tensor mixed       = nn::reshape(nn::matmul(probability, v), {-1, 4});
  const Tensor projected   = nn::reshape(SpectralDense(p, state, "attention/project", mixed, 4, 8), {-1, 16, 8});
  const Tensor gamma       = p.get("attention/gamma", {}, nn::init::zeros());
  return {input, probability, input + gamma * projected};
}

struct SagGenerated {
  Tensor           pixels;
  SpatialAttention attention;
};
/// tflow SagGenerate: z [n, 16] -> 128 tanh = 16x8 tokens -> attention -> shared 8 -> 49 sigmoid patches -> [n, 784].
inline SagGenerated SagGenerate(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& z) {
  const Tensor tokens    = nn::reshape(SpectralDense(p, state, "input", z, 16, 128, nn::Activation::Tanh), {-1, 16, 8});
  auto         attention = SagAttention(p, state, tokens);
  const Tensor patches =
      SpectralDense(p, state, "patch", nn::reshape(attention.output, {-1, 8}), 8, 49, nn::Activation::Sigmoid);
  return {nn::reshape(nn::transpose(nn::reshape(patches, {-1, 4, 4, 7, 7}), {0, 1, 3, 2, 4}), {-1, 784}), attention};
}

struct SagDiscriminated {
  Tensor           score;
  SpatialAttention attention;
};
/// tflow SagDiscriminate: [n, 784] -> 16 patches 49 -> 8 tanh + positions [16, 8] -> attention -> mean -> score 1.
inline SagDiscriminated SagDiscriminate(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& image) {
  const Tensor patches = nn::reshape(nn::transpose(nn::reshape(image, {-1, 4, 7, 4, 7}), {0, 1, 3, 2, 4}), {-1, 49});
  const Tensor encoded = nn::reshape(SpectralDense(p, state, "patch", patches, 49, 8, nn::Activation::Tanh), {-1, 16, 8});
  const Tensor positions = p.get("position", {16, 8}, nn::init::normal(.01));
  auto         attention = SagAttention(p, state, encoded + positions);
  return {SpectralDense(p, state, "output", nn::mean(attention.output, {1}), 8, 1), attention};
}

/// tflow ml::SelfAttentionGanModel: hinge objectives; 16 warm-up power
/// iterations of both networks; Advance(group) before each group's Adam step.
class SelfAttentionGanModel : public GroupedGan {
 public:
  explicit SelfAttentionGanModel(bool training = true, std::vector<std::uint64_t> seeds = {5903, 5923, 0, 0})
      : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      SagDiscriminate(Group(1), Group(3), SagGenerate(Group(0), Group(2), nn::zeros({2, 16})).pixels);
    }
    if (training) {
      MakeAdam(.0002, .5, .999, {0, 1});
      for (int i = 0; i < 16; ++i) Advance(0), Advance(1);
    }
  }
  /// One ordered power iteration of G's (0) or D's (1) spectral vectors.
  void Advance(int group) { BigAdvance(Group(group), Group(group + 2)); }

  Arrays Forward(const Tensor& x, const Tensor& z) {
    const auto g    = SagGenerate(Group(0), Group(2), z);
    const auto real = SagDiscriminate(Group(1), Group(3), Flat784(x)), fake = SagDiscriminate(Group(1), Group(3), g.pixels);
    const auto loss = HingeGanObjectives(real.score, fake.score);
    return {{"fake", g.pixels},
            {"real_score", real.score},
            {"fake_score", fake.score},
            {"discriminator_loss", loss.discriminator},
            {"generator_loss", loss.generator},
            {"g_attention", g.attention.probability},
            {"g_attention_input", g.attention.input},
            {"g_attention_output", g.attention.output},
            {"real_attention", real.attention.probability},
            {"real_attention_input", real.attention.input},
            {"real_attention_output", real.attention.output},
            {"fake_attention", fake.attention.probability}};
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
    return SagGenerate(Group(0), Group(2), z).pixels;
  }
  /// Attention block of G or D from tokens [n, 16, 8] ONLY (no image/noise).
  Arrays Attention(bool use_generator, const Tensor& tokens) {
    nn::NoGradGuard guard;
    const auto      a = use_generator ? SagAttention(Group(0), Group(2), tokens) : SagAttention(Group(1), Group(3), tokens);
    return {{"probability", a.probability}, {"output", a.output}};
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& z) {
    return GradientOf(group, Forward(x, z).at(group == 0 ? "generator_loss" : "discriminator_loss"));
  }
  void Update(int group, const Tensor& x, const Tensor& z) { Apply(group, Gradient(group, x, z)); }
};

}  // namespace gan
