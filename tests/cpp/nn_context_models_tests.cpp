#include <gtest/gtest.h>

#include <dm/dm.hpp>

#include "../fixtures/nn_context_encoder_checks.hpp"

TEST(ContextTensor, IndependentPartialValuesAndEveryJacobian) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(PartialConvChecks());
}
TEST(ContextTensor, MaskAlignmentLeakageCompositeAndHoleMetrics) {
  EXPECT_NO_THROW(DataCompositeChecks());
}
TEST(ContextTensor, StableExtremeAdversarialGradients) {
  EXPECT_NO_THROW(AdversarialLossChecks());
}
TEST(ContextTensor, FullCompactGraphsAndEveryLearnedGradient) {
  EXPECT_NO_THROW(ModelChecks());
}
TEST(ContextTensor, OriginalBytePreprocessingAndEveryPreviewPixel) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<int32_t> raw(3 * 3072);
  for (size_t i = 0; i < raw.size(); ++i)
    raw[i] = (i * 31 + 17) % 256;
  const auto pairs = m::ContextPairs(n::from_vector(raw, {3, 32, 32, 3}), {2, 0, 2});
  const auto truth = pairs.truth.to_vector<float>(), masked = pairs.masked.to_vector<float>(),
             patch    = pairs.patch.to_vector<float>();
  const auto   valid  = pairs.valid.to_vector<float>();
  const int    ids[]  = {2, 0, 2};
  const double fill[] = {117. / 255, 104. / 255, 123. / 255};
  for (int b = 0; b < 3; ++b)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x) {
        const bool hole = y >= 8 && y < 24 && x >= 8 && x < 24;
        EXPECT_EQ(valid[(b * 32 + y) * 32 + x], !hole);
        for (int c = 0; c < 3; ++c) {
          const int   i = ((b * 32 + y) * 32 + x) * 3 + c;
          const float v = float(raw[((ids[b] * 32 + y) * 32 + x) * 3 + c]) / 255.f;
          EXPECT_EQ(truth[i], v);
          EXPECT_EQ(masked[i], float(2 * (hole ? fill[c] : v) - 1));
          if (hole)
            EXPECT_EQ(patch[((b * 16 + y - 8) * 16 + x - 8) * 3 + c], 2 * v - 1);
        }
      }
  const auto mean      = m::ContextFillMean(pairs);
  const auto predicted = n::clip(mean + .1, 0., 1.);
  const auto tile      = m::ContextPreviewTiles(pairs, predicted, mean).to_vector<float>();
  const auto mv = mean.to_vector<float>(), pv = predicted.to_vector<float>();
  for (int b = 0; b < 3; ++b)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x)
        for (int c = 0; c < 3; ++c) {
          const int    i        = ((b * 32 + y) * 32 + x) * 3 + c;
          const double values[] = {
              (double(masked[i]) + 1) / 2,
              mv[i],
              pv[i],
              truth[i],
              c == 0 ? 4 * std::abs(double(pv[i]) - truth[i]) : 0};
          for (int j = 0; j < 5; ++j)
            EXPECT_EQ(
                tile[((b * 32 + y) * 160 + j * 32 + x) * 3 + c],
                float(std::lround(255 * std::clamp(values[j], 0., 1.))) / 255.f);
        }
  const auto rows = m::InpaintBatch(pairs, {2, 0});
  EXPECT_EQ(rows.truth.shape(), (n::Shape{2, 32, 32, 3}));
  EXPECT_THROW(
      m::ContextPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {2}), std::invalid_argument);
  EXPECT_THROW(m::ContextConfig({31, 8, 32}).Encoder(), std::invalid_argument);
}
TEST(ContextTensor, AlternationFreezesOtherWeightsAndRunningState) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::ParameterStore generator(71), gstate(72), discriminator(73), dstate(74);
  const auto        config = m::ContextConfig::Compact();
  const auto        data   = m::ContextPairs(
      n::arange(0, 8 * 3072, 1, n::DType::Int32).reshape({8, 32, 32, 3}), {0, 1, 2, 3, 4, 5, 6, 7});
  {
    n::NoGradGuard guard;
    const auto     patch = m::ContextEncoder(generator, gstate, data.masked, config, false).patch;
    m::ContextDiscriminator(discriminator, dstate, patch, config, false);
  }
  auto snapshot = [](const n::ParameterStore& p, const n::ParameterStore& state) {
    auto out = p.flatten(), v = state.flatten();
    out.insert(out.end(), v.begin(), v.end());
    return out;
  };
  n::OptimizerConfig adam;
  adam.kind  = n::OptimizerKind::Adam;
  adam.beta1 = .5;
  n::Optimizer g_optimizer(generator, .002, adam), d_optimizer(discriminator, .0002, adam);
  const auto   before_g = snapshot(generator, gstate);
  n::Tensor    fake;
  {
    n::NoGradGuard guard;
    fake = m::ContextEncoder(generator, gstate, data.masked, config, true, false).patch;
  }
  const auto d = m::ContextDiscriminator(
      discriminator, dstate, n::concat({data.patch, fake}, 0), config, true, true);
  const auto labels = n::concat({n::ones({8}), n::zeros({8})}, 0);
  const auto d_loss = 2. * n::mean(n::softplus(d.logits) - labels * d.logits);
  d_optimizer.step(n::grad(d_loss, discriminator.parameters()));
  EXPECT_EQ(before_g, snapshot(generator, gstate));
  const auto before_d  = snapshot(discriminator, dstate);
  const auto generated = m::ContextEncoder(generator, gstate, data.masked, config, true, true);
  const auto fool = m::ContextDiscriminator(discriminator, dstate, generated.patch, config, false);
  const auto g_loss = .999 * m::ContextReconstruction(generated.patch, data.patch) +
                      .001 * m::ContextBCE(fool.logits, true);
  g_optimizer.step(n::grad(g_loss, generator.parameters()));
  EXPECT_EQ(before_d, snapshot(discriminator, dstate));
  EXPECT_NE(before_g, snapshot(generator, gstate));
}
