#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace n = dm::nn;
namespace m = n::models;
namespace {
using A = std::map<std::string, n::Tensor>;
A Snapshot(const n::ParameterStore& p) {
  A a;
  for (const auto& name : p.names())
    a.emplace(name, p.at(name).detach().clone());
  return a;
}
bool Equal(const A& a, const A& b) {
  if (a.size() != b.size())
    return false;
  for (const auto& [k, v] : a)
    if (!b.count(k) || !n::equal_bits(v, b.at(k)))
      return false;
  return true;
}
double Difference(const n::Tensor& a, const n::Tensor& b) {
  return n::max(n::abs(a - b)).item();
}
n::Tensor Gaussian(const n::Shape& shape, std::uint32_t seed, n::DType type = n::DType::Float64) {
  return n::cast(n::reshape(m::GaussianNoise(shape, 1, seed), shape), type);
}
} // namespace
TEST(NnStyleGan, SpatialNormalizationHasPopulationMomentsIndependentRowsAndLiveHigherPaths) {
  auto x   = n::tensor({1., 3., 5., 7.}, {1, 2, 2, 1}, n::DType::Float64).requires_grad_();
  auto out = n::spatial_instance_normalize(x);
  EXPECT_NEAR(n::mean(out).item(), 0, 1e-14);
  EXPECT_NEAR(n::mean(n::square(out)).item(), 5. / (5. + double(1e-8f)), 1e-14);
  EXPECT_TRUE(n::equal_bits(out, n::spatial_instance_normalize(x + 100.)));
  auto doubled = n::concat({x, x * 20. + 10.}, 0);
  EXPECT_TRUE(
      n::equal_bits(n::select(n::spatial_instance_normalize(doubled), 0, 0), n::select(out, 0, 0)));
  n::GradOptions graph;
  graph.create_graph = true;
  auto first =
      n::grad(n::sum(out * n::tensor({1., 0., 2., -1.}, {1, 2, 2, 1}, x.dtype())), {x}, graph)[0];
  auto second = n::grad(n::sum(n::square(first)), {x})[0];
  EXPECT_GT(n::max(n::abs(second)).item(), 0);
  EXPECT_EQ(n::sum(n::spatial_instance_normalize(n::ones({2, 1, 1, 3}))).item(), 0);
}
TEST(NnStyleGan, ModulatedLinearKernelHasItsNormalizationDerivativeAndHigherPaths) {
  auto t             = n::DType::Float64;
  auto x             = n::tensor({1., 0.}, {1, 1, 1, 2}, t).requires_grad_(),
       w             = n::tensor({3., 4.}, {1, 1, 2, 1}, t).requires_grad_(),
       s             = n::ones({1, 2}, t).requires_grad_();
  auto           out = n::modulated_conv2d(x, w, s);
  n::GradOptions graph;
  graph.create_graph = true;
  graph.retain_graph = true;
  auto gradients     = n::grad(n::sum(out.image), {x, w, s}, graph);
  EXPECT_NEAR(out.image.item(), .6, 1e-9);
  EXPECT_NEAR(out.demodulator.item(), .2, 1e-9);
  EXPECT_NEAR(gradients[0].at({0, 0, 0, 0}), .6, 1e-9);
  EXPECT_NEAR(gradients[0].at({0, 0, 0, 1}), .8, 1e-9);
  EXPECT_NEAR(gradients[1].at({0, 0, 0, 0}), .128, 1e-9);
  EXPECT_NEAR(gradients[1].at({0, 0, 1, 0}), -.096, 1e-9);
  EXPECT_NEAR(gradients[2].at({0, 0}), .384, 1e-9);
  EXPECT_NEAR(gradients[2].at({0, 1}), -.384, 1e-9);
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(n::square(gradients[1])), {w})[0])).item(), 0);
}
TEST(NnStyleGan, PerRowDemodulationPositiveScalingSignZeroAndNonDemodulatedRGB) {
  auto t = n::DType::Float64;
  auto x = n::ones({2, 2, 3, 2}, t), w = n::full({3, 3, 2, 3}, .125, t),
       s          = n::tensor({1., 2., 2., 4.}, {2, 2}, t);
  auto normalized = n::modulated_conv2d(x, w, s), plain = n::modulated_conv2d(x, w, s, false);
  EXPECT_LT(Difference(n::select(normalized.image, 0, 0), n::select(normalized.image, 0, 1)), 1e-7);
  EXPECT_TRUE(n::equal_bits(n::select(plain.image, 0, 1), 2. * n::select(plain.image, 0, 0)));
  EXPECT_TRUE(n::equal_bits(plain.demodulator, n::ones({2, 3}, t)));
  auto negative = n::modulated_conv2d(x, w, n::tensor({1., 2., -1., -2.}, {2, 2}, t));
  EXPECT_TRUE(n::equal_bits(n::select(negative.image, 0, 0), -n::select(negative.image, 0, 1)));
  auto zero = n::modulated_conv2d(x, w, n::zeros(s.shape(), t));
  EXPECT_EQ(n::sum(n::abs(zero.image)).item(), 0);
  EXPECT_TRUE(std::isfinite(n::max(zero.demodulator).item()));
  auto single = n::modulated_conv2d(n::take_rows(x, {0}), w, n::take_rows(s, {0}));
  EXPECT_LT(Difference(single.image, n::take_rows(normalized.image, {0})), 1e-13);
}
TEST(NnStyleGan, PrimitiveAndSpatialNormalizationRejectInvalidGeometryGuardsAndOverflow) {
  auto x = n::ones({2, 2, 3, 2}), w = n::ones({3, 3, 2, 3}), s = n::ones({2, 2});
  EXPECT_THROW(n::modulated_conv2d(x, n::ones({2, 2, 2, 3}), s), std::invalid_argument);
  EXPECT_THROW(n::modulated_conv2d(x, w, n::ones({1, 2})), std::invalid_argument);
  EXPECT_THROW(n::modulated_conv2d(x, w, s, true, 0), std::invalid_argument);
  EXPECT_THROW(n::modulated_conv2d(x, w, s, true, 1e-100), std::invalid_argument);
  EXPECT_THROW(n::modulated_conv2d(x, n::full(w.shape(), 1e30), s), std::invalid_argument);
  EXPECT_THROW(n::modulated_conv2d(x, w, n::full(s.shape(), NAN)), std::invalid_argument);
  EXPECT_THROW(n::spatial_instance_normalize(n::ones({2, 3})), std::invalid_argument);
  EXPECT_THROW(n::spatial_instance_normalize(x, -1), std::invalid_argument);
  EXPECT_THROW(n::spatial_instance_normalize(n::full(x.shape(), INFINITY)), std::invalid_argument);
  EXPECT_THROW(n::spatial_instance_normalize(n::full(x.shape(), 1e30)), std::invalid_argument);
}
TEST(NnStyleGan, VariantCensusSourceStyleShapesDemodulatorsAndUnequalRealRows) {
  for (bool version2 : {false, true}) {
    n::ParameterStore g(6607), d(6619);
    auto              z = Gaussian({3, 8}, 6659, n::DType::Float32),
         n4             = Gaussian({3, 4, 4, 1}, 6653, n::DType::Float32),
         n8             = Gaussian({3, 8, 8, 1}, 6661, n::DType::Float32);
    auto out            = m::StyleGan(g, d, n::ones({2, 8, 8, 1}), z, n4, n8, version2);
    EXPECT_EQ(g.count(), version2 ? 625 : 661);
    EXPECT_EQ(g.size(), version2 ? 19 : 17);
    EXPECT_EQ(d.count(), 493);
    EXPECT_EQ(d.size(), 6);
    EXPECT_EQ(out.fake.shape(), (n::Shape{3, 8, 8, 1}));
    EXPECT_EQ(out.mapped.shape(), (n::Shape{3, 8}));
    EXPECT_EQ(out.real_score.shape(), (n::Shape{2, 1}));
    EXPECT_EQ(out.first_style.shape(), (n::Shape{3, version2 ? 4 : 8}));
    EXPECT_EQ(out.first_demodulator.shape(), version2 ? n::Shape({3, 4}) : n::Shape({}));
    if (!version2)
      EXPECT_EQ(out.first_demodulator.item(), 1);
  }
}
TEST(NnStyleGan, InitialNoiseHasZeroEffectAndZeroInputPartialButLearnableStrengthPartial) {
  for (bool version2 : {false, true}) {
    auto              t = n::DType::Float64;
    n::ParameterStore g(6607, t), d(6619, t);
    auto z = Gaussian({2, 8}, 6659), n4 = Gaussian({2, 4, 4, 1}, 6653).requires_grad_(),
         n8  = Gaussian({2, 8, 8, 1}, 6661).requires_grad_();
    auto out = m::StyleGan(g, d, n::ones({2, 8, 8, 1}, t), z, n4, n8, version2);
    auto w   = m::StyleGanMap(g, z);
    EXPECT_TRUE(
        n::equal_bits(out.fake, m::StyleGanSynthesize(g, w, w, n4 * 2., n8 * 3., version2).image));
    n::GradOptions keep;
    keep.retain_graph = true;
    auto noise        = n::grad(out.g_loss, {n4, n8}, keep);
    EXPECT_EQ(n::max(n::abs(noise[0])).item(), 0);
    EXPECT_EQ(n::max(n::abs(noise[1])).item(), 0);
    auto strength =
        n::grad(out.g_loss, {g.at("layer4/noise_strength"), g.at("layer8/noise_strength")});
    EXPECT_GT(n::sum(n::abs(strength[0])).item() + n::sum(n::abs(strength[1])).item(), 1e-12);
    g.set("layer4/noise_strength", n::full({4}, .2, t));
    g.set("layer8/noise_strength", n::full({4}, .2, t));
    auto active = m::StyleGan(g, d, n::ones({2, 8, 8, 1}, t), z, n4, n8, version2);
    EXPECT_GT(n::max(n::abs(n::grad(active.g_loss, {n4})[0])).item(), 0);
  }
}
TEST(NnStyleGan, MappingZAndWOnlySynthesisMixingAndPureImmutableFieldsAgree) {
  for (bool version2 : {false, true}) {
    n::ParameterStore g(6607), d(6619);
    auto              t = g.dtype();
    auto              z = Gaussian({2, 8}, 6659, t), n4 = Gaussian({2, 4, 4, 1}, 6653, t),
         n8     = Gaussian({2, 8, 8, 1}, 6661, t);
    auto out    = m::StyleGan(g, d, n::zeros({2, 8, 8, 1}, t), z, n4, n8, version2);
    auto before = Snapshot(g), dbefore = Snapshot(d);
    auto w = m::StyleGanMap(g, z), other = m::StyleGanMap(g, Gaussian({2, 8}, 6719, t));
    auto plain = m::StyleGanSynthesize(g, w, w, n4, n8, version2),
         mixed = m::StyleGanSynthesize(g, w, other, n4, n8, version2);
    EXPECT_TRUE(n::equal_bits(w, out.mapped));
    EXPECT_TRUE(n::equal_bits(plain.image, out.fake));
    EXPECT_TRUE(n::equal_bits(plain.first_style, mixed.first_style));
    EXPECT_TRUE(n::equal_bits(plain.first, mixed.first));
    EXPECT_GT(Difference(plain.second_style, mixed.second_style), 1e-8);
    EXPECT_GT(Difference(plain.image, mixed.image), 1e-8);
    EXPECT_LT(
        Difference(
            m::StyleGanSynthesize(
                g,
                n::take_rows(w, {0}),
                n::take_rows(w, {0}),
                n::take_rows(n4, {0}),
                n::take_rows(n8, {0}),
                version2)
                .image,
            n::take_rows(plain.image, {0})),
        3e-6);
    EXPECT_TRUE(Equal(before, Snapshot(g)));
    EXPECT_TRUE(Equal(dbefore, Snapshot(d)));
  }
}
TEST(NnStyleGan, DoubledFirstActualStylePreservesDemodulatedOutputAndChangesItsFactor) {
  n::ParameterStore g(6607);
  auto              w = m::StyleGanMap(g, Gaussian({2, 8}, 6659, g.dtype()));
  auto n4 = Gaussian({2, 4, 4, 1}, 6653, g.dtype()), n8 = Gaussian({2, 8, 8, 1}, 6661, g.dtype());
  auto original = m::StyleGanSynthesize(g, w, w, n4, n8, true);
  auto doubled  = m::StyleGanSynthesize(g, w, w, n4, n8, true, 2. * original.first_style);
  EXPECT_LT(Difference(original.image, doubled.image), 3e-6);
  EXPECT_LT(
      n::max(n::abs(doubled.first_demodulator * 2. / original.first_demodulator - 1.)).item(),
      2e-4);
  EXPECT_TRUE(n::equal_bits(original.second_style, doubled.second_style));
}
TEST(NnStyleGan, ZeroTruncationNeedsSharedNoiseToCollapseImagesWithActiveStrengths) {
  for (bool version2 : {false, true}) {
    auto              t = n::DType::Float64;
    n::ParameterStore g(6607, t);
    auto              mapped = m::StyleGanMap(g, Gaussian({2, 8}, 6659));
    auto              n4 = Gaussian({2, 4, 4, 1}, 6653), n8 = Gaussian({2, 8, 8, 1}, 6661);
    m::StyleGanSynthesize(g, mapped, mapped, n4, n8, version2);
    g.set("layer4/noise_strength", n::full({4}, .2, t));
    g.set("layer8/noise_strength", n::full({4}, .2, t));
    auto center    = n::tile(n::mean(mapped, {0}, true), {2, 1});
    auto different = m::StyleGanSynthesize(g, center, center, n4, n8, version2).image;
    auto shared =
        m::StyleGanSynthesize(
            g, center, center, n::take_rows(n4, {0, 0}), n::take_rows(n8, {0, 0}), version2)
            .image;
    EXPECT_GT(Difference(n::select(different, 0, 0), n::select(different, 0, 1)), 1e-8);
    EXPECT_LT(Difference(n::select(shared, 0, 0), n::select(shared, 0, 1)), 1e-13);
  }
}
TEST(NnStyleGan, FullGeneratorHalfCriticLossAndExplicitDThenFreshGWriteOwnership) {
  for (bool version2 : {false, true}) {
    auto              t = n::DType::Float64;
    n::ParameterStore g(6607, t), d(6619, t);
    auto              z = Gaussian({2, 8}, 6659), n4 = Gaussian({2, 4, 4, 1}, 6653),
         n8 = Gaussian({2, 8, 8, 1}, 6661), x = n::full({2, 8, 8, 1}, .3, t);
    auto           out = m::StyleGan(g, d, x, z, n4, n8, version2);
    n::GradOptions keep;
    keep.retain_graph        = true;
    auto               stale = n::grad(out.g_loss, g.parameters(), keep);
    auto               dg    = n::grad(out.d_loss, d.parameters());
    n::OptimizerConfig cfg;
    cfg.kind  = n::OptimizerKind::Adam;
    cfg.beta1 = .5;
    n::Optimizer go(g, .0002, cfg), do_(d, .0002, cfg);
    auto         gbefore = Snapshot(g);
    do_.step(dg);
    EXPECT_TRUE(Equal(gbefore, Snapshot(g)));
    auto   dafter = Snapshot(d);
    auto   fresh  = m::StyleGan(g, d, x, z, n4, n8, version2);
    auto   gg     = n::grad(fresh.g_loss, g.parameters());
    double change = 0;
    for (std::size_t i = 0; i < gg.size(); ++i)
      change = std::max(change, Difference(stale[i], gg[i]));
    EXPECT_GT(change, 1e-12);
    go.step(gg);
    EXPECT_TRUE(Equal(dafter, Snapshot(d)));
    EXPECT_EQ(go.iterations(), 1);
    EXPECT_EQ(do_.iterations(), 1);
    for (const auto& name : d.names())
      d.set(name, n::zeros(d.at(name).shape(), t));
    auto zero = m::StyleGan(g, d, x, z, n4, n8, version2);
    EXPECT_EQ(zero.d_loss.item(), .5);
    EXPECT_EQ(zero.g_loss.item(), 1);
  }
}
TEST(NnStyleGan, OwningFieldsKeepHigherPathsThroughBothVariantsAfterStoreDestruction) {
  for (bool version2 : {false, true}) {
    auto      z = Gaussian({2, 8}, 6659).requires_grad_();
    n::Tensor loss, style, image;
    {
      auto              t = z.dtype();
      n::ParameterStore g(6607, t), d(6619, t);
      auto              out = m::StyleGan(
          g,
          d,
          n::zeros({2, 8, 8, 1}, t),
          z,
          Gaussian({2, 4, 4, 1}, 6653),
          Gaussian({2, 8, 8, 1}, 6661),
          version2);
      loss  = out.g_loss;
      style = out.first_style;
      image = out.fake;
    }
    n::GradOptions graph;
    graph.create_graph = true;
    auto first         = n::grad(loss, {z}, graph)[0];
    auto second        = n::grad(n::sum(n::square(first)), {z})[0];
    EXPECT_GT(n::max(n::abs(first)).item(), 0);
    EXPECT_GT(n::max(n::abs(second)).item(), 0);
    EXPECT_EQ(image.shape(), (n::Shape{2, 8, 8, 1}));
    EXPECT_EQ(style.shape(), (n::Shape{2, version2 ? 4 : 8}));
  }
}
TEST(NnStyleGan, InvalidStoreNoiseStylesAndVariantOverrideFailBeforeLeaves) {
  n::ParameterStore g(1), d(2);
  auto              z = n::ones({2, 8}), n4 = n::zeros({2, 4, 4, 1}), n8 = n::zeros({2, 8, 8, 1}),
       x = n::ones({2, 8, 8, 1});
  EXPECT_THROW(m::StyleGan(g, g, x, z, n4, n8), std::invalid_argument);
  EXPECT_THROW(m::StyleGan(g, d, x, z, n4, n::zeros({1, 8, 8, 1})), std::invalid_argument);
  EXPECT_THROW(m::StyleGanMap(g, n::ones({2, 7})), std::invalid_argument);
  EXPECT_THROW(
      m::StyleGanSynthesize(g, z, z, n4, n8, false, n::ones({2, 4})), std::invalid_argument);
  EXPECT_THROW(
      m::StyleGanSynthesize(g, z, z, n4, n8, true, n::ones({1, 4})), std::invalid_argument);
  EXPECT_THROW(m::StyleGan(g, d, x, z, n4, n::full(n8.shape(), NAN)), std::invalid_argument);
  EXPECT_EQ(g.size() + d.size(), 0);
}
