#include <gtest/gtest.h>

#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/self_attention_gan.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Snapshot(const n::ParameterStore& p) {
  std::vector<n::Tensor> a;
  for (const auto& v : p.parameters())
    a.push_back(v.detach().clone());
  return a;
}
void Same(const n::ParameterStore& p, const std::vector<n::Tensor>& a) {
  ASSERT_EQ(p.size(), a.size());
  auto values = p.parameters();
  for (std::size_t i = 0; i < a.size(); ++i)
    EXPECT_TRUE(n::equal_bits(values[i], a[i]));
}
void Init(
    n::ParameterStore& g,
    n::ParameterStore& d,
    n::ParameterStore& gs,
    n::ParameterStore& ds) {
  m::SelfAttentionGan(g, d, gs, ds, n::zeros({2, 784}, g.dtype()), n::zeros({2, 16}, g.dtype()));
  n::advance_cached_spectral_state(g, gs, 16);
  n::advance_cached_spectral_state(d, ds, 16);
}
} // namespace
TEST(NnSelfAttentionGan, CachedDenseInitializesSourceConstantsAndNeverAdvancesExistingState) {
  n::ParameterStore p(3), s(0);
  auto              x = n::ones({2, 3});
  auto              y = n::cached_spectral_dense(p, s, "dense", x, 4, n::Activation::Tanh);
  EXPECT_EQ(y.shape(), n::Shape({2, 4}));
  EXPECT_EQ(p.count(), 16);
  EXPECT_EQ(s.count(), 7);
  EXPECT_EQ(s.at("dense/u").at({0, 0}), double(1.f / std::sqrt(4.f)));
  EXPECT_EQ(s.at("dense/v").at({0, 0}), double(1.f / std::sqrt(3.f)));
  auto state = Snapshot(s);
  n::cached_spectral_dense(p, s, "dense", x, 4);
  Same(s, state);
  auto weight = Snapshot(p);
  n::advance_cached_spectral_state(p, s, 16);
  Same(p, weight);
  s.set("dense/u", n::ones({4, 1}));
  p.set("dense/b", n::full({4}, NAN));
  EXPECT_THROW(
      n::cached_spectral_dense(p, s, "dense", x, 4, n::Activation::Tanh), std::invalid_argument);
}
TEST(NnSelfAttentionGan, CachedDenseRejectsInputsStoresNamesAndProductsBeforeLeaves) {
  n::ParameterStore p(3), s(0), wide(0, n::DType::Float64);
  auto              x = n::ones({2, 3});
  EXPECT_THROW(n::cached_spectral_dense(p, p, "a", x, 2), std::invalid_argument);
  EXPECT_THROW(n::cached_spectral_dense(p, wide, "a", x, 2), std::invalid_argument);
  EXPECT_THROW(n::cached_spectral_dense(p, s, "", x, 2), std::invalid_argument);
  EXPECT_THROW(n::cached_spectral_dense(p, s, "a", x, 0), std::invalid_argument);
  EXPECT_THROW(
      n::cached_spectral_dense(p, s, "a", x, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::ones({3}),
        n::zeros({0, 3}),
        n::full({2, 3}, INFINITY),
        n::ones({2, 3}, n::DType::Float64)})
    EXPECT_THROW(n::cached_spectral_dense(p, s, "a", bad, 2), std::invalid_argument);
  EXPECT_EQ(p.size() + s.size(), 0);
}
TEST(NnSelfAttentionGan, AttentionUsesUnscaledQueryKeyAndInitialZeroGateIsExactIdentity) {
  auto              t = n::DType::Float64;
  n::ParameterStore p(4, t), s(0, t);
  auto x = n::tensor({.1, .4, .8, .2, -.2, .9, .6, .3, .5}, {1, 3, 3}, t).requires_grad_();
  auto a = m::SpectralSpatialAttention(p, s, x, 2, 2);
  EXPECT_TRUE(n::equal_bits(a.input, a.output));
  EXPECT_EQ(a.probability.shape(), n::Shape({1, 3, 3}));
  n::advance_cached_spectral_state(p, s, 16);
  a          = m::SpectralSpatialAttention(p, s, x, 2, 2);
  auto flat  = x.reshape({3, 3});
  auto dense = [&](const std::string& name) {
    return n::cached_spectral_dense(p, s, "attention/" + name, flat, 2).reshape({1, 3, 2});
  };
  auto q = dense("query"), k = dense("key");
  auto score = n::matmul(q, k, false, true);
  auto exact = n::softmax(score, -1);
  EXPECT_TRUE(n::equal_bits(a.probability, exact));
  EXPECT_GT(n::max(n::abs(a.probability - n::softmax(score / std::sqrt(2.), -1))).item(), 1e-4);
  auto grad  = n::grad(n::sum(a.output), p.parameters());
  auto names = p.names();
  bool gate  = false;
  for (std::size_t i = 0; i < grad.size(); ++i) {
    if (names[i] == "attention/gamma") {
      gate = n::max(n::abs(grad[i])).item() > 1e-10;
    } else
      EXPECT_EQ(n::max(n::abs(grad[i])).item(), 0);
  }
  EXPECT_TRUE(gate);
}
TEST(NnSelfAttentionGan, LearnedAttentionMixesOtherTokensAndPreservesFirstHigherPaths) {
  auto              t = n::DType::Float64;
  n::ParameterStore p(4, t), s(0, t);
  auto x = n::tensor({.1, .4, .8, .2, -.2, .9, .6, .3, .5}, {1, 3, 3}, t).requires_grad_();
  m::SpectralSpatialAttention(p, s, x, 2, 2);
  n::advance_cached_spectral_state(p, s, 16);
  p.set("attention/gamma", n::scalar(.3, t));
  auto a    = m::SpectralSpatialAttention(p, s, x, 2, 2);
  auto data = x.to_vector<double>();
  data[0] += 1.;
  auto changed = m::SpectralSpatialAttention(p, s, n::from_vector(data, x.shape()), 2, 2);
  EXPECT_GT(
      n::max(n::abs(n::slice_axis(a.output, 1, 1, 3) - n::slice_axis(changed.output, 1, 1, 3)))
          .item(),
      1e-6);
  EXPECT_LT(n::max(n::abs(n::sum(a.probability, {-1}) - 1.)).item(), 1e-14);
  n::TensorFunction fn = [&](const auto& values) {
    auto o = m::SpectralSpatialAttention(p, s, values[0], 2, 2);
    return n::sum(n::square(o.output)) + n::sum(n::square(o.probability));
  };
  EXPECT_TRUE(n::gradcheck(fn, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  auto base      = p.flatten();
  auto objective = [&]() {
    auto o = m::SpectralSpatialAttention(p, s, x, 2, 2);
    return n::sum(n::square(o.output)) + n::sum(n::square(o.probability));
  };
  auto                values = n::grad(objective(), p.parameters());
  std::vector<double> analytic;
  for (const auto& v : values) {
    auto a = v.to_vector<double>();
    analytic.insert(analytic.end(), a.begin(), a.end());
  }
  for (std::size_t i = 0; i < base.size(); ++i) {
    auto perturbed = base;
    perturbed[i] += 1e-6;
    p.unflatten(perturbed);
    auto plus = objective().item();
    perturbed[i] -= 2e-6;
    p.unflatten(perturbed);
    auto minus = objective().item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 2e-8);
  }
  p.unflatten(base);
  auto o       = m::SpectralSpatialAttention(p, s, x, 2, 2);
  auto stopped = n::grad(n::sum(o.output), s.parameters(), {.allow_unused = true});
  for (const auto& g : stopped)
    EXPECT_FALSE(g.defined());
}
TEST(NnSelfAttentionGan, AttentionStridesCustomPrefixAndBadGeometry) {
  n::ParameterStore p(1), s(0);
  auto              x = n::transpose(n::ones({2, 3, 4}), {0, 2, 1});
  auto              a = m::SpectralSpatialAttention(p, s, x, 2, 2, "custom");
  auto              b = m::SpectralSpatialAttention(p, s, x.contiguous(), 2, 2, "custom");
  EXPECT_TRUE(n::equal_bits(a.output, b.output));
  EXPECT_TRUE(p.contains("custom/gamma"));
  n::ParameterStore fresh(1), fs(0);
  for (auto bad :
       {n::Tensor{},
        n::ones({2, 3}),
        n::zeros({2, 0, 3}),
        n::full({2, 3, 3}, NAN),
        n::ones({2, 3, 3}, n::DType::Float64)})
    EXPECT_THROW(m::SpectralSpatialAttention(fresh, fs, bad), std::invalid_argument);
  EXPECT_THROW(
      m::SpectralSpatialAttention(fresh, fs, n::ones({2, 3, 3}), 0), std::invalid_argument);
  EXPECT_THROW(
      m::SpectralSpatialAttention(
          fresh, fs, n::ones({2, 3, 3}), std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_THROW(
      m::SpectralSpatialAttention(fresh, fs, n::ones({2, 3, 3}), 2, 4, ""), std::invalid_argument);
  EXPECT_EQ(fresh.size() + fs.size(), 0);
}
TEST(NnSelfAttentionGan, DefaultCensusPatchOrderingSeedAndUnequalPools) {
  n::ParameterStore g(5903), d(5923), gs(0), ds(0);
  auto              o = m::SelfAttentionGan(g, d, gs, ds, n::zeros({2, 784}), n::zeros({3, 16}));
  EXPECT_EQ(g.count(), 2730);
  EXPECT_EQ(d.count(), 650);
  EXPECT_EQ(gs.count(), 245);
  EXPECT_EQ(ds.count(), 110);
  EXPECT_EQ(g.size(), 13);
  EXPECT_EQ(d.size(), 14);
  EXPECT_EQ(gs.size() + ds.size(), 24);
  EXPECT_EQ(o.fake.shape(), n::Shape({3, 784}));
  EXPECT_EQ(o.real_score.shape(), n::Shape({2, 1}));
  EXPECT_TRUE(n::equal_bits(o.g_attention_input, o.g_attention_output));
  EXPECT_TRUE(n::equal_bits(o.real_attention_input, o.real_attention_output));
  auto patch = n::arange(0, 784, 1).reshape({1, 16, 49});
  auto image = m::MaeImage(patch);
  EXPECT_TRUE(n::equal_bits(patch, m::MaePatches(image.reshape({1, 28, 28, 1}))));
  EXPECT_EQ(image.at({0, 7}), 49);
  EXPECT_EQ(image.at({0, 28}), 7);
  EXPECT_EQ(image.at({0, 196}), 196);
  n::ParameterStore g2(5903), d2(5923), gs2(0), ds2(0);
  auto again = m::SelfAttentionGan(g2, d2, gs2, ds2, n::zeros({2, 784}), n::zeros({3, 16}));
  Same(g2, Snapshot(g));
  Same(d2, Snapshot(d));
  EXPECT_TRUE(n::equal_bits(o.fake, again.fake));
}
TEST(NnSelfAttentionGan, ImmutableEvaluationAndFourExplicitStateDStateFreshGWrites) {
  n::ParameterStore g(5903), d(5923), gs(0), ds(0);
  Init(g, d, gs, ds);
  auto x = n::full({2, 784}, .3), z = n::full({3, 16}, -.2);
  auto gb = Snapshot(g), db = Snapshot(d), gsb = Snapshot(gs), dsb = Snapshot(ds);
  m::SelfAttentionGan(g, d, gs, ds, x, z);
  Same(g, gb);
  Same(d, db);
  Same(gs, gsb);
  Same(ds, dsb);
  n::OptimizerConfig config;
  config.kind  = n::OptimizerKind::Adam;
  config.beta1 = .5;
  n::Optimizer go(g, .0002, config), disc(d, .0002, config);
  n::advance_cached_spectral_state(d, ds);
  Same(g, gb);
  Same(d, db);
  Same(gs, gsb);
  dsb         = Snapshot(ds);
  auto before = m::SelfAttentionGan(g, d, gs, ds, x, z);
  disc.step(n::grad(before.discriminator_loss, d.parameters()));
  Same(g, gb);
  Same(gs, gsb);
  Same(ds, dsb);
  db = Snapshot(d);
  n::advance_cached_spectral_state(g, gs);
  Same(g, gb);
  Same(d, db);
  Same(ds, dsb);
  gsb        = Snapshot(gs);
  auto fresh = m::SelfAttentionGan(g, d, gs, ds, x, z);
  EXPECT_FALSE(n::equal_bits(before.fake_score, fresh.fake_score));
  go.step(n::grad(fresh.generator_loss, g.parameters()));
  Same(d, db);
  Same(gs, gsb);
  Same(ds, dsb);
  EXPECT_EQ(go.iterations(), 1);
  EXPECT_EQ(disc.iterations(), 1);
}
TEST(NnSelfAttentionGan, GeneratorAndDiscriminatorRetainLiveContinuousAndRecordPaths) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(5903, t), d(5923, t), gs(0, t), ds(0, t);
  Init(g, d, gs, ds);
  auto z      = n::full({1, 16}, .2, t).requires_grad_();
  auto a      = m::SelfAttentionGanGenerate(g, gs, z);
  auto pixels = a.pixels, probability = a.attention_probability;
  auto derivative = n::grad(n::sum(pixels), {z});
  EXPECT_GT(n::max(n::abs(derivative[0])).item(), 1e-6);
  n::TensorFunction fn = [&](const auto& values) {
    auto out = m::SelfAttentionGanGenerate(g, gs, values[0]);
    return n::sum(n::square(out.pixels));
  };
  EXPECT_TRUE(n::gradcheck(fn, {z}, {.atol = 1e-7, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {z}, {.atol = 1e-7, .rtol = 1e-5}).ok);
  auto image = n::full({1, 784}, .3, t).requires_grad_();
  auto score = m::SelfAttentionGanDiscriminate(d, ds, image).score;
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(score), {image})[0])).item(), 1e-6);
}
TEST(NnSelfAttentionGan, InvalidModelInputsStoresAndNoiseWidthsPrecedeLeaves) {
  n::ParameterStore g(1), d(2), gs(0), ds(0), wide(0, n::DType::Float64);
  auto              x = n::zeros({2, 784}), z = n::zeros({3, 16});
  EXPECT_THROW(m::SelfAttentionGan(g, g, gs, ds, x, z), std::invalid_argument);
  EXPECT_THROW(m::SelfAttentionGan(g, d, gs, gs, x, z), std::invalid_argument);
  EXPECT_THROW(m::SelfAttentionGan(g, d, wide, ds, x, z), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 784}),
        n::zeros({2, 783}),
        n::full({2, 784}, NAN),
        n::zeros({2, 784}, n::DType::Float64)})
    EXPECT_THROW(m::SelfAttentionGan(g, d, gs, ds, bad, z), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 16}),
        n::zeros({2, 15}),
        n::full({2, 16}, INFINITY),
        n::zeros({2, 16}, n::DType::Float64)})
    EXPECT_THROW(m::SelfAttentionGan(g, d, gs, ds, x, bad), std::invalid_argument);
  EXPECT_EQ(g.size() + d.size() + gs.size() + ds.size(), 0);
}
