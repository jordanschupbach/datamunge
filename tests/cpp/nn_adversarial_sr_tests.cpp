#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace n = dm::nn;
namespace m = n::models;
namespace {
using A = std::map<std::string, n::Tensor>;
A Snapshot(const n::ParameterStore& p) {
  A out;
  for (const auto& name : p.names())
    out.emplace(name, p.at(name).detach().clone());
  return out;
}
bool Equal(const A& a, const A& b) {
  if (a.size() != b.size())
    return false;
  for (const auto& [k, v] : a)
    if (!b.count(k) || !n::equal_bits(v, b.at(k)))
      return false;
  return true;
}
double Delta(const n::Tensor& a, const n::Tensor& b) {
  return n::max(n::abs(a - b)).item();
}
struct Fixture {
  n::DType          t = n::DType::Float64;
  n::ParameterStore g{7001, t}, d{7013, t}, s{0, t};
  n::Tensor         lr = n::full({2, 16, 16, 3}, .2, t), hr = n::full({2, 32, 32, 3}, .4, t),
            bic = n::full({2, 32, 32, 3}, .2, t), w = n::full({3, 3, 3, 64}, .001, t),
            b = n::zeros({64}, t);
  auto Forward(bool enhanced = false, bool phase = true) {
    return m::AdversarialSR(g, d, s, lr, hr, bic, w, b, enhanced, phase);
  }
};
const std::vector<std::string> Names{
    "residual0/bn1",
    "residual0/bn2",
    "residual1/bn1",
    "residual1/bn2",
    "body_bn"};
} // namespace
TEST(NnAdversarialSR, RelativisticUnequalPoolsKeepCrossMeanPartialsAndCommonShift) {
  auto t           = n::DType::Float64;
  auto r           = n::tensor({-2., 0., 3.}, {3, 1}, t).requires_grad_(),
       f           = n::tensor({-1., 2.}, {2, 1}, t).requires_grad_();
  auto           o = m::RelativisticAverageGan(r, f);
  n::GradOptions keep;
  keep.retain_graph = true;
  auto           dd = n::grad(o.discriminator, {r, f}, keep);
  n::GradOptions graph;
  graph.create_graph = true;
  auto gg            = n::grad(o.generator, {r, f}, graph);
  auto a             = n::sigmoid(o.real_relative).to_vector<double>(),
       b             = n::sigmoid(o.fake_relative).to_vector<double>();
  double ma          = std::accumulate(a.begin(), a.end(), 0.) / 3,
         mb          = std::accumulate(b.begin(), b.end(), 0.) / 2;
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(dd[0].at({i, 0}), .5 * ((a[i] - 1) - mb) / 3, 1e-14);
    EXPECT_NEAR(gg[0].at({i, 0}), .5 * (a[i] - (mb - 1)) / 3, 1e-14);
  }
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(dd[1].at({i, 0}), .5 * (b[i] - (ma - 1)) / 2, 1e-14);
    EXPECT_NEAR(gg[1].at({i, 0}), .5 * ((b[i] - 1) - ma) / 2, 1e-14);
  }
  auto shifted = m::RelativisticAverageGan(r.detach() + 100, f.detach() + 100);
  EXPECT_NEAR(o.generator.item(), shifted.generator.item(), 1e-14);
  EXPECT_NEAR(n::sum(gg[0]).item() + n::sum(gg[1]).item(), 0, 1e-14);
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(n::square(gg[0])), {f})[0])).item(), 0);
}
TEST(NnAdversarialSR, CRDShuffleRetainsSourceChannelOrderAndInverseGradient) {
  auto x = n::reshape(n::arange(0, 8, 1, n::DType::Float64), {1, 1, 1, 8}).requires_grad_();
  auto y = m::SRPixelShuffle(x, 2, 2);
  EXPECT_EQ(y.to_vector<double>(), (std::vector<double>{0, 4, 1, 5, 2, 6, 3, 7}));
  EXPECT_TRUE(n::equal_bits(n::grad(n::sum(y), {x})[0], n::ones(x.shape(), x.dtype())));
}
TEST(NnAdversarialSR, FrozenVGGUsesRoundedBGRMeansAndStopsAllWeightBiasPaths) {
  auto                t = n::DType::Float64;
  std::vector<double> weights(3 * 3 * 3 * 64);
  for (int c = 0; c < 3; ++c)
    weights[((1 * 3 + 1) * 3 + c) * 64 + c] = 1;
  auto w = n::from_vector(weights, {3, 3, 3, 64}).requires_grad_();
  auto b = n::zeros({64}, t).requires_grad_(),
       x = n::tensor({1., 0., .5}, {1, 1, 1, 3}, t).requires_grad_();
  auto f = m::AdversarialSRFeatures(x, w, b);
  EXPECT_NEAR(f.at({0, 0, 0, 0}), 127.5 - double(103.939f), 1e-13);
  EXPECT_NEAR(f.at({0, 0, 0, 1}), -double(116.779f), 1e-13);
  EXPECT_NEAR(f.at({0, 0, 0, 2}), 255 - double(123.68f), 1e-13);
  n::GradOptions unused;
  unused.allow_unused = true;
  auto derivative     = n::grad(n::sum(f), {x, w, b}, unused);
  EXPECT_TRUE(n::equal_bits(derivative[0], n::full(x.shape(), 255, t)));
  EXPECT_TRUE(!derivative[1].defined() || n::sum(n::abs(derivative[1])).item() == 0);
  EXPECT_TRUE(!derivative[2].defined() || n::sum(n::abs(derivative[2])).item() == 0);
}
TEST(NnAdversarialSR, VariantCensusExplicitBicubicResidualAndSourceBoundaryShapes) {
  for (bool enhanced : {false, true}) {
    Fixture q;
    auto    o = q.Forward(enhanced);
    EXPECT_EQ(q.g.count(), enhanced ? 3511 : 1591);
    EXPECT_EQ(q.g.size(), enhanced ? 40 : 25);
    EXPECT_EQ(q.d.count(), enhanced ? 1672 : 1673);
    EXPECT_EQ(q.d.size(), enhanced ? 7 : 8);
    EXPECT_EQ(q.s.count(), enhanced ? 0 : 40);
    EXPECT_EQ(q.s.size(), enhanced ? 0 : 10);
    auto p = m::AdversarialSRGenerate(q.g, q.s, q.lr, q.bic, enhanced);
    EXPECT_TRUE(n::equal_bits(p.image, q.bic + p.residual));
    EXPECT_EQ(p.packed.shape(), enhanced ? n::Shape({2, 32, 32, 4}) : n::Shape({2, 16, 16, 16}));
    EXPECT_EQ(p.shuffled.shape(), (n::Shape{2, 32, 32, 4}));
    EXPECT_EQ(o.prediction_feature.shape(), (n::Shape{2, 32, 32, 64}));
    EXPECT_FALSE(o.target_feature.requires_grad());
  }
}
TEST(NnAdversarialSR, BatchPathsDoNotWriteStatsAndIgnoreSavedValuesWhileInferenceReadsThem) {
  Fixture q;
  auto    o      = q.Forward();
  auto    before = Snapshot(q.s);
  auto    batch  = m::AdversarialSRGenerate(q.g, q.s, q.lr, q.bic, false, true);
  EXPECT_TRUE(Equal(before, Snapshot(q.s)));
  EXPECT_EQ(batch.moments_mean.shape(), (n::Shape{5, 4}));
  EXPECT_FALSE(batch.moments_mean.requires_grad());
  for (const auto& name : Names) {
    q.s.set(name + "/moving_mean", n::ones({4}, q.t));
    q.s.set(name + "/moving_var", n::full({4}, 2., q.t));
  }
  auto changed   = m::AdversarialSRGenerate(q.g, q.s, q.lr, q.bic, false, true);
  auto inference = m::AdversarialSRGenerate(q.g, q.s, q.lr, q.bic);
  EXPECT_TRUE(n::equal_bits(batch.image, changed.image));
  EXPECT_GT(Delta(o.prediction, inference.image), 1e-7);
  EXPECT_FALSE(inference.moments_mean.defined());
}
TEST(NnAdversarialSR, EnhancedHasNoBNStateOrBatchCouplingAndSavedBatchPathsAgree) {
  Fixture q;
  auto    a = m::AdversarialSRGenerate(q.g, q.s, q.lr, q.bic, true, false),
       b    = m::AdversarialSRGenerate(q.g, q.s, q.lr, q.bic, true, true);
  EXPECT_TRUE(n::equal_bits(a.image, b.image));
  EXPECT_FALSE(b.moments_mean.defined());
  EXPECT_EQ(q.s.size(), 0);
  EXPECT_TRUE(n::equal_bits(a.packed, a.shuffled));
  auto single =
      m::AdversarialSRGenerate(q.g, q.s, n::take_rows(q.lr, {0}), n::take_rows(q.bic, {0}), true);
  EXPECT_LT(Delta(single.image, n::take_rows(a.image, {0})), 3e-6);
}
TEST(NnAdversarialSR, TargetCanaryLeavesBothGPathsExactAndSourceRoundedPhaseLossesAgree) {
  for (bool enhanced : {false, true}) {
    Fixture q;
    q.hr.requires_grad_();
    auto o       = q.Forward(enhanced, false);
    auto changed = m::AdversarialSR(
        q.g, q.d, q.s, q.lr, n::zeros(q.hr.shape(), q.t), q.bic, q.w, q.b, enhanced, true);
    EXPECT_TRUE(n::equal_bits(o.prediction, changed.prediction));
    EXPECT_TRUE(n::equal_bits(o.prediction_training, changed.prediction_training));
    EXPECT_TRUE(n::equal_bits(o.g_loss, o.pixel_loss));
    EXPECT_FALSE(o.target_feature.requires_grad());
    auto   gan = q.Forward(enhanced, true);
    double expected =
        gan.content_loss.item() + double(enhanced ? .005f : .001f) * gan.adversarial_loss.item();
    if (enhanced)
      expected += double(.01f) * gan.pixel_loss.item();
    EXPECT_NEAR(gan.g_loss.item(), expected, 1e-13);
    EXPECT_GT(n::max(n::abs(n::grad(o.pixel_loss, {q.hr})[0])).item(), 0);
  }
}
TEST(NnAdversarialSR, CommitBNUsesSamePreWriteCaptureAndPostWriteRecomputationDiffers) {
  Fixture q;
  auto    o                  = q.Forward(false, false);
  auto    capture_mean       = o.moments_mean.detach().clone(),
       capture_var           = o.moments_unbiased_variance.detach().clone();
  auto               before  = Snapshot(q.s);
  auto               partial = n::grad(o.g_loss, q.g.parameters());
  n::OptimizerConfig c;
  c.kind  = n::OptimizerKind::Adam;
  c.beta1 = .9;
  n::Optimizer optimizer(q.g, .0001, c);
  optimizer.step(partial);
  EXPECT_TRUE(Equal(before, Snapshot(q.s)));
  auto wrong = m::AdversarialSRGenerate(q.g, q.s, q.lr, q.bic, false, true);
  EXPECT_GT(Delta(wrong.moments_mean, capture_mean), 1e-8);
  for (int i = 0; i < 5; ++i) {
    auto next = n::batch_norm_moving_values(
        q.s.at(Names[i] + "/moving_mean"),
        q.s.at(Names[i] + "/moving_var"),
        n::select(capture_mean, 0, i),
        n::select(capture_var, 0, i));
    q.s.set(Names[i] + "/moving_mean", next.mean);
    q.s.set(Names[i] + "/moving_var", next.variance);
    EXPECT_TRUE(n::equal_bits(next.mean, n::select(capture_mean, 0, i) * double(1.f - .9f)));
  }
  EXPECT_FALSE(Equal(before, Snapshot(q.s)));
  EXPECT_EQ(optimizer.iterations(), 1);
}
TEST(NnAdversarialSR, WarmupLeavesCriticFrozenAndColdOptimizersResetWithoutTouchingWeights) {
  Fixture q;
  q.Forward();
  auto               before = Snapshot(q.d);
  n::OptimizerConfig c;
  c.kind   = n::OptimizerKind::Adam;
  c.beta1  = .9;
  auto go  = std::make_unique<n::Optimizer>(q.g, .0001, c);
  auto do_ = std::make_unique<n::Optimizer>(q.d, .0001, c);
  for (int i = 0; i < 2; ++i)
    go->step(n::grad(q.Forward(false, false).g_loss, q.g.parameters()));
  EXPECT_TRUE(Equal(before, Snapshot(q.d)));
  EXPECT_EQ(go->iterations(), 2);
  EXPECT_EQ(do_->iterations(), 0);
  auto g = Snapshot(q.g), s = Snapshot(q.s);
  go  = std::make_unique<n::Optimizer>(q.g, .0001, c);
  do_ = std::make_unique<n::Optimizer>(q.d, .0001, c);
  EXPECT_EQ(go->iterations(), 0);
  EXPECT_EQ(do_->iterations(), 0);
  EXPECT_TRUE(Equal(g, Snapshot(q.g)));
  EXPECT_TRUE(Equal(s, Snapshot(q.s)));
}
TEST(NnAdversarialSR, DThenFreshGChangesPartialAndPreservesUnselectedStores) {
  for (bool enhanced : {false, true}) {
    Fixture        q;
    auto           out = q.Forward(enhanced);
    n::GradOptions keep;
    keep.retain_graph = true;
    auto stale        = n::grad(out.g_loss, q.g.parameters(), keep),
         dd           = n::grad(out.d_loss, q.d.parameters());
    n::OptimizerConfig c;
    c.kind  = n::OptimizerKind::Adam;
    c.beta1 = .9;
    n::Optimizer go(q.g, .0001, c), do_(q.d, .0001, c);
    auto         gb = Snapshot(q.g), sb = Snapshot(q.s);
    do_.step(dd);
    EXPECT_TRUE(Equal(gb, Snapshot(q.g)));
    EXPECT_TRUE(Equal(sb, Snapshot(q.s)));
    auto   db     = Snapshot(q.d);
    auto   gg     = n::grad(q.Forward(enhanced).g_loss, q.g.parameters());
    double change = 0;
    for (std::size_t i = 0; i < gg.size(); ++i)
      change = std::max(change, Delta(stale[i], gg[i]));
    EXPECT_GT(change, 1e-13);
    go.step(gg);
    EXPECT_TRUE(Equal(db, Snapshot(q.d)));
    EXPECT_TRUE(Equal(sb, Snapshot(q.s)));
  }
}
TEST(NnAdversarialSR, OwningPathsRetainHigherLRDerivativesAfterAllParentsAndStoresExpire) {
  for (bool enhanced : {false, true}) {
    n::Tensor lr, loss, image;
    {
      Fixture q;
      q.lr.requires_grad_();
      lr     = q.lr;
      auto o = q.Forward(enhanced, false);
      loss   = n::mean(n::square(o.prediction_training));
      image  = o.prediction;
    }
    n::GradOptions graph;
    graph.create_graph = true;
    auto first         = n::grad(loss, {lr}, graph)[0];
    auto second        = n::grad(n::sum(n::square(first)), {lr})[0];
    EXPECT_GT(n::max(n::abs(second)).item(), 0);
    EXPECT_EQ(image.shape(), (n::Shape{2, 32, 32, 3}));
  }
}
TEST(NnAdversarialSR, InvalidAliasesGeometryFeaturesAndRelativisticPoolsFailBeforeLeaves) {
  Fixture q;
  EXPECT_THROW(m::AdversarialSR(q.g, q.g, q.s, q.lr, q.hr, q.bic, q.w, q.b), std::invalid_argument);
  EXPECT_THROW(m::AdversarialSR(q.g, q.d, q.g, q.lr, q.hr, q.bic, q.w, q.b), std::invalid_argument);
  EXPECT_THROW(m::AdversarialSR(q.g, q.d, q.d, q.lr, q.hr, q.bic, q.w, q.b), std::invalid_argument);
  EXPECT_THROW(
      m::AdversarialSR(q.g, q.d, q.s, q.lr, n::zeros({1, 32, 32, 3}, q.t), q.bic, q.w, q.b),
      std::invalid_argument);
  EXPECT_THROW(
      m::AdversarialSR(q.g, q.d, q.s, q.lr, q.hr, q.bic, n::zeros({3, 3, 3, 63}, q.t), q.b),
      std::invalid_argument);
  EXPECT_THROW(
      m::AdversarialSRGenerate(q.g, q.s, n::full(q.lr.shape(), NAN, q.t), q.bic),
      std::invalid_argument);
  EXPECT_THROW(m::RelativisticAverageGan(n::ones({2, 2}), n::ones({2, 1})), std::invalid_argument);
  EXPECT_THROW(
      m::RelativisticAverageGan(n::ones({2, 1}), n::full({1, 1}, INFINITY)), std::invalid_argument);
  EXPECT_EQ(q.g.size(), 0);
  EXPECT_EQ(q.d.size(), 0);
  EXPECT_EQ(q.s.size(), 0);
}
