#include <gtest/gtest.h>
#include <dm/dm.hpp>

namespace n = dm::nn;
namespace m = n::models;
namespace {
using A = std::map<std::string, n::Tensor>;
A Snapshot(const n::ParameterStore& p) {
  A out;
  for (const auto& name : p.names()) out.emplace(name, p.at(name).detach().clone());
  return out;
}
bool Equal(const A& a, const A& b) {
  if (a.size() != b.size()) return false;
  for (const auto& [key, value] : a)
    if (!b.count(key) || !n::equal_bits(value, b.at(key))) return false;
  return true;
}
double Delta(const n::Tensor& a, const n::Tensor& b) {
  return n::max(n::abs(a - b)).item();
}
struct Fixture {
  n::DType t = n::DType::Float64;
  n::ParameterStore g{7307, t}, d{7309, t};
  n::Tensor real = n::full({2, 5, 7, 3}, .3, t), previous = n::full({3, 5, 7, 3}, -.2, t),
            noise = m::SingleImageNoise(3, 5, 7, false, 7517, t),
            rp = n::full({2, 5, 7, 3}, .1, t), rz = m::SingleImageNoise(2, 5, 7, true, 7411, t),
            amp = n::full({1}, .2, t);
  auto Forward() { return m::SingleImageGan(g, d, real, previous, noise, rp, rz, amp); }
};
} // namespace

TEST(NnSingleImageGan, NoiseUsesOneSourceStreamAndTilesOnlyCoarseRGB) {
  for (bool shared : {false, true}) {
    auto z = m::SingleImageNoise(2, 3, 5, shared, 7411);
    auto expected = n::cast(m::GaussianNoise({3, 5, shared ? 1 : 3}, 2, 7411), n::DType::Float32);
    if (shared) expected = n::tile(expected, {1, 1, 1, 3});
    EXPECT_TRUE(n::equal_bits(z, expected));
    EXPECT_TRUE(n::equal_bits(z, m::SingleImageNoise(2, 3, 5, shared, 7411)));
    EXPECT_FALSE(z.requires_grad());
    EXPECT_GT(Delta(z, m::SingleImageNoise(2, 3, 5, shared, 7412)), .1);
  }
  EXPECT_THROW(m::SingleImageNoise(0, 2, 3, true, 1), std::invalid_argument);
  EXPECT_THROW(m::SingleImageNoise(1, 2, 3, true, 1, n::DType::Int64), std::invalid_argument);
  EXPECT_THROW(m::SingleImageNoise(INT64_MAX, 2, 3, true, 1), std::invalid_argument);
}
TEST(NnSingleImageGan, AmplitudeIsDetachedDoubleRMSWithOneFinalNarrowing) {
  auto t = n::DType::Float64;
  auto x = n::full({1, 2, 3, 3}, -2, t).requires_grad_();
  auto p = n::ones(x.shape(), t).requires_grad_();
  auto a = m::SingleImageAmplitude(x, p);
  EXPECT_EQ(a.shape(), (n::Shape{1}));
  EXPECT_NEAR(a.item(), .3, 1e-15);
  EXPECT_FALSE(a.requires_grad());
  EXPECT_EQ(m::SingleImageAmplitude(x, p, 0).item(), 0);
  EXPECT_THROW(m::SingleImageAmplitude(x, p, -1), std::invalid_argument);
  EXPECT_THROW(m::SingleImageAmplitude(x, n::zeros({1, 2, 4, 3}, t)), std::invalid_argument);
  auto xf = n::cast(x.detach(), n::DType::Float32), pf = n::cast(p.detach(), n::DType::Float32);
  EXPECT_EQ(m::SingleImageAmplitude(xf, pf).item(), double(float(.3)));
  EXPECT_THROW(m::SingleImageAmplitude(n::full(x.shape(), 1e300, t), p), std::invalid_argument);
}
TEST(NnSingleImageGan, SourceCensusRectangularBoundariesAndScalarConventions) {
  Fixture q;
  auto o = q.Forward();
  EXPECT_EQ(q.g.count(), 371); EXPECT_EQ(q.g.size(), 6);
  EXPECT_EQ(q.d.count(), 297); EXPECT_EQ(q.d.size(), 6);
  EXPECT_EQ(o.fake.shape(), q.previous.shape());
  EXPECT_EQ(o.reconstruction.shape(), q.real.shape());
  EXPECT_EQ(o.fake_score.shape(), (n::Shape{3, 5, 7, 1}));
  auto g = m::SingleImageGenerate(q.g, q.previous, q.noise, q.amp);
  EXPECT_EQ(g.head.shape(), (n::Shape{3, 5, 7, 4}));
  EXPECT_EQ(g.body.shape(), g.head.shape());
  EXPECT_TRUE(n::equal_bits(g.image, q.previous + g.residual));
  EXPECT_TRUE(n::equal_bits(g.image, m::SingleImageGenerate(q.g, q.previous, q.noise, n::full({}, .2, q.t)).image));
  auto tiny = n::zeros({1, 1, 1, 3}, q.t);
  EXPECT_EQ(m::SingleImageGenerate(q.g, tiny, tiny, q.amp).image.shape(), tiny.shape());
  auto strided = n::transpose(q.previous, {0, 2, 1, 3});
  EXPECT_TRUE(n::equal_bits(m::SingleImageDiscriminate(q.d, strided),
                            m::SingleImageDiscriminate(q.d, strided.contiguous())));
}
TEST(NnSingleImageGan, ZeroWeightsExposeUnclippedResidualAndExactLossScales) {
  Fixture q;
  q.Forward();
  for (auto* p : {&q.g, &q.d})
    for (const auto& name : p->names()) p->set(name, n::zeros(p->at(name).shape(), q.t));
  q.previous = n::full(q.previous.shape(), 4, q.t);
  auto o = q.Forward();
  EXPECT_TRUE(n::equal_bits(o.fake, q.previous));
  EXPECT_TRUE(n::equal_bits(o.reconstruction, q.rp));
  EXPECT_EQ(o.adversarial_loss.item(), 1);
  EXPECT_EQ(o.d_loss.item(), .5);
  EXPECT_NEAR(o.reconstruction_loss.item(), .04, 1e-15);
  EXPECT_NEAR(o.g_loss.item(), 1.4, 1e-14);
  q.g.set("output/b", n::full({3}, 1., q.t));
  o = q.Forward();
  EXPECT_GT(n::min(o.fake).item(), 4.7);
}
TEST(NnSingleImageGan, ResidualSkipAndNoiseAmplitudePathsMatchFiniteDifferences) {
  Fixture q;
  auto p = n::full({1, 3, 5, 3}, .2, q.t).requires_grad_();
  auto z = m::SingleImageNoise(1, 3, 5, false, 7571, q.t).requires_grad_();
  auto a = q.amp.clone().requires_grad_();
  auto loss = n::mean(n::square(m::SingleImageGenerate(q.g, p, z, a).image));
  auto gradient = n::grad(loss, {p, z, a});
  std::vector<n::Tensor> inputs{p, z, a};
  for (int slot = 0; slot < 3; ++slot) {
    auto v = inputs[slot].to_vector<double>();
    for (auto index : {std::size_t(0), v.size() / 2, v.size() - 1}) {
      double old = v[index], h = 1e-6;
      v[index] = old + h; auto plus = inputs;
      plus[slot] = n::from_vector(v, inputs[slot].shape());
      v[index] = old - h; auto minus = inputs;
      minus[slot] = n::from_vector(v, inputs[slot].shape());
      v[index] = old;
      double lp = n::mean(n::square(m::SingleImageGenerate(q.g, plus[0], plus[1], plus[2]).image)).item();
      double lm = n::mean(n::square(m::SingleImageGenerate(q.g, minus[0], minus[1], minus[2]).image)).item();
      EXPECT_NEAR((lp - lm) / (2 * h), gradient[slot].to_vector<double>()[index], 2e-8);
    }
  }
  for (const auto& name : q.g.names()) q.g.set(name, n::zeros(q.g.at(name).shape(), q.t));
  auto skip = n::grad(n::sum(m::SingleImageGenerate(q.g, p, z, a).image), {p, z, a});
  EXPECT_TRUE(n::equal_bits(skip[0], n::ones(p.shape(), q.t)));
  EXPECT_EQ(n::sum(n::abs(skip[1])).item(), 0);
  EXPECT_EQ(skip[2].item(), 0);
}
TEST(NnSingleImageGan, SharedGeneratorReceivesBothObjectiveBranchesWithUnequalPools) {
  Fixture q;
  auto o = q.Forward();
  n::GradOptions keep; keep.retain_graph = true;
  auto total = n::grad(o.g_loss, q.g.parameters(), keep);
  auto adv = n::grad(o.adversarial_loss, q.g.parameters(), keep);
  auto rec = n::grad(o.reconstruction_loss, q.g.parameters());
  double adv_norm = 0, rec_norm = 0;
  for (std::size_t i = 0; i < total.size(); ++i) {
    EXPECT_LT(Delta(total[i], adv[i] + 10 * rec[i]), 1e-12);
    adv_norm += n::sum(n::abs(adv[i])).item();
    rec_norm += n::sum(n::abs(rec[i])).item();
  }
  EXPECT_GT(adv_norm, 1e-6); EXPECT_GT(rec_norm, 1e-6);
  EXPECT_NEAR(o.adversarial_loss.item(), n::mean(n::square(o.fake_score - 1)).item(), 1e-15);
  EXPECT_NEAR(o.d_loss.item(), .5 * (n::mean(n::square(o.real_score - 1)) + n::mean(n::square(o.fake_score))).item(), 1e-15);
}
TEST(NnSingleImageGan, GenerationAndPatchScoresHaveNoBatchCouplingOrHiddenWrites) {
  Fixture q;
  auto o = q.Forward();
  auto gb = Snapshot(q.g), db = Snapshot(q.d);
  auto one = m::SingleImageGenerate(q.g, n::take_rows(q.previous, {1}), n::take_rows(q.noise, {1}), q.amp);
  EXPECT_TRUE(n::equal_bits(one.image, n::take_rows(o.fake, {1})));
  EXPECT_TRUE(n::equal_bits(m::SingleImageDiscriminate(q.d, one.image), n::take_rows(o.fake_score, {1})));
  { n::NoGradGuard guard;
    EXPECT_FALSE(m::SingleImageGenerate(q.g, q.previous, q.noise, q.amp).image.requires_grad()); }
  EXPECT_TRUE(Equal(gb, Snapshot(q.g))); EXPECT_TRUE(Equal(db, Snapshot(q.d)));
}
TEST(NnSingleImageGan, ExplicitDThenFreshGStepHasIndependentOwnershipAndClocks) {
  Fixture q;
  auto o = q.Forward();
  n::GradOptions keep; keep.retain_graph = true;
  auto stale = n::grad(o.g_loss, q.g.parameters(), keep);
  auto dd = n::grad(o.d_loss, q.d.parameters());
  n::OptimizerConfig cfg; cfg.kind = n::OptimizerKind::Adam; cfg.beta1 = .5;
  n::Optimizer go(q.g, .0005, cfg), do_(q.d, .0005, cfg);
  auto gb = Snapshot(q.g); do_.step(dd);
  EXPECT_TRUE(Equal(gb, Snapshot(q.g)));
  auto db = Snapshot(q.d);
  auto fresh = n::grad(q.Forward().g_loss, q.g.parameters());
  double change = 0;
  for (std::size_t i = 0; i < fresh.size(); ++i) change = std::max(change, Delta(stale[i], fresh[i]));
  EXPECT_GT(change, 1e-9);
  go.step(fresh);
  EXPECT_TRUE(Equal(db, Snapshot(q.d)));
  EXPECT_EQ(go.iterations(), 1); EXPECT_EQ(do_.iterations(), 1);
}
TEST(NnSingleImageGan, WarmStartCopiesBothGroupsButFreshOptimizersRemainCold) {
  Fixture q; q.Forward();
  n::OptimizerConfig cfg; cfg.kind = n::OptimizerKind::Adam; cfg.beta1 = .5;
  n::Optimizer opt(q.g, .0005, cfg);
  opt.step(n::grad(q.Forward().g_loss, q.g.parameters()));
  n::ParameterStore nextg(7307, q.t), nextd(7309, q.t);
  m::SingleImageGenerate(nextg, q.previous, q.noise, q.amp);
  m::SingleImageDiscriminate(nextd, q.real);
  for (const auto& [name, value] : Snapshot(q.g)) nextg.set(name, value);
  for (const auto& [name, value] : Snapshot(q.d)) nextd.set(name, value);
  n::Optimizer nextgo(nextg, .0005, cfg), nextdo(nextd, .0005, cfg);
  EXPECT_TRUE(Equal(Snapshot(q.g), Snapshot(nextg)));
  EXPECT_TRUE(Equal(Snapshot(q.d), Snapshot(nextd)));
  EXPECT_EQ(opt.iterations(), 1); EXPECT_EQ(nextgo.iterations(), 0); EXPECT_EQ(nextdo.iterations(), 0);
}
TEST(NnSingleImageGan, RawPatchHasNominalSevenBySevenReceptiveField) {
  n::ParameterStore d(7309, n::DType::Float64);
  auto x = n::full({1, 11, 13, 3}, .5, n::DType::Float64).requires_grad_();
  auto score = m::SingleImageDiscriminate(d, x);
  auto center = n::select(n::select(n::select(n::select(score, 0, 0), 0, 5), 0, 6), 0, 0);
  auto dx = n::grad(center, {x})[0].to_vector<double>();
  double inside = 0;
  for (int y = 0; y < 11; ++y) for (int col = 0; col < 13; ++col) for (int c = 0; c < 3; ++c) {
    double v = dx[(y * 13 + col) * 3 + c];
    if (std::abs(y - 5) > 3 || std::abs(col - 6) > 3) EXPECT_EQ(v, 0);
    else inside += std::abs(v);
  }
  EXPECT_GT(inside, 1e-4);
}
TEST(NnSingleImageGan, OwningFieldsKeepHigherPathsAfterParentAndStoresExpire) {
  auto x = n::full({1, 3, 5, 3}, .2, n::DType::Float64).requires_grad_();
  auto z = m::SingleImageNoise(1, 3, 5, false, 7411, x.dtype()).requires_grad_();
  auto a = n::full({}, .3, x.dtype()).requires_grad_();
  n::Tensor image, loss;
  { n::ParameterStore g(7307, x.dtype()), d(7309, x.dtype());
    auto out = m::SingleImageGan(g, d, x, x, z, x, z, a);
    image = out.fake; loss = out.g_loss; }
  n::GradOptions graph; graph.create_graph = true;
  auto first = n::grad(loss + n::mean(n::square(image)), {x, z, a}, graph);
  for (auto value : first) EXPECT_GT(n::sum(n::abs(value)).item(), 0);
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(n::square(first[0])), {z})[0])).item(), 1e-10);
}
TEST(NnSingleImageGan, AllInvalidInputsFailBeforeAnyParameterLeaves) {
  n::ParameterStore g(7307), d(7309), wrong(1, n::DType::Float64);
  auto x = n::zeros({1, 3, 5, 3}), a = n::ones({1});
  EXPECT_THROW(m::SingleImageGan(g, g, x, x, x, x, x, a), std::invalid_argument);
  EXPECT_THROW(m::SingleImageGan(g, wrong, x, x, x, x, x, a), std::invalid_argument);
  EXPECT_THROW(m::SingleImageGan(g, d, x, x, x, x, n::zeros({1, 3, 4, 3}), a), std::invalid_argument);
  EXPECT_THROW(m::SingleImageGan(g, d, x, x, x, x, x, n::ones({1, 1})), std::invalid_argument);
  EXPECT_THROW(m::SingleImageGenerate(g, x, n::zeros({1, 3, 5, 1}), a), std::invalid_argument);
  EXPECT_THROW(m::SingleImageGenerate(g, x, x, n::full({1}, NAN)), std::invalid_argument);
  EXPECT_THROW(m::SingleImageDiscriminate(d, n::full(x.shape(), INFINITY)), std::invalid_argument);
  EXPECT_THROW(m::SingleImageDiscriminate(d, n::zeros({1, 3, 5, 3}, n::DType::Int64)), std::invalid_argument);
  EXPECT_EQ(g.size(), 0); EXPECT_EQ(d.size(), 0); EXPECT_EQ(wrong.size(), 0);
}
