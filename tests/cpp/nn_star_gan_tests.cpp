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
} // namespace
TEST(NnStarGan, ThreeTransformsPreserveOrientationSignedValuesAndSelectedInputPartials) {
  auto t      = n::DType::Float64;
  auto image  = n::tensor({0., .25, .5, .75}, {1, 2, 2, 1}, t);
  auto x      = n::tile(image, {3, 1, 1, 1}).requires_grad_();
  auto domain = n::tensor({0, 1, 2}, {3}, n::DType::Int64);
  auto y      = m::StarDomainImages(x, domain);
  EXPECT_TRUE(n::equal_bits(n::take_rows(y, {0}), image));
  EXPECT_TRUE(n::equal_bits(n::take_rows(y, {1}), 1. - image));
  EXPECT_TRUE(n::equal_bits(n::take_rows(y, {2}), n::tensor({.5, 0., .75, .25}, {1, 2, 2, 1}, t)));
  auto dx = n::grad(n::sum(y), {x})[0];
  EXPECT_TRUE(n::equal_bits(n::take_rows(dx, {0, 2}), n::ones({2, 2, 2, 1}, t)));
  EXPECT_TRUE(n::equal_bits(n::take_rows(dx, {1}), n::full({1, 2, 2, 1}, -1, t)));
  EXPECT_EQ(
      n::min(m::StarDomainImages(n::full({1, 2, 2, 1}, 2, t), n::ones({1}, domain.dtype()))).item(),
      -1);
}
TEST(NnStarGan, SplitRecordsDisjointObservedOccurrencesUnusedIdentitiesAndLiveGathers) {
  auto t      = n::DType::Float64;
  auto x      = n::full({40, 28, 28, 1}, .25, t).requires_grad_();
  auto labels = n::cast(n::arange(0, 40, 1, t) / 4., n::DType::Int64);
  auto ids    = n::arange(100, 140, 1, n::DType::Int64);
  auto p      = m::SplitStarDomains(x, labels, ids, 1);
  EXPECT_EQ(p.image.shape(), (n::Shape{30, 28, 28, 1}));
  EXPECT_EQ(p.unused.dtype(), n::DType::Int64);
  EXPECT_EQ(p.source.dtype(), n::DType::Int64);
  EXPECT_EQ(p.domain.dtype(), n::DType::Int32);
  EXPECT_EQ(
      p.domain.to_vector<int>(), (std::vector<int>{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                                                   0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}));
  EXPECT_EQ(
      p.unused.to_vector<std::int64_t>(),
      (std::vector<std::int64_t>{103, 107, 111, 115, 119, 123, 127, 131, 135, 139}));
  EXPECT_TRUE(n::equal_bits(p.image, m::StarDomainImages(p.original, p.domain)));
  auto dx = n::grad(n::sum(p.original), {x})[0];
  EXPECT_EQ(n::sum(dx).item(), 30 * 784);
  EXPECT_EQ(n::sum(n::take_rows(dx, {3, 7, 11})).item(), 0);
  auto selected = m::SplitStarDomains(
      n::zeros({30, 28, 28, 1}),
      n::cast(n::arange(0, 30, 1, t) / 3., n::DType::Int32),
      n::arange(0, 30, 1, n::DType::Int32),
      1);
  EXPECT_EQ(selected.unused.numel(), 0);
}
TEST(NnStarGan, CategoricalDrawUsesRawModuloAndLocalRepeatableStream) {
  auto a = m::UniformCategorical(900, 6338), b = m::UniformCategorical(900, 6338);
  EXPECT_TRUE(n::equal_bits(a.raw, b.raw));
  EXPECT_TRUE(n::equal_bits(a.labels, b.labels));
  auto raw    = a.raw.to_vector<double>();
  auto labels = a.labels.to_vector<std::int32_t>();
  for (int i = 0; i < 900; ++i) {
    EXPECT_LT(raw[i], 4294967295.);
    EXPECT_EQ(labels[i], std::uint64_t(raw[i]) % 3);
  }
  auto rejected = m::UniformCategorical(5, 0, 1073741825);
  EXPECT_EQ(
      rejected.raw.to_vector<double>(),
      (std::vector<double>{2357136044., 2546248239., 3071714933., 2588848963., 2340255427.}));
  EXPECT_FALSE(a.raw.requires_grad());
  EXPECT_EQ(n::max(m::UniformCategorical(8, 6338, 1).labels).item(), 0);
  EXPECT_THROW(m::UniformCategorical(0, 1), std::invalid_argument);
  EXPECT_THROW(m::UniformCategorical(1, 1, 0), std::invalid_argument);
  EXPECT_THROW(m::UniformCategorical(1, 1, std::int64_t{1} << 32), std::invalid_argument);
}
TEST(NnStarGan, NativeCensusBothActualSkipShapesTargetConditionAndImageOnlyCritic) {
  n::ParameterStore g(6301), d(6311);
  auto              x      = n::full({2, 28, 28, 1}, .3);
  auto              target = n::tensor({0, 2}, {2}, n::DType::Int64);
  auto              o      = m::StarGanGenerate(g, x, target);
  auto              c      = m::StarGanDiscriminate(d, x);
  EXPECT_EQ(g.count(), 1609);
  EXPECT_EQ(g.size(), 10);
  EXPECT_EQ(d.count(), 688);
  EXPECT_EQ(d.size(), 8);
  EXPECT_EQ(o.code.shape(), (n::Shape{2, 28, 28, 3}));
  EXPECT_EQ(o.input.shape(), (n::Shape{2, 28, 28, 4}));
  EXPECT_EQ(o.joined14.shape(), (n::Shape{2, 14, 14, 8}));
  EXPECT_EQ(o.joined28.shape(), (n::Shape{2, 28, 28, 6}));
  EXPECT_EQ(c.domain_logits.shape(), (n::Shape{2, 3}));
  EXPECT_EQ(c.source.shape(), (n::Shape{2, 7, 7, 1}));
  EXPECT_GT(Delta(n::select(o.image, 0, 0), n::select(o.image, 0, 1)), 1e-5);
  EXPECT_LT(
      Delta(
          m::StarGanGenerate(g, n::take_rows(x, {0}), n::take_rows(target, {0})).image,
          n::take_rows(o.image, {0})),
      2e-6);
}
TEST(NnStarGan, ThreeWayCrossEntropyIsStableAndHasLiveHessian) {
  auto           t      = n::DType::Float64;
  auto           logits = n::zeros({1, 3}, t).requires_grad_();
  auto           target = n::zeros({1}, n::DType::Int64);
  n::GradOptions graph;
  graph.create_graph = true;
  auto loss          = m::StarDomainLoss(logits, target);
  auto first         = n::grad(loss, {logits}, graph)[0];
  auto second        = n::grad(n::sum(n::square(first)), {logits})[0];
  EXPECT_NEAR(loss.item(), std::log(3.), 1e-14);
  EXPECT_NEAR(first.at({0, 0}), -2. / 3, 1e-14);
  EXPECT_GT(n::max(n::abs(second)).item(), 0);
  auto huge   = n::tensor({1000., -1000., 0.}, {1, 3}, t).requires_grad_();
  auto stable = m::StarDomainLoss(huge, n::ones({1}, target.dtype()));
  EXPECT_EQ(stable.item(), 2000);
  auto derivative = n::grad(stable, {huge})[0];
  EXPECT_EQ(derivative.at({0, 0}), 1);
  EXPECT_EQ(derivative.at({0, 1}), -1);
}
TEST(NnStarGan, FullSourceGHalfSourceDRealCEOnlyAndSharedRoundtripGrouping) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(6301, t), d(6311, t);
  auto              x = n::full({2, 28, 28, 1}, .3, t);
  auto original = n::zeros({2}, n::DType::Int32), target = n::ones({2}, n::DType::Int64);
  m::StarGan(g, d, x, original, target);
  for (const auto& name : d.names())
    d.set(name, n::zeros(d.at(name).shape(), t));
  auto o = m::StarGan(g, d, x, original, target);
  EXPECT_NEAR(o.discriminator_loss.item(), .5 + std::log(3.), 1e-14);
  EXPECT_EQ(o.source_loss_g.item(), 1);
  EXPECT_EQ(o.source_loss_d.item(), .5);
  EXPECT_NEAR(
      o.generator_loss.item(), (1. + std::log(3.)) + 10. * o.reconstruction_loss.item(), 1e-14);
  EXPECT_TRUE(n::equal_bits(o.fake, m::StarGanGenerate(g, x, target).image));
  EXPECT_TRUE(n::equal_bits(o.reconstruction, m::StarGanGenerate(g, o.fake, original).image));
  auto same = m::StarGan(g, d, x, original, original);
  EXPECT_TRUE(std::isfinite(same.generator_loss.item()));
}
TEST(NnStarGan, ReconstructionPartialIncludesBothUsesOfTheSharedGenerator) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(6301, t), d(6311, t);
  auto              x = n::full({2, 28, 28, 1}, .3, t);
  auto   original = n::zeros({2}, n::DType::Int64), target = n::ones({2}, original.dtype());
  auto   out     = m::StarGan(g, d, x, original, target);
  auto   full    = n::grad(out.reconstruction_loss, g.parameters());
  auto   stopped = m::StarGanGenerate(g, out.fake.detach(), original).image;
  auto   wrong   = n::grad(n::mean(n::abs(stopped - x)), g.parameters());
  double change  = 0;
  for (std::size_t i = 0; i < full.size(); ++i)
    change = std::max(change, Delta(full[i], wrong[i]));
  EXPECT_GT(change, 1e-10);
}
TEST(NnStarGan, DThenFreshGPreservesEachStoreAndChangesTheCapturedVector) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(6301, t), d(6311, t);
  auto              x     = n::full({2, 28, 28, 1}, .3, t);
  auto           original = n::zeros({2}, n::DType::Int64), target = n::ones({2}, original.dtype());
  auto           out = m::StarGan(g, d, x, original, target);
  n::GradOptions keep;
  keep.retain_graph        = true;
  auto               stale = n::grad(out.generator_loss, g.parameters(), keep);
  auto               dd    = n::grad(out.discriminator_loss, d.parameters());
  n::OptimizerConfig cfg;
  cfg.kind  = n::OptimizerKind::Adam;
  cfg.beta1 = .5;
  n::Optimizer go(g, .0002, cfg), do_(d, .0002, cfg);
  auto         before = Snapshot(g);
  do_.step(dd);
  EXPECT_TRUE(Equal(before, Snapshot(g)));
  auto   dafter = Snapshot(d);
  auto   gg     = n::grad(m::StarGan(g, d, x, original, target).generator_loss, g.parameters());
  double change = 0;
  for (std::size_t i = 0; i < gg.size(); ++i)
    change = std::max(change, Delta(stale[i], gg[i]));
  EXPECT_GT(change, 1e-10);
  go.step(gg);
  EXPECT_TRUE(Equal(dafter, Snapshot(d)));
  EXPECT_EQ(go.iterations(), 1);
  EXPECT_EQ(do_.iterations(), 1);
}
TEST(NnStarGan, PureGenerationDependsOnTargetAloneAndInferenceDoesNotWrite) {
  n::ParameterStore g(6301), d(6311);
  auto              x = n::ones({2, 28, 28, 1}), original = n::zeros({2}, n::DType::Int64),
       target = n::ones({2}, original.dtype());
  auto o      = m::StarGan(g, d, x, original, target);
  auto gb = Snapshot(g), db = Snapshot(d);
  auto changed = m::StarGan(g, d, x, target, target);
  EXPECT_TRUE(n::equal_bits(o.fake, changed.fake));
  EXPECT_TRUE(n::equal_bits(o.real_source, changed.real_source));
  EXPECT_TRUE(n::equal_bits(o.real_domain_logits, changed.real_domain_logits));
  EXPECT_TRUE(Equal(gb, Snapshot(g)));
  EXPECT_TRUE(Equal(db, Snapshot(d)));
}
TEST(NnStarGan, OwningFieldsKeepLiveHigherSourcePathsAfterParentAndStoresExpire) {
  auto      x = n::full({2, 28, 28, 1}, .3, n::DType::Float64).requires_grad_();
  n::Tensor image, loss, probability;
  {
    n::ParameterStore g(6301, x.dtype()), d(6311, x.dtype());
    auto o = m::StarGan(g, d, x, n::zeros({2}, n::DType::Int64), n::ones({2}, n::DType::Int64));
    image  = o.reconstruction;
    loss   = o.generator_loss;
    probability = o.fake_domain_probability;
  }
  n::GradOptions graph;
  graph.create_graph = true;
  auto first         = n::grad(loss, {x}, graph)[0];
  auto second        = n::grad(n::sum(n::square(first)), {x})[0];
  EXPECT_GT(n::max(n::abs(second)).item(), 0);
  EXPECT_EQ(image.shape(), x.shape());
  EXPECT_EQ(probability.shape(), (n::Shape{2, 3}));
}
TEST(NnStarGan, DomainAndSplitErrorsRejectRangesShapesDtypesBalanceAndRepeatedIds) {
  auto x      = n::zeros({40, 28, 28, 1}),
       labels = n::cast(n::arange(0, 40, 1, n::DType::Float64) / 4., n::DType::Int64),
       ids    = n::arange(0, 40, 1, n::DType::Int64);
  EXPECT_THROW(m::SplitStarDomains(x, labels, ids, 2), std::invalid_argument);
  EXPECT_THROW(
      m::SplitStarDomains(x, labels, n::zeros(ids.shape(), ids.dtype()), 1), std::invalid_argument);
  EXPECT_THROW(
      m::SplitStarDomains(x, n::zeros(labels.shape(), labels.dtype()), ids, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::SplitStarDomains(n::full(x.shape(), 1.01), labels, ids, 1), std::invalid_argument);
  EXPECT_THROW(m::SplitStarDomains(n::ones({}), labels, ids, 1), std::invalid_argument);
  EXPECT_THROW(
      m::StarDomainImages(n::ones({1, 2, 3, 1}), n::zeros({1}, ids.dtype())),
      std::invalid_argument);
  EXPECT_THROW(
      m::StarDomainImages(n::ones({1, 2, 2, 1}), n::full({1}, 3, ids.dtype())),
      std::invalid_argument);
}
TEST(NnStarGan, GameErrorsRejectAliasesAndEveryDomainBeforeOneHotOrLeaves) {
  n::ParameterStore g(6301), d(6311);
  auto              x = n::zeros({2, 28, 28, 1}), o = n::zeros({2}, n::DType::Int64),
       t = n::ones({2}, o.dtype());
  EXPECT_THROW(m::StarGan(g, g, x, o, t), std::invalid_argument);
  EXPECT_THROW(m::StarGan(g, d, x, n::full({2}, -1, o.dtype()), t), std::invalid_argument);
  EXPECT_THROW(m::StarGan(g, d, x, o, n::full({2}, 3, o.dtype())), std::invalid_argument);
  EXPECT_THROW(m::StarGan(g, d, x, o, n::ones({2})), std::invalid_argument);
  EXPECT_THROW(m::StarGanGenerate(g, x, n::zeros({1}, o.dtype())), std::invalid_argument);
  EXPECT_THROW(m::StarGan(g, d, n::full(x.shape(), NAN), o, t), std::invalid_argument);
  EXPECT_EQ(g.size(), 0);
  EXPECT_EQ(d.size(), 0);
}
