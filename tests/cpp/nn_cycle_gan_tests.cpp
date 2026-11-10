#include <gtest/gtest.h>

#include <dm/dm.hpp>

namespace n = dm::nn;
namespace m = n::models;
namespace {
using A = std::map<std::string, n::Tensor>;
A Snapshot(const std::vector<n::ParameterStore*>& stores) {
  A out;
  for (std::size_t k = 0; k < stores.size(); ++k)
    for (const auto& name : stores[k]->names())
      out.emplace(std::to_string(k) + "/" + name, stores[k]->at(name).detach().clone());
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
double Difference(const n::Tensor& a, const n::Tensor& b) {
  return n::max(n::abs(a - b)).item();
}
} // namespace
TEST(NnCycleGan, RotationInversionOrientationRectangleAndLiveInputDerivative) {
  auto x        = n::tensor({0., .25, .5, .75}, {1, 2, 2, 1}, n::DType::Float64).requires_grad_();
  auto expected = n::tensor({.5, 1., .25, .75}, {1, 2, 2, 1}, x.dtype());
  auto y        = m::RotateInvertGrayscale(x);
  EXPECT_TRUE(n::equal_bits(y, expected));
  EXPECT_TRUE(n::equal_bits(m::RotateInvertGrayscale(y, true), x));
  auto gradient = n::grad(n::sum(y), {x})[0];
  EXPECT_TRUE(n::equal_bits(gradient, n::full(x.shape(), -1., x.dtype())));
  auto rectangle = n::reshape(n::arange(0, 6, 1, n::DType::Float64), {1, 2, 3, 1});
  auto turned    = m::RotateInvertGrayscale(rectangle);
  EXPECT_EQ(turned.shape(), (n::Shape{1, 3, 2, 1}));
  EXPECT_TRUE(n::equal_bits(m::RotateInvertGrayscale(turned, true), rectangle));
  EXPECT_THROW(m::RotateInvertGrayscale(n::ones({2, 3})), std::invalid_argument);
  EXPECT_THROW(m::RotateInvertGrayscale(n::full({1, 2, 2, 1}, NAN)), std::invalid_argument);
  EXPECT_THROW(m::RotateInvertGrayscale(n::ones({1, 2, 2, 3})), std::invalid_argument);
}
TEST(NnCycleGan, DomainsPreserveObservedOrderProvenanceAndOriginalBWithoutPairing) {
  auto image =
      n::reshape(n::arange(0, 40 * 784, 1, n::DType::Float64) / (40 * 784), {40, 28, 28, 1})
          .requires_grad_();
  std::vector<std::int32_t> labels;
  for (int i = 0; i < 40; ++i)
    labels.push_back((i * 3) % 10);
  auto ids = n::arange(1000, 1040, 1, n::DType::Int64);
  auto d   = m::SplitCycleDomains(image, n::from_vector(labels, {40}), ids);
  EXPECT_EQ(d.a.shape(), (n::Shape{20, 28, 28, 1}));
  EXPECT_TRUE(n::equal_bits(d.a, n::slice_axis(image, 0, 0, 20)));
  EXPECT_TRUE(n::equal_bits(d.original_b, n::slice_axis(image, 0, 20, 40)));
  EXPECT_TRUE(n::equal_bits(d.b, m::RotateInvertGrayscale(d.original_b)));
  EXPECT_TRUE(n::equal_bits(d.ids_a, n::slice_axis(ids, 0, 0, 20)));
  EXPECT_TRUE(n::equal_bits(d.ids_b, n::slice_axis(ids, 0, 20, 40)));
  auto g = n::grad(n::sum(d.a) + n::sum(d.original_b), {image})[0];
  EXPECT_TRUE(n::equal_bits(g, n::ones(image.shape(), image.dtype())));
}
TEST(NnCycleGan, DomainsRejectBadBalanceDuplicateIdsUnitRangeAndIntegerMetadata) {
  auto x      = n::ones({20, 28, 28, 1});
  auto ids    = n::arange(0, 20, 1, n::DType::Int64);
  auto labels = n::cast(n::arange(0, 20, 1, n::DType::Float64) / 2., n::DType::Int32);
  EXPECT_THROW(
      m::SplitCycleDomains(x, n::zeros({20}, n::DType::Int32), ids), std::invalid_argument);
  EXPECT_THROW(
      m::SplitCycleDomains(x, labels, n::zeros({20}, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(
      m::SplitCycleDomains(x, labels, n::full({20}, -1, n::DType::Int64)), std::invalid_argument);
  EXPECT_THROW(
      m::SplitCycleDomains(x, n::cast(labels, n::DType::Float32), ids), std::invalid_argument);
  EXPECT_THROW(m::SplitCycleDomains(n::full(x.shape(), 1.01), labels, ids), std::invalid_argument);
  EXPECT_THROW(m::SplitCycleDomains(n::ones({21, 28, 28, 1}), labels, ids), std::invalid_argument);
}
TEST(NnCycleGan, UnpairedUnequalRowsAndFourStoreParameterCensus) {
  n::ParameterStore ab(6203), ba(6211), da(6217), db(6221);
  auto              a = n::full({2, 28, 28, 1}, .2), b = n::full({3, 28, 28, 1}, .8);
  auto              out = m::CycleGan(ab, ba, da, db, a, b);
  EXPECT_EQ(ab.count(), 1390);
  EXPECT_EQ(ba.count(), 1390);
  EXPECT_EQ(da.count(), 661);
  EXPECT_EQ(db.count(), 661);
  EXPECT_EQ(ab.size() + ba.size() + da.size() + db.size(), 32);
  EXPECT_EQ(out.fake_a.shape(), b.shape());
  EXPECT_EQ(out.fake_b.shape(), a.shape());
  EXPECT_EQ(out.cycle_a.shape(), a.shape());
  EXPECT_EQ(out.cycle_b.shape(), b.shape());
  EXPECT_EQ(out.real_a_score.shape(), (n::Shape{2, 7, 7, 1}));
  EXPECT_EQ(out.fake_a_score.shape(), (n::Shape{3, 7, 7, 1}));
}
TEST(NnCycleGan, FullGeneratorAdversarialAndHalfCriticScalesWithExplicitWeights) {
  n::ParameterStore ab(6203), ba(6211), da(6217), db(6221);
  auto              a = n::full({2, 28, 28, 1}, .2), b = n::full({3, 28, 28, 1}, .8);
  m::CycleGan(ab, ba, da, db, a, b);
  for (auto* p : {&da, &db})
    for (const auto& name : p->names())
      p->set(name, n::zeros(p->at(name).shape()));
  auto out = m::CycleGan(ab, ba, da, db, a, b);
  EXPECT_EQ(out.d_a_loss.item(), .5);
  EXPECT_EQ(out.d_b_loss.item(), .5);
  EXPECT_EQ(out.adversarial_a.item(), 1);
  EXPECT_EQ(out.adversarial_b.item(), 1);
  auto weighted =
      (out.adversarial_a + out.adversarial_b) + (10. * (out.cycle_loss_a + out.cycle_loss_b) +
                                                 5. * (out.identity_loss_a + out.identity_loss_b));
  EXPECT_TRUE(n::equal_bits(out.g_loss, weighted));
  EXPECT_EQ(m::CycleGan(ab, ba, da, db, a, b, 0, 0).g_loss.item(), 2);
  da.set("output/b", n::full({1}, 3.));
  EXPECT_EQ(n::mean(m::CycleGanDiscriminate(da, a)).item(), 3.);
}
TEST(NnCycleGan, TranslationsReadTheirSourceAloneAndAllForwardStoresStayImmutable) {
  n::ParameterStore ab(6203), ba(6211), da(6217), db(6221);
  auto              a = n::full({2, 28, 28, 1}, .2), b = n::full({3, 28, 28, 1}, .8);
  auto              out    = m::CycleGan(ab, ba, da, db, a, b);
  auto              before = Snapshot({&ab, &ba, &da, &db});
  EXPECT_TRUE(n::equal_bits(out.fake_b, m::Pix2PixGenerate(ab, a).image));
  EXPECT_TRUE(n::equal_bits(out.fake_a, m::Pix2PixGenerate(ba, b).image));
  EXPECT_TRUE(
      n::equal_bits(out.fake_b, m::CycleGan(ab, ba, da, db, a, n::zeros(b.shape())).fake_b));
  EXPECT_TRUE(n::equal_bits(out.identity_a, m::Pix2PixGenerate(ba, a).image));
  EXPECT_TRUE(n::equal_bits(out.identity_b, m::Pix2PixGenerate(ab, b).image));
  EXPECT_TRUE(Equal(before, Snapshot({&ab, &ba, &da, &db})));
}
TEST(NnCycleGan, JointGeneratorCapturesPrecedeEitherWriteAndSequentialRecomputeDiffers) {
  n::ParameterStore ab(6203, n::DType::Float64), ba(6211, n::DType::Float64),
      da(6217, n::DType::Float64), db(6221, n::DType::Float64);
  auto a = n::full({2, 28, 28, 1}, .2, n::DType::Float64),
       b = n::full({2, 28, 28, 1}, .8, n::DType::Float64);
  m::CycleGan(ab, ba, da, db, a, b);
  std::vector<n::ParameterStore*> p{&ab, &ba, &da, &db};
  n::OptimizerConfig              config;
  config.kind  = n::OptimizerKind::Adam;
  config.beta1 = .5;
  std::vector<std::unique_ptr<n::Optimizer>> opt;
  for (auto* store : p)
    opt.push_back(std::make_unique<n::Optimizer>(*store, .0002, config));
  for (int group : {2, 3}) {
    auto out      = m::CycleGan(ab, ba, da, db, a, b);
    auto before   = Snapshot(p);
    auto gradient = n::grad(group == 2 ? out.d_a_loss : out.d_b_loss, p[group]->parameters());
    opt[group]->step(gradient);
    auto after = Snapshot(p);
    for (const auto& [key, value] : before)
      if (key[0] != '0' + group)
        EXPECT_TRUE(n::equal_bits(value, after.at(key)));
  }
  auto           joint = m::CycleGan(ab, ba, da, db, a, b);
  n::GradOptions keep;
  keep.retain_graph = true;
  auto first        = n::grad(joint.g_loss, ab.parameters(), keep),
       second       = n::grad(joint.g_loss, ba.parameters(), keep);
  auto before_ba    = Snapshot({&ba});
  auto critics      = Snapshot({&da, &db});
  opt[0]->step(first);
  EXPECT_TRUE(Equal(before_ba, Snapshot({&ba})));
  auto   wrong  = n::grad(m::CycleGan(ab, ba, da, db, a, b).g_loss, ba.parameters());
  double change = 0;
  for (std::size_t i = 0; i < wrong.size(); ++i)
    change = std::max(change, Difference(wrong[i], second[i]));
  EXPECT_GT(change, 1e-10);
  auto after_ab = Snapshot({&ab});
  opt[1]->step(second);
  EXPECT_TRUE(Equal(after_ab, Snapshot({&ab})));
  EXPECT_TRUE(Equal(critics, Snapshot({&da, &db})));
  for (auto& o : opt)
    EXPECT_EQ(o->iterations(), 1);
}
TEST(NnCycleGan, ACycleHasBothGeneratorPathsAndNoOppositeInputPath) {
  n::ParameterStore ab(6203, n::DType::Float64), ba(6211, n::DType::Float64),
      da(6217, n::DType::Float64), db(6221, n::DType::Float64);
  auto a             = n::full({2, 28, 28, 1}, .2, n::DType::Float64).requires_grad_(),
       b             = n::full({3, 28, 28, 1}, .8, n::DType::Float64).requires_grad_();
  auto           out = m::CycleGan(ab, ba, da, db, a, b);
  n::GradOptions keep;
  keep.retain_graph = true;
  auto g0           = n::grad(out.cycle_loss_a, ab.parameters(), keep),
       g1           = n::grad(out.cycle_loss_a, ba.parameters(), keep);
  double size0 = 0, size1 = 0;
  for (const auto& g : g0)
    size0 += n::sum(n::abs(g)).item();
  for (const auto& g : g1)
    size1 += n::sum(n::abs(g)).item();
  EXPECT_GT(size0, 0);
  EXPECT_GT(size1, 0);
  keep.allow_unused = true;
  EXPECT_FALSE(n::grad(n::sum(out.cycle_a), {b}, keep)[0].defined());
  EXPECT_FALSE(n::grad(n::sum(out.fake_a), {a}, keep)[0].defined());
}
TEST(NnCycleGan, OwnedFieldsKeepHigherPathsAfterParentsAndStoresAreDestroyed) {
  auto      a = n::full({2, 28, 28, 1}, .2, n::DType::Float64).requires_grad_();
  n::Tensor loss, cycle;
  {
    n::ParameterStore ab(6203, n::DType::Float64), ba(6211, n::DType::Float64),
        da(6217, n::DType::Float64), db(6221, n::DType::Float64);
    auto out = m::CycleGan(ab, ba, da, db, a, n::full({2, 28, 28, 1}, .8, n::DType::Float64));
    loss     = out.g_loss;
    cycle    = out.cycle_a;
  }
  n::GradOptions graph;
  graph.create_graph = true;
  auto first         = n::grad(loss, {a}, graph)[0];
  auto second        = n::grad(n::sum(n::square(first)), {a})[0];
  EXPECT_EQ(cycle.shape(), a.shape());
  EXPECT_EQ(second.shape(), a.shape());
  EXPECT_GT(n::max(n::abs(second)).item(), 0);
}
TEST(NnCycleGan, AllAliasesInvalidWeightsAndBadInputsFailBeforeLeaves) {
  n::ParameterStore               ab(1), ba(2), da(3), db(4);
  std::vector<n::ParameterStore*> p{&ab, &ba, &da, &db};
  auto                            a = n::ones({2, 28, 28, 1});
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < i; ++j) {
      auto alias = p;
      alias[i]   = alias[j];
      EXPECT_THROW(
          m::CycleGan(*alias[0], *alias[1], *alias[2], *alias[3], a, a), std::invalid_argument);
    }
  EXPECT_THROW(m::CycleGan(ab, ba, da, db, a, a, -1, 5), std::invalid_argument);
  EXPECT_THROW(m::CycleGan(ab, ba, da, db, a, a, 10, INFINITY), std::invalid_argument);
  EXPECT_THROW(m::CycleGan(ab, ba, da, db, a, a, 1e300, 5), std::invalid_argument);
  EXPECT_THROW(m::CycleGan(ab, ba, da, db, a, n::full(a.shape(), NAN)), std::invalid_argument);
  EXPECT_THROW(m::CycleGan(ab, ba, da, db, a, n::ones({2, 27, 28, 1})), std::invalid_argument);
  for (auto* store : p)
    EXPECT_EQ(store->size(), 0);
}
