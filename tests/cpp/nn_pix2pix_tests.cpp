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
} // namespace
TEST(NnPix2Pix, SobelIntegerRecoveryInclusiveThresholdZeroPaddingAndDetachedState) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    auto white = n::ones({1, 3, 3, 1}, type).requires_grad_();
    auto edge  = n::quantized_sobel_edges(white);
    EXPECT_EQ(edge.shape(), white.shape());
    EXPECT_EQ(edge.dtype(), type);
    EXPECT_FALSE(edge.requires_grad());
    EXPECT_EQ(edge.at({0, 1, 1, 0}), 0);
    EXPECT_EQ(n::sum(edge).item(), 8);
    auto impulse = n::tensor({0., 0., 0., 0., 64. / 255, 0., 0., 0., 0.}, {1, 3, 3, 1}, type);
    EXPECT_EQ(n::sum(n::quantized_sobel_edges(impulse, 128)).item(), 8);
    EXPECT_EQ(n::sum(n::quantized_sobel_edges(impulse, 129)).item(), 0);
    EXPECT_EQ(n::sum(n::quantized_sobel_edges(white, 0)).item(), 9);
    EXPECT_EQ(n::sum(n::quantized_sobel_edges(white, 2041)).item(), 0);
    auto strided = n::transpose(n::tile(white, {2, 1, 1, 1}), {0, 2, 1, 3});
    EXPECT_EQ(n::sum(n::quantized_sobel_edges(strided)).item(), 16);
  }
  // A Float32 half-bin can round upward under Float32 multiplication; source quantizes in Double.
  const float value   = static_cast<float>(63.5 / 255.);
  auto        tiny    = n::zeros({1, 3, 3, 1});
  auto        values  = tiny.to_vector<float>();
  values[4]           = value;
  tiny                = n::from_vector(values, tiny.shape());
  const int recovered = int(std::floor(double(value) * 255 + .5));
  EXPECT_EQ(n::sum(n::quantized_sobel_edges(tiny, 128)).item(), recovered >= 64 ? 8 : 0);
}
TEST(NnPix2Pix, SobelRejectsRangeNonfiniteDtypeAndGeometry) {
  EXPECT_THROW(n::quantized_sobel_edges(n::ones({1, 2, 2, 3})), std::invalid_argument);
  EXPECT_THROW(n::quantized_sobel_edges(n::ones({2, 2})), std::invalid_argument);
  EXPECT_THROW(n::quantized_sobel_edges(n::full({1, 2, 2, 1}, -1)), std::invalid_argument);
  EXPECT_THROW(n::quantized_sobel_edges(n::full({1, 2, 2, 1}, 1.01)), std::invalid_argument);
  EXPECT_THROW(n::quantized_sobel_edges(n::full({1, 2, 2, 1}, NAN)), std::invalid_argument);
  EXPECT_THROW(n::quantized_sobel_edges(n::ones({1, 2, 2, 1}), -1), std::invalid_argument);
  EXPECT_THROW(
      n::quantized_sobel_edges(n::ones({1, 2, 2, 1}, n::DType::Int32)), std::invalid_argument);
}
TEST(NnPix2Pix, RealConcatenationSkipsAndSourceParameterCensus) {
  n::ParameterStore g(6101), d(6113);
  auto              source = n::ones({2, 28, 28, 1});
  auto              out    = m::Pix2PixGenerate(g, source);
  auto              score  = m::Pix2PixDiscriminate(d, source, out.image);
  EXPECT_EQ(g.count(), 1390);
  EXPECT_EQ(g.size(), 10);
  EXPECT_EQ(d.count(), 725);
  EXPECT_EQ(d.size(), 6);
  EXPECT_EQ(score.shape(), (n::Shape{2, 7, 7, 1}));
  EXPECT_TRUE(n::equal_bits(n::slice_axis(out.joined14, 3, 0, 4), out.up14));
  EXPECT_TRUE(n::equal_bits(n::slice_axis(out.joined14, 3, 4, 8), out.down14));
  EXPECT_TRUE(n::equal_bits(n::slice_axis(out.joined28, 3, 0, 2), out.up28));
  EXPECT_TRUE(n::equal_bits(n::slice_axis(out.joined28, 3, 2, 3), source));
  EXPECT_TRUE(
      n::equal_bits(
          m::Pix2PixGenerate(g, n::take_rows(source, {0})).image, n::take_rows(out.image, {0})));
}
TEST(NnPix2Pix, LossesArePatchMeansWithoutDiscriminatorHalfAndPairedL1Weight100) {
  n::ParameterStore g(6101), d(6113);
  auto              x      = n::full({2, 28, 28, 1}, .2);
  auto              source = n::ones(x.shape());
  m::Pix2Pix(g, d, x, source);
  for (const auto& name : d.names())
    d.set(name, n::zeros(d.at(name).shape()));
  auto out = m::Pix2Pix(g, d, x, source);
  EXPECT_NEAR(out.discriminator_loss.item(), 2 * std::log(2), 2e-7);
  EXPECT_NEAR(out.g_adversarial.item(), std::log(2), 1e-7);
  EXPECT_FLOAT_EQ(out.generator_loss.item(), (out.g_adversarial + 100 * out.l1).item());
  auto zero = m::Pix2Pix(g, d, x, source, 0);
  EXPECT_TRUE(n::equal_bits(zero.generator_loss, zero.g_adversarial));
  auto paired = m::Pix2Pix(g, d, n::ones(x.shape()), source);
  EXPECT_GT(std::abs(out.l1.item() - paired.l1.item()), .1);
}
TEST(NnPix2Pix, SourceOnlyGenerationAndConditionalScoresAreImmutable) {
  n::ParameterStore g(6101), d(6113);
  auto              x = n::full({2, 28, 28, 1}, .2), c = n::ones(x.shape());
  auto              out = m::Pix2Pix(g, d, x, c);
  auto              gs = Snapshot(g), ds = Snapshot(d);
  auto              alternate = m::Pix2Pix(g, d, 1. - x, c);
  EXPECT_TRUE(n::equal_bits(out.fake, alternate.fake));
  EXPECT_GT(Difference(m::Pix2PixGenerate(g, n::zeros(c.shape())).image, out.fake), 1e-6);
  EXPECT_GT(Difference(m::Pix2PixDiscriminate(d, n::zeros(c.shape()), x), out.real_logit), 1e-6);
  EXPECT_TRUE(Equal(gs, Snapshot(g)));
  EXPECT_TRUE(Equal(ds, Snapshot(d)));
}
TEST(NnPix2Pix, ExplicitDThenFreshGUpdatesOnlyItsStoreAndChangesTheGeneratorGradient) {
  n::ParameterStore g(6101, n::DType::Float64), d(6113, n::DType::Float64);
  auto x = n::full({2, 28, 28, 1}, .2, n::DType::Float64), c = n::ones(x.shape(), x.dtype());
  auto out = m::Pix2Pix(g, d, x, c);
  n::GradOptions keep;
  keep.retain_graph          = true;
  auto               stale   = n::grad(out.generator_loss, g.parameters(), keep);
  auto               gd      = n::grad(out.discriminator_loss, d.parameters());
  auto               beforeG = Snapshot(g), beforeD = Snapshot(d);
  n::OptimizerConfig config;
  config.kind  = n::OptimizerKind::Adam;
  config.beta1 = .5;
  n::Optimizer od(d, .0002, config), og(g, .0002, config);
  od.step(gd);
  EXPECT_TRUE(Equal(beforeG, Snapshot(g)));
  EXPECT_FALSE(Equal(beforeD, Snapshot(d)));
  auto   afterD  = Snapshot(d);
  auto   fresh   = m::Pix2Pix(g, d, x, c);
  auto   gg      = n::grad(fresh.generator_loss, g.parameters());
  double changed = 0;
  for (std::size_t i = 0; i < gg.size(); ++i)
    changed = std::max(changed, Difference(gg[i], stale[i]));
  EXPECT_GT(changed, 1e-12);
  og.step(gg);
  EXPECT_TRUE(Equal(afterD, Snapshot(d)));
  EXPECT_FALSE(Equal(beforeG, Snapshot(g)));
  EXPECT_EQ(og.iterations(), 1);
  EXPECT_EQ(od.iterations(), 1);
}
TEST(NnPix2Pix, SourcePathsAndHigherDerivativesSurviveParentAndStoreDestruction) {
  auto      source = n::full({2, 28, 28, 1}, .3, n::DType::Float64).requires_grad_();
  n::Tensor fake, loss, skip;
  {
    n::ParameterStore g(6101, n::DType::Float64), d(6113, n::DType::Float64);
    auto              generated = m::Pix2PixGenerate(g, source);
    auto              out = m::Pix2Pix(g, d, n::zeros(source.shape(), source.dtype()), source);
    fake                  = out.fake;
    loss                  = out.generator_loss;
    skip                  = generated.joined28;
  }
  n::GradOptions graph;
  graph.create_graph = true;
  auto first         = n::grad(loss, {source}, graph)[0];
  auto second        = n::grad(n::sum(n::square(first)), {source})[0];
  EXPECT_EQ(second.shape(), source.shape());
  EXPECT_GT(n::max(n::abs(first)).item(), 0);
  EXPECT_GT(n::max(n::abs(second)).item(), 0);
  EXPECT_EQ(fake.shape(), source.shape());
  EXPECT_TRUE(n::equal_bits(n::slice_axis(skip, 3, 2, 3), source));
}
TEST(NnPix2Pix, InvalidPairsAndWeightsFailBeforeAnyLeafCreation) {
  n::ParameterStore g(1), d(2);
  auto              x = n::ones({2, 28, 28, 1});
  EXPECT_THROW(m::Pix2Pix(g, g, x, x), std::invalid_argument);
  EXPECT_THROW(m::Pix2Pix(g, d, x, n::ones({1, 28, 28, 1})), std::invalid_argument);
  EXPECT_THROW(m::Pix2Pix(g, d, x, x, -1), std::invalid_argument);
  EXPECT_THROW(m::Pix2Pix(g, d, x, x, INFINITY), std::invalid_argument);
  EXPECT_THROW(m::Pix2Pix(g, d, x, x, 1e300), std::invalid_argument);
  EXPECT_THROW(m::Pix2PixDiscriminate(d, x, n::ones({1, 28, 28, 1})), std::invalid_argument);
  EXPECT_THROW(m::Pix2PixGenerate(g, n::ones({2, 27, 28, 1})), std::invalid_argument);
  EXPECT_THROW(m::Pix2Pix(g, d, n::full(x.shape(), NAN), x), std::invalid_argument);
  EXPECT_EQ(g.size() + d.size(), 0);
  n::ParameterStore dd(2, n::DType::Float64);
  EXPECT_THROW(m::Pix2Pix(g, dd, x, x), std::invalid_argument);
  EXPECT_EQ(g.size() + dd.size(), 0);
}
