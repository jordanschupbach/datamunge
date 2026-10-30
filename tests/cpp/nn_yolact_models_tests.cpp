#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
namespace m = nn::models;
using nn::Tensor;
using Box = nn::Box;
namespace check {
template <typename A, typename B> void Near(A a, B b, double t, const std::string &msg) {
  EXPECT_NEAR(double(a), double(b), t) << msg;
}
inline void Require(bool ok, const std::string &msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
} // namespace check
namespace {
m::YolactConfig Compact() {
  m::YolactConfig c;
  c.size = 64;
  c.classes = 10;
  c.depth = 18;
  c.divisor = 8;
  c.width = 32;
  c.prototypes = 8;
  c.grayscale = true;
  return c;
}

std::int64_t ParameterReference(const m::YolactConfig &c) {
  // Independently enumerate canonical ResNet blocks and convolution shapes.
  const bool bottleneck = c.depth == 101;
  const int blocks101[] = {3, 4, 23, 3}, blocks18[] = {2, 2, 2, 2};
  int in = 64 / c.divisor;
  std::int64_t n = 49 * 3 * in + 2 * in;
  for (int stage = 0; stage < 4; ++stage) {
    const int w = (64 << stage) / c.divisor, out = w * (bottleneck ? 4 : 1);
    for (int block = 0; block < (bottleneck ? blocks101[stage] : blocks18[stage]); ++block) {
      if (bottleneck)
        n += std::int64_t(in) * w + 2 * w + 9LL * w * w + 2 * w + std::int64_t(w) * out +
             2 * out;
      else
        n += 9LL * in * w + 2 * w + 9LL * w * out + 2 * out;
      if (in != out)
        n += std::int64_t(in) * out + 2 * out;
      in = out;
    }
  }
  if (c.depth == 101 && c.divisor == 1)
    check::Require(n == 42500160, "canonical ResNet101 without classifier");
  const int w = c.width, k = c.prototypes;
  for (int i = 0; i < 3; ++i)
    n += std::int64_t((128 << i) * (bottleneck ? 4 : 1) / c.divisor) * w + w;
  n += 10LL * (9LL * w * w + w); // three smooth, two down, one prediction, four protonet.
  n += (9LL * w + 1) * (3 * (c.classes + 1) + 12 + 3 * k) +
       (std::int64_t(w) + 1) * (k + c.classes);
  return n;
}

void LinearMaskChecks() {
  const std::vector<float> p_values = {1, 2, -1, 3, 2, -2, 4, 1, 0, 2, -3, 1},
                           c_values = {.5f, -.25f}, t_values = {0, 1, 0, 1, 0, 1};
  std::vector<float> m_values = {1, 1, 0, 0, 1, 0};
  struct Result {
    std::vector<float> logits, dp, dc;
    double loss;
  };
  auto run = [&] {
    Tensor proto = nn::from_vector(p_values, {1, 2, 3, 2}),
           coeff = nn::from_vector(c_values, {1, 2});
    proto.requires_grad_(true);
    coeff.requires_grad_(true);
    const Tensor logits = m::LinearInstanceMasks(proto, coeff);
    const Tensor loss = m::CroppedInstanceMaskLoss(
        logits, nn::from_vector(t_values, {1, 2, 3}), nn::from_vector(m_values, {1, 2, 3}));
    const auto g = nn::grad(loss, {proto, coeff});
    return Result{logits.to_vector<float>(), g[0].to_vector<float>(), g[1].to_vector<float>(),
                  loss.item<double>()};
  };
  auto out = run();
  double expected = 0, dc[2] = {0, 0};
  for (int i = 0; i < 6; ++i) {
    const double z =
        p_values[std::size_t(2 * i)] * .5 - p_values[std::size_t(2 * i + 1)] * .25;
    check::Near(out.logits[std::size_t(i)], z, 1e-7, "signed prototype linear combination");
    const double g =
        (1 / (1 + std::exp(-z)) - t_values[std::size_t(i)]) * m_values[std::size_t(i)] / 3;
    expected += (std::log1p(std::exp(z)) - z * t_values[std::size_t(i)]) *
                m_values[std::size_t(i)] / 3;
    for (int k = 0; k < 2; ++k) {
      check::Near(out.dp[std::size_t(2 * i + k)], g * c_values[std::size_t(k)], 1e-7,
                  "every prototype derivative and zero outside crop");
      dc[k] += g * p_values[std::size_t(2 * i + k)];
    }
  }
  check::Near(out.loss, expected, 1e-7, "area-normalized cropped BCE");
  for (int k = 0; k < 2; ++k)
    check::Near(out.dc[std::size_t(k)], dc[k], 1e-7, "every coefficient derivative");
  m_values.assign(6, 0.f);
  out = run();
  check::Near(out.loss, 0, 0, "empty crop has zero mask loss");
  for (float v : out.dp)
    check::Near(v, 0, 0, "empty crop masks prototype gradients");
  const Tensor empty = m::CroppedInstanceMaskLoss(nn::zeros({0, 2, 3}), nn::zeros({0, 2, 3}),
                                                  nn::zeros({0, 2, 3}));
  check::Near(empty.item<double>(), 0, 0, "zero positive masks produce finite zero loss");
  std::cout
      << "  signed prototype assembly, cropped BCE and every analytic prototype/coefficient "
         "derivative; empty "
         "crop/positive set passed"
      << std::endl;
}

void ModelChecks() {
  {
    nn::ParameterStore p(17), state(17);
    m::YolactConfig cfg; // full ResNet101/FPN256/32 prototypes/80 classes
    check::Require(m::YolactCells(550) == std::vector<int>({69, 35, 18, 9, 5}),
                   "full 550px level grids");
    int anchors = 0;
    for (int c : m::YolactCells(550))
      anchors += 3 * c * c;
    check::Require(anchors == 19248 && 2 * m::YolactCells(550)[0] == 138,
                   "full YOLACT prior/prototype geometry");
    cfg.size = 64;
    std::vector<float> v(64 * 64 * 3);
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = std::sin(.13f * float(i));
    const auto net = m::Yolact(p, state, nn::from_vector(v, {1, 64, 64, 3}), cfg, true);
    check::Require(p.count() == ParameterReference(cfg), "full YOLACT parameter count");
    Tensor objective = nn::sum(nn::square(net.logits));
    for (const auto &x : {net.deltas, net.coefficients, net.prototypes, net.semantic})
      objective = objective + nn::sum(nn::square(x));
    const auto grads = nn::grad(objective, p.parameters());
    for (std::size_t k = 0; k < grads.size(); ++k)
      check::Require(std::isfinite(nn::sum(nn::square(grads[k])).item<double>()),
                     "finite full YOLACT derivative");
    check::Require(grads.size() == 348 && p.size() == 348,
                   "full ResNet101 and all YOLACT branches connected");
    std::cout
        << "  full ResNet101 YOLACT: " << p.count()
        << " parameters, 19248 priors, 138px/32 prototypes at 550px, all 348 derivatives "
           "(executed on 64x64)"
        << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  const auto cfg = Compact();
  std::vector<float> v(2 * 64 * 64);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = std::sin(.13f * float(i));
  const auto net = m::Yolact(p, state, nn::from_vector(v, {2, 64, 64, 1}), cfg, true);
  check::Require(p.count() == ParameterReference(cfg) &&
                     net.cells == std::vector<int>({8, 4, 2, 1, 1}) && net.anchors == 258,
                 "compact YOLACT independent parameter/anchor count");
  Tensor objective = nn::mean(nn::square(net.logits));
  for (const auto &x : {net.deltas, net.coefficients, net.prototypes, net.semantic})
    objective = objective + nn::mean(nn::square(x));
  const std::vector<nn::Shape> shapes = {
      {2, 258, 11}, {2, 258, 4}, {2, 258, 8}, {2, 16, 16, 8}, {2, 8, 8, 10}};
  const std::vector<Tensor> outs = {net.logits, net.deltas, net.coefficients, net.prototypes,
                                    net.semantic};
  for (int i = 0; i < 5; ++i)
    check::Require(outs[std::size_t(i)].shape() == shapes[std::size_t(i)],
                   "YOLACT branch output shape");
  const auto grads = nn::grad(objective, p.parameters());
  double low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm =
        std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(std::isfinite(norm) && norm > 0,
                   "finite nonzero YOLACT derivative: " + p.names()[k]);
    low = std::min(low, norm);
    high = std::max(high, norm);
  }
  for (float c : net.coefficients.to_vector<float>())
    check::Require(std::abs(c) <= 1, "bounded tanh coefficients");
  for (float x : net.prototypes.to_vector<float>())
    check::Require(x >= 0, "nonnegative ReLU prototypes");
  std::cout << "  compact YOLACT: " << p.count()
            << " parameters, 258 priors, every branch/gradient passed, norms [" << low << ','
            << high << ']' << std::endl;
}

} // namespace
TEST(NnYolact, IndependentEverySignedPrototypeCoefficientGradientAndCropNormalization) {
  LinearMaskChecks();
}
TEST(NnYolact, FullIndependentCensus348GradientsCompactEveryNonzeroBranchGradient) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(NnYolact, AllPredecessorSuppressionChainClassImageIsolationStableTiesAndCaps) {
  m::TensorDetections d{nn::tensor({0.,  0.,  10., 10., 0.,  4.,  10., 14., 0.,  8.,
                                    10., 18., 0.,  0.,  10., 10., 0.,  0.,  10., 10.},
                                   {5, 4}),
                        nn::tensor({.9, .8, .7, .8, .6}),
                        nn::from_vector(std::vector<std::int64_t>{0, 0, 0, 0, 1}),
                        nn::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 0})};
  EXPECT_EQ(m::TensorFastNms(d, .3).to_vector<int>(), (std::vector<int>{0, 4, 3}));
  EXPECT_EQ(m::TensorFastNms(d, .3, 200, 1).to_vector<int>(), (std::vector<int>{0, 3}));
  EXPECT_EQ(m::TensorFastNms(d, .3, 0).numel(), 0);
  const auto greedy = nn::non_max_suppression(nn::slice_axis(d.boxes, 0, 0, 3),
                                              nn::slice_axis(d.scores, 0, 0, 3), 3, .3);
  EXPECT_EQ(greedy.indices.to_vector<int>(), (std::vector<int>{0, 2}));
  const auto match = m::YolactMatching(
      nn::tensor({0., 0., 10., 10., 20., 20., 30., 30., 0., 0., 10., 22., 50., 50., 60., 60.},
                 {4, 4}),
      nn::tensor({0., 0., 10., 10., 20., 20., 30., 30.}, {2, 4}));
  EXPECT_EQ(match.to_vector<int>(), (std::vector<int>{0, 1, -2, -1}));
}
TEST(NnYolact, IndependentWeightedLossClassMaskDerivativesIgnoredMiningAndNoPositives) {
  auto z = nn::zeros({1, 4, 3}).requires_grad_(), c = nn::zeros({1, 4, 2}).requires_grad_();
  m::YolactOutputs net;
  net.logits = z;
  net.deltas = nn::zeros({1, 4, 4});
  net.coefficients = c;
  net.prototypes = nn::ones({1, 2, 2, 2});
  net.semantic = nn::zeros({1, 1, 1, 2});
  m::TensorYolactTargets t{nn::from_vector(std::vector<std::int64_t>{0, 2, 2, -1}, {1, 4}),
                           nn::zeros({1, 4, 4}),
                           nn::from_vector(std::vector<std::int64_t>{0, 0}, {1, 2}),
                           nn::tensor({1., 0., 0., 0.}, {1, 2, 2}),
                           nn::tensor({1., 0., 0., 0.}, {1, 2, 2}),
                           nn::zeros({1, 1, 1, 2})};
  const auto loss = m::BuildTensorYolactLoss(net, t);
  EXPECT_NEAR(loss.loss.item(), 3 * std::log(3.) + 7.125 * std::log(2.), 2e-6);
  const auto g = nn::grad(loss.loss, {z, c});
  const auto dz = g[0].to_vector<float>(), dc = g[1].to_vector<float>();
  for (int a = 0; a < 4; ++a)
    for (int k = 0; k < 3; ++k)
      EXPECT_NEAR(dz[a * 3 + k], a == 3 ? 0 : 1. / 3 - (k == (a == 0 ? 0 : 2)), 1e-7);
  for (int a = 0; a < 4; ++a)
    for (int k = 0; k < 2; ++k)
      EXPECT_NEAR(dc[a * 2 + k], a == 0 ? -3.0625 : 0, 1e-7);
  EXPECT_EQ(loss.negatives.to_vector<int>(), (std::vector<int>{0, 1, 1, 0}));
  t.labels = nn::full({1, 4}, 2., nn::DType::Int64);
  t.pairs = nn::zeros({0, 2}, nn::DType::Int64);
  t.mask_targets = nn::zeros({0, 2, 2});
  t.crop = nn::zeros({0, 2, 2});
  EXPECT_NEAR(m::BuildTensorYolactLoss(net, t).loss.item(), std::log(2.), 1e-7);
}
TEST(NnYolact, ReorderedSceneNearestTargetsCropsClassUnionsAndZeroOffsets) {
  m::TensorInstanceScenes data;
  data.images = nn::zeros({2, 4, 4, 1});
  data.boxes = {nn::tensor({0., 0., 2., 2.}, {1, 4}), nn::tensor({2., 2., 4., 4.}, {1, 4})};
  data.labels = {nn::zeros({1}, nn::DType::Int64), nn::ones({1}, nn::DType::Int64)};
  data.masks = {
      nn::tensor({1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}, {1, 4, 4})
          .to(nn::DType::Bool),
      nn::tensor({0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1.}, {1, 4, 4})
          .to(nn::DType::Bool)};
  m::YolactOutputs net;
  net.logits = nn::zeros({2, 4, 3});
  net.proto_side = 2;
  net.cells = {1};
  nn::Generator rng(41);
  const auto target = m::MakeTensorYolactTargets(
      net,
      nn::tensor({0., 0., 2., 2., 2., 2., 4., 4., 8., 8., 9., 9., 10., 10., 11., 11.}, {4, 4}),
      data, {1, 0}, 2, rng);
  EXPECT_EQ(target.labels.to_vector<int>(), (std::vector<int>{2, 1, 2, 2, 0, 2, 2, 2}));
  EXPECT_EQ(target.pairs.to_vector<int>(), (std::vector<int>{0, 1, 1, 0}));
  EXPECT_EQ(target.mask_targets.to_vector<float>(),
            (std::vector<float>{0, 0, 0, 1, 1, 0, 0, 0}));
  EXPECT_EQ(target.crop.to_vector<float>(), target.mask_targets.to_vector<float>());
  EXPECT_EQ(target.semantic.to_vector<float>(), (std::vector<float>{0, 1, 0, 0}));
  EXPECT_EQ(nn::max(nn::abs(target.deltas)).item(), 0);
}
TEST(NnYolact, DecodeKnownSignedPrototypeSupportScoreCutoffAndImageIds) {
  m::YolactOutputs net;
  net.probabilities = nn::tensor({.9, .1}, {1, 1, 2});
  net.deltas = nn::zeros({1, 1, 4});
  net.coefficients = nn::ones({1, 1, 1});
  net.prototypes = nn::tensor({-2., 2., 2., -2.}, {1, 2, 2, 1});
  net.proto_side = 2;
  const auto anchors = nn::tensor({0., 0., 4., 4.}, {1, 4});
  const auto result = m::DecodeTensorYolact(net, anchors, {5}, 4, 4);
  EXPECT_EQ(result.objects.images.to_vector<int>(), (std::vector<int>{5}));
  EXPECT_EQ(result.objects.labels.to_vector<int>(), (std::vector<int>{0}));
  EXPECT_EQ(result.masks.to_vector<int>(),
            (std::vector<int>{0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0}));
  net.probabilities = nn::tensor({.05, .95}, {1, 1, 2});
  EXPECT_EQ(m::DecodeTensorYolact(net, anchors, {5}, 4, 4).objects.scores.numel(), 0);
}
