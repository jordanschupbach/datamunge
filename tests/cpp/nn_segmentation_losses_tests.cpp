#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include <random>
namespace nn = dm::nn;
namespace seg = nn::models;
using nn::Tensor;
namespace check {
template <typename A, typename B>
void Near(A a, B b, double tolerance, const std::string &message) {
  EXPECT_NEAR(double(a), double(b), tolerance) << message;
}
inline void Require(bool value, const std::string &message) {
  EXPECT_TRUE(value) << message;
}
} // namespace check
namespace {
void LossChecks() {
  std::vector<float> p_values = {.2f, .8f, .7f, .3f, .6f, .4f, .9f, .1f};
  std::vector<std::int32_t> y_values = {1, 0, 1, 0};
  const Tensor distance = nn::from_vector(std::vector<float>{-2, 0, 1, 3}, {1, 1, 4, 1});
  std::vector<double> values;
  std::vector<std::vector<float>> gradients;
  auto run = [&] {
    Tensor probabilities = nn::from_vector(p_values, {1, 1, 4, 2});
    probabilities.requires_grad_(true);
    const Tensor labels = nn::from_vector(y_values, {1, 1, 4});
    const std::vector<Tensor> losses = {
        seg::SoftDiceLoss(probabilities, labels, 2),
        seg::SoftTverskyLoss(probabilities, labels, 2),
        seg::LovaszSoftmaxLoss(probabilities, labels, 2, 1),
        seg::BoundaryLoss(nn::slice(probabilities, {0, 0, 0, 1}, {1, 1, 4, 1}), distance)};
    values.clear();
    gradients.clear();
    for (const auto &loss : losses) {
      values.push_back(loss.item<double>());
      nn::GradOptions options;
      options.allow_unused =
          true; // a constant (all-void/absent) loss does not depend on the probabilities
      const Tensor g =
          loss.requires_grad() ? nn::grad(loss, {probabilities}, options)[0] : Tensor();
      gradients.push_back(g.defined() ? g.to_vector<float>() : std::vector<float>(8, 0.f));
    }
  };
  run();
  constexpr double epsilon = 1e-6;
  check::Near(values[0], 1 - (1.2 + epsilon) / (1.8 + epsilon), 1e-7,
              "soft Dice overlap value");
  check::Near(values[1], 1 - (1.2 + epsilon) / (1.88 + epsilon), 1e-7,
              "asymmetric Tversky overlap value");
  check::Near(values[2], 5. / 12, 1e-7,
              "Lovasz sorted errors [.6,.3,.2,.1] with Jaccard increments [.5,1/6,1/3,0]");
  check::Near(values[3], -.225, 1e-7, "boundary loss may be negative");
  const double lovasz[] = {-1. / 3, 1. / 6, -.5, 0}, boundary[] = {-.5, 0, .25, .75};
  for (int i = 0; i < 4; ++i) {
    const double target = i % 2 == 0 ? 1 : 0;
    for (int which = 0; which < 2; ++which) {
      const double alpha = which ? .3 : .5, beta = which ? .7 : .5,
                   den = which ? 1.88 + epsilon : 1.8 + epsilon, numerator = 1.2 + epsilon;
      const double derivative =
          -(target * den - numerator * ((1 - beta) * target + alpha * (1 - target))) /
          (den * den);
      check::Near(gradients[std::size_t(which)][std::size_t(2 * i + 1)], derivative, 2e-7,
                  "independent overlap quotient-rule derivative");
    }
    check::Near(gradients[2][std::size_t(2 * i + 1)], lovasz[i], 1e-7,
                "every Lovasz fixed-sort subgradient");
    check::Near(gradients[3][std::size_t(2 * i + 1)], boundary[i], 1e-7,
                "signed-distance boundary derivative");
    for (int k = 0; k < 4; ++k)
      check::Near(gradients[std::size_t(k)][std::size_t(2 * i)], 0, 0,
                  "background omitted from foreground loss");
  }
  y_values.assign(4, 255);
  run();
  for (int k = 0; k < 3; ++k) {
    check::Near(values[std::size_t(k)], 0, 0, "all-void image has zero region loss");
    for (int i = 0; i < 8; ++i)
      check::Near(gradients[std::size_t(k)][std::size_t(i)], 0, 0,
                  "all-void image has zero region derivatives");
  }
  y_values.assign(4, 0);
  run();
  check::Near(values[2], 0, 0, "Lovasz skips absent foreground class");
  y_values = {1, 0, 1, 0};
  p_values = {0, 1, 1, 0, 0, 1, 1, 0};
  run();
  for (int k = 0; k < 3; ++k)
    check::Near(values[std::size_t(k)], 0, 0, "perfect probabilities give zero region loss");
  std::cout << "  Dice/Tversky values and analytic gradients; Lovasz sorted subgradients; "
               "boundary/absent/perfect/void "
               "cases passed"
            << std::endl;
}

void DistanceChecks() {
  std::mt19937 rng(23);
  std::bernoulli_distribution pick(.2);
  for (int trial = 0; trial < 100; ++trial) {
    const int h = 1 + trial % 8, w = 1 + (trial / 8) % 9;
    std::vector<bool> seeds(std::size_t(h * w));
    for (int i = 0; i < h * w; ++i)
      seeds[std::size_t(i)] = trial % 10 ? pick(rng) : false;
    const auto distance =
        seg::SquaredDistance2D(nn::from_vector(seeds, {h, w})).to_vector<double>();
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) {
        double expected = INFINITY;
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            if (seeds[std::size_t(y * w + x)])
              expected = std::min(expected, double((r - y) * (r - y) + (c - x) * (c - x)));
        if (std::isfinite(expected))
          check::Near(distance[std::size_t(r * w + c)], expected, 1e-12,
                      "tensor EDT equals exhaustive nearest site");
        else
          check::Require(std::isinf(distance[std::size_t(r * w + c)]),
                         "no-seed EDT is infinite");
      }
  }
  std::vector<std::int32_t> labels(5 * 7, 0);
  for (int r = 1; r < 4; ++r)
    for (int c = 2; c < 5; ++c)
      labels[std::size_t(r * 7 + c)] = 1;
  auto distance =
      seg::SignedBoundaryDistances(nn::from_vector(labels, {1, 5, 7}), 2).to_vector<float>();
  check::Near(distance[2 * 7 + 3], -1, 0, "interior distance uses -(EDT-1)");
  check::Near(distance[1 * 7 + 2], 0, 0, "foreground boundary distance zero");
  check::Near(distance[0], std::sqrt(5.), 1e-7, "exterior Euclidean distance");
  check::Near(distance[2 * 7 + 0], 2, 0, "exterior horizontal distance");
  for (int fill : {0, 1}) {
    distance = seg::SignedBoundaryDistances(
                   nn::from_vector(std::vector<std::int32_t>(35, fill), {1, 5, 7}), 2)
                   .to_vector<float>();
    for (float v : distance)
      check::Near(v, 0, 0,
                  fill ? "full foreground has no internal boundary"
                       : "empty foreground boundary map zero");
  }
  std::cout << "  100 exact EDT/exhaustive rectangular comparisons and signed "
               "interior/boundary/exterior/empty/full maps "
               "passed"
            << std::endl;
}

} // namespace
TEST(NnSegmentationLosses, IndependentQuotientRuleSortedSubgradientsVoidPerfectAbsent) {
  LossChecks();
}
TEST(NnSegmentationLosses, HundredRectangularExactDistanceOraclesAndSignedMaps) {
  DistanceChecks();
}
TEST(NnSegmentationLosses, BoundaryMapsDetachedAndProbabilityDerivative) {
  const auto
      p = nn::from_vector(std::vector<double>{.1, .2, .3, .4}, {1, 2, 2, 1}).requires_grad_(),
      distance =
          nn::from_vector(std::vector<double>{-2, 0, 1, 3}, {1, 2, 2, 1}).requires_grad_();
  const auto loss = seg::BoundaryLoss(p, distance);
  nn::GradOptions o;
  o.allow_unused = true;
  const auto g = nn::grad(loss, {p, distance}, o);
  EXPECT_EQ(g[0].to_vector<double>(), std::vector<double>({-.5, 0, .25, .75}));
  EXPECT_FALSE(g[1].defined());
}
TEST(NnSegmentationLosses, OneDimensionalEnvelopeExactWithInfiniteAndNegativeCosts) {
  const auto input = nn::from_vector(std::vector<double>{INFINITY, 2, -1, INFINITY, 4});
  const auto result = seg::SquaredDistance1D(input).to_vector<double>();
  for (int i = 0; i < 5; ++i) {
    double expected = INFINITY;
    for (int j = 0; j < 5; ++j)
      expected = std::min(expected, input.at({j}) + (i - j) * (i - j));
    EXPECT_EQ(result[i], expected);
  }
  EXPECT_EQ(seg::SquaredDistance1D(nn::zeros({0})).numel(), 0);
}
TEST(NnSegmentationLosses, InvalidShapesDtypesAndLossSettingsReject) {
  EXPECT_THROW(seg::SoftDiceLoss(nn::zeros({1, 2}), nn::zeros({1}), 2), std::invalid_argument);
  EXPECT_THROW(
      seg::SoftTverskyLoss(nn::zeros({1, 2}), nn::zeros({1}, nn::DType::Int64), 2, -.1),
      std::invalid_argument);
  EXPECT_THROW(seg::SquaredDistance1D(nn::full({1}, -INFINITY)), std::invalid_argument);
  EXPECT_THROW(seg::SquaredDistance2D(nn::ones({2, 2})), std::invalid_argument);
}
