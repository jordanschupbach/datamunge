
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
m::WordTree Digits() {
  return m::WordTree({-1, -1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
}
TEST(NnYoloV2, WordTreeSiblingMarginalsCoarseFineEveryDerivativeAndWideCapacity) {
  const auto tree = Digits();
  const auto x = n::zeros({1, 12}, n::DType::Float64).requires_grad_();
  for (int label : {0, 2}) {
    const auto out = m::WordTreeProbabilities(x, tree);
    const auto loss =
        n::mean(m::WordTreeLoss(out, n::full({1}, label, n::DType::Int64), tree));
    EXPECT_NEAR(loss.item(), std::log(label ? 10. : 2.), 1e-12);
    const auto g = n::grad(loss, {x})[0];
    for (int node = 0; node < 12; ++node) {
      EXPECT_NEAR(out.conditional.at({0, node}), node < 2 ? .5 : .2, 1e-12);
      EXPECT_NEAR(out.marginal.at({0, node}), node < 2 ? .5 : .1, 1e-12);
      const double expected = node == 0       ? -.5
                              : node == 1     ? .5
                              : label == 0    ? 0
                              : node == 2     ? -.8
                              : node % 2 == 0 ? .2
                                              : 0;
      EXPECT_NEAR(g.at({0, node}), expected, 1e-12);
    }
  }
  const m::WordTree wide(std::vector<int>(9418, -1));
  const auto p = m::WordTreeProbabilities(n::zeros({1, 9418}), wide);
  EXPECT_NEAR(m::WordTreeLoss(p, n::full({1}, 9417., n::DType::Int64), wide).item(),
              std::log(9418.), 2e-6);
}
TEST(NnYoloV2, PriorClusteringKnownPopulationsDuplicateEmptyClusterAndShapeIoU) {
  const auto x = n::tensor({2., 4., 8., 4., 2., 4., 8., 4., 2., 4., 8., 4.}, {6, 2});
  EXPECT_EQ(m::ClusterYoloDimensions(x, 2).to_vector<int>(), std::vector<int>({2, 4, 8, 4}));
  EXPECT_EQ(m::YoloDimensionIoU(n::tensor({2., 4.}), n::tensor({8., 4.})).item(), .25);
  EXPECT_EQ(m::ClusterYoloDimensions(n::tensor({2., 4., 2., 4.}, {2, 2}), 2).to_vector<int>(),
            std::vector<int>({2, 4, 2, 4}));
  const auto mean = m::ClusterYoloDimensions(
      n::from_vector(std::vector<double>{1, 2, 2, 4, 3, 6}, {3, 2}), 1);
  EXPECT_EQ(mean.to_vector<int>(), std::vector<int>({2, 4}));
}
TEST(NnYoloV2, RectangularCanvasTargetsLargestCollisionIgnoredOverlapEmptyImageAndCurrentIou) {
  m::TensorDetectionScenes data;
  data.images = n::zeros({2, 40, 80, 1});
  data.boxes = {n::tensor({7., 42., 23., 58., 5., 40., 25., 60.}, {2, 4}), n::zeros({0, 4})};
  data.labels = {n::from_vector(std::vector<std::int64_t>{3, 4}),
                 n::zeros({0}, n::DType::Int64)};
  const auto prior = n::tensor({1., 1., 2., 1.}, {2, 2});
  std::vector<float> values(2 * 4 * 4 * 2 * 15);
  const int base = ((1 * 4 + 2) * 2) * 15;
  values[base + 2] = std::log(2.f);
  const auto raw = n::from_vector(values, {2, 4, 4, 2, 15});
  for (bool coarse : {false, true}) {
    const auto t = m::MakeYoloV2Targets(raw, prior, data, {0, 1}, coarse);
    EXPECT_EQ(t.collisions, 1);
    EXPECT_EQ(t.labels.at({0, 1, 2, 1}), coarse ? 0 : 4);
    EXPECT_EQ(n::sum(t.positive).item(), 1);
    EXPECT_NEAR(t.geometry.at({0, 1, 2, 1, 0}), .5, 1e-6);
    EXPECT_NEAR(t.geometry.at({0, 1, 2, 1, 1}), .5, 1e-6);
    EXPECT_NEAR(t.geometry.at({0, 1, 2, 1, 2}), 0, 1e-6);
    EXPECT_NEAR(t.geometry.at({0, 1, 2, 1, 3}), 0, 1e-6);
    EXPECT_NEAR(t.confidence.at({0, 1, 2, 1}), 1, 1e-6);
    EXPECT_EQ(t.weights.at({0, 1, 2, 1}), 1);
    EXPECT_EQ(t.weights.at({0, 1, 2, 0}), 0);
    EXPECT_EQ(t.weights.at({1, 0, 0, 0}), .5);
    EXPECT_FALSE(t.confidence.requires_grad());
  }
  values[base + 15 + 3] = std::log(.5f);
  const auto changed =
      m::MakeYoloV2Targets(n::from_vector(values, {2, 4, 4, 2, 15}), prior, data, {0, 1});
  EXPECT_NEAR(changed.confidence.at({0, 1, 2, 1}), .5, 1e-6);
  EXPECT_EQ(changed.geometry.at({0, 1, 2, 1, 3}), 0);
}
TEST(NnYoloV2, AllSevenAnalyticDetectionDerivativesPositiveOffsetBackgroundIgnored) {
  const auto type = n::DType::Float64;
  const auto raw = n::zeros({1, 1, 1, 1, 7}, type).requires_grad_();
  m::YoloV2Targets t{n::from_vector(std::vector<double>{.5, .5, 0, 0}, {1, 1, 1, 1, 4}),
                     n::ones({1, 1, 1, 1}, type),
                     n::ones({1, 1, 1, 1}, type),
                     n::ones({1, 1, 1, 1}, type),
                     n::ones({1, 1, 1, 1}, n::DType::Int64),
                     0};
  const auto check = [&](double expected, const std::vector<double> &gradient) {
    const auto loss = m::YoloV2Loss(raw, t, 2, {});
    EXPECT_NEAR(loss.item(), expected, 1e-12);
    const auto g = n::grad(loss, {raw})[0].to_vector<double>();
    for (int k = 0; k < 7; ++k)
      EXPECT_NEAR(g[k], gradient[k], 1e-12);
  };
  check(.25 + std::log(2.), {0, 0, 0, 0, -.25, .5, -.5});
  t.geometry = n::from_vector(std::vector<double>{.25, .75, .2, -.3}, {1, 1, 1, 1, 4});
  check(1.525 + std::log(2.), {.625, -.625, -2, 3, -.25, .5, -.5});
  t.positive = n::zeros({1, 1, 1, 1}, type);
  t.confidence = n::zeros({1, 1, 1, 1}, type);
  t.weights = n::full({1, 1, 1, 1}, .5, type);
  check(.125, {0, 0, 0, 0, .125, 0, 0});
  t.weights = n::zeros({1, 1, 1, 1}, type);
  check(0, {0, 0, 0, 0, 0, 0, 0});
}
TEST(NnYoloV2, ClassificationOnlyJointSelectionEveryDerivativeAndWeakObjectnessFloor) {
  const auto type = n::DType::Float64;
  const auto tree = Digits();
  std::vector<double> x(34);
  x[22] = std::log(2.);
  m::YoloV2Targets t{n::zeros({1, 1, 2, 1, 4}, type),
                     n::zeros({1, 1, 2, 1}, type),
                     n::zeros({1, 1, 2, 1}, type),
                     n::zeros({1, 1, 2, 1}, type),
                     n::zeros({1, 1, 2, 1}, n::DType::Int64),
                     0};
  m::YoloV2Weights w;
  w.detection = 0;
  w.classification = 1;
  w.classification_labels = n::full({1}, 2., n::DType::Int64);
  const auto check = [&](double expected, int selected, double root_gradient,
                         double weak_gradient) {
    const auto raw = n::from_vector(x, {1, 1, 2, 1, 17}).requires_grad_();
    const auto loss = m::YoloV2Loss(raw, t, 12, w, &tree);
    EXPECT_NEAR(loss.item(), expected, 1e-12);
    const auto g = n::grad(loss, {raw})[0].to_vector<double>();
    for (int a = 0; a < 2; ++a)
      for (int c = 0; c < 17; ++c) {
        double derivative = 0;
        if (a == selected && c >= 5) {
          const int node = c - 5;
          derivative = node == 0       ? root_gradient
                       : node == 1     ? -root_gradient
                       : node == 2     ? -.8
                       : node % 2 == 0 ? .2
                                       : 0;
        }
        if (a == selected && c == 4)
          derivative = weak_gradient;
        EXPECT_NEAR(g[a * 17 + c], derivative, 1e-12);
      }
  };
  check(std::log(7.5), 1, -1. / 3, 0);
  x[4] = std::log(.9 / .1);
  x[21] = std::log(.1 / .9);
  check(std::log(10.), 0, -.5, 0);
  x[4] = std::log(.1 / .9);
  x[21] = std::log(.2 / .8);
  check(std::log(7.5) + .01, 1, -1. / 3, -.032);
}
TEST(NnYoloV2, DecodeBoxesClassScoresHierarchyLeavesAndNoGradSuppression) {
  const auto raw = n::zeros({1, 1, 1, 2, 15}), prior = n::full({2, 2}, .5);
  const auto out = m::DecodeYoloV2TensorDetections(raw, prior, m::YoloV2Probabilities(raw, 10),
                                                   n::full({1}, 7., n::DType::Int64), 16, 24);
  EXPECT_EQ(out.scores.numel(), 10);
  for (int c = 0; c < 10; ++c) {
    EXPECT_EQ(out.labels.at({c}), c);
    EXPECT_NEAR(out.scores.at({c}), .05, 1e-7);
  }
  EXPECT_EQ(out.boxes.at({0, 0}), 4);
  EXPECT_EQ(out.boxes.at({0, 1}), 6);
  EXPECT_FALSE(out.boxes.requires_grad());
  const auto tree = Digits();
  const auto joint = n::zeros({1, 1, 1, 1, 17});
  const auto fine = m::DecodeYoloV2TensorDetections(
      joint, n::full({1, 2}, .5), m::YoloV2Probabilities(joint, 12, &tree),
      n::zeros({1}, n::DType::Int64), 16, 24, true);
  EXPECT_EQ(fine.labels.to_vector<int>(), std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}
TEST(NnYoloV2, FullV2And9418NodeHeadsEveryGradientCountsAndCompactMultiscale) {
  n::set_num_threads(2);
  for (bool large : {false, true}) {
    n::ParameterStore p(17), state(17);
    m::YoloV2Config c;
    c.update_stats = false;
    if (large) {
      c.classes = 9418;
      c.anchors = 3;
    }
    const auto net = m::YoloV2(
        p, state, n::sin(.13 * n::arange(0, 128 * 128 * 3)).reshape({1, 128, 128, 3}), c,
        true);
    EXPECT_EQ(p.count(), large ? 79502989 : 50655389);
    EXPECT_EQ(state.count(), 20672);
    const auto g = n::grad(n::mean(n::square(net.raw)), p.parameters());
    for (const auto &v : g) {
      EXPECT_TRUE(v.defined());
      EXPECT_TRUE(std::isfinite(n::sum(n::square(v)).item()));
    }
    std::cout << "full " << c.classes << "-class tensors=" << g.size() << '\n';
  }
  n::ParameterStore p(17), state(17);
  m::YoloV2Config c;
  c.classes = 10;
  c.compact = true;
  c.width_divisor = 8;
  c.update_stats = false;
  for (int side : {64, 80, 96}) {
    const auto net = m::YoloV2(
        p, state, n::sin(.13 * n::arange(0, 2 * side * side)).reshape({2, side, side, 1}), c,
        true);
    EXPECT_EQ(net.raw.shape(), n::Shape({2, side / 8, side / 8, 5, 15}));
    EXPECT_EQ(net.passthrough.shape(), n::Shape({2, side / 8, side / 8, 32}));
    if (side == 64)
      for (const auto &v : n::grad(n::mean(n::square(net.raw)), p.parameters())) {
        const double norm = n::sum(n::square(v)).item();
        EXPECT_TRUE(std::isfinite(norm));
        EXPECT_GT(norm, 0);
      }
  }
  std::cout << "compact learned=" << p.count() << " state=" << state.count() << '\n';
}
TEST(NnYoloV2, InvalidHierarchyLabelsPriorsAndShapesRejected) {
  EXPECT_THROW(m::WordTree({-1, 1}), std::invalid_argument);
  EXPECT_THROW(m::WordTree({}), std::invalid_argument);
  const auto tree = Digits();
  EXPECT_THROW(m::WordTreeProbabilities(n::zeros({1, 11}), tree), std::invalid_argument);
  const auto p = m::WordTreeProbabilities(n::zeros({1, 12}), tree);
  EXPECT_THROW(m::WordTreeLoss(p, n::full({1}, 12., n::DType::Int64), tree),
               std::invalid_argument);
  EXPECT_THROW(m::ClusterYoloDimensions(n::zeros({3, 2}), 2), std::invalid_argument);
  EXPECT_THROW(m::DecodeYoloV2Boxes(n::zeros({1, 2, 2, 3, 15}), n::zeros({3, 2}), 64, 64),
               std::invalid_argument);
}
