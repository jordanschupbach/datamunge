
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnYoloV3, AllMultiLabelCoordinateBackgroundIgnoredAnalyticDerivatives) {
  const auto type = n::DType::Float64;
  std::array<n::Tensor, 3> raw;
  m::YoloV3Targets t;
  for (int l = 0; l < 3; ++l) {
    raw[l] = n::zeros({1, 1, 1, 3, 7}, type).requires_grad_();
    t.levels[l] = {n::zeros({1, 1, 1, 3, 4}, type), n::zeros({1, 1, 1, 3, 2}, type),
                   n::zeros({1, 1, 1, 3}, type),    n::zeros({1, 1, 1, 3}, type),
                   n::zeros({1, 1, 1, 3}, type),    n::zeros({1, 1, 1, 3}, type)};
  }
  const auto positive = n::from_vector(std::vector<double>{1, 0, 0}, {1, 1, 1, 3});
  auto &v = t.levels[0];
  v.geometry = n::from_vector(std::vector<double>{.5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                              {1, 1, 1, 3, 4});
  v.classes = n::from_vector(std::vector<double>{1, 1, 0, 0, 0, 0}, {1, 1, 1, 3, 2});
  v.positive = v.objectness_weight = v.geometry_weight = positive;
  const auto check = [&](double expected, const std::vector<double> &derivative) {
    const auto loss = m::YoloV3Loss(raw, t, 2);
    EXPECT_NEAR(loss.item(), expected, 1e-12);
    const auto g = n::grad(loss, {raw[0], raw[1], raw[2]});
    for (int l = 0; l < 3; ++l)
      for (int i = 0; i < 21; ++i)
        EXPECT_NEAR(g[l].reshape({21}).at({i}), l == 0 && i < 7 ? derivative[i] : 0, 1e-12);
  };
  check(3 * std::log(2.), {0, 0, 0, 0, -.5, -.5, -.5});
  v.geometry = n::from_vector(std::vector<double>{.25, .75, .2, -.3, 0, 0, 0, 0, 0, 0, 0, 0},
                              {1, 1, 1, 3, 4});
  v.geometry_weight = positive * 2.;
  check(3 * std::log(2.) + .255, {.125, -.125, -.4, .6, -.5, -.5, -.5});
  v.positive = n::zeros({1, 1, 1, 3}, type);
  v.geometry_weight = v.positive;
  check(std::log(2.), {0, 0, 0, 0, .5, 0, 0});
  v.objectness_weight = v.positive;
  check(0, {0, 0, 0, 0, 0, 0, 0});
}
TEST(NnYoloV3, LogisticZeroSoftTargetAndExtremeValuesHaveCorrectDerivatives) {
  const auto x = n::from_vector(std::vector<double>{0, 0, 0, -80, 80}).requires_grad_(),
             target = n::from_vector(std::vector<double>{0, .25, 1, 0, 1});
  const auto loss = m::YoloLogisticCrossEntropy(x, target);
  const auto g = n::grad(n::sum(loss), {x})[0].to_vector<double>();
  EXPECT_EQ(g[0], .5);
  EXPECT_EQ(g[1], .25);
  EXPECT_EQ(g[2], -.5);
  EXPECT_NEAR(g[3], 0, 1e-30);
  EXPECT_NEAR(g[4], 0, 1e-30);
  EXPECT_TRUE(std::isfinite(n::sum(loss).item()));
}
TEST(NnYoloV3, OneGlobalPriorPerObjectCrossScaleIgnoreAreaWeightAndFirstCollisionTies) {
  std::array<n::Tensor, 3> raw;
  for (int l = 0; l < 3; ++l)
    raw[l] = n::zeros({1, 2 << l, 2 << l, 3, 8});
  const auto priors = n::tensor(
      {4., 4., 6., 6., 8., 8., 12., 12., 16., 16., 20., 20., 24., 24., 32., 32., 48., 48.},
      {9, 2});
  m::TensorDetectionScenes data;
  data.images = n::zeros({1, 64, 64, 1});
  data.boxes = {n::tensor({24., 24., 32., 32., 16., 16., 32., 32., 0., 0., 32., 32.}, {3, 4})};
  data.labels = {n::from_vector(std::vector<std::int64_t>{0, 1, 2})};
  const auto t = m::MakeYoloV3Targets(raw, priors, data, {0});
  EXPECT_EQ(t.collisions, 0);
  int ignored = 0;
  for (int l = 0; l < 3; ++l) {
    const auto &g = t.levels[l];
    EXPECT_EQ(n::sum(g.positive).item(), 1);
    ignored += n::count_nonzero(
                   n::logical_and(n::equal(g.positive, 0.), n::equal(g.objectness_weight, 0.)))
                   .item<int>();
    const double side = 32 >> l, area = side * side;
    EXPECT_NEAR(n::sum(g.areas).item(), area, 1e-5);
    EXPECT_NEAR(n::sum(g.geometry_weight).item(), 2 - area / 4096., 1e-6);
    EXPECT_EQ(n::argmax(n::sum(g.classes, {0, 1, 2, 3})).item(), 2 - l);
    EXPECT_FALSE(g.geometry.requires_grad());
  }
  EXPECT_GE(ignored, 3);
  data.boxes[0] = n::concat({data.boxes[0], n::tensor({24., 24., 32., 32.}, {1, 4})}, 0);
  data.labels[0] = n::from_vector(std::vector<std::int64_t>{0, 1, 2, 1});
  const auto collision = m::MakeYoloV3Targets(raw, priors, data, {0});
  EXPECT_EQ(collision.collisions, 1);
  EXPECT_EQ(n::sum(n::slice_axis(collision.levels[2].classes, 4, 0, 1)).item(), 1);
}
TEST(NnYoloV3, EmptyTargetsRetainAllHeadObjectnessAndNoGeometryClassGradients) {
  std::array<n::Tensor, 3> raw;
  for (int l = 0; l < 3; ++l)
    raw[l] = n::zeros({1, 2 << l, 2 << l, 3, 7}).requires_grad_();
  m::TensorDetectionScenes data;
  data.images = n::zeros({1, 64, 64, 1});
  data.boxes = {n::zeros({0, 4})};
  data.labels = {n::zeros({0}, n::DType::Int64)};
  const auto t = m::MakeYoloV3Targets(raw, n::ones({9, 2}), data, {0});
  const auto loss = m::YoloV3Loss(raw, t, 2);
  EXPECT_NEAR(loss.item(), 252 * std::log(2.), 2e-5);
  const auto grads = n::grad(loss, {raw[0], raw[1], raw[2]});
  for (int l = 0; l < 3; ++l) {
    EXPECT_EQ(n::max(n::abs(n::slice_axis(grads[l], 4, 0, 4))).item(), 0);
    EXPECT_EQ(n::max(n::abs(n::slice_axis(grads[l], 4, 5, 7))).item(), 0);
    EXPECT_EQ(n::min(n::slice_axis(grads[l], 4, 4, 5)).item(), .5);
  }
}
TEST(NnYoloV3, DecodeCombinesHeadsIndependentClassesAndPerImageCap) {
  std::array<n::Tensor, 3> raw;
  for (int l = 0; l < 3; ++l)
    raw[l] = n::zeros({2, 2 << l, 2 << l, 3, 7});
  const auto images = n::from_vector(std::vector<std::int64_t>{7, 2}),
             priors = n::full({9, 2}, 16.);
  const auto candidates = m::DecodeYoloV3Detections(raw, priors, images, 64, 64, false);
  EXPECT_EQ(candidates.scores.numel(), 1008);
  EXPECT_NEAR(n::min(candidates.scores).item(), .25, 1e-7);
  const auto out = m::DecodeYoloV3Detections(raw, priors, images, 64, 64);
  EXPECT_EQ(out.scores.numel(), 200);
  EXPECT_EQ(n::count_nonzero(n::equal(out.images, 2.)).item(), 100);
  EXPECT_EQ(n::count_nonzero(n::equal(out.images, 7.)).item(), 100);
  EXPECT_FALSE(out.boxes.requires_grad());
}
TEST(NnYoloV3, Full80And20ClassEvery222TensorGradientCountsAndCompactMultiscale) {
  n::set_num_threads(2);
  for (int classes : {80, 20}) {
    n::ParameterStore p(17), state(17);
    m::YoloV3Config c;
    c.classes = classes;
    c.update_stats = false;
    const auto net = m::YoloV3(
        p, state, n::sin(.13 * n::arange(0, 128 * 128 * 3)).reshape({1, 128, 128, 3}), c,
        true);
    EXPECT_EQ(p.count(), classes == 80 ? 61949149 : 61626049);
    EXPECT_EQ(state.count(), 52608);
    const auto objective = n::mean(n::square(net.raw[0])) + n::mean(n::square(net.raw[1])) +
                           n::mean(n::square(net.raw[2]));
    const auto g = n::grad(objective, p.parameters());
    EXPECT_EQ(g.size(), 222);
    for (const auto &v : g) {
      EXPECT_TRUE(v.defined());
      EXPECT_TRUE(std::isfinite(n::sum(n::square(v)).item()));
    }
  }
  n::ParameterStore p(17), state(17);
  m::YoloV3Config c;
  c.classes = 10;
  c.width_divisor = 8;
  c.grayscale = true;
  c.update_stats = false;
  for (int side : {64, 96, 128}) {
    const auto net = m::YoloV3(
        p, state, n::sin(.13 * n::arange(0, 2 * side * side)).reshape({2, side, side, 1}), c,
        true);
    for (int l = 0; l < 3; ++l)
      EXPECT_EQ(net.raw[l].shape(), n::Shape({2, side / (32 >> l), side / (32 >> l), 3, 15}));
    if (side == 64)
      for (const auto &v :
           n::grad(n::mean(n::square(net.raw[0])) + n::mean(n::square(net.raw[1])) +
                       n::mean(n::square(net.raw[2])),
                   p.parameters())) {
        const double norm = n::sum(n::square(v)).item();
        EXPECT_TRUE(std::isfinite(norm));
        EXPECT_GT(norm, 0);
      }
  }
  std::cout << "compact YOLO v3 learned=" << p.count() << " state=" << state.count() << '\n';
}
TEST(NnYoloV3, InvalidTargetsHeadsAndLogisticLabelsAreRejected) {
  EXPECT_THROW(m::YoloLogisticCrossEntropy(n::zeros({1}), n::full({1}, 1.1)),
               std::invalid_argument);
  EXPECT_THROW(m::DecodeYoloV3LevelDetections(n::zeros({1, 1}), 0, n::ones({9, 2}),
                                              n::zeros({1}, n::DType::Int64), 64, 64),
               std::invalid_argument);
  std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 7}), n::zeros({1, 3, 3, 3, 7}),
                                  n::zeros({1, 8, 8, 3, 7})};
  EXPECT_THROW(
      m::DecodeYoloV3Detections(raw, n::ones({9, 2}), n::zeros({1}, n::DType::Int64), 64, 64),
      std::invalid_argument);
}
