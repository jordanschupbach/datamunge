
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnRetinaDetection, AnchorsMatchIndependentAllCoordinateOracle) {
  const std::vector<int> cells = {16, 8, 4, 2, 1};
  const auto a = m::RetinaAnchors(cells, 16, n::DType::Float64);
  EXPECT_EQ(a.shape(), n::Shape({3069, 4}));
  std::int64_t j = 0;
  for (int l = 0; l < 5; ++l)
    for (int y = 0; y < cells[l]; ++y)
      for (int x = 0; x < cells[l]; ++x)
        for (double scale : {1., std::cbrt(2.), std::cbrt(4.)})
          for (double ratio : {.5, 1., 2.}) {
            const double side = (16 << l) * scale, h = side * std::sqrt(ratio),
                         w = side / std::sqrt(ratio), cy = (y + .5) * (8 << l),
                         cx = (x + .5) * (8 << l);
            const double expected[] = {cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2};
            for (int k = 0; k < 4; ++k)
              EXPECT_NEAR(a.at({j, k}), expected[k], 1e-12);
            ++j;
          }
  EXPECT_EQ(m::RetinaAnchors({80, 40, 20, 10, 5}).shape()[0], 76725);
  EXPECT_FALSE(a.requires_grad());
}
TEST(NnRetinaDetection, FocalEveryValueAndGradientMatchIndependentCentralDifferences) {
  const auto reference = [](double x, double y, double gamma) {
    const auto sp = [](double z) {
      return std::max(z, 0.) + std::log1p(std::exp(-std::abs(z)));
    };
    return (y ? .25 : .75) * std::exp(-gamma * sp((2 * y - 1) * x)) *
           (std::max(x, 0.) - x * y + std::log1p(std::exp(-std::abs(x))));
  };
  for (double gamma : {0., .5, 2.})
    for (double x : {-80., -10., -2., 0., 2., 10., 80.})
      for (double y : {0., 1.}) {
        const auto v = n::full({}, x, n::DType::Float64).requires_grad_(),
                   t = n::full({}, y, n::DType::Float64),
                   loss = m::SigmoidFocalLoss(v, t, .25, gamma);
        EXPECT_NEAR(loss.item(), reference(x, y, gamma), 1e-12);
        const double eps = 1e-4,
                     expected = (reference(x + eps, y, gamma) - reference(x - eps, y, gamma)) /
                                (2 * eps);
        EXPECT_NEAR(n::grad(loss, {v})[0].item(), expected, 1e-9);
      }
}
TEST(NnRetinaDetection, LossAllDerivativesPositiveBackgroundIgnoredAndEmpty) {
  const auto type = n::DType::Float64;
  auto x = n::zeros({1, 3, 2}, type).requires_grad_(),
       d = n::from_vector(std::vector<double>{.05, .2, 0, -.2, .3, .4, .5, .6, .7, .8, .9, 1},
                          {1, 3, 4})
               .requires_grad_();
  m::RetinaTargets t{n::from_vector(std::vector<double>{1, 0, 0, 0, 0, 0}, {1, 3, 2}),
                     n::from_vector(std::vector<double>{1, 0, 0}, {1, 3}),
                     n::from_vector(std::vector<double>{1, 1, 0}, {1, 3}),
                     n::zeros({1, 3, 4}, type)};
  const auto loss = m::RetinaLoss(x, d, t);
  EXPECT_NEAR(loss.item(), .625 * std::log(2.) + .5 * .05 * .05 * 9 + 2 * (.2 - .5 / 9),
              1e-12);
  const auto g = n::grad(loss, {x, d});
  for (int i = 0; i < 6; ++i)
    EXPECT_NEAR(g[0].reshape({6}).at({i}),
                i == 0  ? -.25 * (.125 + .25 * std::log(2.))
                : i < 4 ? .75 * (.125 + .25 * std::log(2.))
                        : 0.,
                1e-12);
  const double expected[] = {.45, 1, 0, -1};
  for (int i = 0; i < 12; ++i)
    EXPECT_NEAR(g[1].reshape({12}).at({i}), i < 4 ? expected[i] : 0., 1e-12);
  t.classes = n::zeros({1, 3, 2}, type);
  t.positive = n::zeros({1, 3}, type);
  EXPECT_NEAR(m::RetinaLoss(x, d, t).item(), .75 * std::log(2.), 1e-12);
  t.weights = n::zeros({1, 3}, type);
  const auto ignored = m::RetinaLoss(x, d, t);
  EXPECT_EQ(ignored.item(), 0);
  for (const auto &v : n::grad(ignored, {x, d}))
    EXPECT_EQ(n::count_nonzero(v).item(), 0);
}
TEST(NnRetinaDetection, TargetsPreserveIgnoredBandForcedCollisionAndUnitOffsets) {
  m::TensorDetectionScenes data;
  data.images = n::zeros({2, 64, 64, 1});
  data.boxes = {n::tensor({0., 0., 10., 10., 0., 0., 10., 10.}, {2, 4}), n::zeros({0, 4})};
  data.labels = {n::from_vector(std::vector<std::int64_t>{0, 1}),
                 n::zeros({0}, n::DType::Int64)};
  const auto a = n::tensor(
      {0., 0., 10., 10., 0., 1., 10., 11., 0., 0., 10., 22., 30., 30., 40., 40.}, {4, 4});
  const auto t = m::MakeRetinaTargets(a, data, {0, 1}, 2);
  EXPECT_EQ(t.positive.to_vector<int>(), std::vector<int>({1, 1, 0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(t.weights.to_vector<int>(), std::vector<int>({1, 1, 0, 1, 1, 1, 1, 1}));
  EXPECT_EQ(n::sum(t.classes).item(), 2);
  const auto restored = n::decode_boxes(
      n::slice_axis(a, 0, 0, 2),
      n::slice_axis(n::slice_axis(t.boxes, 0, 0, 1).reshape({4, 4}), 0, 0, 2), {1, 1, 1, 1});
  EXPECT_LT(n::max(n::abs(restored - data.boxes[0])).item(), 1e-5);
}
TEST(NnRetinaDetection, IndependentSigmoidClassesUnitCodingClipAndRawNms) {
  const auto a = n::tensor({0., 0., 10., 10., 0., 0., 10., 10.}, {2, 4}),
             p = n::tensor({.9, .8, .7, .01}, {1, 2, 2});
  const auto raw = m::DecodeRetinaDetections(p, n::zeros({1, 2, 4}), a,
                                             n::zeros({1}, n::DType::Int64), 64, 64, false);
  EXPECT_EQ(raw.scores.shape()[0], 3);
  const auto out = m::DecodeRetinaDetections(p, n::zeros({1, 2, 4}), a,
                                             n::zeros({1}, n::DType::Int64), 64, 64);
  EXPECT_EQ(out.labels.to_vector<int>(), std::vector<int>({0, 1}));
  EXPECT_NEAR(out.scores.at({0}), .9, 1e-6);
  EXPECT_EQ(out.boxes.to_vector<int>(), std::vector<int>({0, 0, 10, 10, 0, 0, 10, 10}));
  const auto d = n::tensor({.2, .1, 0., 0.}, {1, 1, 4});
  const auto shifted =
      m::DecodeRetinaDetections(n::full({1, 1, 1}, .9), d, n::slice_axis(a, 0, 0, 1),
                                n::zeros({1}, n::DType::Int64), 64, 64);
  EXPECT_LT(n::max(n::abs(shifted.boxes - n::tensor({2., 1., 12., 11.}, {1, 4}))).item(),
            1e-5);
}
TEST(NnRetinaDetection, CompactEveryLearnedGradientCountsAndPairedStatePolicy) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  m::RetinaNetConfig c;
  c.classes = 10;
  c.image_size = 64;
  c.backbone_depth = 18;
  c.width_divisor = 8;
  c.pyramid_width = 32;
  c.grayscale = true;
  c.update_stats = false;
  const auto x = n::sin(.13 * n::arange(0, 2 * 64 * 64)).reshape({2, 64, 64, 1});
  const auto net = m::RetinaNet(p, state, x, c, true);
  EXPECT_EQ(p.count(), 346246);
  EXPECT_EQ(state.count(), 1200);
  EXPECT_EQ(net.anchors, 774);
  const auto before = state.at(state.names()[0]).clone();
  for (const auto &g : n::grad(n::mean(n::square(net.logits)) + n::mean(n::square(net.deltas)),
                               p.parameters())) {
    EXPECT_TRUE(g.defined());
    EXPECT_TRUE(std::isfinite(n::sum(n::square(g)).item()));
    EXPECT_GT(n::sum(n::square(g)).item(), 0);
  }
  EXPECT_EQ(n::max(n::abs(state.at(state.names()[0]) - before)).item(), 0);
  c.update_stats = true;
  {
    n::NoGradGuard guard;
    m::RetinaNet(p, state, x, c, true);
  }
  double change = 0;
  for (const auto &name : state.names())
    change += n::sum(n::abs(state.at(name))).item();
  EXPECT_GT(change, 0);
}
TEST(NnRetinaDetection, FullWeightsAll195GradientsAndC5P6Count) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  m::RetinaNetConfig c;
  c.image_size = 128;
  c.update_stats = false;
  const auto net = m::RetinaNet(
      p, state, n::sin(.13 * n::arange(0, 128 * 128 * 3)).reshape({1, 128, 128, 3}), c, true);
  EXPECT_EQ(p.count(), 37968692);
  EXPECT_EQ(state.count(), 53120);
  EXPECT_EQ(net.anchors, 3069);
  const auto g =
      n::grad(n::mean(n::square(net.logits)) + n::mean(n::square(net.deltas)), p.parameters());
  EXPECT_EQ(g.size(), 195);
  for (const auto &v : g) {
    EXPECT_TRUE(v.defined());
    EXPECT_TRUE(std::isfinite(n::sum(n::square(v)).item()));
  }
}
TEST(NnRetinaDetection, InvalidDtypesClassesShapesAndFocalTargets) {
  EXPECT_THROW(m::RetinaAnchors({1, 1}, 8), std::invalid_argument);
  EXPECT_THROW(m::SigmoidFocalLoss(n::ones({1}), n::full({1}, .5)), std::invalid_argument);
  EXPECT_THROW(m::SigmoidFocalLoss(n::ones({1}), n::ones({1}), .25, -1),
               std::invalid_argument);
  n::ParameterStore p(17), state(17);
  m::RetinaNetConfig c;
  c.classes = 0;
  EXPECT_THROW(m::RetinaNet(p, state, n::zeros({1, 640, 640, 3}), c), std::invalid_argument);
}
