
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnFcosDetection, AllPointCoordinatesRangesAndCenternessMatchIndependentOracle) {
  const std::vector<int> grids = {16, 8, 4, 2, 1};
  const auto p = m::FcosPoints(grids, .25, n::DType::Float64);
  EXPECT_EQ(p.shape(), n::Shape({341, 5}));
  std::int64_t j = 0;
  for (int l = 0; l < 5; ++l)
    for (int r = 0; r < grids[l]; ++r)
      for (int c = 0; c < grids[l]; ++c) {
        EXPECT_EQ(p.at({j, 0}), (r + .5) * (8 << l));
        EXPECT_EQ(p.at({j, 1}), (c + .5) * (8 << l));
        EXPECT_EQ(p.at({j, 2}), 8 << l);
        EXPECT_EQ(p.at({j, 3}), l ? (32 << l) * .25 : 0.);
        EXPECT_EQ(p.at({j, 4}), l < 4 ? (64 << l) * .25 : INFINITY);
        ++j;
      }
  EXPECT_EQ(m::FcosPoints({80, 40, 20, 10, 5}).shape()[0], 8525);
  EXPECT_NEAR(m::FcosCenterness(n::tensor({2., 3., 8., 12.})).item(), .25, 1e-7);
  EXPECT_EQ(m::FcosCenterness(n::tensor({0., 2., 4., 2.})).item(), 0);
}
TEST(NnFcosDetection, StrictInteriorInclusiveRangesSmallestAreaFirstTiesAndCentreSampling) {
  m::TensorDetectionScenes data;
  data.images = n::zeros({2, 64, 64, 1});
  data.boxes = {n::tensor({0., 0., 24., 24., 4., 4., 12., 12., 4., 4., 12., 12.}, {3, 4}),
                n::zeros({0, 4})};
  data.labels = {n::from_vector(std::vector<std::int64_t>{0, 1, 0}),
                 n::zeros({0}, n::DType::Int64)};
  const auto points = n::tensor({8.,  8.,  8., 0., 16., 4., 8.,  8., 0., 24., 8.,  8., 8.,
                                 16., 32., 2., 2., 1.,  0., 32., 8., 8., 8.,  17., 32.},
                                {5, 5});
  const auto t = m::MakeFcosTargets(points, data, {0, 1}, 2);
  EXPECT_EQ(t.positive.to_vector<int>(), std::vector<int>({1, 1, 1, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(n::slice_axis(t.classes, 0, 0, 1).to_vector<int>(),
            std::vector<int>({0, 1, 1, 0, 1, 0, 0, 0, 0, 0}));
  EXPECT_EQ(t.centerness.at({0, 0}), 1);
  EXPECT_EQ(t.distances.at({0, 0, 0}), 4);
  const auto original = m::MakeFcosTargets(points, data, {0}, 2, 0);
  EXPECT_EQ(original.positive.at({0, 3}), 1);
  EXPECT_FALSE(t.classes.requires_grad());
}
TEST(NnFcosDetection, AllThreeLossTermsAndEveryAnalyticDerivative) {
  const auto type = n::DType::Float64;
  const auto l = n::zeros({1, 2, 2}, type).requires_grad_(),
             d = n::from_vector(std::vector<double>{.8, 1.3, 2.1, 2.7, 2, 3, 4, 5}, {1, 2, 4})
                     .requires_grad_(),
             c = n::zeros({1, 2}, type).requires_grad_();
  m::FcosTargets t{n::from_vector(std::vector<double>{1, 0, 0, 0}, {1, 2, 2}),
                   n::from_vector(std::vector<double>{1, 2, 3, 4, 0, 0, 0, 0}, {1, 2, 4}),
                   n::from_vector(std::vector<double>{1, 0}, {1, 2}),
                   n::from_vector(std::vector<double>{.6, 0}, {1, 2})};
  const auto loss = m::FcosLoss(l, d, c, t);
  EXPECT_NEAR(loss.item(), 1.625 * std::log(2.) + std::log(25. / 12.6), 1e-12);
  const auto g = n::grad(loss, {l, d, c});
  const double a = .125 + .25 * std::log(2.);
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(g[0].reshape({4}).at({i}), i ? .75 * a : -.25 * a, 1e-12);
  for (int i = 0; i < 8; ++i)
    EXPECT_NEAR(g[1].reshape({8}).at({i}),
                i >= 4  ? 0
                : i % 2 ? -2.9 / 12.6
                        : -4. / 12.6,
                1e-12);
  EXPECT_NEAR(g[2].at({0, 0}), -.1, 1e-12);
  EXPECT_EQ(g[2].at({0, 1}), 0);
  t.classes = n::zeros({1, 2, 2}, type);
  t.positive = n::zeros({1, 2}, type);
  t.centerness = n::zeros({1, 2}, type);
  const auto bg = m::FcosLoss(l, d, c, t);
  EXPECT_NEAR(bg.item(), .75 * std::log(2.), 1e-12);
  const auto empty = n::grad(bg, {d, c});
  for (const auto &v : empty)
    EXPECT_EQ(n::count_nonzero(v).item(), 0);
}
TEST(NnFcosDetection, DecodeClassThresholdPrecedesCenternessAndPreservesClassIsolation) {
  m::FcosOutputs net;
  net.logits = n::tensor({0., -8., 0., 0.}, {1, 2, 2});
  net.distances = n::full({1, 2, 4}, 4.);
  net.centerness = n::tensor({-20., 0.}, {1, 2});
  const auto points = n::tensor({8., 8., 8., 0., 64., 8., 8., 8., 0., 64.}, {2, 5});
  const auto raw =
      m::DecodeFcosDetections(net, points, n::zeros({1}, n::DType::Int64), 64, 64, false);
  EXPECT_EQ(raw.scores.numel(), 3);
  EXPECT_LT(raw.scores.at({2}), .01);
  const auto out =
      m::DecodeFcosDetections(net, points, n::zeros({1}, n::DType::Int64), 64, 64);
  EXPECT_EQ(out.labels.to_vector<int>(), std::vector<int>({0, 1}));
  EXPECT_NEAR(out.scores.at({0}), .5, 1e-7);
  EXPECT_EQ(out.boxes.to_vector<int>(), std::vector<int>({4, 4, 12, 12, 4, 4, 12, 12}));
}
TEST(NnFcosDetection, GlobalPerImageTop1000HasStablePointClassTies) {
  m::FcosOutputs net;
  net.logits = n::zeros({1, 600, 2});
  net.distances = n::ones({1, 600, 4});
  net.centerness = n::zeros({1, 600});
  const auto point = n::tensor({8., 8., 8., 0., 64.}, {1, 5}) * n::ones({600, 1});
  const auto raw =
      m::DecodeFcosDetections(net, point, n::zeros({1}, n::DType::Int64), 64, 64, false);
  EXPECT_EQ(raw.scores.numel(), 1000);
  const auto labels = raw.labels.to_vector<int>();
  for (int i = 0; i < 1000; ++i)
    EXPECT_EQ(labels[i], i % 2);
}
m::FcosConfig Compact() {
  m::FcosConfig c;
  c.classes = 10;
  c.image_size = 64;
  c.backbone_depth = 18;
  c.width_divisor = 8;
  c.pyramid_width = 32;
  c.groups = 8;
  c.grayscale = true;
  c.stride_scaled_distances = true;
  c.update_stats = false;
  return c;
}
TEST(NnFcosDetection, CompactWholeGraphCountsPositiveDistancesAndEveryGradient) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  const auto net =
      m::Fcos(p, state, n::sin(.13 * n::arange(0, 2 * 64 * 64)).reshape({2, 64, 64, 1}),
              Compact(), true);
  EXPECT_EQ(p.count(), 314684);
  EXPECT_EQ(state.count(), 1200);
  EXPECT_EQ(net.points, 86);
  EXPECT_EQ(net.distances.shape(), n::Shape({2, 86, 4}));
  EXPECT_GT(n::min(net.distances).item(), 0);
  const auto objective = n::mean(n::square(net.logits)) + n::mean(n::square(net.distances)) +
                         n::mean(n::square(net.centerness));
  for (const auto &g : n::grad(objective, p.parameters())) {
    EXPECT_TRUE(g.defined());
    const double norm = n::sum(n::square(g)).item();
    EXPECT_TRUE(std::isfinite(norm));
    EXPECT_GT(norm, 0);
  }
}
TEST(NnFcosDetection, FullWeightsEvery218GradientAndInvalidMetadata) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  m::FcosConfig c;
  c.image_size = 128;
  c.update_stats = false;
  const auto net = m::Fcos(
      p, state, n::sin(.13 * n::arange(0, 128 * 128 * 3)).reshape({1, 128, 128, 3}), c, true);
  EXPECT_EQ(p.count(), 36426138);
  EXPECT_EQ(state.count(), 53120);
  EXPECT_EQ(net.points, 341);
  const auto objective = n::mean(n::square(net.logits)) + n::mean(n::square(net.distances)) +
                         n::mean(n::square(net.centerness));
  const auto g = n::grad(objective, p.parameters());
  EXPECT_EQ(g.size(), 218);
  for (const auto &v : g) {
    EXPECT_TRUE(v.defined());
    EXPECT_TRUE(std::isfinite(n::sum(n::square(v)).item()));
  }
  EXPECT_THROW(m::FcosPoints({1, 1}, 1), std::invalid_argument);
  EXPECT_THROW(m::FcosPoints({1, 1, 1, 1, 1}, 0), std::invalid_argument);
  c.groups = 3;
  EXPECT_THROW(m::Fcos(p, state, n::zeros({1, 128, 128, 3}), c), std::invalid_argument);
}
