
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnCenterDetection, AuthorsRadiusAndEveryGaussianPixelMatchIndependentOracle) {
  const auto scalar = [](double h, double w) {
    const double b1 = h + w, c1 = h * w * .3 / 1.7,
                 r1 = (b1 + std::sqrt(b1 * b1 - 4 * c1)) / 2, b2 = 2 * (h + w),
                 c2 = .3 * h * w, r2 = (b2 + std::sqrt(b2 * b2 - 16 * c2)) / 2,
                 b3 = -1.4 * (h + w), c3 = -.3 * h * w,
                 r3 = (b3 + std::sqrt(b3 * b3 - 11.2 * c3)) / 2;
    return std::min({r1, r2, r3});
  };
  for (double h : {1., 5., 10., 30.})
    for (double w : {2., 10., 17.})
      EXPECT_NEAR(m::CenterGaussianRadius(n::full({}, h, n::DType::Float64),
                                          n::full({}, w, n::DType::Float64))
                      .item(),
                  scalar(h, w), 1e-12);
  const auto centres = n::from_vector(std::vector<std::int64_t>{3, 3, 0, 0, 3, 4}, {3, 2}),
             labels = n::zeros({3}, n::DType::Int64),
             radii = n::full({3}, 2., n::DType::Int64);
  const auto heat =
      m::CenterGaussianHeatmap(7, 7, 1, centres, labels, radii, n::DType::Float64);
  for (int y = 0; y < 7; ++y)
    for (int x = 0; x < 7; ++x) {
      double expected = 0;
      for (const auto &point : std::vector<std::array<int, 2>>{{3, 3}, {0, 0}, {3, 4}}) {
        const int dy = y - point[0], dx = x - point[1];
        if (std::abs(dy) <= 2 && std::abs(dx) <= 2)
          expected =
              std::max(expected, std::exp(-(dy * dy + dx * dx) / (2. * std::pow(5. / 6., 2))));
      }
      EXPECT_NEAR(heat.at({y, x, 0}), expected, 1e-15);
    }
  EXPECT_EQ(heat.at({0, 6, 0}), 0);
}
TEST(NnCenterDetection, CollisionsKeepLargestFirstAreaTiesAndOnlyWinnersDraw) {
  m::TensorDetectionScenes data;
  data.images = n::zeros({2, 64, 64, 1});
  data.boxes = {
      n::tensor({8., 12., 24., 28., 9., 13., 25., 29., 7., 11., 27., 31., 7., 11., 27., 31.},
                {4, 4}),
      n::zeros({0, 4})};
  data.labels = {n::from_vector(std::vector<std::int64_t>{1, 2, 3, 0}),
                 n::zeros({0}, n::DType::Int64)};
  const auto t = m::MakeCenterTargets(data, {0, 1}, 4);
  EXPECT_EQ(t.collisions, 3);
  EXPECT_EQ(n::sum(t.mask).item(), 1);
  EXPECT_EQ(t.heatmap.at({0, 4, 5, 3}), 1);
  EXPECT_EQ(t.heatmap.at({0, 4, 5, 0}), 0);
  for (int k = 0; k < 2; ++k) {
    EXPECT_EQ(t.size.at({0, 4, 5, k}), 5);
    EXPECT_EQ(t.offset.at({0, 4, 5, k}), .25);
  }
  EXPECT_EQ(n::sum(n::slice_axis(t.heatmap, 0, 1, 2)).item(), 0);
  EXPECT_FALSE(t.heatmap.requires_grad());
}
TEST(NnCenterDetection, GaussianFocalAndAllMaskedL1DerivativesAreAnalytic) {
  const auto type = n::DType::Float64;
  const auto l = n::zeros({1, 1, 3, 1}, type).requires_grad_(),
             s = n::from_vector(std::vector<double>{.2, 4, 2, 3, 4, 5}, {1, 1, 3, 2})
                     .requires_grad_(),
             o = n::from_vector(std::vector<double>{.1, .8, 2, 3, 4, 5}, {1, 1, 3, 2})
                     .requires_grad_();
  const auto heat = n::from_vector(std::vector<double>{1, .5, 0}, {1, 1, 3, 1}),
             ts = n::from_vector(std::vector<double>{1, 3, 0, 0, 0, 0}, {1, 1, 3, 2}),
             to = n::from_vector(std::vector<double>{.3, .5, 0, 0, 0, 0}, {1, 1, 3, 2}),
             mask = n::from_vector(std::vector<double>{1, 0, 0}, {1, 1, 3, 1});
  const auto loss = m::CenterLoss(l, s, o, heat, ts, to, mask);
  EXPECT_NEAR(loss.item(), .515625 * std::log(2.) + .68, 1e-12);
  const auto g = n::grad(loss, {l, s, o});
  const double a = .125 + .25 * std::log(2.), expected[] = {-a, a / 16, a};
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(g[0].reshape({3}).at({i}), expected[i], 1e-12);
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(g[1].reshape({6}).at({i}), i == 0 ? -.1 : i == 1 ? .1 : 0., 1e-12);
    EXPECT_NEAR(g[2].reshape({6}).at({i}), i == 0 ? -1. : i == 1 ? 1. : 0., 1e-12);
  }
  const auto bg = m::CenterLoss(l, s, o, n::zeros({1, 1, 3, 1}, type), ts, to,
                                n::zeros({1, 1, 3, 1}, type));
  EXPECT_NEAR(bg.item(), .75 * std::log(2.), 1e-12);
  for (const auto &v : n::grad(bg, {s, o}))
    EXPECT_EQ(n::count_nonzero(v).item(), 0);
  const auto extreme = m::CenterHeatmapFocal(n::from_vector(std::vector<double>{80, -80, -80}),
                                             n::from_vector(std::vector<double>{1, 0, 0}));
  EXPECT_LT(n::sum(extreme).item(), 1e-10);
}
TEST(NnCenterDetection, PlateausKeepAllClassMaximaHigherIndexTiesAndNoNms) {
  const auto peaks = m::CenterPeaks(n::zeros({1, 2, 2, 2}));
  EXPECT_EQ(n::count_nonzero(peaks).item(), 8);
  const auto out =
      m::DecodeCenterDetections(peaks, n::full({1, 2, 2, 2}, 4.), n::full({1, 2, 2, 2}, .5),
                                n::zeros({1}, n::DType::Int64), 64, 64);
  EXPECT_EQ(out.scores.numel(), 8);
  EXPECT_EQ(out.labels.to_vector<int>(), std::vector<int>({1, 0, 1, 0, 1, 0, 1, 0}));
  EXPECT_EQ(out.boxes.at({0, 2}), 14);
  EXPECT_EQ(out.boxes.at({0, 3}), 14);
}
TEST(NnCenterDetection, Top100PrecedesInvalidSizeFilterAndThresholdIsStrict) {
  const auto peaks = n::full({1, 1, 101, 1}, .5);
  auto sizes = n::ones({1, 1, 101, 2});
  const auto first =
      n::one_hot(n::zeros({1}, n::DType::Int64), 101, 1, 0).reshape({1, 1, 101, 1});
  sizes = sizes * first;
  const auto out = m::DecodeCenterDetections(peaks, sizes, n::zeros({1, 1, 101, 2}),
                                             n::zeros({1}, n::DType::Int64), 64, 500);
  EXPECT_EQ(out.scores.numel(), 0);
  const auto cutoff = m::DecodeCenterDetections(n::full({1, 1, 1, 1}, .01),
                                                n::ones({1, 1, 1, 2}), n::zeros({1, 1, 1, 2}),
                                                n::zeros({1}, n::DType::Int64), 64, 64);
  EXPECT_EQ(cutoff.scores.numel(), 0);
}
TEST(NnCenterDetection, FullAndCompactWholeGraphCountsEveryGradientAndGeometry) {
  n::set_num_threads(2);
  for (bool compact : {false, true}) {
    n::ParameterStore p(17), state(17);
    m::CenterNetConfig c;
    c.update_stats = false;
    const int side = compact ? 64 : 128;
    if (compact) {
      c.classes = 10;
      c.width_divisor = 8;
      c.head_width = 32;
      c.upsample_widths = {64, 32, 32};
      c.grayscale = true;
    }
    const auto net =
        m::CenterNet(p, state,
                     n::sin(.13 * n::arange(0, 2 * side * side * (compact ? 1 : 3)))
                         .reshape({2, side, side, compact ? 1 : 3}),
                     c, true);
    EXPECT_EQ(p.count(), compact ? 319862 : 14046164);
    EXPECT_EQ(state.count(), compact ? 1456 : 10496);
    EXPECT_EQ(net.heatmap.shape(), n::Shape({2, side / 4, side / 4, c.classes}));
    const auto objective = n::mean(n::square(net.heatmap)) + n::mean(n::square(net.size)) +
                           n::mean(n::square(net.offset));
    const auto g = n::grad(objective, p.parameters());
    EXPECT_EQ(g.size(), 81);
    for (const auto &v : g) {
      EXPECT_TRUE(v.defined());
      const double norm = n::sum(n::square(v)).item();
      EXPECT_TRUE(std::isfinite(norm));
      if (compact)
        EXPECT_GT(norm, 0);
    }
  }
}
TEST(NnCenterDetection, InvalidTargetsAndGeometryAreRejected) {
  EXPECT_THROW(m::CenterGaussianRadius(n::zeros({}), n::ones({})), std::invalid_argument);
  EXPECT_THROW(m::CenterHeatmapFocal(n::zeros({1}), n::full({1}, 1.2)), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  EXPECT_THROW(m::CenterNet(p, state, n::zeros({1, 33, 32, 3})), std::invalid_argument);
}
