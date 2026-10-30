
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnYoloV5, CompleteIoUHasFrozenAlphaFiniteDifferenceDerivative) {
  const auto type = n::DType::Float64;
  auto x = n::from_vector(std::vector<double>{.53, .48, .3, .2}, {1, 4}).requires_grad_();
  const auto target = n::from_vector(std::vector<double>{.5, .5, .4, .32}, {1, 4});
  const auto q = m::CompleteBoxQuality(x, target);
  const auto g = n::grad(n::sum(1. - q.ciou), {x})[0].to_vector<double>();
  const auto host = [&](std::vector<double> p) {
    const double py = p[0], px = p[1], ph = p[2], pw = p[3], ty = .5, tx = .5, th = .4,
                 tw = .32;
    const double i = std::max(0., std::min(py + ph / 2, ty + th / 2) -
                                      std::max(py - ph / 2, ty - th / 2)) *
                     std::max(0., std::min(px + pw / 2, tx + tw / 2) -
                                      std::max(px - pw / 2, tx - tw / 2)),
                 iou = i / (ph * pw + th * tw - i),
                 h = std::max(py + ph / 2, ty + th / 2) - std::min(py - ph / 2, ty - th / 2),
                 w = std::max(px + pw / 2, tx + tw / 2) - std::min(px - pw / 2, tx - tw / 2),
                 a = std::atan(tw / th) - std::atan(pw / ph);
    return iou - ((py - ty) * (py - ty) + (px - tx) * (px - tx)) / (h * h + w * w + 1e-7) -
           q.alpha.item() * 4 / (std::acos(-1.) * std::acos(-1.)) * a * a;
  };
  EXPECT_NEAR(q.ciou.item(), host(x.to_vector<double>()), 1e-14);
  for (int k = 0; k < 4; ++k) {
    auto a = x.to_vector<double>(), b = a;
    a[k] += 1e-6;
    b[k] -= 1e-6;
    EXPECT_NEAR(g[k], -(host(a) - host(b)) / 2e-6, 1e-8);
  }
  EXPECT_FALSE(q.alpha.requires_grad());
}
TEST(NnYoloV5, RatioNeighbour21SlotsFallbackAndFirstCollisionTies) {
  std::array<n::Tensor, 3> raw;
  for (int l = 0; l < 3; ++l)
    raw[l] = n::zeros({1, 2 << l, 2 << l, 3, 7});
  const auto p = n::tensor(
      {4., 4., 6., 6., 8., 8., 12., 12., 16., 16., 20., 20., 24., 24., 32., 32., 48., 48.},
      {9, 2});
  m::TensorDetectionScenes data;
  data.images = n::zeros({1, 64, 64, 1});
  data.boxes = {n::tensor({22., 22., 30., 30.}, {1, 4})};
  data.labels = {n::ones({1}, n::DType::Int64)};
  const auto t = m::MakeYoloV5Targets(raw, p, data, {0});
  int count = 0;
  for (const auto &g : t.levels)
    count += n::sum(g.positive).item<int>();
  EXPECT_EQ(count, 21);
  EXPECT_EQ(t.collisions, 0);
  data.boxes[0] = n::concat({data.boxes[0], data.boxes[0]}, 0);
  data.labels[0] = n::from_vector(std::vector<std::int64_t>{1, 0});
  const auto ties = m::MakeYoloV5Targets(raw, p, data, {0});
  EXPECT_EQ(ties.collisions, 21);
  for (const auto &g : ties.levels)
    EXPECT_EQ(n::sum(n::slice_axis(g.classes, 4, 0, 1)).item(), 0);
  data.boxes[0] = n::slice_axis(data.boxes[0], 0, 0, 1);
  data.labels[0] = n::ones({1}, n::DType::Int64);
  const auto fallback = m::MakeYoloV5Targets(raw, n::full({9, 2}, .01), data, {0});
  count = 0;
  for (const auto &g : fallback.levels)
    count += n::sum(g.positive).item<int>();
  EXPECT_EQ(count, 3);
}
TEST(NnYoloV5, EmptyAndPerfectPositiveLossEveryAnalyticDerivative) {
  const auto type = n::DType::Float64;
  for (bool positive : {false, true}) {
    std::array<n::Tensor, 3> raw;
    m::YoloV5Targets t;
    for (int l = 0; l < 3; ++l) {
      raw[l] = n::zeros({1, 1, 1, 3, 7}, type).requires_grad_();
      t.levels[l] = {n::zeros({1, 1, 1, 3, 4}, type), n::zeros({1, 1, 1, 3, 2}, type),
                     n::zeros({1, 1, 1, 3}, type), n::full({1, 1, 1, 3}, -1., type)};
    }
    if (positive) {
      t.levels[0].geometry = n::from_vector(
          std::vector<double>{.5, .5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 3, 4});
      t.levels[0].classes =
          n::from_vector(std::vector<double>{1, 0, 0, 0, 0, 0}, {1, 1, 1, 3, 2});
      t.levels[0].positive = n::from_vector(std::vector<double>{1, 0, 0}, {1, 1, 1, 3});
    }
    const auto loss = m::YoloV5Loss(raw, t, 2, n::full({9, 2}, 32., type));
    EXPECT_NEAR(loss.item(), (positive ? 10.4 : 5.4) * std::log(2.), 1e-10);
    const auto g = n::grad(loss, {raw[0], raw[1], raw[2]});
    const double balance[] = {.4, 1, 4};
    for (int l = 0; l < 3; ++l)
      for (int a = 0; a < 3; ++a)
        for (int k = 0; k < 7; ++k) {
          double expected = k == 4 ? balance[l] / 6 : 0;
          if (positive && l == 0 && a == 0) {
            if (k == 4)
              expected *= -1;
            if (k == 5)
              expected = -1.25;
            if (k == 6)
              expected = 1.25;
          }
          EXPECT_NEAR(g[l].reshape({21}).at({a * 7 + k}), expected, 1e-12);
        }
  }
}
TEST(NnYoloV5, MosaicPixelRoundingAnnotationQuadrantsAndFlipInvolution) {
  m::TensorDetectionScenes data;
  data.images = n::arange(0, 4 * 4 * 6).reshape({4, 4, 6, 1}) / 255.;
  for (int i = 0; i < 4; ++i) {
    data.boxes.push_back(n::tensor({0., 0., 2., 4.}, {1, 4}));
    data.labels.push_back(n::full({1}, double(i), n::DType::Int64));
  }
  const auto mosaic = m::MosaicTensorDetectionScenes(
      data, n::from_vector(std::vector<std::int64_t>{0, 1, 2, 3}, {1, 4}));
  const auto pixels = mosaic.images.to_vector<float>();
  for (int q = 0; q < 4; ++q)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 3; ++x) {
        double sum = 0;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            sum += q * 24 + (y * 2 + dy) * 6 + x * 2 + dx;
        EXPECT_NEAR(pixels[((q / 2) * 2 + y) * 6 + (q % 2) * 3 + x],
                    float(std::floor(sum / 4 + .5) / 255.), 0.);
      }
  EXPECT_EQ(mosaic.labels[0].to_vector<int>(), std::vector<int>({0, 1, 2, 3}));
  EXPECT_EQ(mosaic.boxes[0].to_vector<float>(),
            std::vector<float>({0, 0, 1, 2, 0, 3, 1, 5, 2, 0, 3, 2, 2, 3, 3, 5}));
  const auto twice =
      m::FlipTensorDetectionScenes(m::FlipTensorDetectionScenes(data, {3, 0}), {0, 1});
  EXPECT_EQ(twice.images.to_vector<float>(),
            n::gather(data.images, n::from_vector(std::vector<std::int64_t>{3, 0}))
                .to_vector<float>());
  EXPECT_EQ(twice.boxes[0].to_vector<float>(), data.boxes[3].to_vector<float>());
  auto copy = m::SelectTensorDetectionScenes(data, {0});
  copy.boxes[0].fill_(0);
  EXPECT_GT(n::sum(data.boxes[0]).item(), 0);
}
TEST(NnYoloV5, Full177GradientsAndCompactMultiscaleBiasStateCounts) {
  n::set_num_threads(2);
  for (bool compact : {false, true}) {
    n::ParameterStore p(17), state(17);
    m::YoloV5Config c;
    c.update_stats = false;
    if (compact) {
      c.classes = 10;
      c.width_divisor = 4;
      c.grayscale = true;
      c.reference_size = 64;
      c.expected_objects = 3;
    }
    const int side = compact ? 64 : 128, channels = compact ? 1 : 3;
    const auto net = m::YoloV5(p, state,
                               n::sin(.13 * n::arange(0, 2 * side * side * channels))
                                   .reshape({2, side, side, channels}),
                               c, true);
    if (!compact) {
      EXPECT_EQ(p.count(), 7235389);
      EXPECT_EQ(state.count(), 19008);
    }
    const auto g = n::grad(n::mean(n::square(net.raw[0])) + n::mean(n::square(net.raw[1])) +
                               n::mean(n::square(net.raw[2])),
                           p.parameters());
    EXPECT_EQ(g.size(), 177);
    for (const auto &v : g) {
      EXPECT_TRUE(v.defined());
      EXPECT_TRUE(std::isfinite(n::sum(n::square(v)).item()));
      if (compact)
        EXPECT_GT(n::sum(n::square(v)).item(), 0);
    }
    std::cout << (compact ? "compact" : "full") << " learned=" << p.count()
              << " state=" << state.count() << '\n';
    if (compact)
      for (int size : {96, 128}) {
        const auto o = m::YoloV5(p, state, n::zeros({1, size, size, 1}), c);
        EXPECT_EQ(o.raw[0].shape()[1], size / 32);
      }
  }
}
TEST(NnYoloV5, DecodeUsesExpandedCentreSquaredSigmoidSizeAndAllClasses) {
  std::array<n::Tensor, 3> raw;
  for (int l = 0; l < 3; ++l)
    raw[l] = n::zeros({2, 2 << l, 2 << l, 3, 7});
  const auto priors = n::full({9, 2}, 16.),
             images = n::from_vector(std::vector<std::int64_t>{7, 2});
  const auto boxes = m::DecodeYoloV5Boxes(raw[0], priors, 0);
  const auto b = boxes.reshape({24, 4}).to_vector<float>();
  EXPECT_EQ(b[0], .25);
  EXPECT_EQ(b[1], .25);
  EXPECT_EQ(b[2], .25);
  EXPECT_EQ(b[3], .25);
  const auto candidates = m::DecodeYoloV5Detections(raw, priors, images, 64, 64, false);
  EXPECT_EQ(candidates.scores.numel(), 1008);
  EXPECT_EQ(n::min(candidates.scores).item(), .25);
  const auto out = m::DecodeYoloV5Detections(raw, priors, images, 64, 64);
  EXPECT_LE(out.scores.numel(), 200);
  EXPECT_FALSE(out.boxes.requires_grad());
}
TEST(NnYoloV5, InvalidGeometryAndMosaicInputsReject) {
  EXPECT_THROW(m::CompleteBoxQuality(n::zeros({2}), n::zeros({2})), std::invalid_argument);
  EXPECT_THROW(m::DecodeYoloV5Boxes(n::zeros({1}), n::ones({9, 2}), 0), std::invalid_argument);
  m::TensorDetectionScenes data;
  data.images = n::zeros({1, 3, 4, 1});
  data.boxes = {n::zeros({0, 4})};
  data.labels = {n::zeros({0}, n::DType::Int64)};
  EXPECT_THROW(m::MosaicTensorDetectionScenes(data, n::zeros({1, 4}, n::DType::Int64)),
               std::invalid_argument);
}
