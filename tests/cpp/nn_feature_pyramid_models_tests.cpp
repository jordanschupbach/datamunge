
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnFeaturePyramid, OddRectangularFusionAndEveryInputGradientMatchIndependentOracle) {
  const int H[] = {7, 4, 2}, W[] = {9, 5, 3};
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  std::vector<n::Tensor> input;
  std::vector<std::vector<double>> values;
  for (int l = 0; l < 3; ++l) {
    const auto x = n::sin(.4 * n::arange(0, H[l] * W[l]).to(type) + l)
                       .reshape({1, H[l], W[l], 1})
                       .detach()
                       .requires_grad_();
    input.push_back(x);
    values.push_back(x.to_vector<double>());
  }
  {
    n::NoGradGuard guard;
    m::FeaturePyramid(p, "p", input, 1);
    for (int l = 0; l < 3; ++l) {
      const auto prefix = "p/level" + std::to_string(l + 2);
      p.set(prefix + "/lateral/W", n::ones({1, 1, 1, 1}, type));
      const auto center =
          n::one_hot(n::from_vector(std::vector<std::int64_t>{4}), 9, 1, 0, type)
              .reshape({3, 3, 1, 1}) *
          double(l + 2);
      p.set(prefix + "/smooth/W", center);
    }
  }
  const auto out = m::FeaturePyramid(p, "p", input, 1);
  n::Tensor loss = n::zeros({}, type);
  std::vector<std::vector<double>> expected;
  for (int l = 0; l < 3; ++l) {
    expected.emplace_back(H[l] * W[l], 0);
    loss = loss + n::sum(out.levels[l]);
  }
  int negative = 0;
  for (int l = 0; l < 3; ++l)
    for (int r = 0; r < H[l]; ++r)
      for (int c = 0; c < W[l]; ++c) {
        int rr = r, cc = c;
        double sum = 0;
        for (int j = l; j < 3; ++j) {
          if (j > l) {
            rr = rr * H[j] / H[j - 1];
            cc = cc * W[j] / W[j - 1];
          }
          sum += values[j][rr * W[j] + cc];
          expected[j][rr * W[j] + cc] += l + 2;
        }
        EXPECT_NEAR(out.inner[l].at({0, r, c, 0}), sum, 1e-12);
        EXPECT_NEAR(out.levels[l].at({0, r, c, 0}), (l + 2) * sum, 1e-12);
        negative += out.levels[l].at({0, r, c, 0}) < 0;
      }
  const auto grad = n::grad(loss, input);
  for (int l = 0; l < 3; ++l) {
    const auto g = grad[l].to_vector<double>();
    for (std::size_t i = 0; i < g.size(); ++i)
      EXPECT_EQ(g[i], expected[l][i]);
  }
  EXPECT_GT(negative, 0);
  EXPECT_EQ(out.levels[3].shape(), n::Shape({1, 1, 2, 1}));
  EXPECT_EQ(out.levels[3].at({0, 0, 0, 0}), out.levels[2].at({0, 0, 0, 0}));
  EXPECT_EQ(out.levels[3].at({0, 0, 1, 0}), out.levels[2].at({0, 0, 2, 0}));
}
TEST(NnFeaturePyramid, PyramidRoutingRestoresOrderAndRoutesFeatureGradientsOnly) {
  const auto type = n::DType::Float64;
  std::vector<m::DetectionPyramidLevel> levels;
  std::vector<n::Tensor> vars;
  for (int k = 2; k <= 5; ++k) {
    const int side = 2048 / (1 << k);
    const auto f = (n::full({2, side, side, 1}, double(k), type) +
                    n::from_vector(std::vector<double>{0, 10}, {2, 1, 1, 1}))
                       .detach()
                       .requires_grad_();
    levels.push_back({f, 1 << k});
    vars.push_back(f);
  }
  const std::vector<double> sides = {896, 32, 224, 112, 64};
  std::vector<double> values;
  for (double side : sides)
    values.insert(values.end(), {64, 64, 64 + side, 64 + side});
  const auto boxes = n::from_vector(values, {5, 4}).requires_grad_();
  const auto indices = n::from_vector(std::vector<std::int64_t>{1, 0, 1, 0, 1});
  EXPECT_EQ(m::PyramidDetectionLevels(boxes, 224).to_vector<int>(),
            std::vector<int>({5, 2, 4, 3, 2}));
  const auto pool = m::PyramidDetectionRoiPool(levels, boxes, indices, 224);
  const std::vector<double> expected = {15, 2, 14, 3, 12};
  for (int i = 0; i < 5; ++i)
    EXPECT_LT(n::max(n::abs(n::slice_axis(pool, 0, i, i + 1) - expected[i])).item(), 1e-6);
  vars.push_back(boxes);
  n::GradOptions options;
  options.allow_unused = true;
  const auto grad = n::grad(n::sum(pool), vars, options);
  for (int l = 0; l < 4; ++l)
    EXPECT_NEAR(n::sum(grad[l]).item(), l == 0 ? 98. : 49., 1e-6);
  EXPECT_FALSE(grad.back().defined());
}
TEST(NnFeaturePyramid, CompleteSourceDetectorCountsShapesAnchorsAndEveryLearnedGradient) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  const auto image = n::sin(.17 * n::arange(0, 2 * 64 * 64))
                         .reshape({2, 64, 64, 1})
                         .detach()
                         .requires_grad_();
  const auto f = m::FpnDigitDetectorForward(p, image);
  EXPECT_EQ(p.count(), 1292911);
  ASSERT_EQ(f.pyramid.levels.size(), 5);
  ASSERT_EQ(f.roi_levels.size(), 4);
  for (int l = 0; l < 5; ++l)
    EXPECT_EQ(f.pyramid.levels[l].shape(), n::Shape({2, 16 / (1 << l), 16 / (1 << l), 64}));
  EXPECT_EQ(f.rpn.anchors, 1023);
  const auto anchors = m::FpnDigitAnchors();
  EXPECT_EQ(anchors.shape(), n::Shape({1023, 4}));
  EXPECT_EQ(n::boxes_to_center(n::slice_axis(anchors, 0, 1020, 1023)).at({1, 0}), 32);
  const auto boxes = n::tensor(
      {0., 0., 8., 8., 0., 0., 16., 16., 0., 0., 32., 32., 0., 0., 64., 64.}, {4, 4});
  const auto indices = n::from_vector(std::vector<std::int64_t>{0, 1, 0, 1});
  const auto pool = m::PyramidDetectionRoiPool(f.roi_levels, boxes, indices, 32);
  const auto head = m::PooledDetectionRoiHead(p, pool);
  EXPECT_EQ(p.count(), 2174882);
  const auto loss =
      m::DetectionRoiLoss(head, n::from_vector(std::vector<std::int64_t>{1, 3, 5, 7}),
                          n::ones({4, 4}) * .1, n::ones({4})) +
      m::RpnLoss(f.rpn, n::ones({2, 1023}), n::ones({2, 1023}), n::zeros({2, 1023, 4}),
                 n::ones({2, 1023}));
  auto vars = p.parameters();
  vars.push_back(image);
  for (const auto &g : n::grad(loss, vars)) {
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_GT(n::sum(n::abs(g)).item(), 0);
  }
}
TEST(NnFeaturePyramid, CanonicalSideClampsZeroAreaAndExtraLevelCanBeDisabled) {
  EXPECT_EQ(m::PyramidDetectionLevels(
                n::tensor({0., 0., 0.,  0.,  0., 0., 8.,  8.,  0., 0., 16.,  16.,
                           0., 0., 32., 32., 0., 0., 64., 64., 0., 0., 512., 512.},
                          {6, 4}),
                32)
                .to_vector<int>(),
            std::vector<int>({2, 2, 3, 4, 5, 5}));
  n::ParameterStore p(17);
  const auto f =
      m::FeaturePyramid(p, "p", {n::ones({1, 5, 7, 4}), n::ones({1, 3, 4, 8})}, 4, false);
  EXPECT_EQ(f.levels.size(), 2);
}
TEST(NnFeaturePyramid, InvalidPyramidFeaturesRoutingAndFinisherFail) {
  n::ParameterStore p(17);
  EXPECT_THROW(m::FeaturePyramid(p, "p", {}, 4), std::invalid_argument);
  EXPECT_THROW((m::FeaturePyramid(p, "p", {n::ones({1, 3, 3, 4}), n::ones({1, 5, 5, 4})}, 4)),
               std::invalid_argument);
  EXPECT_THROW((m::FeaturePyramid(p, "p", {n::ones({1, 3, 3, 4}), n::ones({2, 2, 2, 4})}, 4)),
               std::invalid_argument);
  EXPECT_THROW(m::PyramidDetectionLevels(n::zeros({1, 4}), 0), std::invalid_argument);
  EXPECT_THROW(
      m::PyramidDetectionRoiPool({}, n::zeros({0, 4}), n::zeros({0}, n::DType::Int64)),
      std::invalid_argument);
  EXPECT_THROW(m::PooledDetectionRoiHead(p, n::ones({1, 6, 7, 64})), std::invalid_argument);
  EXPECT_THROW(m::FpnDigitDetectorForward(p, n::ones({1, 65, 65, 1})), std::invalid_argument);
}
