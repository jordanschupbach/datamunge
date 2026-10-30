
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnEfficientDet, FastFusionAllValuesInputAndScalarDerivativesMatchIndependentOracle) {
  const auto type = n::DType::Float64;
  std::vector<n::Tensor> x, w, leaves;
  const double values[3][2] = {{1, -2}, {3, 5}, {7, 11}}, initial[] = {2, -1, 3};
  for (int i = 0; i < 3; ++i) {
    x.push_back(
        n::from_vector(std::vector<double>{values[i][0], values[i][1]}).requires_grad_());
    w.push_back(n::full({}, initial[i], type).requires_grad_());
  }
  leaves = x;
  leaves.insert(leaves.end(), w.begin(), w.end());
  const auto y = m::FastFeatureFusion(x, w);
  const auto g = n::grad(n::sum(y), leaves);
  const double denominator = 5.0001;
  for (int k = 0; k < 2; ++k)
    EXPECT_NEAR(y.at({k}), (2 * values[0][k] + 3 * values[2][k]) / denominator, 1e-12);
  for (int i = 0; i < 3; ++i) {
    double expected = 0;
    for (int k = 0; k < 2; ++k) {
      EXPECT_NEAR(g[i].at({k}), std::max(0., initial[i]) / denominator, 1e-12);
      expected += (values[i][k] * denominator - 2 * values[0][k] - 3 * values[2][k]) /
                  (denominator * denominator);
    }
    EXPECT_NEAR(g[i + 3].item(), initial[i] > 0 ? expected : 0, 1e-12);
  }
  for (auto &v : w)
    v = n::zeros({}, type).requires_grad_();
  leaves = x;
  leaves.insert(leaves.end(), w.begin(), w.end());
  const auto zero = m::FastFeatureFusion(x, w);
  EXPECT_EQ(n::count_nonzero(zero).item(), 0);
  for (const auto &v : n::grad(n::sum(zero), leaves))
    EXPECT_EQ(n::count_nonzero(v).item(), 0);
}
TEST(NnEfficientDet, OddGridsEightNodesNineteenEdgesAndWholeInputGradients) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  std::vector<n::Tensor> x;
  const int sides[] = {17, 9, 5, 3, 2};
  for (int s : sides)
    x.push_back(n::sin(.13 * n::arange(0, 2 * s * s * 4))
                    .reshape({2, s, s, 4})
                    .detach()
                    .requires_grad_());
  const auto y = m::BiFpnCell(p, state, "cell", x, 4);
  EXPECT_EQ(p.count(), 531);
  EXPECT_EQ(state.count(), 64);
  n::Tensor objective = n::zeros({});
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(y[i].shape(), x[i].shape());
    objective = objective + n::mean(n::square(y[i]));
  }
  for (const auto &g : n::grad(objective, x)) {
    EXPECT_TRUE(std::isfinite(n::sum(n::square(g)).item()));
    EXPECT_GT(n::sum(n::square(g)).item(), 0);
  }
  int edges = 0;
  for (const auto &name : p.names())
    if (name.ends_with("/weight"))
      ++edges;
  EXPECT_EQ(edges, 19);
}
TEST(NnEfficientDet, EveryCompoundVariantMatchesIndependentSourceTable) {
  const int widths[] = {64, 88, 112, 160, 224, 288, 384, 384},
            backbones[] = {0, 1, 2, 3, 4, 5, 6, 6}, repeats[] = {3, 4, 5, 6, 7, 7, 8, 8},
            heads[] = {3, 3, 3, 4, 4, 4, 5, 5},
            resolution[] = {512, 640, 768, 896, 1024, 1280, 1280, 1536};
  for (int i = 0; i < 8; ++i) {
    const auto c = m::EfficientDetScaling(i);
    EXPECT_EQ(c.width, widths[i]);
    EXPECT_EQ(c.backbone, backbones[i]);
    EXPECT_EQ(c.repeats, repeats[i]);
    EXPECT_EQ(c.head_repeats, heads[i]);
    EXPECT_EQ(c.resolution, resolution[i]);
  }
}
TEST(NnEfficientDet, FullD0AndCompactEveryLearnedGradientCountsAndHeadSharing) {
  n::set_num_threads(2);
  for (bool compact : {false, true}) {
    n::ParameterStore p(17), state(17);
    m::EfficientDetConfig c;
    c.update_stats = false;
    c.image_size = compact ? 64 : 128;
    if (compact) {
      c.classes = 10;
      c.width = 32;
      c.repeats = 2;
      c.head_repeats = 2;
      c.backbone_multiplier = .25;
      c.grayscale = true;
    }
    const auto net = m::EfficientDet(
        p, state,
        n::sin(.13 * n::arange(0, 2 * c.image_size * c.image_size * (compact ? 1 : 3)))
            .reshape({2, c.image_size, c.image_size, compact ? 1 : 3}),
        c, true);
    EXPECT_EQ(p.count(), compact ? 342038 : 3874217);
    EXPECT_EQ(state.count(), compact ? 13568 : 47136);
    EXPECT_EQ(net.anchors, compact ? 774 : 3069);
    const auto grads = n::grad(n::mean(n::square(net.logits)) + n::mean(n::square(net.deltas)),
                               p.parameters());
    if (!compact)
      EXPECT_EQ(grads.size(), 493);
    for (std::size_t i = 0; i < grads.size(); ++i) {
      EXPECT_TRUE(grads[i].defined());
      const double norm = n::sum(n::square(grads[i])).item();
      EXPECT_TRUE(std::isfinite(norm));
      if (compact && p.names()[i] != "bifpn1/node7/bn/beta")
        EXPECT_GT(norm, 0) << p.names()[i];
    }
    for (int l = 0; l < 5; ++l)
      EXPECT_TRUE(state.contains("classification/layer0/bn_level" + std::to_string(l) +
                                 "/moving_mean"));
  }
}
TEST(NnEfficientDet, InvalidFusionVariantsAndPyramidGeometryRejected) {
  EXPECT_THROW(m::EfficientDetScaling(8), std::invalid_argument);
  EXPECT_THROW(m::FastFeatureFusion({n::ones({2})}, {n::ones({1})}), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  std::vector<n::Tensor> x;
  for (int side : {17, 8, 5, 3, 2})
    x.push_back(n::ones({1, side, side, 4}));
  EXPECT_THROW(m::BiFpnCell(p, state, "bad", x, 4), std::invalid_argument);
  m::EfficientDetConfig c;
  c.backbone_multiplier = INFINITY;
  EXPECT_THROW(m::EfficientDet(p, state, n::zeros({1, 512, 512, 3}), c),
               std::invalid_argument);
}
