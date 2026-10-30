
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
TEST(NnDetectionComponents, SsdDistinctMatchingMatchesIndependentGreedyOracle) {
  for (const auto type : {n::DType::Float32, n::DType::Float64})
    for (int A : {1, 4, 9})
      for (int G = 0; G <= A; ++G) {
        const auto iou = n::abs(n::sin(.37 * n::arange(0, A * G).to(type))).reshape({A, G});
        std::vector<int> expected(A, -1);
        std::vector<bool> used(G, false);
        for (int k = 0; k < G; ++k) {
          double best = -1;
          int a = -1, g = -1;
          for (int i = 0; i < A; ++i)
            if (expected[i] < 0)
              for (int j = 0; j < G; ++j)
                if (!used[j] && iou.at({i, j}) > best) {
                  best = iou.at({i, j});
                  a = i;
                  g = j;
                }
          expected[a] = g;
          used[g] = true;
        }
        for (int a = 0; a < A; ++a)
          if (expected[a] < 0) {
            double best = .5;
            for (int g = 0; g < G; ++g)
              if (iou.at({a, g}) >= best) {
                best = iou.at({a, g});
                expected[a] = g;
              }
          }
        EXPECT_EQ(n::models::MatchSsd(iou).to_vector<int>(), expected);
      }
  EXPECT_EQ(n::models::MatchSsd(n::ones({4, 2})).to_vector<int>(),
            std::vector<int>({0, 1, 1, 1}));
  EXPECT_EQ(n::models::MatchSsd(n::zeros({4, 2})).to_vector<int>(),
            std::vector<int>({0, 1, -1, -1}));
}
TEST(NnDetectionComponents, ZeroLogitRpnDerivativeAndAllMaskedLoss) {
  n::ParameterStore p(17, n::DType::Float64);
  const auto r = n::models::BuildRpn(p, {{n::zeros({1, 1, 1, 4}, p.dtype()), 4, 1, 2}}, 8);
  const auto labels = n::from_vector(std::vector<double>{1, 0}, {1, 2});
  const auto loss =
      n::models::RpnLoss(r, labels, n::ones({1, 2}, p.dtype()), n::zeros({1, 2, 4}, p.dtype()),
                         n::zeros({1, 2}, p.dtype()));
  const auto g = n::grad(loss, {p.at("rpn/objectness/b")})[0];
  EXPECT_NEAR(loss.item(), std::log(2.), 1e-12);
  EXPECT_NEAR(g.at({0}), -.25, 1e-12);
  EXPECT_NEAR(g.at({1}), .25, 1e-12);
  const auto out = n::models::BuildRpn(p, {{n::zeros({1, 1, 1, 4}, p.dtype()), 4, 1, 2}}, 8);
  EXPECT_EQ(n::models::RpnLoss(out, labels, n::zeros({1, 2}, p.dtype()),
                               n::zeros({1, 2, 4}, p.dtype()), n::zeros({1, 2}, p.dtype()))
                .item(),
            0);
}
TEST(NnDetectionComponents, SharedMultiLevelRpnValuesAndAllGradients) {
  n::ParameterStore p(17, n::DType::Float64);
  const auto x = n::sin(.17 * n::arange(0, 2 * 3 * 3 * 4).to(p.dtype()))
                     .reshape({2, 3, 3, 4})
                     .detach()
                     .requires_grad_();
  const auto z = n::sin(.23 * n::arange(0, 2 * 2 * 2 * 4).to(p.dtype()))
                     .reshape({2, 2, 2, 4})
                     .detach()
                     .requires_grad_();
  const auto r = n::models::BuildRpn(p, {{x, 4, 3, 2}, {z, 4, 2, 2}}, 8);
  EXPECT_EQ(r.anchors, 26);
  EXPECT_EQ(r.deltas.shape(), n::Shape({2, 26, 4}));
  EXPECT_EQ(p.count(), 386);
  auto vars = p.parameters();
  vars.push_back(x);
  vars.push_back(z);
  const auto loss =
      n::models::RpnLoss(r, n::ones({2, 26}, p.dtype()), n::ones({2, 26}, p.dtype()),
                         n::ones({2, 26, 4}, p.dtype()) * .2, n::ones({2, 26}, p.dtype()));
  for (const auto &g : n::grad(loss, vars)) {
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_GT(n::sum(n::abs(g)).item(), 0);
  }
}
TEST(NnDetectionComponents, SmoothL1ValuesAndDerivatives) {
  const auto x =
      n::from_vector(std::vector<double>{-2, -1, -.5, 0, .5, 1, 2}).requires_grad_();
  const auto y = n::models::DetectorSmoothL1(x);
  EXPECT_EQ(y.to_vector<double>(), std::vector<double>({1.5, .5, .125, 0, .125, .5, 1.5}));
  EXPECT_EQ(n::grad(n::sum(y), {x})[0].to_vector<double>(),
            std::vector<double>({-1, -1, -.5, 0, .5, 1, 1}));
}
TEST(NnDetectionComponents, PatchClassifierCountAndCompleteGradient) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  const auto x = n::sin(.17 * n::arange(0, 2 * 28 * 28))
                     .reshape({2, 28, 28, 1})
                     .detach()
                     .requires_grad_();
  const auto y = n::models::PatchDetectorLogits(p, x);
  EXPECT_EQ(y.shape(), n::Shape({2, 11}));
  EXPECT_EQ(p.count(), 421771);
  auto vars = p.parameters();
  vars.push_back(x);
  for (const auto &g : n::grad(n::mean(n::square(y)), vars)) {
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_GT(n::sum(n::abs(g)).item(), 0);
  }
}
TEST(NnDetectionComponents, InvalidMatchingAndRpnGeometryFail) {
  EXPECT_THROW(n::models::MatchSsd(n::zeros({1, 2})), std::invalid_argument);
  EXPECT_THROW(n::models::MatchSsd(n::full({2, 2}, NAN)), std::invalid_argument);
  EXPECT_THROW(n::models::MatchSsd(n::ones({2, 2}), -1), std::invalid_argument);
  EXPECT_THROW(n::models::MatchSsd(n::ones({2, 2}, n::DType::Int64)), std::invalid_argument);
  n::ParameterStore p(17);
  EXPECT_THROW(n::models::BuildRpn(p, {}), std::invalid_argument);
  EXPECT_THROW((n::models::BuildRpn(p, {{n::ones({1, 2, 3, 4}), 4, 2, 2}})),
               std::invalid_argument);
  EXPECT_THROW((n::models::BuildRpn(
                   p, {{n::ones({1, 2, 2, 4}), 4, 2, 2}, {n::ones({2, 1, 1, 4}), 4, 1, 2}})),
               std::invalid_argument);
  EXPECT_THROW(n::models::PatchDetectorLogits(p, n::ones({1, 27, 28, 1})),
               std::invalid_argument);
  EXPECT_THROW(n::models::DetectorSmoothL1(n::ones({2}, n::DType::Int64)),
               std::invalid_argument);
}
