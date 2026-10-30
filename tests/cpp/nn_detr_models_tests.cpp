
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include <random>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnDetr, TensorHungarianMatches100ExhaustiveRectangularOraclesAndDeterministicTies) {
  std::mt19937 generator(17);
  std::uniform_real_distribution<double> draw(-10, 10);
  for (int trial = 0; trial < 100; ++trial) {
    const int R = 1 + trial % 4, C = R + trial % 3;
    std::vector<double> cost(R * C);
    for (auto &v : cost)
      v = draw(generator);
    const auto chosen =
        m::TensorHungarianAssignment(n::from_vector(cost, {R, C})).to_vector<int>();
    std::vector<bool> used(C);
    double actual = 0;
    for (int r = 0; r < R; ++r) {
      ASSERT_GE(chosen[r], 0);
      ASSERT_LT(chosen[r], C);
      EXPECT_FALSE(used[chosen[r]]);
      used[chosen[r]] = true;
      actual += cost[r * C + chosen[r]];
    }
    std::fill(used.begin(), used.end(), false);
    double best = INFINITY;
    std::function<void(int, double)> enumerate = [&](int r, double sum) {
      if (r == R) {
        best = std::min(best, sum);
        return;
      }
      for (int c = 0; c < C; ++c)
        if (!used[c]) {
          used[c] = true;
          enumerate(r + 1, sum + cost[r * C + c]);
          used[c] = false;
        }
    };
    enumerate(0, 0);
    EXPECT_NEAR(actual, best, 1e-10);
  }
  EXPECT_EQ(m::TensorHungarianAssignment(n::zeros({3, 5})).to_vector<int>(),
            std::vector<int>({0, 1, 2}));
  EXPECT_EQ(m::TensorHungarianAssignment(n::zeros({0, 3})).numel(), 0);
  EXPECT_THROW(m::TensorHungarianAssignment(n::zeros({3, 2})), std::invalid_argument);
  EXPECT_THROW(m::TensorHungarianAssignment(n::full({1, 1}, NAN)), std::invalid_argument);
  const double M = std::numeric_limits<double>::max();
  EXPECT_THROW(
      m::TensorHungarianAssignment(n::from_vector(std::vector<double>{-M, M, -M, M}, {2, 2})),
      std::invalid_argument);
}
TEST(NnDetr, EveryNormalizedSinePositionMatchesIndependentSourceOracle) {
  const auto p = m::DetrPositions(3, 5, 16, n::DType::Float64);
  const double tau = 2 * std::acos(-1.);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 5; ++c)
      for (int k = 0; k < 8; ++k) {
        const double scale = std::pow(10000., 2 * (k / 2) / 8.),
                     y = (r + 1) * tau / (3 + 1e-6) / scale,
                     x = (c + 1) * tau / (5 + 1e-6) / scale;
        EXPECT_NEAR(p.at({0, r * 5 + c, k}), k % 2 ? std::cos(y) : std::sin(y), 1e-15);
        EXPECT_NEAR(p.at({0, r * 5 + c, k + 8}), k % 2 ? std::cos(x) : std::sin(x), 1e-15);
      }
  EXPECT_FALSE(p.requires_grad());
}
TEST(NnDetr, MatchedLossAllLogitDerivativesAndDisjointBoxCentralDifferences) {
  const auto type = n::DType::Float64;
  const auto x = n::zeros({1, 2, 3}, type).requires_grad_(),
             b = n::from_vector(std::vector<double>{.25, .3, .2, .2, .2, .3, .4, .5},
                                {1, 2, 4})
                     .requires_grad_();
  m::DetrTargets t{
      n::from_vector(std::vector<double>{1, 0, 0, 0, 0, 1}, {1, 2, 3}),
      n::from_vector(std::vector<double>{.7, .65, .25, .3, 0, 0, 0, 0}, {1, 2, 4}),
      n::from_vector(std::vector<double>{1, 0}, {1, 2})};
  const auto loss = m::DetrLayerLoss(x, b, t);
  const auto g = n::grad(loss, {x, b});
  for (int q = 0; q < 2; ++q)
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(g[0].at({0, q, c}),
                  (1. / 3. - ((q == 0 && c == 0) || (q == 1 && c == 2))) * (q ? .1 : 1.) / 1.1,
                  1e-12);
  for (int k = 0; k < 8; ++k) {
    auto plus = b.to_vector<double>(), minus = plus;
    const double step = 1e-6;
    plus[k] += step;
    minus[k] -= step;
    const double expected = (m::DetrLayerLoss(x, n::from_vector(plus, {1, 2, 4}), t).item() -
                             m::DetrLayerLoss(x, n::from_vector(minus, {1, 2, 4}), t).item()) /
                            (2 * step);
    EXPECT_NEAR(g[1].reshape({8}).at({k}), expected, 1e-8);
  }
  t.labels = n::one_hot(n::full({1, 2}, 2., n::DType::Int64), 3, 1, 0, type);
  t.positive = n::zeros({1, 2}, type);
  t.boxes = n::zeros({1, 2, 4}, type);
  const auto empty = m::DetrLayerLoss(x, b, t);
  EXPECT_NEAR(empty.item(), std::log(3.), 1e-12);
  EXPECT_EQ(n::count_nonzero(n::grad(empty, {b})[0]).item(), 0);
}
TEST(NnDetr, MatchingIsOneToOneDetachedUsesClassesAndHandlesEmptyImages) {
  m::TensorDetectionScenes data;
  data.images = n::zeros({2, 64, 64, 1});
  data.boxes = {n::tensor({0., 0., 16., 16., 32., 32., 48., 48.}, {2, 4}), n::zeros({0, 4})};
  data.labels = {n::from_vector(std::vector<std::int64_t>{1, 0}),
                 n::zeros({0}, n::DType::Int64)};
  const auto boxes =
      n::tensor({.625, .625, .25, .25, .125, .125, .25, .25, .5, .5, .1, .1}, {1, 3, 4}) *
      n::ones({2, 1, 1});
  const auto logits = n::zeros({2, 3, 3}).requires_grad_();
  const auto t = m::MatchDetrTargets(logits, boxes, data, {0, 1}, 2);
  EXPECT_EQ(n::argmax(t.labels, 2).to_vector<int>(), std::vector<int>({0, 1, 2, 2, 2, 2}));
  EXPECT_EQ(t.positive.to_vector<int>(), std::vector<int>({1, 1, 0, 0, 0, 0}));
  EXPECT_FALSE(t.labels.requires_grad());
  EXPECT_FALSE(t.boxes.requires_grad());
  EXPECT_THROW(m::MatchDetrTargets(n::zeros({1, 1, 3}), n::full({1, 1, 4}, .25), data, {0}, 2),
               std::invalid_argument);
  m::DetrOutputs net;
  net.logits = {logits, logits};
  net.boxes = {boxes, boxes};
  const auto layer = m::DetrLayerLoss(logits, boxes, t);
  EXPECT_NEAR(m::DetrLoss(net, data, {0, 1}, 2).item(), 2 * layer.item(), 1e-6);
}
TEST(NnDetr, DecoderForegroundChoiceIncludesNoObjectDenominatorWithoutNms) {
  const auto logits = n::tensor({0., 0., 3., 0., 0., 3.}, {1, 2, 3}),
             boxes = n::full({1, 2, 4}, .5);
  const auto d = m::DecodeDetrDetections(logits, boxes,
                                         n::from_vector(std::vector<std::int64_t>{7}), 64, 64);
  EXPECT_EQ(d.scores.numel(), 2);
  EXPECT_EQ(d.labels.to_vector<int>(), std::vector<int>({0, 0}));
  EXPECT_EQ(d.images.to_vector<int>(), std::vector<int>({7, 7}));
  EXPECT_NEAR(d.scores.at({0}), 1. / (2 + std::exp(3.)), 1e-7);
  EXPECT_EQ(d.boxes.to_vector<int>(), std::vector<int>({16, 16, 48, 48, 16, 16, 48, 48}));
}
TEST(NnDetr, FullAndCompactAuxiliaryGraphsEveryParameterAndExpectedZeroGradients) {
  n::set_num_threads(2);
  for (bool compact : {false, true}) {
    n::ParameterStore p(17), state(17);
    m::DetrConfig c;
    c.image_size = 128;
    c.update_stats = false;
    if (compact) {
      c.classes = 10;
      c.image_size = 64;
      c.backbone_depth = 18;
      c.width_divisor = 8;
      c.dimension = 64;
      c.heads = 4;
      c.feedforward = 128;
      c.encoder_layers = 2;
      c.decoder_layers = 2;
      c.queries = 10;
      c.dropout = 0;
      c.grayscale = true;
      c.small_stem = true;
    }
    const int N = compact ? 2 : 1, C = compact ? 1 : 3;
    const auto net = m::Detr(p, state,
                             n::sin(.13 * n::arange(0, N * c.image_size * c.image_size * C))
                                 .reshape({N, c.image_size, c.image_size, C}),
                             c, true);
    EXPECT_EQ(net.logits.size(), compact ? 2 : 6);
    if (compact) {
      EXPECT_EQ(net.memory.shape(), n::Shape({2, 64, 64}));
      EXPECT_EQ(net.logits.back().shape(), n::Shape({2, 10, 11}));
    }
    n::Tensor objective = n::zeros({});
    for (std::size_t k = 0; k < net.logits.size(); ++k)
      objective =
          objective + n::mean(n::square(net.logits[k])) + n::mean(n::square(net.boxes[k]));
    const auto g = n::grad(objective, p.parameters(), {false, false, true});
    int nonzero = 0;
    for (std::size_t k = 0; k < g.size(); ++k) {
      const double norm = g[k].defined() ? n::sum(n::square(g[k])).item() : 0.;
      EXPECT_TRUE(std::isfinite(norm));
      if (norm > 0)
        ++nonzero;
      else if (compact) {
        const auto name = p.names()[k];
        EXPECT_TRUE(name.find("decoder0/self/") != std::string::npos ||
                    name == "decoder0/norm1/gamma" || name.ends_with("/k/b"))
            << name;
      }
    }
    EXPECT_GT(nonzero, 100);
    std::cout << (compact ? "compact" : "full") << " DETR learned=" << p.count()
              << " state=" << state.count() << " tensors=" << g.size()
              << " nonzero=" << nonzero << '\n';
  }
}
