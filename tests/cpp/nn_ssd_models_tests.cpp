
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnSsdModels, EveryDefaultBoxMatchesIndependentShapeAndGridOracle) {
  const std::vector<int> cells = {38, 19, 10, 5, 3, 1}, counts = {4, 6, 6, 6, 4, 4};
  const std::vector<double> sizes = {21, 45, 99, 153, 207, 261, 315},
                            steps = {8, 16, 32, 64, 100, 300};
  const auto boxes = m::SsdDefaultBoxes(cells, sizes, steps, counts, n::DType::Float64);
  EXPECT_EQ(boxes.shape(), n::Shape({8732, 4}));
  std::int64_t index = 0;
  for (int l = 0; l < 6; ++l) {
    const double mean = std::sqrt(sizes[l] * sizes[l + 1]);
    std::vector<std::pair<double, double>> shapes = {{sizes[l], sizes[l]}, {mean, mean}};
    for (int ratio = 2; ratio <= (counts[l] == 6 ? 3 : 2); ++ratio) {
      const double h = sizes[l] / std::sqrt(double(ratio)),
                   w = sizes[l] * std::sqrt(double(ratio));
      shapes.push_back({h, w});
      shapes.push_back({w, h});
    }
    for (int r = 0; r < cells[l]; ++r)
      for (int c = 0; c < cells[l]; ++c)
        for (const auto &[h, w] : shapes) {
          const double y = (r + .5) * steps[l], x = (c + .5) * steps[l];
          const std::array<double, 4> expected = {y - h / 2, x - w / 2, y + h / 2, x + w / 2};
          for (int k = 0; k < 4; ++k)
            EXPECT_NEAR(boxes.at({index, k}), expected[k], 1e-12);
          ++index;
        }
  }
  EXPECT_EQ(index, 8732);
  const auto compact = m::SsdDefaultBoxes({16, 8, 4, 2, 1, 1}, {8, 16, 24, 32, 48, 64, 80},
                                          {4, 8, 16, 32, 64, 64}, {4, 6, 6, 6, 4, 4});
  EXPECT_EQ(compact.shape(), n::Shape({1536, 4}));
}
TEST(NnSsdModels, LearnedL2RescalingHasKnownValuesAndFiniteZeroVectorDerivative) {
  n::ParameterStore p(17, n::DType::Float64);
  const auto x =
      n::from_vector(std::vector<double>{3, 4, 0, 0}, {1, 1, 2, 2}).requires_grad_();
  const auto y = m::SsdNormalize(p, "n", x);
  EXPECT_EQ(y.to_vector<double>(), std::vector<double>({12, 16, 0, 0}));
  const auto g = n::grad(n::sum(y), {x, p.at("n/scale")});
  EXPECT_NEAR(g[0].at({0, 0, 1, 0}), 2e7, 1e-8);
  EXPECT_NEAR(g[0].at({0, 0, 1, 1}), 2e7, 1e-8);
  EXPECT_NEAR(g[1].at({0}), .6, 1e-12);
  EXPECT_NEAR(g[1].at({1}), .8, 1e-12);
}
TEST(NnSsdModels, PerImageMiningLossAndEveryDerivativeMatchIndependentOracle) {
  const auto type = n::DType::Float64;
  const int A = 8, C = 3;
  const auto raw = .8 * n::sin(.7 * n::arange(0, 2 * A * C).to(type));
  const auto logits = raw.reshape({2, A, C}).detach().requires_grad_(),
             deltas = n::full({2, A, 4}, .3, type).requires_grad_();
  std::vector<std::int64_t> labels(2 * A, 2);
  labels[0] = 0;
  labels[A] = labels[A + 1] = 1;
  const auto out = m::SsdMultiboxLoss(logits, deltas, n::from_vector(labels, {2, A}),
                                      n::zeros({2, A, 4}, type));
  const auto gradients = n::grad(out.loss, {logits, deltas});
  EXPECT_EQ(out.positive_counts.to_vector<int>(), std::vector<int>({1, 2}));
  const auto selected = out.selected_negatives.to_vector<bool>();
  const auto x = raw.to_vector<double>();
  double expected_loss = 0;
  for (int row = 0; row < 2; ++row) {
    std::vector<double> ce(A);
    std::vector<std::array<double, 3>> prob(A);
    std::vector<int> backgrounds;
    int positives = 0;
    for (int a = 0; a < A; ++a) {
      double total = 0;
      for (int c = 0; c < C; ++c) {
        prob[a][c] = std::exp(x[(row * A + a) * C + c]);
        total += prob[a][c];
      }
      for (int c = 0; c < C; ++c)
        prob[a][c] /= total;
      ce[a] = -std::log(prob[a][labels[row * A + a]]);
      if (labels[row * A + a] == 2)
        backgrounds.push_back(a);
      else
        ++positives;
    }
    std::stable_sort(backgrounds.begin(), backgrounds.end(), [&](int a, int b) {
      return ce[a] > ce[b];
    });
    backgrounds.resize(std::min(int(backgrounds.size()), 3 * positives));
    for (int a = 0; a < A; ++a) {
      const bool positive = labels[row * A + a] != 2,
                 negative =
                     std::find(backgrounds.begin(), backgrounds.end(), a) != backgrounds.end();
      EXPECT_EQ(selected[row * A + a], negative);
      if (positive || negative)
        expected_loss += ce[a] / 3.;
      if (positive)
        expected_loss += 4 * .5 * .3 * .3 / 3.;
      for (int c = 0; c < C; ++c)
        EXPECT_NEAR(gradients[0].at({row, a, c}),
                    (positive || negative) ? (prob[a][c] - (c == labels[row * A + a])) / 3.
                                           : 0.,
                    1e-12);
      for (int c = 0; c < 4; ++c)
        EXPECT_NEAR(gradients[1].at({row, a, c}), positive ? .1 : 0., 1e-12);
    }
  }
  EXPECT_NEAR(out.loss.item(), expected_loss, 1e-12);
  EXPECT_NEAR(out.regression.item(), .18, 1e-12);
  EXPECT_FALSE(out.selected_negatives.requires_grad());
}
TEST(NnSsdModels, MiningTiesPreferLowerIndicesAndAllBackgroundGradientsAreExactZero) {
  const auto labels = n::from_vector(std::vector<std::int64_t>{0, 2, 2, 2, 2}, {1, 5});
  const auto tied = m::SsdMultiboxLoss(n::zeros({1, 5, 3}), n::zeros({1, 5, 4}), labels,
                                       n::zeros({1, 5, 4}));
  EXPECT_EQ(tied.selected_negatives.to_vector<int>(), std::vector<int>({0, 1, 1, 1, 0}));
  const auto logits = n::ones({2, 8, 3}).requires_grad_(),
             deltas = n::ones({2, 8, 4}).requires_grad_();
  const auto empty = m::SsdMultiboxLoss(logits, deltas, n::full({2, 8}, 2., n::DType::Int64),
                                        n::zeros({2, 8, 4}));
  EXPECT_EQ(empty.loss.item(), 0);
  EXPECT_EQ(n::count_nonzero(empty.selected_negatives).item(), 0);
  for (const auto &g : n::grad(empty.loss, {logits, deltas}))
    EXPECT_EQ(n::max(n::abs(g)).item(), 0);
  const auto zero_logits = n::zeros({1, 0, 3}).requires_grad_(),
             zero_deltas = n::zeros({1, 0, 4}).requires_grad_();
  const auto zero = m::SsdMultiboxLoss(zero_logits, zero_deltas,
                                       n::zeros({1, 0}, n::DType::Int64), n::zeros({1, 0, 4}));
  EXPECT_EQ(zero.loss.item(), 0);
  for (const auto &g : n::grad(zero.loss, {zero_logits, zero_deltas}))
    EXPECT_EQ(g.numel(), 0);
}
TEST(NnSsdModels, TargetsPreserveCollidingTruthLabelsAndRoundTripOffsets) {
  const auto anchors = n::tensor(
      {0., 0., 10., 10., 0., 1., 10., 11., 30., 30., 40., 40., 50., 50., 60., 60.}, {4, 4});
  m::TensorDetectionScenes data;
  data.images = n::zeros({1, 64, 64, 1});
  data.boxes = {n::tensor({0., 0., 10., 10., 0., 0., 10., 10.}, {2, 4})};
  data.labels = {n::from_vector(std::vector<std::int64_t>{3, 7})};
  const auto targets = m::MakeSsdTargets(anchors, data, {0}, 10);
  EXPECT_EQ(targets.labels.to_vector<int>(), std::vector<int>({3, 7, 10, 10}));
  const auto restored =
      n::decode_boxes(n::slice_axis(anchors, 0, 0, 2),
                      n::slice_axis(targets.targets.reshape({4, 4}), 0, 0, 2));
  EXPECT_LT(n::max(n::abs(restored - data.boxes[0])).item(), 1e-6);
  EXPECT_FALSE(targets.targets.requires_grad());
  data.boxes = {n::zeros({0, 4})};
  data.labels = {n::zeros({0}, n::DType::Int64)};
  const auto bg = m::MakeSsdTargets(anchors, data, {0}, 10);
  EXPECT_EQ(n::min(bg.labels).item(), 10);
  EXPECT_EQ(n::max(n::abs(bg.targets)).item(), 0);
  EXPECT_EQ(m::MakeSsdTargets(anchors, data, {}, 10).labels.shape(), n::Shape({0, 4}));
}
TEST(NnSsdModels, InferenceTiesPreferHigherIndicesAndTopCapPrecedesAreaFilter) {
  const auto anchors =
      n::tensor({0., 0., 10., 10., 20., 20., 30., 30., 40., 40., 50., 50.}, {3, 4});
  const auto prob = n::tensor({.8, .2, .8, .2, .7, .3}, {1, 3, 2});
  const auto ids = n::from_vector(std::vector<std::int64_t>{7});
  const auto d = m::DecodeSsdDetections(prob, n::zeros({1, 3, 4}), anchors, ids, 64, 64, 1);
  ASSERT_EQ(d.scores.numel(), 1);
  EXPECT_EQ(d.boxes.at({0, 0}), 20);
  EXPECT_EQ(d.images.at({0}), 7);
  const auto outside = n::tensor({80., 80., 90., 90., 0., 0., 10., 10.}, {2, 4}),
             p = n::tensor({.9, .1, .8, .2}, {1, 2, 2});
  EXPECT_EQ(
      m::DecodeSsdDetections(p, n::zeros({1, 2, 4}), outside, ids, 64, 64, 1).scores.numel(),
      0);
  EXPECT_EQ(
      m::DecodeSsdDetections(p, n::zeros({1, 2, 4}), outside, ids, 64, 64, 2).scores.numel(),
      1);
  const auto both =
      m::DecodeSsdDetections(n::tensor({.2, .3, .5}, {1, 1, 3}), n::zeros({1, 1, 4}),
                             n::tensor({-2., -2., 10., 10.}, {1, 4}), ids, 64, 64);
  EXPECT_EQ(both.labels.to_vector<int>(), std::vector<int>({1, 0}));
  EXPECT_EQ(n::min(both.boxes).item(), 0);
  EXPECT_FALSE(both.boxes.requires_grad());
  const auto empty = m::DecodeSsdDetections(n::zeros({1, 0, 3}), n::zeros({1, 0, 4}),
                                            n::zeros({0, 4}), ids, 64, 64);
  EXPECT_EQ(empty.scores.numel(), 0);
}
TEST(NnSsdModels, CompactArchitectureCountAllFeaturesAndEveryLearnedTensorGradient) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  const auto image = n::sin(.13 * n::arange(0, 2 * 64 * 64)).reshape({2, 64, 64, 1});
  const auto net = m::SSD(p, image, 11, true);
  EXPECT_EQ(p.count(), 1167362);
  EXPECT_EQ(net.anchors, 1536);
  EXPECT_EQ(net.logits.shape(), n::Shape({2, 1536, 11}));
  for (int l = 0; l < 6; ++l)
    EXPECT_EQ(net.features[l].shape(),
              n::Shape({2, net.cells[l], net.cells[l], net.channels[l]}));
  for (const auto &g : n::grad(n::mean(n::square(net.logits)) + n::mean(n::square(net.deltas)),
                               p.parameters())) {
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_GT(n::sum(n::abs(g)).item(), 0);
  }
}
TEST(NnSsdModels, FullSsd300CountGridsPredictionsAndEveryLearnedTensorGradient) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  const auto image = n::sin(.13 * n::arange(0, 300 * 300 * 3)).reshape({1, 300, 300, 3});
  const auto net = m::SSD(p, image);
  EXPECT_EQ(p.count(), 26285486);
  EXPECT_EQ(net.anchors, 8732);
  EXPECT_EQ(net.cells, std::vector<int>({38, 19, 10, 5, 3, 1}));
  EXPECT_EQ(net.logits.shape(), n::Shape({1, 8732, 21}));
  for (int l = 0; l < 6; ++l)
    EXPECT_EQ(net.features[l].shape(),
              n::Shape({1, net.cells[l], net.cells[l], net.channels[l]}));
  for (const auto &g : n::grad(n::mean(n::square(net.logits)) + n::mean(n::square(net.deltas)),
                               p.parameters())) {
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_GT(n::sum(n::abs(g)).item(), 0);
  }
}
TEST(NnSsdModels, InvalidModelDefaultLossAndInferenceInputsFail) {
  n::ParameterStore p(17);
  EXPECT_THROW(m::SSD(p, n::zeros({1, 64, 64, 1})), std::invalid_argument);
  EXPECT_THROW(m::SsdNormalize(p, "n", n::ones({2, 3})), std::invalid_argument);
  EXPECT_THROW(m::SsdDefaultBoxes({1}, {8, 16}, {4}, {5}), std::invalid_argument);
  EXPECT_THROW(m::SsdDefaultBoxes({1}, {8, NAN}, {4}, {4}), std::invalid_argument);
  EXPECT_THROW(m::SsdMultiboxLoss(n::zeros({1, 1, 3}), n::zeros({1, 1, 4}),
                                  n::full({1, 1}, 3., n::DType::Int64), n::zeros({1, 1, 4})),
               std::invalid_argument);
  EXPECT_THROW(m::DecodeSsdDetections(n::full({1, 1, 3}, NAN), n::zeros({1, 1, 4}),
                                      n::zeros({1, 4}), n::zeros({1}, n::DType::Int64), 64,
                                      64),
               std::invalid_argument);
}

TEST(NnSsdModels, OptionalRawDecodeDefersFinalSuppressionAndImageCap) {
 const auto boxes=n::tensor({0.,0.,10.,10.,0.,0.,10.,10.},{2,4});
 const auto prob=n::tensor({.9,.1,.8,.2},{1,2,2});
 const auto raw=m::DecodeSsdDetections(prob,n::zeros({1,2,4}),boxes,n::zeros({1},n::DType::Int64),64,64,200,.05,.45,1,false);
 EXPECT_EQ(raw.scores.numel(),2);
}
