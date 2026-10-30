
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnTwoStageDetection, FeatureCoordinatesAndBilinearPoolMatchCellCentreOracle) {
  const auto rr = n::arange(0, 16).reshape({16, 1}) * n::ones({16, 16}),
             cc = n::arange(0, 16).reshape({1, 16}) * n::ones({16, 16});
  const auto features =
      n::stack({rr, cc}, 2).reshape({1, 16, 16, 2}).detach().requires_grad_();
  const auto rois = m::DetectionFeatureBoxes(n::tensor({8., 12., 40., 36.}, {1, 4}), 4, 16, 16)
                        .detach()
                        .requires_grad_();
  const auto pooled =
      n::max_pool2d(n::crop_and_resize(features, rois,
                                       n::from_vector(std::vector<std::int64_t>{0}), {14, 14}),
                    2, 2, n::Padding::Valid);
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 7; ++j) {
      EXPECT_NEAR(pooled.at({0, i, j, 0}), 1.5 + (2 * i + 1) * 8. / 13., 2e-6);
      EXPECT_NEAR(pooled.at({0, i, j, 1}), 2.5 + (2 * j + 1) * 6. / 13., 2e-6);
    }
  n::GradOptions options;
  options.allow_unused = true;
  const auto gradients = n::grad(n::sum(pooled), {features, rois}, options);
  EXPECT_GT(n::sum(n::abs(gradients[0])).item(), 0);
  EXPECT_FALSE(gradients[1].defined());
}
TEST(NnTwoStageDetection, PaddingPreservesCentreAndAnisotropicExtents) {
  const auto b = n::from_vector(std::vector<double>{10, 20, 30, 60}, {1, 4}).requires_grad_();
  const auto p = m::PadDetectionBoxes(b, .07), c = n::boxes_to_center(p);
  EXPECT_NEAR(p.at({0, 0}), 8.6, 1e-12);
  EXPECT_NEAR(p.at({0, 1}), 17.2, 1e-12);
  EXPECT_NEAR(c.at({0, 0}), 20, 1e-12);
  EXPECT_NEAR(c.at({0, 2}), 22.8, 1e-12);
  EXPECT_GT(n::sum(n::abs(n::grad(n::sum(n::square(p)), {b})[0])).item(), 0);
}
TEST(NnTwoStageDetection, ExactMultitaskLossAndSelectedRegressorGradient) {
  const auto type = n::DType::Float64;
  const auto logits = n::zeros({2, 3}, type).requires_grad_(),
             deltas = n::full({2, 2, 4}, .5, type).requires_grad_();
  m::DetectionRoiOutputs h{logits, n::softmax(logits, 1), deltas};
  const auto loss =
      m::DetectionRoiLoss(h, n::from_vector(std::vector<std::int64_t>{0, 2}),
                          n::zeros({2, 4}, type), n::from_vector(std::vector<double>{1, 0}));
  EXPECT_NEAR(loss.item(), std::log(3.) + .5, 1e-12);
  const auto g = n::grad(loss, {logits, deltas});
  EXPECT_NEAR(g[0].at({0, 0}), -1. / 3, 1e-12);
  EXPECT_NEAR(g[0].at({1, 2}), -1. / 3, 1e-12);
  for (int coord = 0; coord < 4; ++coord) {
    EXPECT_EQ(g[1].at({0, 0, coord}), .5);
    EXPECT_EQ(g[1].at({0, 1, coord}), 0);
    EXPECT_EQ(g[1].at({1, 0, coord}), 0);
    EXPECT_EQ(g[1].at({1, 1, coord}), 0);
  }
}
TEST(NnTwoStageDetection, RcnnHeadCountAndEveryLearnedTensorGradient) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  const auto x = n::sin(.37 * n::arange(0, 2 * 28 * 28))
                     .reshape({2, 28, 28, 1})
                     .detach()
                     .requires_grad_();
  const auto h = m::RcnnPatchHead(p, x);
  EXPECT_EQ(p.count(), 426931);
  EXPECT_EQ(h.deltas.shape(), n::Shape({2, 10, 4}));
  EXPECT_LT(n::max(n::abs(n::sum(h.probabilities, {1}) - 1.)).item(), 1e-6);
  auto vars = p.parameters();
  vars.push_back(x);
  for (const auto &g :
       n::grad(m::DetectionRoiLoss(h, n::from_vector(std::vector<std::int64_t>{2, 7}),
                                   n::ones({2, 4}) * .1, n::ones({2})),
               vars)) {
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_GT(n::sum(n::abs(g)).item(), 0);
  }
}
TEST(NnTwoStageDetection, FullSharedBackboneHeadAndRpnCountsAndGradients) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  const auto x = n::sin(.17 * n::arange(0, 2 * 64 * 64))
                     .reshape({2, 64, 64, 1})
                     .detach()
                     .requires_grad_();
  const auto features = m::StrideFourDetectionBackbone(p, x);
  EXPECT_EQ(features.shape(), n::Shape({2, 16, 16, 64}));
  EXPECT_EQ(p.count(), 64992);
  const auto head =
      m::DetectionRoiHead(p, features, n::tensor({0., 0., 1., 1., .1, .2, .7, .9}, {2, 4}),
                          n::from_vector(std::vector<std::int64_t>{0, 1}));
  EXPECT_EQ(p.count(), 946963);
  const auto rpn = m::BuildRpn(p, {{features, 64, 16, 9}});
  EXPECT_EQ(p.count(), 1026624);
  EXPECT_EQ(rpn.anchors, 2304);
  auto vars = p.parameters();
  vars.push_back(x);
  const auto loss = m::DetectionRoiLoss(head, n::from_vector(std::vector<std::int64_t>{2, 7}),
                                        n::ones({2, 4}) * .1, n::ones({2})) +
                    m::RpnLoss(rpn, n::ones({2, 2304}), n::ones({2, 2304}),
                               n::zeros({2, 2304, 4}), n::ones({2, 2304}));
  for (const auto &g : n::grad(loss, vars)) {
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_GT(n::sum(n::abs(g)).item(), 0);
  }
}
TEST(NnTwoStageDetection, RcnnSamplesMatchIndependentOverlapAndTargetRules) {
  m::DetectionSceneConfig cfg;
  cfg.scenes = 2;
  cfg.max_objects = 1;
  const auto scenes = m::MakeTensorDetectionScenes(
      n::ones({1, 28, 28, 1}), n::from_vector(std::vector<std::int64_t>{3}), cfg);
  const auto samples = m::MakeRcnnTrainingSamples(scenes, {1});
  EXPECT_EQ(samples.images.shape()[0], samples.labels.numel());
  EXPECT_FALSE(samples.images.requires_grad());
  const auto boxes = samples.boxes.to_vector<float>(),
             truth = scenes.boxes[1].to_vector<float>();
  for (std::int64_t i = 0; i < samples.labels.numel(); ++i) {
    double best = 0;
    int target = -1;
    for (std::int64_t j = 0; j < scenes.boxes[1].shape()[0]; ++j) {
      const double y1 = std::max(boxes[i * 4], truth[j * 4]),
                   x1 = std::max(boxes[i * 4 + 1], truth[j * 4 + 1]),
                   y2 = std::min(boxes[i * 4 + 2], truth[j * 4 + 2]),
                   x2 = std::min(boxes[i * 4 + 3], truth[j * 4 + 3]);
      const double intersection = std::max(0., y2 - y1) * std::max(0., x2 - x1);
      const double area = (boxes[i * 4 + 2] - boxes[i * 4]) *
                          (boxes[i * 4 + 3] - boxes[i * 4 + 1]),
                   ta = (truth[j * 4 + 2] - truth[j * 4]) *
                        (truth[j * 4 + 3] - truth[j * 4 + 1]);
      const double overlap = intersection / (area + ta - intersection);
      if (overlap > best) {
        best = overlap;
        target = int(j);
      }
    }
    EXPECT_EQ(samples.labels.at({i}), best >= .5 ? scenes.labels[1].at({target}) : 10);
    EXPECT_EQ(samples.mask.at({i}), best >= .6 ? 1 : 0);
    if (best >= .6) {
      const auto decoded = n::decode_boxes(n::slice_axis(samples.boxes, 0, i, i + 1),
                                           n::slice_axis(samples.targets, 0, i, i + 1));
      EXPECT_LT(n::max(n::abs(decoded - n::slice_axis(scenes.boxes[1], 0, target, target + 1)))
                    .item(),
                1e-5);
    }
    EXPECT_EQ(samples.indices.at({i}), 1);
  }
  const auto empty = m::MakeRcnnTrainingSamples(scenes, {});
  EXPECT_EQ(empty.images.shape(), n::Shape({0, 28, 28, 1}));
}
TEST(NnTwoStageDetection, InvalidGeometryHeadLossAndDataFail) {
  n::ParameterStore p(17);
  EXPECT_THROW(m::DetectionFeatureBoxes(n::zeros({1, 4}), 0, 16, 16), std::invalid_argument);
  EXPECT_THROW(m::PadDetectionBoxes(n::zeros({1, 4}), -.1), std::invalid_argument);
  EXPECT_THROW(m::RcnnPatchHead(p, n::ones({1, 27, 28, 1})), std::invalid_argument);
  EXPECT_THROW(m::StrideFourDetectionBackbone(p, n::ones({1, 64, 64, 3})),
               std::invalid_argument);
  EXPECT_THROW(m::DetectionRoiHead(p, n::ones({1, 16, 16, 64}), n::ones({1, 4}),
                                   n::from_vector(std::vector<std::int64_t>{1})),
               std::invalid_argument);
  m::DetectionRoiOutputs h{n::zeros({1, 3}), n::ones({1, 3}) / 3., n::zeros({1, 2, 4})};
  EXPECT_THROW(m::DetectionRoiLoss(h, n::from_vector(std::vector<std::int64_t>{2}),
                                   n::zeros({1, 4}), n::ones({1})),
               std::invalid_argument);
}

TEST(NnTwoStageDetection, ClassificationThresholdCanPrecedeRegressionThreshold) {
  m::TensorDetectionScenes scenes;
  scenes.images = n::ones({1, 64, 64, 1});
  scenes.boxes = {n::tensor({0., 0., 48., 48.}, {1, 4})};
  scenes.labels = {n::from_vector(std::vector<std::int64_t>{7})};
  const auto samples = m::MakeRcnnTrainingSamples(scenes, {0});
  EXPECT_EQ(samples.labels.to_vector<int>(), std::vector<int>({7, 7}));
  EXPECT_EQ(samples.mask.to_vector<int>(), std::vector<int>({0, 1}));
  EXPECT_EQ(n::max(n::abs(samples.targets)).item(), 0);
}

TEST(NnTwoStageDetection, RoiSamplingQuotasReorderedImageIndicesAndSeed) {
  m::TensorDetectionScenes scenes;
  scenes.images = n::ones({2, 64, 64, 1});
  scenes.boxes = {n::tensor({0., 0., 64., 64.}, {1, 4}),
                  n::tensor({0., 0., 64., 64.}, {1, 4})};
  scenes.labels = {n::from_vector(std::vector<std::int64_t>{3}),
                   n::from_vector(std::vector<std::int64_t>{7})};
  const std::vector<n::Tensor> candidates = {scenes.boxes[1] * n::ones({40, 4}),
                                             scenes.boxes[0] * n::ones({40, 4})};
  n::Generator rng(7), repeat(7);
  const auto a = m::SampleDetectionRois(scenes, {1, 0}, candidates, rng),
             b = m::SampleDetectionRois(scenes, {1, 0}, candidates, repeat);
  EXPECT_EQ(a.labels.numel(), 128);
  EXPECT_EQ(n::max(n::abs(a.pixels - b.pixels)).item(), 0);
  EXPECT_FALSE(a.targets.requires_grad());
  for (int image = 0; image < 2; ++image) {
    const auto labels = n::slice_axis(a.labels, 0, image * 64, image * 64 + 64);
    EXPECT_EQ(n::count_nonzero(n::less(labels, 10.)).item(), 16);
    EXPECT_EQ(n::count_nonzero(n::equal(labels, double(image == 0 ? 7 : 3))).item(), 16);
    EXPECT_EQ(n::min(n::slice_axis(a.indices, 0, image * 64, image * 64 + 64)).item(), image);
    EXPECT_EQ(n::max(n::slice_axis(a.indices, 0, image * 64, image * 64 + 64)).item(), image);
  }
  EXPECT_EQ(n::sum(a.mask).item(), 32);
  EXPECT_EQ(n::max(n::abs(a.targets)).item(), 0);
}
TEST(NnTwoStageDetection, RoiEmptyTruthIsBackgroundAndEmptyRowsAreWellShaped) {
  m::TensorDetectionScenes scenes;
  scenes.images = n::zeros({1, 64, 64, 1});
  scenes.boxes = {n::zeros({0, 4})};
  scenes.labels = {n::zeros({0}, n::DType::Int64)};
  n::Generator rng(7);
  const auto r = m::SampleDetectionRois(scenes, {0}, {n::zeros({0, 4})}, rng);
  EXPECT_EQ(r.labels.numel(), 64);
  EXPECT_EQ(n::min(r.labels).item(), 10);
  EXPECT_EQ(n::sum(r.mask).item(), 0);
  const auto empty = m::SampleDetectionRois(scenes, {}, {}, rng);
  EXPECT_EQ(empty.rois.shape(), n::Shape({0, 4}));
  EXPECT_THROW(m::SampleDetectionRois(scenes, {0}, {}, rng), std::invalid_argument);
  EXPECT_THROW(m::SampleDetectionRois(scenes, {0}, {n::zeros({0, 4})}, rng, 0),
               std::invalid_argument);
}
TEST(NnTwoStageDetection, PackingPreservesImageAndBoxOrderIncludingEmptyImages) {
  const auto boxes = n::tensor({8., 12., 40., 36., 4., 4., 20., 20.}, {2, 4});
  const auto p =
      m::PackDetectionRois({n::zeros({0, 4}), boxes, n::slice_axis(boxes, 0, 0, 1)});
  EXPECT_EQ(p.indices.to_vector<int>(), std::vector<int>({1, 1, 2}));
  EXPECT_EQ(p.pixels.shape(), n::Shape({3, 4}));
  EXPECT_NEAR(p.rois.at({0, 0}), 1.5 / 15., 1e-7);
  const auto empty = m::PackDetectionRois({n::zeros({0, 4})});
  EXPECT_EQ(empty.rois.shape(), n::Shape({0, 4}));
  EXPECT_THROW(m::PackDetectionRois({}), std::invalid_argument);
}
TEST(NnTwoStageDetection, RoiDecodeKeepsEveryClassAndUsesItsOwnOffsets) {
  m::DetectionRoiOutputs head{n::zeros({1, 3}), n::tensor({.4, .5, .1}, {1, 3}),
                              n::tensor({0., 0., 0., 0., 1., 0., 0., 0.}, {1, 2, 4})};
  const auto boxes = n::tensor({0., 0., 10., 10.}, {1, 4}),
             images = n::from_vector(std::vector<std::int64_t>{7});
  const auto refined = m::DecodeDetectionRoiOutputs(head, boxes, images),
             raw = m::DecodeDetectionRoiOutputs(head, boxes, images, false);
  EXPECT_EQ(refined.labels.to_vector<int>(), std::vector<int>({1, 0}));
  EXPECT_EQ(refined.images.to_vector<int>(), std::vector<int>({7, 7}));
  EXPECT_NEAR(refined.boxes.at({0, 0}), 1, 1e-6);
  EXPECT_EQ(raw.boxes.at({0, 0}), 0);
  EXPECT_FALSE(refined.boxes.requires_grad());
  m::DetectionRoiOutputs empty{n::zeros({0, 3}), n::zeros({0, 3}), n::zeros({0, 2, 4})};
  EXPECT_EQ(
      m::DecodeDetectionRoiOutputs(empty, n::zeros({0, 4}), n::zeros({0}, n::DType::Int64))
          .scores.numel(),
      0);
}
TEST(NnTwoStageDetection, RpnTargetsMatchKnownPositiveIgnoredForcedAndNegativeAnchors) {
  const auto anchors = n::tensor(
      {0., 0., 10., 10., 0., 0., 10., 20., 30., 30., 40., 40., 60., 60., 70., 70.}, {4, 4});
  m::TensorDetectionScenes scenes;
  scenes.images = n::zeros({1, 80, 80, 1});
  scenes.boxes = {n::tensor({0., 0., 10., 10., 30., 30., 45., 45.}, {2, 4})};
  scenes.labels = {n::from_vector(std::vector<std::int64_t>{0, 1})};
  n::Generator rng(7);
  const auto t = m::MakeDetectionRpnTargets(anchors, scenes, {0}, rng, 4);
  EXPECT_EQ(t.labels.to_vector<int>(), std::vector<int>({1, 0, 1, 0}));
  EXPECT_EQ(t.weights.to_vector<int>(), std::vector<int>({1, 0, 1, 1}));
  EXPECT_EQ(t.positives.to_vector<int>(), std::vector<int>({1, 0, 1, 0}));
  EXPECT_FALSE(t.targets.requires_grad());
  const auto restored = n::decode_boxes(n::slice_axis(anchors, 0, 2, 3),
                                        n::slice_axis(t.targets.reshape({4, 4}), 0, 2, 3));
  EXPECT_LT(n::max(n::abs(restored - n::slice_axis(scenes.boxes[0], 0, 1, 2))).item(), 1e-5);
  scenes.boxes = {n::zeros({0, 4})};
  scenes.labels = {n::zeros({0}, n::DType::Int64)};
  const auto empty = m::MakeDetectionRpnTargets(anchors, scenes, {0}, rng, 2);
  EXPECT_EQ(n::sum(empty.weights).item(), 2);
  EXPECT_EQ(n::sum(empty.labels).item(), 0);
  EXPECT_EQ(n::sum(empty.targets).item(), 0);
  const auto rows = m::MakeDetectionRpnTargets(anchors, scenes, {}, rng);
  EXPECT_EQ(rows.targets.shape(), n::Shape({0, 4, 4}));
}
TEST(NnTwoStageDetection, RpnProposalsClipFilterSuppressAndRespectTopBudgets) {
  const auto boxes = n::tensor({-2., -2., 10., 10., 0., 0., 10., 10., 15., 15., 18., 18., 20.,
                                20., 30., 30.},
                               {4, 4}),
             scores = n::tensor({.9, .8, .7, .6});
  const auto deltas = n::zeros({4, 4}).requires_grad_();
  const auto proposals = m::DetectionRpnProposals(scores, deltas, boxes, 64, 64);
  EXPECT_EQ(proposals.shape(), n::Shape({2, 4}));
  EXPECT_EQ(proposals.at({0, 0}), 0);
  EXPECT_EQ(proposals.at({1, 0}), 20);
  EXPECT_FALSE(proposals.requires_grad());
  EXPECT_EQ(m::DetectionRpnProposals(scores, deltas, boxes, 64, 64, 2).shape(),
            n::Shape({1, 4}));
  EXPECT_EQ(m::DetectionRpnProposals(scores, deltas, boxes, 64, 64, 0).shape(),
            n::Shape({0, 4}));
  m::Rpn r;
  r.objectness = n::stack({scores, scores}, 0);
  r.deltas = n::zeros({2, 4, 4});
  const auto batch = m::ComputeDetectionProposals(r, boxes, 64, 64);
  ASSERT_EQ(batch.size(), 2);
  EXPECT_EQ(n::max(n::abs(batch[0] - batch[1])).item(), 0);
  EXPECT_THROW(m::DetectionRpnProposals(scores, deltas, boxes, 0, 64), std::invalid_argument);
  EXPECT_THROW(m::DetectionRpnProposals(n::full({4}, NAN), deltas, boxes, 64, 64),
               std::invalid_argument);
}

TEST(NnTwoStageDetection, HeadLossDoesNotDifferentiateThroughRpnProposalSelection) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  const auto image = n::sin(.17 * n::arange(0, 64 * 64)).reshape({1, 64, 64, 1});
  const auto features = m::StrideFourDetectionBackbone(p, image);
  const auto rpn = m::BuildRpn(p, {{features, 64, 16, 9}});
  const auto proposals = m::ComputeDetectionProposals(
      rpn, n::anchor_grid(16, 16, 4, {12, 20, 30}, {.5, 1, 2}), 64, 64);
  const auto packed = m::PackDetectionRois(proposals);
  ASSERT_GT(packed.indices.numel(), 0);
  const auto head = m::DetectionRoiHead(p, features, packed.rois, packed.indices);
  n::GradOptions options;
  options.allow_unused = true;
  const auto derivatives =
      n::grad(n::mean(n::square(head.logits)),
              {p.at("rpn/objectness/W"), p.at("backbone/conv1/W")}, options);
  EXPECT_FALSE(derivatives[0].defined());
  EXPECT_GT(n::sum(n::abs(derivatives[1])).item(), 0);
}
