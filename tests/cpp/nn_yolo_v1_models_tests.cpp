
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnDetectionAugmentation, IdentityTranslationBilinearExposureClippingAndRowIsolation) {
  m::TensorDetectionScenes source;
  std::vector<int> pixels(48, 200);
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 6; ++c)
      pixels[r * 6 + c] = 4 + 10 * r + c;
  source.images = n::from_vector(pixels, {2, 4, 6, 1}).to(n::DType::Float32) / 255.;
  source.boxes = {n::tensor({1., 2., 3., 4.}, {1, 4}), n::tensor({0., 0., 1., 2.}, {1, 4})};
  source.labels = {n::from_vector(std::vector<std::int64_t>{3}),
                   n::from_vector(std::vector<std::int64_t>{7})};
  const auto identity = m::TransformTensorDetectionScenes(
      source, {0, 1}, n::tensor({1., 0., 0., 1., 1., 0., 0., 1.}, {2, 4}));
  EXPECT_EQ(identity.images.to_vector<float>(), source.images.to_vector<float>());
  const auto shifted =
      m::TransformTensorDetectionScenes(source, {0}, n::tensor({1., 1., -1., 1.}, {1, 4}));
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 6; ++c)
      EXPECT_NEAR(shifted.images.at({0, r, c, 0}) * 255.,
                  r > 0 && c < 5 ? pixels[(r - 1) * 6 + c + 1] : 0, 1e-5);
  EXPECT_EQ(shifted.boxes[0].to_vector<int>(), std::vector<int>({2, 1, 4, 3}));
  const auto half =
      m::TransformTensorDetectionScenes(source, {0}, n::tensor({1., .5, .5, 1.}, {1, 4}));
  EXPECT_NEAR(half.images.at({0, 1, 1, 0}) * 255., 10, 1e-5);
  EXPECT_NEAR(half.images.at({0, 0, 0, 0}) * 255., 1, 1e-5);
  const auto scaled =
      m::TransformTensorDetectionScenes(source, {0}, n::tensor({2., 0., 0., 1.}, {1, 4}));
  EXPECT_NEAR(scaled.images.at({0, 1, 2, 0}) * 255., 19, 1e-5);
  EXPECT_EQ(scaled.boxes[0].to_vector<int>(), std::vector<int>({0, 1, 4, 5}));
  const auto reorder = m::TransformTensorDetectionScenes(
      source, {1, 0}, n::tensor({1., 0., 0., 1.5, 1., 0., 0., 1.}, {2, 4}));
  EXPECT_EQ(reorder.labels[0].at({0}), 7);
  EXPECT_NEAR(reorder.images.at({0, 0, 0, 0}) * 255., 255, 1e-5);
  const auto outside =
      m::TransformTensorDetectionScenes(source, {0}, n::tensor({1., 0., 10., 1.}, {1, 4}));
  EXPECT_EQ(outside.boxes[0].shape()[0], 0);
  EXPECT_EQ(n::count_nonzero(outside.images).item(), 0);
  EXPECT_EQ(source.labels[0].at({0}), 3);
}
TEST(NnYoloV1, EveryLossDerivativeExactImperfectSwappedAndEmptyResponsibility) {
  const auto type = n::DType::Float64;
  std::vector<double> values = {.2, .8, .5, .5, .5, .5, .4, .2, .2, .3, .3, .2};
  m::YoloV1Targets t{n::ones({1, 1, 1}, type), n::full({1, 1, 1, 4}, .5, type),
                     n::ones({1, 1, 1}, n::DType::Int64), 0};
  const auto check = [&](double expected, const std::vector<double> &derivative,
                         int responsible) {
    const auto x = n::from_vector(values, {1, 1, 1, 12}).requires_grad_();
    const auto out = m::YoloV1Loss(x, t, 2, 1);
    EXPECT_NEAR(out.loss.item(), expected, 1e-10);
    const auto g = n::grad(out.loss, {x})[0].to_vector<double>();
    for (int i = 0; i < 12; ++i)
      EXPECT_NEAR(g[i], derivative[i], 1e-10);
    for (int i = 0; i < 2; ++i)
      EXPECT_EQ(out.responsible.at({0, 0, 0, i}), i == responsible ? 1 : 0);
    EXPECT_FALSE(out.iou.requires_grad());
    EXPECT_FALSE(out.responsible.requires_grad());
  };
  check(.46, {.4, -.4, 0, 0, 0, 0, -1.2, 0, 0, 0, 0, .2}, 0);
  values[4] = values[5] = .4;
  check(.20009216, {.4, -.4, 0, 0, -1, -1, -.0192, 0, 0, 0, 0, .2}, 0);
  for (int i = 0; i < 4; ++i) {
    values[2 + i] = .1;
    values[7 + i] = .5;
  }
  check(.8, {.4, -.4, 0, 0, 0, 0, .4, 0, 0, 0, 0, -1.6}, 1);
  t.object = n::zeros({1, 1, 1}, type);
  check(.1, {0, 0, 0, 0, 0, 0, .4, 0, 0, 0, 0, .2}, -1);
}
TEST(NnYoloV1, LargestCellCollisionFirstTieAndRootSizeEncoding) {
  m::TensorDetectionScenes data;
  data.images = n::zeros({2, 56, 56, 1});
  data.boxes = {n::tensor({8., 8., 16., 16., 7., 7., 17., 17., 7., 7., 17., 17.}, {3, 4}),
                n::zeros({0, 4})};
  data.labels = {n::from_vector(std::vector<std::int64_t>{3, 4, 5}),
                 n::zeros({0}, n::DType::Int64)};
  const auto t = m::MakeYoloV1Targets(data, {0, 1}, 10);
  EXPECT_EQ(t.discarded, 2);
  EXPECT_EQ(t.labels.at({0, 1, 1}), 4);
  EXPECT_EQ(n::sum(t.object).item(), 1);
  EXPECT_NEAR(t.boxes.at({0, 1, 1, 0}), .5, 1e-6);
  EXPECT_NEAR(t.boxes.at({0, 1, 1, 2}), std::sqrt(10. / 56.), 1e-6);
}
TEST(NnYoloV1, DecoderSquaresNegativeRootsClipsScoresAndSeparatesClasses) {
  const auto x = n::tensor({2., .5, .5, .5, -.5, -.5, 2., .5, .5, .5, .5, .8}, {1, 1, 1, 12});
  const auto raw =
      m::DecodeYoloV1Detections(x, n::zeros({1}, n::DType::Int64), 56, 56, 2, false);
  EXPECT_EQ(raw.scores.numel(), 4);
  EXPECT_EQ(raw.boxes.to_vector<int>(), std::vector<int>({21, 21, 35, 35, 21, 21, 35, 35, 21,
                                                          21, 35, 35, 21, 21, 35, 35}));
  const auto out = m::DecodeYoloV1Detections(x, n::zeros({1}, n::DType::Int64), 56, 56, 2);
  EXPECT_EQ(out.labels.to_vector<int>(), std::vector<int>({0, 1}));
  EXPECT_EQ(out.scores.at({0}), 1);
  EXPECT_EQ(out.scores.at({1}), .5);
}
TEST(NnYoloV1, FullPaper448BackboneAll48DerivativesAndAnalyticDenseHeadCount) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  const auto net = m::YoloV1Backbone(
      p, nullptr, n::sin(.13 * n::arange(0, 448 * 448 * 3)).reshape({1, 448, 448, 3}));
  EXPECT_EQ(net.convolutions.size(), 24);
  EXPECT_EQ(net.features.shape(), n::Shape({1, 7, 7, 1024}));
  const std::int64_t head = std::int64_t(7 * 7 * 1024) * 4096 + 4096 + 4096 * 1470 + 1470;
  EXPECT_EQ(p.count() + head, 271703550);
  const auto g = n::grad(n::mean(n::square(net.features)), p.parameters());
  EXPECT_EQ(g.size(), 48);
  for (const auto &v : g) {
    EXPECT_TRUE(v.defined());
    EXPECT_TRUE(std::isfinite(n::sum(n::square(v)).item()));
  }
}
TEST(NnYoloV1, CompactBothNormalizationModesEveryLearnedGradientAndStateCounts) {
  n::set_num_threads(2);
  for (bool bn : {false, true}) {
    n::ParameterStore p(17), state(17);
    m::YoloV1Config c;
    c.classes = 10;
    c.compact = true;
    c.width_divisor = 8;
    c.dense_width = 256;
    c.batch_norm = bn;
    c.keep_probability = 1;
    c.update_stats = false;
    const auto net = m::YoloV1(
        p, &state, n::sin(.13 * n::arange(0, 2 * 56 * 56)).reshape({2, 56, 56, 1}), c, true);
    EXPECT_EQ(p.count(), 2799372 + (bn ? 1648 : 0));
    EXPECT_EQ(state.count(), bn ? 3296 : 0);
    EXPECT_EQ(net.prediction.shape(), n::Shape({2, 7, 7, 20}));
    for (const auto &v : n::grad(n::mean(n::square(net.prediction)), p.parameters())) {
      EXPECT_TRUE(v.defined());
      const double norm = n::sum(n::square(v)).item();
      EXPECT_TRUE(std::isfinite(norm));
      EXPECT_GT(norm, 0);
    }
  }
}
TEST(NnYoloV1, PretrainingCropsKeepRowsSeparatePaddingBytesAndClassLabels) {
  m::TensorDetectionScenes source;
  source.images =
      n::concat({n::full({1, 4, 4, 1}, 128. / 255.), n::full({1, 4, 4, 1}, 200. / 255.)}, 0);
  source.boxes = {n::tensor({1., 1., 3., 3.}, {1, 4}), n::tensor({1., 1., 3., 3.}, {1, 4})};
  source.labels = {n::from_vector(std::vector<std::int64_t>{3}),
                   n::from_vector(std::vector<std::int64_t>{7})};
  const auto fit = m::YoloV1PretrainingCrops(source, {0}),
             held = m::YoloV1PretrainingCrops(source, {1});
  EXPECT_EQ(fit.images.shape(), n::Shape({1, 56, 56, 1}));
  EXPECT_EQ(fit.images.at({0, 0, 0, 0}), 0);
  EXPECT_NEAR(fit.images.at({0, 28, 28, 0}) * 255., 128, 1e-5);
  EXPECT_NEAR(held.images.at({0, 28, 28, 0}) * 255., 200, 1e-5);
  EXPECT_EQ(fit.labels[0].at({0}), 3);
  source.labels[1] = n::from_vector(std::vector<std::int64_t>{9});
  source.images = n::concat({n::slice_axis(source.images, 0, 0, 1), n::ones({1, 4, 4, 1})}, 0);
  EXPECT_EQ(m::YoloV1PretrainingCrops(source, {0}).images.to_vector<float>(),
            fit.images.to_vector<float>());
}
TEST(NnYoloV1,
     VisiblePrefixOptimizerAndTransferPreserveLastFourHeadLeavesAndDestinationIdentity) {
  n::set_num_threads(2);
  for (bool bn : {false, true}) {
    n::ParameterStore pc(17), sc(17), p(17), state(17);
    m::YoloV1Config c;
    c.classes = 10;
    c.compact = true;
    c.width_divisor = 8;
    c.dense_width = 8;
    c.keep_probability = 1;
    c.batch_norm = bn;
    {
      n::NoGradGuard guard;
      m::YoloV1PretrainingLogits(pc, sc, n::zeros({1, 56 * 56}), c, false);
      m::YoloV1(p, &state, n::zeros({1, 56, 56, 1}), c, false);
    }
    EXPECT_EQ(pc.count(), bn ? 353714 : 352578);
    const auto untouched = p.at("fc1/W").clone(), handle = p.at("conv1/W");
    n::OptimizerConfig config;
    config.kind = n::OptimizerKind::Adam;
    n::Optimizer adam(pc, .001, config);
    const auto input = n::sin(.13 * n::arange(0, 2 * 56 * 56)).reshape({2, 56 * 56}),
               labels = n::from_vector(std::vector<std::int64_t>{3, 7});
    const auto logits = m::YoloV1PretrainingLogits(pc, sc, input, c, true);
    const auto loss = n::mean(n::sparse_softmax_cross_entropy(logits, labels));
    adam.minimize(loss);
    m::CopyYoloV1Prefix(pc, sc, p, state);
    for (const auto &name : pc.names())
      if (name.rfind("conv", 0) == 0)
        EXPECT_TRUE(n::all(n::equal(p.at(name), pc.at(name))).item<bool>());
    EXPECT_TRUE(n::all(n::equal(handle, pc.at("conv1/W"))).item<bool>());
    EXPECT_TRUE(n::all(n::equal(untouched, p.at("fc1/W"))).item<bool>());
    for (const auto &name : sc.names())
      EXPECT_TRUE(n::all(n::equal(state.at(name), sc.at(name))).item<bool>());
    if (bn) {
      n::ParameterStore incomplete(17), empty(17);
      for (int i = 1; i <= 20; ++i) {
        const auto base = "conv" + std::to_string(i);
        incomplete.set(base + "/W", pc.at(base + "/W"));
        incomplete.set(base + "/bn/gamma", pc.at(base + "/bn/gamma"));
      }
      EXPECT_THROW(m::CopyYoloV1Prefix(incomplete, empty, p, state), std::invalid_argument);
    }
  }
}
TEST(NnYoloV1, InvalidGeometryAffineValuesAndMissingBatchNormStateRejected) {
  n::ParameterStore p(17);
  m::YoloV1Config c;
  c.batch_norm = true;
  EXPECT_THROW(m::YoloV1(p, nullptr, n::zeros({1, 448, 448, 3}), c), std::invalid_argument);
  EXPECT_THROW(m::YoloV1Backbone(p, nullptr, n::zeros({1, 56, 56, 1})), std::invalid_argument);
  m::TensorDetectionScenes source;
  source.images = n::zeros({1, 4, 4, 1});
  source.boxes = {n::zeros({0, 4})};
  source.labels = {n::zeros({0}, n::DType::Int64)};
  EXPECT_THROW(
      m::TransformTensorDetectionScenes(source, {0}, n::tensor({0., 0., 0., 1.}, {1, 4})),
      std::invalid_argument);
}
