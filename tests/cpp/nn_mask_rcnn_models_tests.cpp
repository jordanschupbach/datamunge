#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
namespace m = nn::models;
using nn::Tensor;
using Box = nn::Box;
namespace check {
template <typename A, typename B> void Near(A a, B b, double t, const std::string &msg) {
  EXPECT_NEAR(double(a), double(b), t) << msg;
}
inline void Require(bool ok, const std::string &msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
} // namespace check
namespace {
void AlignChecks() {
  std::vector<float> values(25);
  for (int i = 0; i < 25; ++i)
    values[std::size_t(i)] = std::sin(.3f * i);
  Tensor input = nn::from_vector(values, {1, 5, 5, 1});
  input.requires_grad_(true);
  const std::vector<Box> boxes = {{0, 0, 20, 20}, {2.3f, 3.1f, 15.7f, 18.6f}};
  const Tensor aligned = m::MaskRoiAlign(input, nn::boxes_from_vector(boxes),
                                         nn::zeros({2}, nn::DType::Int64), 4, 3);
  const auto gradient = nn::grad(nn::sum(aligned), {input})[0].to_vector<float>();
  const auto out = aligned.to_vector<float>();
  std::vector<double> derivatives(25, 0);
  double error = 0;
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        double expected = 0;
        const Box &box = boxes[std::size_t(b)];
        for (int a = 0; a < 2; ++a)
          for (int z = 0; z < 2; ++z) {
            const double sy = std::clamp(
                             (box[0] + (y + (a + .5) / 2) / 3 * (box[2] - box[0])) / 4 - .5,
                             0., 4.),
                         sx = std::clamp(
                             (box[1] + (x + (z + .5) / 2) / 3 * (box[3] - box[1])) / 4 - .5,
                             0., 4.);
            const int iy = int(sy), ix = int(sx);
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx) {
                const int i = std::min(iy + dy, 4) * 5 + std::min(ix + dx, 4);
                const double w =
                    (dy ? sy - iy : 1 - (sy - iy)) * (dx ? sx - ix : 1 - (sx - ix)) * .25;
                expected += w * values[std::size_t(i)];
                derivatives[std::size_t(i)] += w;
              }
          }
        error = std::max(error, std::abs(expected - out[std::size_t((b * 3 + y) * 3 + x)]));
      }
  for (int i = 0; i < 25; ++i)
    error = std::max(error, std::abs(derivatives[std::size_t(i)] - gradient[std::size_t(i)]));
  check::Near(error, 0, 3e-6, "fractional RoIAlign every value and feature derivative");
  std::cout << "  aligned2x2 samples/bin: independent border/fractional value/gradient error="
            << error << std::endl;
}

void MaskChecks() {
  {
    Tensor logits = nn::zeros({2, 2, 2, 3});
    logits.requires_grad_(true);
    const std::vector<float> targets = {0, 1, 0, 1, 1, 1, 0, 0};
    const Tensor loss =
        m::SelectedInstanceMaskLoss(logits, nn::from_vector(std::vector<std::int64_t>{0, 2}),
                                    nn::from_vector(targets, {2, 2, 2}));
    const auto g = nn::grad(loss, {logits})[0].to_vector<float>();
    check::Near(loss.item<double>(), std::log(2.), 1e-7, "zero-logit mask BCE");
    for (int n = 0; n < 2; ++n)
      for (int pixel = 0; pixel < 4; ++pixel)
        for (int c = 0; c < 3; ++c)
          check::Near(g[std::size_t((n * 4 + pixel) * 3 + c)],
                      c == (n ? 2 : 0) ? (.5 - targets[std::size_t(n * 4 + pixel)]) / 8 : 0,
                      1e-7, "only matched class gets mask gradient");
  }
  for (bool full : {true, false}) {
    nn::ParameterStore p(17);
    const int in = full ? 256 : 64;
    std::vector<float> values(std::size_t(2 * 14 * 14 * in));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = std::sin(.13f * float(i));
    const Tensor output = m::MaskRcnnHead(p, nn::from_vector(values, {2, 14, 14, in}),
                                          full ? 256 : 32, full ? 80 : 10);
    const auto grads = nn::grad(nn::mean(nn::square(output)), p.parameters());
    check::Require(grads.size() == 12, "four conv,upconv,score parameters all connected");
    check::Require(output.shape() == nn::Shape({2, 28, 28, full ? 80 : 10}),
                   "learned upconv produces28px masks");
    for (const auto &g : grads) {
      const double norm = nn::sum(nn::square(g)).item<double>();
      check::Require(std::isfinite(norm) && (full || norm > 0),
                     "finite nonzero mask parameter derivative");
    }
    std::cout << "  " << (full ? "full" : "compact") << " mask head: " << p.count()
              << " parameters, " << grads.size() << " derivatives" << std::endl;
  }
}

} // namespace
TEST(NnMaskRcnn, IndependentFractionalBorderEveryAlignedValueAndFeatureDerivative) {
  AlignChecks();
}
TEST(NnMaskRcnn, SelectedClassBceWrongClassZerosFullCompactEveryHeadGradient) {
  nn::set_num_threads(2);
  MaskChecks();
}
TEST(NnMaskRcnn, NearestSupportTargetsUseBestBoxFirstTieAndDetach) {
  m::TensorInstanceScenes data;
  data.images = nn::zeros({1, 4, 6, 1});
  data.boxes = {nn::tensor({0., 0., 4., 6., 0., 0., 4., 6.}, {2, 4})};
  data.labels = {nn::from_vector(std::vector<std::int64_t>{1, 1})};
  const auto support =
      nn::equal(nn::arange(0, 24) - nn::floor(nn::arange(0, 24) / 3.) * 3., 0.)
          .reshape({1, 4, 6});
  data.masks = {nn::concat({support, nn::logical_not(support)}, 0)};
  m::TensorRoiBatch batch;
  batch.pixels = nn::tensor({-1., 1., 5., 7., 0., 0., 4., 6.}, {2, 4});
  batch.indices = nn::zeros({2}, nn::DType::Int64);
  batch.labels = nn::ones({2}, nn::DType::Int64);
  batch.mask = nn::tensor({1., 0.});
  const auto result = m::PositiveInstanceMaskTargets(data, {0}, batch, 3);
  EXPECT_EQ(result.targets.shape(), nn::Shape({1, 3, 3}));
  const auto values = result.targets.to_vector<float>();
  const auto host = support.to_vector<bool>();
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) {
      const int yy = std::clamp(int(-1 + (y + .5) * 6 / 3), 0, 3),
                xx = std::clamp(int(1 + (x + .5) * 6 / 3), 0, 5);
      EXPECT_EQ(values[y * 3 + x], float(host[yy * 6 + xx]));
    }
  EXPECT_FALSE(result.targets.requires_grad());
  batch.labels = nn::zeros({2}, nn::DType::Int64);
  EXPECT_THROW(m::PositiveInstanceMaskTargets(data, {0}, batch), std::invalid_argument);
}
TEST(NnMaskRcnn, EmptyRoisAndSelectedMaskLossRemainDifferentiable) {
  auto feature = nn::ones({1, 5, 7, 2}).requires_grad_();
  const auto aligned =
      m::MaskRoiAlign(feature, nn::zeros({0, 4}), nn::zeros({0}, nn::DType::Int64), 4, 3);
  EXPECT_EQ(aligned.shape(), nn::Shape({0, 3, 3, 2}));
  EXPECT_EQ(nn::sum(nn::grad(nn::sum(aligned), {feature})[0]).item(), 0);
  auto z = nn::zeros({0, 3, 3, 2}).requires_grad_();
  const auto loss =
      m::SelectedInstanceMaskLoss(z, nn::zeros({0}, nn::DType::Int64), nn::zeros({0, 3, 3}));
  EXPECT_EQ(loss.item(), 0);
  EXPECT_TRUE(nn::grad(loss, {z})[0].defined());
  EXPECT_THROW(m::MaskAlignmentBoxes(nn::zeros({1, 4}), 5, 7, 4, 3), std::invalid_argument);
}
TEST(NnMaskRcnn, RawDecodeKeepsAllClassesInputOrderAndDefaultNmsUnchanged) {
  m::DetectionRoiOutputs head;
  head.probabilities = nn::tensor({.7, .2, .1, .8, .15, .05}, {2, 3});
  head.deltas = nn::zeros({2, 2, 4});
  const auto boxes = nn::tensor({0., 0., 10., 10., 0., 0., 10., 10.}, {2, 4}),
             ids = nn::full({2}, 3., nn::DType::Int64);
  const auto raw = m::DecodeDetectionRoiOutputs(head, boxes, ids, true, .3, .05, 0, false);
  EXPECT_EQ(raw.scores.to_vector<float>(), (std::vector<float>{.7f, .2f, .8f, .15f}));
  EXPECT_EQ(raw.labels.to_vector<int>(), (std::vector<int>{0, 1, 0, 1}));
  const auto final = m::DecodeDetectionRoiOutputs(head, boxes, ids);
  EXPECT_EQ(final.scores.to_vector<float>(), (std::vector<float>{.8f, .2f}));
  EXPECT_EQ(final.images.to_vector<int>(), (std::vector<int>{3, 3}));
}

TEST(NnMaskRcnn, PreviewKnownPaletteScoreOverlayAndInvalidShapes) {
  m::TensorInstanceScenes data;
  data.images = nn::full({1, 2, 2, 1}, .25);
  data.boxes = {nn::tensor({0., 0., 2., 2.}, {1, 4})};
  data.labels = {nn::zeros({1}, nn::DType::Int64)};
  data.masks = {nn::tensor({1., 0., 0., 0.}, {1, 2, 2}).to(nn::DType::Bool)};
  m::TensorMaskDetections prediction{{nn::zeros({2, 4}), nn::tensor({.9, .1}),
                                      nn::from_vector(std::vector<std::int64_t>{1, 2}),
                                      nn::zeros({2}, nn::DType::Int64)},
                                     nn::ones({2, 2, 2}, nn::DType::Bool)};
  const auto preview = m::InstanceMaskPreview(data, &prediction);
  EXPECT_EQ(preview.shape(), nn::Shape({1, 2, 6, 3}));
  const auto v = preview.to_vector<float>();
  for (int k = 0; k < 3; ++k)
    EXPECT_FLOAT_EQ(v[k], .25f);
  EXPECT_FLOAT_EQ(v[6], 80 / 255.f);
  EXPECT_FLOAT_EQ(v[12], 117 / 255.f);
  EXPECT_THROW(m::InstanceMaskPreview(data, nullptr, 0), std::invalid_argument);
  nn::ParameterStore p(17);
  EXPECT_THROW(m::DetectCompactMaskRcnn(p, data, {0}), std::invalid_argument);
}
