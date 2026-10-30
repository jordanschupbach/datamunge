
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnInstanceScenes, RetentionPreservesEveryScenePixelBoxLabelAndTightSupportBounds) {
  const auto y = n::arange(0, 28).reshape({28, 1}), x = n::arange(0, 28).reshape({1, 28});
  const auto a = n::logical_and(n::logical_and(n::greater(y, 4.), n::less(y, 23.)),
                                n::logical_and(n::greater(x, 9.), n::less(x, 18.)))
                     .to(n::DType::Float32),
             b = n::less(n::abs(y - x), 3.).to(n::DType::Float32);
  const auto digits = n::stack({a, b, n::zeros({28, 28})}, 0).reshape({3, 28, 28, 1}),
             labels = n::from_vector(std::vector<std::int64_t>{1, 2, 3});
  m::DetectionSceneConfig c;
  c.scenes = 6;
  c.height = 32;
  c.width = 40;
  c.min_size = 8;
  c.max_size = 20;
  c.seed = 19;
  const auto plain = m::MakeTensorDetectionScenes(digits, labels, c);
  const auto data = m::MakeTensorInstanceScenes(digits, labels, c);
  EXPECT_EQ(data.images.to_vector<float>(), plain.images.to_vector<float>());
  EXPECT_EQ(data.masks.size(), 6);
  for (int row = 0; row < 6; ++row) {
    EXPECT_EQ(data.boxes[row].to_vector<float>(), plain.boxes[row].to_vector<float>());
    EXPECT_EQ(data.labels[row].to_vector<int>(), plain.labels[row].to_vector<int>());
    const auto K = data.boxes[row].shape()[0];
    EXPECT_EQ(data.masks[row].shape(), n::Shape({K, 32, 40}));
    EXPECT_EQ(data.masks[row].dtype(), n::DType::Bool);
    auto combined = n::zeros({32, 40}, n::DType::Bool);
    for (std::int64_t k = 0; k < K; ++k) {
      const auto mask = n::slice_axis(data.masks[row], 0, k, k + 1).reshape({32, 40});
      combined = n::logical_or(combined, mask);
      const auto pixels = mask.to_vector<int>();
      int y0 = 32, x0 = 40, y1 = 0, x1 = 0;
      for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 40; ++c)
          if (pixels[r * 40 + c]) {
            y0 = std::min(y0, r);
            x0 = std::min(x0, c);
            y1 = std::max(y1, r + 1);
            x1 = std::max(x1, c + 1);
          }
      EXPECT_EQ(data.boxes[row].at({k, 0}), y0);
      EXPECT_EQ(data.boxes[row].at({k, 1}), x0);
      EXPECT_EQ(data.boxes[row].at({k, 2}), y1);
      EXPECT_EQ(data.boxes[row].at({k, 3}), x1);
    }
    const auto image = n::slice_axis(data.images, 0, row, row + 1).reshape({32, 40});
    const auto expected = n::greater(n::floor(image.to(n::DType::Float64) * 255. + .5), 32.);
    EXPECT_EQ(combined.to_vector<int>(), expected.to_vector<int>());
    EXPECT_FALSE(data.masks[row].requires_grad());
  }
  EXPECT_FALSE(data.images.requires_grad());
}
TEST(NnInstanceScenes, BlankDigitsProduceAlignedEmptySupportsAndZeroCanvas) {
  m::DetectionSceneConfig c;
  c.scenes = 2;
  const auto data =
      m::MakeTensorInstanceScenes(n::zeros({1, 28, 28, 1}), n::zeros({1}, n::DType::Int64), c);
  EXPECT_EQ(n::sum(data.images).item(), 0);
  for (const auto &masks : data.masks)
    EXPECT_EQ(masks.shape(), n::Shape({0, 64, 64}));
}
TEST(NnInstanceScenes, LogicalIoUAndEmptyUnionAreExact) {
  const auto a = n::from_vector(std::vector<bool>{true, false}),
             b = n::from_vector(std::vector<bool>{true, true});
  EXPECT_DOUBLE_EQ(m::InstanceMaskIoU(a, b).item(), .5);
  EXPECT_EQ(
      m::InstanceMaskIoU(n::zeros({0}, n::DType::Bool), n::zeros({0}, n::DType::Bool)).item(),
      0);
  EXPECT_THROW(m::InstanceMaskIoU(n::zeros({2}), n::zeros({2})), std::invalid_argument);
}
TEST(NnInstanceScenes, PastedMaskEveryPixelMatchesIndependentBorderBilinearAndClassSelection) {
  const std::vector<double> values = {.15, .9, .85, .1, .7, .3, .25, .8};
  const auto p = n::from_vector(values, {2, 2, 2}),
             box = n::from_vector(std::vector<double>{.25, 1.5, 6.75, 8.25});
  for (int label = 0; label < 2; ++label) {
    const auto actual = m::PasteInstanceMask(p, label, box, 7, 9).to_vector<int>();
    for (int y = 0; y < 7; ++y)
      for (int x = 0; x < 9; ++x) {
        int expected = 0;
        if (y + .5 >= .25 && y + .5 < 6.75 && x + .5 >= 1.5 && x + .5 < 8.25) {
          const double sy = std::clamp((y + .5 - .25) / 6.5 * 2 - .5, 0., 1.),
                       sx = std::clamp((x + .5 - 1.5) / 6.75 * 2 - .5, 0., 1.);
          const int iy = int(sy), ix = int(sx);
          double v = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
              v += (dy ? sy - iy : 1 - (sy - iy)) * (dx ? sx - ix : 1 - (sx - ix)) *
                   values[(std::min(iy + dy, 1) * 2 + std::min(ix + dx, 1)) * 2 + label];
          expected = v >= .5;
        }
        EXPECT_EQ(actual[y * 9 + x], expected);
      }
  }
  const auto exact =
      m::PasteInstanceMask(n::full({2, 2, 1}, .5), 0, n::tensor({1., 1., 3., 3.}), 4, 4);
  EXPECT_EQ(n::count_nonzero(exact).item(), 4);
}

TEST(NnInstanceScenes, IntegralMaskApDuplicatesConfidenceAndFirstTruthTieSemantics) {
  m::TensorInstanceScenes data;
  data.images = n::zeros({1, 2, 2, 1});
  data.boxes = {n::tensor({0., 0., 2., 2.}, {1, 4})};
  data.labels = {n::zeros({1}, n::DType::Int64)};
  data.masks = {n::from_vector(std::vector<bool>{true, false, false, true}, {1, 2, 2})};
  m::TensorMaskDetections pred{{n::zeros({2, 4}), n::from_vector(std::vector<float>{.9f, .8f}),
                                n::zeros({2}, n::DType::Int64),
                                n::zeros({2}, n::DType::Int64)},
                               n::concat({data.masks[0], data.masks[0]}, 0)};
  EXPECT_EQ(m::InstanceMaskAveragePrecision(pred, data, 1).item(), 1);
  pred.masks = n::concat({n::logical_not(data.masks[0]), data.masks[0]}, 0);
  EXPECT_EQ(m::InstanceMaskAveragePrecision(pred, data, 1).item(), .5);
  pred.objects.scores = n::full({2}, .5);
  EXPECT_EQ(m::InstanceMaskAveragePrecision(pred, data, 1).item(), .5);
  pred.masks = n::concat({data.masks[0], data.masks[0]}, 0);
  data.boxes[0] = n::concat({data.boxes[0], data.boxes[0]}, 0);
  data.labels[0] = n::zeros({2}, n::DType::Int64);
  data.masks[0] = n::concat({data.masks[0], data.masks[0]}, 0);
  EXPECT_EQ(m::InstanceMaskAveragePrecision(pred, data, 1).item(), .5);
}
TEST(NnInstanceScenes, MaskApEmptyPredictionsAbsentClassesAndInclusiveThreshold) {
  m::TensorInstanceScenes data;
  data.images = n::zeros({1, 1, 2, 1});
  data.boxes = {n::tensor({0., 0., 1., 2.}, {1, 4})};
  data.labels = {n::zeros({1}, n::DType::Int64)};
  data.masks = {n::ones({1, 1, 2}, n::DType::Bool)};
  m::TensorMaskDetections pred{{n::zeros({1, 4}), n::ones({1}), n::zeros({1}, n::DType::Int64),
                                n::zeros({1}, n::DType::Int64)},
                               n::from_vector(std::vector<bool>{true, false}, {1, 1, 2})};
  EXPECT_EQ(m::InstanceMaskAveragePrecision(pred, data, 3, .5).item(), 1);
  EXPECT_EQ(m::InstanceMaskAveragePrecision(pred, data, 3, .50001).item(), 0);
  m::TensorMaskDetections empty{{n::zeros({0, 4}), n::zeros({0}),
                                 n::zeros({0}, n::DType::Int64),
                                 n::zeros({0}, n::DType::Int64)},
                                n::zeros({0, 1, 2}, n::DType::Bool)};
  EXPECT_EQ(m::InstanceMaskAveragePrecision(empty, data, 3).item(), 0);
}
