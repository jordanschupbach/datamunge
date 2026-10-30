#include <deque>
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include <random>

// Independent legacy host pixel/graph oracle; production numerics use tensors.
namespace detector {
using Box = datamunge::nn::Box;
struct DetectionScenes {
  int height = 64, width = 64;
  std::vector<std::uint8_t> pixels;
  std::vector<std::vector<Box>> boxes;
  std::vector<std::vector<int>> labels;
  int size() const {
    return int(boxes.size());
  }
  const std::uint8_t *image(int i) const {
    return pixels.data() + std::size_t(i) * height * width;
  }
};
inline float BoxArea(const Box &b) {
  return std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]);
}
/// tflow's data::BoxIoU (zero for touching or disjoint boxes).
inline float BoxIoU(const Box &a, const Box &b) {
  const float h = std::min(a[2], b[2]) - std::max(a[0], b[0]),
              w = std::min(a[3], b[3]) - std::max(a[1], b[1]);
  if (h <= 0 || w <= 0)
    return 0;
  const float inter = h * w;
  return inter / (BoxArea(a) + BoxArea(b) - inter);
}

struct Digits {
  std::vector<std::uint8_t> pixels;
  std::vector<int> labels;
  [[nodiscard]] int size() const {
    return int(labels.size());
  }
};
struct SceneConfig {
  int scenes = 1000, height = 64, width = 64, max_objects = 4, min_size = 16, max_size = 32;
  float max_overlap = .3f;
  std::uint64_t seed = 1;
};
inline DetectionScenes MakeDetectionScenes(const Digits &digits, const SceneConfig &cfg) {
  if (digits.size() < 1 || cfg.max_objects < 1 || cfg.min_size < 4 ||
      cfg.max_size < cfg.min_size || cfg.max_size > std::min(cfg.height, cfg.width))
    throw std::invalid_argument("invalid detection scene configuration");
  DetectionScenes out;
  out.height = cfg.height;
  out.width = cfg.width;
  out.pixels.assign(std::size_t(cfg.scenes) * cfg.height * cfg.width, 0);
  std::mt19937_64 random(cfg.seed);
  std::uniform_int_distribution<int> count(1, cfg.max_objects), pick(0, digits.size() - 1),
      size(cfg.min_size, cfg.max_size);
  for (int n = 0; n < cfg.scenes; ++n) {
    std::vector<Box> boxes;
    std::vector<int> classes;
    std::uint8_t *canvas = out.pixels.data() + std::size_t(n) * cfg.height * cfg.width;
    const int objects = count(random);
    for (int k = 0; k < objects; ++k) {
      for (int attempt = 0; attempt < 30; ++attempt) {
        const int digit = pick(random), cell = size(random);
        std::uniform_int_distribution<int> top(0, cfg.height - cell),
            left(0, cfg.width - cell);
        const int y0 = top(random), x0 = left(random);
        const std::uint8_t *source = digits.pixels.data() + std::size_t(digit) * 784;
        // Rescale the 28x28 digit to cell x cell (bilinear, pixel centres aligned).
        std::vector<std::uint8_t> patch(std::size_t(cell) * cell);
        int ymin = cell, xmin = cell, ymax = -1, xmax = -1;
        for (int r = 0; r < cell; ++r)
          for (int c = 0; c < cell; ++c) {
            const double sy = (r + .5) * 28. / cell - .5, sx = (c + .5) * 28. / cell - .5;
            const int iy = int(std::floor(sy)), ix = int(std::floor(sx));
            double v = 0;
            for (int a = 0; a <= 1; ++a)
              for (int b = 0; b <= 1; ++b) {
                const int yy = std::clamp(iy + a, 0, 27), xx = std::clamp(ix + b, 0, 27);
                v += (a ? sy - iy : 1 - (sy - iy)) * (b ? sx - ix : 1 - (sx - ix)) *
                     source[yy * 28 + xx];
              }
            auto &p = patch[std::size_t(r) * cell + c];
            p = std::uint8_t(std::clamp(std::lround(v), 0L, 255L));
            if (p > 32) {
              ymin = std::min(ymin, r);
              xmin = std::min(xmin, c);
              ymax = std::max(ymax, r);
              xmax = std::max(xmax, c);
            }
          }
        if (ymax < 0)
          continue;
        const Box box = {float(y0 + ymin), float(x0 + xmin), float(y0 + ymax + 1),
                         float(x0 + xmax + 1)};
        bool clear = true;
        for (const auto &b : boxes)
          if (BoxIoU(b, box) > cfg.max_overlap) {
            clear = false;
            break;
          }
        if (!clear)
          continue;
        for (int r = 0; r < cell; ++r)
          for (int c = 0; c < cell; ++c) {
            std::uint8_t &p = canvas[(y0 + r) * cfg.width + x0 + c];
            p = std::max(p, patch[std::size_t(r) * cell + c]);
          }
        boxes.push_back(box);
        classes.push_back(digits.labels[std::size_t(digit)]);
        break;
      }
    }
    out.boxes.push_back(boxes);
    out.labels.push_back(classes);
  }
  return out;
}

inline std::vector<Box> ComponentProposals(const DetectionScenes &scenes, int row,
                                           const std::vector<int> &thresholds, int gap) {
  const int h = scenes.height, w = scenes.width;
  const std::uint8_t *image = scenes.image(row);
  std::vector<Box> out;
  for (int threshold : thresholds) {
    std::vector<int> component(std::size_t(h) * w, -1);
    std::vector<Box> boxes;
    for (int start = 0; start < h * w; ++start) {
      if (image[start] <= threshold || component[std::size_t(start)] >= 0)
        continue;
      Box b = {float(h), float(w), 0, 0};
      std::deque<int> queue = {start};
      component[std::size_t(start)] = int(boxes.size());
      while (!queue.empty()) {
        const int p = queue.front();
        queue.pop_front();
        const int r = p / w, c = p % w;
        b = {std::min(b[0], float(r)), std::min(b[1], float(c)), std::max(b[2], float(r + 1)),
             std::max(b[3], float(c + 1))};
        for (int dr = -1; dr <= 1; ++dr)
          for (int dc = -1; dc <= 1; ++dc) {
            const int rr = r + dr, cc = c + dc;
            if (rr < 0 || rr >= h || cc < 0 || cc >= w || image[rr * w + cc] <= threshold ||
                component[std::size_t(rr * w + cc)] >= 0)
              continue;
            component[std::size_t(rr * w + cc)] = int(boxes.size());
            queue.push_back(rr * w + cc);
          }
      }
      if ((b[2] - b[0]) * (b[3] - b[1]) >= 9)
        boxes.push_back(b);
    }
    out.insert(out.end(), boxes.begin(), boxes.end());
    for (std::size_t i = 0; i < boxes.size(); ++i)
      for (std::size_t j = i + 1; j < boxes.size(); ++j) {
        const float dy = std::max({0.f, boxes[i][0] - boxes[j][2], boxes[j][0] - boxes[i][2]});
        const float dx = std::max({0.f, boxes[i][1] - boxes[j][3], boxes[j][1] - boxes[i][3]});
        if (dy <= gap && dx <= gap)
          out.push_back(
              {std::min(boxes[i][0], boxes[j][0]), std::min(boxes[i][1], boxes[j][1]),
               std::max(boxes[i][2], boxes[j][2]), std::max(boxes[i][3], boxes[j][3])});
      }
  }
  std::vector<Box> unique;
  for (const auto &b : out)
    if (std::find(unique.begin(), unique.end(), b) == unique.end())
      unique.push_back(b);
  return unique;
}

} // namespace detector

namespace n = dm::nn;
namespace m = n::models;
TEST(NnDetectionScenes, TensorScenesMatchIndependentHostPixelAndAnnotationOracle) {
  detector::Digits old;
  old.labels = {1, 3, 8};
  old.pixels.resize(3 * 784);
  for (int d = 0; d < 3; ++d)
    for (int r = 0; r < 28; ++r)
      for (int c = 0; c < 28; ++c)
        old.pixels[d * 784 + r * 28 + c] =
            (r >= 4 + d && r < 23 - d && c >= 9 - d && c < 18 + d) ? 255 : 0;
  const auto images =
      n::from_vector(std::vector<int>(old.pixels.begin(), old.pixels.end()), {3, 28, 28, 1})
          .to(n::DType::Float32) /
      255.;
  m::DetectionSceneConfig cfg;
  cfg.scenes = 8;
  cfg.max_objects = 3;
  cfg.seed = 11;
  detector::SceneConfig legacy;
  legacy.scenes = 8;
  legacy.max_objects = 3;
  legacy.seed = 11;
  const auto expected = detector::MakeDetectionScenes(old, legacy);
  const auto actual = m::MakeTensorDetectionScenes(
      images, n::from_vector(std::vector<std::int64_t>{1, 3, 8}), cfg);
  EXPECT_FALSE(actual.images.requires_grad());
  EXPECT_EQ(actual.images.shape(), n::Shape({8, 64, 64, 1}));
  const auto bytes = n::floor(actual.images * 255. + .5).to_vector<int>();
  int maxerror = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i)
    maxerror = std::max(maxerror, std::abs(bytes[i] - int(expected.pixels[i])));
  EXPECT_LE(maxerror, 1);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(actual.labels[i].to_vector<int>(), expected.labels[i]);
    const auto boxes = n::boxes_from_vector(expected.boxes[i]);
    EXPECT_EQ(actual.boxes[i].shape(), boxes.shape());
    if (boxes.numel())
      EXPECT_EQ(n::max(n::abs(actual.boxes[i] - boxes)).item(), 0);
  }
  const auto repeat = m::MakeTensorDetectionScenes(
      images, n::from_vector(std::vector<std::int64_t>{1, 3, 8}), cfg);
  EXPECT_EQ(n::max(n::abs(actual.images - repeat.images)).item(), 0);
}
TEST(NnDetectionScenes, BlankDigitsProduceDetachedEmptyAnnotations) {
  m::DetectionSceneConfig cfg;
  cfg.scenes = 2;
  cfg.max_objects = 1;
  const auto d = n::zeros({1, 28, 28, 1}, n::DType::Float64).requires_grad_();
  const auto s = m::MakeTensorDetectionScenes(d, n::zeros({1}, n::DType::Int64), cfg);
  EXPECT_EQ(n::max(s.images).item(), 0);
  EXPECT_EQ(s.images.dtype(), n::DType::Float64);
  EXPECT_FALSE(s.images.requires_grad());
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(s.boxes[i].shape(), n::Shape({0, 4}));
    EXPECT_EQ(s.labels[i].shape(), n::Shape({0}));
  }
}
TEST(NnDetectionScenes, PixelCentreCropsMatchBilinearOracleAndBoxFiniteDifferences) {
  const auto type = n::DType::Float64;
  const auto images = n::sin(.37 * n::arange(0, 2 * 8 * 9 * 2).to(type))
                          .reshape({2, 8, 9, 2})
                          .detach()
                          .requires_grad_();
  const auto rows = n::from_vector(std::vector<std::int64_t>{1, 0});
  const auto original =
      n::from_vector(std::vector<double>{-.7, .3, 7.2, 8.4, 1.1, 2.7, 6.8, 7.2}, {2, 4});
  const auto boxes = original.detach().requires_grad_();
  const auto y = m::CropDetectionPatches(images, rows, boxes, 7);
  for (int b = 0; b < 2; ++b)
    for (int r = 0; r < 7; ++r)
      for (int c = 0; c < 7; ++c)
        for (int ch = 0; ch < 2; ++ch) {
          const double sy = original.at({b, 0}) +
                            (r + .5) * (original.at({b, 2}) - original.at({b, 0})) / 7. - .5,
                       sx = original.at({b, 1}) +
                            (c + .5) * (original.at({b, 3}) - original.at({b, 1})) / 7. - .5;
          const int iy = int(std::floor(sy)), ix = int(std::floor(sx));
          double expected = 0;
          for (int dy = 0; dy <= 1; ++dy)
            for (int dx = 0; dx <= 1; ++dx) {
              const int yy = iy + dy, xx = ix + dx;
              if (yy >= 0 && yy < 8 && xx >= 0 && xx < 9)
                expected += (dy ? sy - iy : 1 - (sy - iy)) * (dx ? sx - ix : 1 - (sx - ix)) *
                            images.at({1 - b, yy, xx, ch});
            }
          EXPECT_NEAR(y.at({b, r, c, ch}), expected, 1e-12);
        }
  const auto derivatives = n::grad(n::sum(n::square(y)), {boxes, images});
  EXPECT_GT(n::sum(n::abs(derivatives[1])).item(), 0);
  const double step = 1e-6;
  n::NoGradGuard guard;
  for (int i = 0; i < 8; ++i) {
    const auto direction =
        n::one_hot(n::from_vector(std::vector<std::int64_t>{i}), 8, 1, 0, type)
            .reshape({2, 4});
    const double plus = n::sum(n::square(m::CropDetectionPatches(
                                   images, rows, original + step * direction, 7)))
                            .item(),
                 minus = n::sum(n::square(m::CropDetectionPatches(
                                    images, rows, original - step * direction, 7)))
                             .item();
    EXPECT_NEAR(derivatives[0].reshape({8}).at({i}), (plus - minus) / (2 * step), 2e-7);
  }
}
TEST(NnDetectionScenes, SquareGeometryPreservesCentreAndGradients) {
  const auto boxes =
      n::from_vector(std::vector<double>{0, 2, 10, 22, 3, 4, 5, 14}, {2, 4}).requires_grad_();
  const auto square = m::SquareDetectionBoxes(boxes, 1.2), c = n::boxes_to_center(square),
             original = n::boxes_to_center(boxes);
  EXPECT_LT(
      n::max(n::abs(n::slice_axis(c, 1, 0, 2) - n::slice_axis(original, 1, 0, 2))).item(),
      1e-12);
  EXPECT_NEAR(c.at({0, 2}), 24, 1e-12);
  EXPECT_NEAR(c.at({1, 3}), 12, 1e-12);
  EXPECT_GT(n::sum(n::abs(n::grad(n::sum(n::square(square)), {boxes})[0])).item(), 0);
}
TEST(NnDetectionScenes, ComponentsMatchIndependentHostThresholdConnectivityAndPairUnions) {
  detector::DetectionScenes old;
  old.height = old.width = 16;
  old.pixels.resize(256);
  old.boxes.resize(1);
  old.labels.resize(1);
  for (int r = 1; r < 4; ++r)
    for (int c = 1; c < 4; ++c)
      old.pixels[r * 16 + c] = 255;
  for (int r = 1; r < 4; ++r)
    for (int c = 7; c < 10; ++c)
      old.pixels[r * 16 + c] = 128;
  old.pixels[15 * 16 + 15] = 255;
  const auto image =
      n::from_vector(std::vector<int>(old.pixels.begin(), old.pixels.end()), {16, 16, 1})
          .to(n::DType::Float32) /
      255.;
  for (int gap : {0, 3})
    for (const auto thresholds :
         {std::vector<int>{32}, std::vector<int>{32, 128, 200}, std::vector<int>{255}}) {
      const auto expected = detector::ComponentProposals(old, 0, thresholds, gap);
      const auto actual = m::ComponentDetectionProposals(image, thresholds, gap);
      const auto oracle = n::boxes_from_vector(expected);
      EXPECT_EQ(actual.shape(), oracle.shape());
      if (oracle.numel())
        EXPECT_EQ(n::max(n::abs(actual - oracle)).item(), 0);
    }
  const auto pairs = m::ComponentDetectionProposals(image, {32}, 3);
  EXPECT_EQ(pairs.shape(), n::Shape({3, 4}));
  EXPECT_EQ(pairs.at({2, 3}), 10);
}
TEST(NnDetectionScenes, BoxDrawingIsPixelExactAndDoesNotMutateImages) {
  const auto image = n::full({1, 16, 16, 1}, .5), box = n::tensor({2., 3., 12., 13.}, {1, 4});
  const auto rgb = m::DrawDetectionBoxes(image, {box}, {1, 0, 0});
  EXPECT_EQ(rgb.shape(), n::Shape({1, 16, 16, 3}));
  EXPECT_EQ(rgb.at({0, 2, 3, 0}), 1);
  EXPECT_EQ(rgb.at({0, 11, 12, 1}), 0);
  EXPECT_EQ(rgb.at({0, 5, 5, 0}), .5);
  EXPECT_EQ(n::min(image).item(), .5);
  const auto empty =
      m::CropDetectionPatches(image, n::zeros({0}, n::DType::Int64), n::zeros({0, 4}), 7);
  EXPECT_EQ(empty.shape(), n::Shape({0, 7, 7, 1}));
}
TEST(NnDetectionScenes, InvalidScenesCropsProposalsAndDrawingsFail) {
  m::DetectionSceneConfig cfg;
  cfg.scenes = 1;
  const auto labels = n::zeros({1}, n::DType::Int64);
  EXPECT_THROW(m::MakeTensorDetectionScenes(n::full({1, 28, 28, 1}, NAN), labels, cfg),
               std::invalid_argument);
  cfg.max_overlap = 1.1;
  EXPECT_THROW(m::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}), labels, cfg),
               std::invalid_argument);
  EXPECT_THROW(m::CropDetectionPatches(n::ones({1, 8, 8, 1}),
                                       n::from_vector(std::vector<std::int64_t>{1}),
                                       n::tensor({0., 0., 4., 4.}, {1, 4})),
               std::invalid_argument);
  EXPECT_THROW(m::SquareDetectionBoxes(n::tensor({4., 0., 1., 4.}, {1, 4})),
               std::invalid_argument);
  EXPECT_THROW(m::ComponentDetectionProposals(n::ones({8, 8, 1}), {256}),
               std::invalid_argument);
  EXPECT_THROW(m::ComponentDetectionProposals(n::full({8, 8, 1}, -1.)), std::invalid_argument);
  EXPECT_THROW(m::DrawDetectionBoxes(n::ones({1, 8, 8, 1}), {}), std::invalid_argument);
}

TEST(NnDetectionScenes, PatchSamplingUsesOnlySelectedRowsAndReproducesTensorSeed) {
  m::DetectionSceneConfig cfg;
  cfg.scenes = 2;
  cfg.max_objects = 1;
  const auto scenes = m::MakeTensorDetectionScenes(
      n::ones({1, 28, 28, 1}), n::from_vector(std::vector<std::int64_t>{3}), cfg);
  const auto a = m::MakeDetectionPatches(scenes, {0}, 3),
             b = m::MakeDetectionPatches(scenes, {0}, 3),
             other = m::MakeDetectionPatches(scenes, {0}, 4);
  EXPECT_EQ(a.images.shape().size(), 4);
  EXPECT_GE(a.labels.numel(), 3);
  EXPECT_EQ(n::count_nonzero(n::equal(a.labels, 3.)).item(), 3);
  EXPECT_EQ(n::max(n::abs(a.images - b.images)).item(), 0);
  EXPECT_EQ(a.labels.to_vector<int>(), b.labels.to_vector<int>());
  EXPECT_FALSE(a.images.requires_grad());
  EXPECT_GE(n::min(a.images).item(), 0);
  EXPECT_LE(n::max(a.images).item(), 1);
  const auto empty = m::MakeDetectionPatches(scenes, {});
  EXPECT_EQ(empty.images.shape(), n::Shape({0, 28, 28, 1}));
  EXPECT_THROW(m::MakeDetectionPatches(scenes, {2}), std::invalid_argument);
  if (a.images.shape() == other.images.shape())
    EXPECT_GT(n::max(n::abs(a.images - other.images)).item(), 0);
}
TEST(NnDetectionScenes, SlidingWindowsHaveSourceOrderAndExactContainment) {
  const auto windows = m::SlidingDetectionWindows(64, 64);
  EXPECT_EQ(windows.shape(), n::Shape({454, 4}));
  EXPECT_EQ(windows.at({0, 0}), 0);
  EXPECT_EQ(windows.at({0, 2}), 16);
  EXPECT_EQ(windows.at({168, 0}), 48);
  EXPECT_EQ(windows.at({168, 1}), 48);
  EXPECT_EQ(windows.at({169, 2}), 22);
  EXPECT_EQ(windows.at({453, 0}), 28);
  EXPECT_EQ(windows.at({453, 3}), 62);
  EXPECT_THROW(m::SlidingDetectionWindows(8, 8, {16}), std::invalid_argument);
  EXPECT_THROW(m::SlidingDetectionWindows(64, 64, {16}, 0), std::invalid_argument);
}
