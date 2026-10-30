
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_partial_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/partial_inpainting.cpp"
#undef main
namespace mask_oracle {
namespace nn = dm::nn;
using dm::nn::models::ContextPairs;
using dm::nn::models::PartialData;
inline PartialData PartialPairs(const nn::Tensor &images, const std::vector<std::size_t> &rows,
                                std::uint32_t seed) {
  const nn::Tensor truth = ContextPairs(images, rows).truth;
  const auto tv = truth.to_vector<float>();
  std::vector<float> input(tv.size()), mask(tv.size());
  for (std::size_t n = 0; n < rows.size(); ++n) {
    std::mt19937 rng(std::uint32_t(seed + 65537 * rows[n]));
    auto integer = [&](int low, int high) {
      return std::uniform_int_distribution<int>(low, high)(rng);
    };
    std::vector<int> valid(1024);
    bool accepted = false;
    for (int attempt = 0; attempt < 100 && !accepted; ++attempt) {
      std::fill(valid.begin(), valid.end(), 1);
      const int y0 = integer(0, 25), x0 = integer(0, 25), rh = integer(5, 12),
                rw = integer(5, 12);
      for (int y = y0; y < std::min(y0 + rh, 32); ++y)
        for (int x = x0; x < std::min(x0 + rw, 32); ++x)
          valid[std::size_t(y * 32 + x)] = 0;
      for (int stroke = 0; stroke < 2; ++stroke) {
        const int ay = integer(0, 31), ax = integer(0, 31), by = integer(0, 31),
                  bx = integer(0, 31), radius = integer(1, 3);
        for (int step = 0; step <= 64; ++step) {
          const double cy = ay + (by - ay) * step / 64., cx = ax + (bx - ax) * step / 64.;
          for (int y = std::max(0, int(cy) - radius); y <= std::min(31, int(cy) + radius + 1);
               ++y)
            for (int x = std::max(0, int(cx) - radius);
                 x <= std::min(31, int(cx) + radius + 1); ++x)
              if ((y - cy) * (y - cy) + (x - cx) * (x - cx) <= radius * radius)
                valid[std::size_t(y * 32 + x)] = 0;
        }
      }
      const auto holes = std::count(valid.begin(), valid.end(), 0);
      accepted = holes >= 123 && holes <= 563;
    }
    check::Require(accepted, "irregular mask hole fraction12..55% with visible context");
    for (std::size_t i = 0; i < 1024; ++i)
      for (std::size_t c = 0; c < 3; ++c) {
        const std::size_t j = (n * 1024 + i) * 3 + c;
        mask[j] = float(valid[i]);
        input[j] = valid[i] ? tv[j] : 0.f;
      }
  }
  return {truth, nn::from_vector(input, truth.shape()), nn::from_vector(mask, truth.shape())};
}

inline nn::Tensor PartialMean(const PartialData &data) {
  const auto in = data.input.to_vector<float>(), m = data.mask.to_vector<float>();
  std::vector<float> out(in.size());
  for (std::int64_t n = 0; n < data.truth.size(0); ++n) {
    double sum[3] = {};
    int count = 0;
    for (std::int64_t i = 0; i < 1024; ++i)
      if (m[std::size_t((n * 1024 + i) * 3)] > 0) {
        ++count;
        for (int c = 0; c < 3; ++c)
          sum[c] += in[std::size_t((n * 1024 + i) * 3 + c)];
      }
    check::Require(count > 0, "visible-only partial mean baseline");
    for (std::int64_t i = 0; i < 1024; ++i)
      for (int c = 0; c < 3; ++c) {
        const std::size_t j = std::size_t((n * 1024 + i) * 3 + c);
        out[j] = m[j] != 0 ? in[j] : float(sum[c] / count);
      }
  }
  return nn::from_vector(out, data.truth.shape());
}

} // namespace mask_oracle
TEST(PartialInpainting, IndependentFullObjectiveEveryAdjoint) {
  LossChecks();
}
TEST(PartialInpainting, OriginalDataIsolationChecks) {
  DataChecks();
}
TEST(PartialInpainting, FullCompactGraphAndMaskGeometry) {
  ModelChecks();
}
TEST(PartialInpainting, GenuineKerasPoolsLossesEveryImageAdjoint) {
  const std::filesystem::path root = "build/cnn-datasets/style_transfer";
  if (!std::filesystem::exists(root / "partial_reference.npz"))
    GTEST_SKIP() << "prepare genuine Keras fixtures";
  dm::nn::set_num_threads(2);
  KerasParity(style::LoadTensorMap((root / "vgg16.npz").string()),
              style::LoadTensorMap((root / "partial_reference.npz").string()));
}
TEST(PartialInpainting, EveryOriginalMaskObservedMeanAndPreviewByte) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  std::vector<std::int32_t> bytes(8 * 3072);
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = (i * 13 + i / 3072 * 37) % 256;
  const auto images = n::from_vector(bytes, {8, 32, 32, 3});
  for (std::uint32_t seed : {0U, 107U, 109U, 113U, 4294967295U}) {
    const std::vector<std::size_t> rows = {7, 0, 3, 1};
    const auto actual = m::PartialPairs(images, rows, seed),
               expected = mask_oracle::PartialPairs(images, rows, seed);
    EXPECT_TRUE(n::all(n::equal(actual.mask, expected.mask)).item<bool>());
    EXPECT_TRUE(n::all(n::equal(actual.truth, expected.truth)).item<bool>());
    EXPECT_TRUE(n::all(n::equal(actual.input, expected.input)).item<bool>());
    const auto mean = m::PartialMean(actual), reference = mask_oracle::PartialMean(expected);
    EXPECT_TRUE(n::all(n::equal(mean, reference)).item<bool>());
    const auto predicted = mean * 1.4 - .2;
    const auto tiles = m::PartialPreviewTiles(actual, predicted, mean).to_vector<float>(),
               in = actual.input.to_vector<float>(), me = mean.to_vector<float>(),
               pr = predicted.to_vector<float>(), truth = actual.truth.to_vector<float>();
    for (int b = 0; b < 4; ++b)
      for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x) {
          const auto index = ((b * 32 + y) * 32 + x) * 3;
          double error = 0;
          for (int c = 0; c < 3; ++c)
            error += std::abs(pr[index + c] - truth[index + c]) / 3;
          for (int c = 0; c < 3; ++c) {
            const double values[] = {in[index + c], me[index + c], pr[index + c],
                                     truth[index + c], c == 0 ? 4 * error : 0.};
            for (int panel = 0; panel < 5; ++panel)
              EXPECT_NEAR(tiles[((b * 32 + y) * 160 + panel * 32 + x) * 3 + c],
                          std::lround(255 * std::clamp(values[panel], 0., 1.)) / 255., 4e-8);
          }
        }
  }
}
TEST(PartialInpainting, BooleanCompositeAndHiddenNaNReferenceSafety) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto known = n::ones({1, 2, 3, 3}, n::DType::Bool);
  EXPECT_TRUE(n::all(n::equal(m::PartialComposite(n::ones({1, 2, 3, 3}),
                                                  n::zeros({1, 2, 3, 3}), known),
                              1.))
                  .item<bool>());
  const auto d = m::PartialPairs(n::ones({2, 32, 32, 3}, n::DType::Int32) * 64, {1}, 107);
  auto changed = d;
  changed.input = n::where(n::greater(d.mask, 0.), d.input, n::full(d.input.shape(), NAN));
  EXPECT_TRUE(n::all(n::equal(m::PartialMean(changed), m::PartialMean(d))).item<bool>());
}
