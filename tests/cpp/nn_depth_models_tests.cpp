#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main cnn_depth_original_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/monocular_depth.cpp"
#undef main
TEST(DepthTensor, IndependentMaskedLogNeighborValuesEveryGradient) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(LossChecks());
}
TEST(DepthTensor, EveryStableExtremePositiveHeadDerivative) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(PositiveHeadChecks());
}
TEST(DepthTensor, IndependentMetricUnitsScaleThresholdsAndPriors) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(MetricPriorChecks());
}
TEST(DepthTensor, Every16LearnedGradientAnd11Stages) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(ModelChecks());
}

TEST(DepthTensor, EveryOriginalSharedSensorSampleAndBoundaryValidity) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<std::int32_t> bytes(480 * 640 * 3);
  std::vector<float> raw(480 * 640);
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = (i * 17 + i / 3) % 256;
  for (size_t i = 0; i < raw.size(); ++i)
    raw[i] = float(i % 101) / 10;
  const int selected[] = {5 * 640 + 5,  5 * 640 + 15, 5 * 640 + 25,
                          5 * 640 + 35, 5 * 640 + 45, 5 * 640 + 55};
  raw[selected[0]] = NAN;
  raw[selected[1]] = INFINITY;
  raw[selected[2]] = .1f;
  raw[selected[3]] = std::nextafter(.1f, 1.f);
  raw[selected[4]] = 10.f;
  raw[selected[5]] = std::nextafter(10.f, 11.f);
  const auto d = m::PrepareDepthSensorFrame(n::from_vector(bytes, {480, 640, 3}),
                                            n::from_vector(raw, {480, 640}), 7, 13);
  const auto rgb = d.rgb.to_vector<float>(), depth = d.depth.to_vector<float>(),
             valid = d.valid.to_vector<float>();
  for (int y = 0; y < 48; ++y)
    for (int x = 0; x < 64; ++x) {
      const int source = (y * 10 + 5) * 640 + x * 10 + 5, dest = y * 64 + x;
      const bool good = std::isfinite(raw[source]) && raw[source] > .1f && raw[source] <= 10;
      EXPECT_EQ(valid[dest], good);
      EXPECT_EQ(depth[dest], good ? raw[source] : 0.f);
      for (int c = 0; c < 3; ++c)
        EXPECT_EQ(rgb[dest * 3 + c], float(bytes[source * 3 + c]) / 255.f);
    }
  EXPECT_EQ(d.room_ids.item<int>(), 7);
  EXPECT_EQ(d.source_rows.item<int>(), 13);
}
TEST(DepthTensor, IndependentSpatialLogMeanFallbackAndMaskedNanOracle) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<float> values(3 * 3072), valid(values.size());
  for (int b = 0; b < 3; ++b)
    for (int i = 0; i < 3072; ++i) {
      const int q = b * 3072 + i;
      values[q] = float(1 + b) + float(i % 7) * .2f;
      valid[q] = i % 11 != 0 && q % 5 != 0;
      if (!valid[q])
        values[q] = NAN;
    }
  m::DepthData data;
  data.depth = n::from_vector(values, {3, 48, 64, 1});
  data.valid = n::from_vector(valid, {3, 48, 64, 1});
  const auto prior = m::FitDepthPriors(data);
  const auto actual = prior.spatial.to_vector<float>();
  double all = 0, total = 0;
  std::vector<double> sum(3072), count(3072);
  for (int b = 0; b < 3; ++b)
    for (int i = 0; i < 3072; ++i)
      if (valid[b * 3072 + i]) {
        const double log = std::log(double(values[b * 3072 + i]));
        all += log;
        ++total;
        sum[i] += log;
        ++count[i];
      }
  EXPECT_NEAR(prior.geometric, std::exp(all / total), 1e-12);
  for (int i = 0; i < 3072; ++i)
    EXPECT_EQ(actual[i], float(std::exp(count[i] ? sum[i] / count[i] : all / total)));
}
TEST(DepthTensor, EveryFivePanelColorErrorMaskAndHalfBytePixel) {
  namespace n = dm::nn;
  namespace m = n::models;
  constexpr int count = 2 * 48 * 64;
  std::vector<float> rgb(count * 3), truth(count), estimate(count), valid(count);
  for (int i = 0; i < count; ++i) {
    truth[i] = float(i % 101) / 10;
    estimate[i] = truth[i] + float(i % 7 - 3) * .17f;
    valid[i] = i % 5 != 0;
    for (int c = 0; c < 3; ++c)
      rgb[3 * i + c] = float((i * 3 + c) % 257) / 256;
  }
  rgb[0] = std::nextafter(.5f / 255.f, 0.f);
  const m::DepthData d{n::from_vector(rgb, {2, 48, 64, 3}),
                       n::from_vector(truth, {2, 48, 64, 1}),
                       n::from_vector(valid, {2, 48, 64, 1}), n::zeros({2}, n::DType::Int32),
                       n::zeros({2}, n::DType::Int32)};
  const auto tiles =
      m::DepthPreviewTiles(d, n::from_vector(estimate, {2, 48, 64, 1})).to_vector<float>();
  auto color = [](double depth) {
    const double t = std::clamp((depth - .1) / 9.9, 0., 1.);
    return std::array<double, 3>{std::clamp(1.5 - std::abs(4 * t - 3), 0., 1.),
                                 std::clamp(1.5 - std::abs(4 * t - 2), 0., 1.),
                                 std::clamp(1.5 - std::abs(4 * t - 1), 0., 1.)};
  };
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 48; ++y)
      for (int x = 0; x < 64; ++x) {
        const int i = (b * 48 + y) * 64 + x;
        const auto gt = color(truth[i]), pr = color(estimate[i]);
        for (int c = 0; c < 3; ++c) {
          const double values[] = {
              rgb[3 * i + c], valid[i] ? gt[c] : 0., pr[c],
              valid[i] && c == 0 ? std::abs(double(estimate[i]) - truth[i]) / 5 : 0.,
              valid[i] ? 1. : 0.};
          for (int k = 0; k < 5; ++k)
            EXPECT_EQ(tiles[((b * 48 + y) * 320 + k * 64 + x) * 3 + c],
                      float(std::lround(255 * std::clamp(values[k], 0., 1.))) / 255.f);
        }
      }
}
TEST(DepthTensor, RgbGradientAndZeroModelHaveNoDepthShortcut) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::ParameterStore p(139);
  const auto rgb = (.5 + .4 * n::sin(n::arange(0, 48 * 64 * 3, 1, n::DType::Float64) * .17))
                       .to(n::DType::Float32)
                       .reshape({1, 48, 64, 3})
                       .requires_grad_();
  const auto out = m::MonocularDepth(p, rgb);
  const auto g = n::grad(n::mean(out.depth) + n::mean(out.coarse), {rgb})[0];
  EXPECT_TRUE(n::all(n::is_finite(g)).item<bool>());
  EXPECT_GT(n::sum(n::abs(g)).item(), 0);
  {
    n::NoGradGuard guard;
    for (size_t i = 0; i < p.size(); ++i)
      p.set(p.names()[i], n::zeros(p.parameters()[i].shape()));
  }
  const auto zero = m::MonocularDepth(p, rgb.detach());
  const auto expected = m::PositiveMetricDepth(n::zeros_like(zero.depth));
  EXPECT_TRUE(n::all(n::equal(zero.depth, expected)).item<bool>());
  EXPECT_TRUE(n::all(n::equal(zero.coarse, expected)).item<bool>());
}
TEST(DepthTensor, PreparedOriginalRoomSourceAndSensorSupport) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto root = std::filesystem::current_path() / "build/cnn-datasets/monocular_depth";
  if (!std::filesystem::exists(root / "pairs.npz"))
    GTEST_SKIP() << "Optional prepared official NYU sensor cache absent";
  const auto split = m::LoadDepthPairs(root);
  std::set<int> rooms, rows;
  const int expected[] = {32, 8, 16};
  int i = 0;
  for (const auto *d : {&split.fit, &split.validation, &split.test}) {
    EXPECT_EQ(d->rgb.size(0), expected[i++]);
    for (int id : d->room_ids.to_vector<int>())
      EXPECT_TRUE(rooms.insert(id).second);
    for (int row : d->source_rows.to_vector<int>()) {
      EXPECT_GE(row, 0);
      EXPECT_LT(row, 1449);
      EXPECT_TRUE(rows.insert(row).second);
    }
    EXPECT_TRUE(n::all(n::greater(n::sum(d->valid, {1, 2, 3}), 307.)).item<bool>());
    EXPECT_TRUE(n::any(n::equal(d->valid, 0.)).item<bool>());
  }
  EXPECT_EQ(rooms.size(), 56);
  EXPECT_EQ(rows.size(), 56);
}
