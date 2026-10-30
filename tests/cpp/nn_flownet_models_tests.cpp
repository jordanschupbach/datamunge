#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main cnn_flownets_original_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/flownets.cpp"
#undef main
TEST(FlowNetTensor, EverySignedCorrelationValueAndBothJacobians) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(CorrelationChecks());
}
TEST(FlowNetTensor, EveryWarpValueCoverageAndImageFlowJacobian) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(WarpChecks());
}
TEST(FlowNetTensor, EveryTransposeValueInputAndKernelJacobian) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(TransposeChecks());
}
TEST(FlowNetTensor, ExplicitUnitsSafeLossAndStrictMetricBoundaries) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(UnitsLossMetricChecks());
}
TEST(FlowNetTensor, FullCompactGraphAllParametersBothImagesAnd27Stages) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(ModelChecks());
}

TEST(FlowNetTensor, LittleEndianPublishedIngressAndPredeterminedCropCoordinates) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto root = std::filesystem::temp_directory_path() / "dm-middlebury-native-scene";
  std::filesystem::create_directories(root);
  n::save_png(n::arange(0, 18).reshape({2, 3, 3}), (root / "frame10.png").string(), 0, 255);
  n::save_png(n::ones({2, 3, 3}), (root / "frame11.png").string());
  const std::vector<float> flow = {3, 4, 1e10f, 0, 0, NAN, -2, 1, 0, 0, 2, -3};
  {
    std::ofstream out(root / "flow10.flo", std::ios::binary);
    const auto put = [&](std::uint32_t word) {
      for (int i = 0; i < 4; ++i)
        out.put(char((word >> (8 * i)) & 255));
    };
    put(0x48454950);
    put(3);
    put(2);
    for (float v : flow)
      put(std::bit_cast<std::uint32_t>(v));
  }
  const auto scene = m::ReadMiddleburyScene(root, 2, 7);
  const auto f = scene.flow.to_vector<float>(), v = scene.valid.to_vector<float>(),
             rgb = scene.first.to_vector<float>();
  for (int i = 0; i < 6; ++i) {
    const bool valid = std::isfinite(flow[2 * i]) && std::isfinite(flow[2 * i + 1]) &&
                       std::abs(flow[2 * i]) < 1e9 && std::abs(flow[2 * i + 1]) < 1e9;
    EXPECT_EQ(v[i], valid);
    for (int c = 0; c < 2; ++c)
      EXPECT_EQ(f[2 * i + c], valid ? flow[2 * i + c] : 0.f);
  }
  for (int i = 0; i < 18; ++i)
    EXPECT_EQ(rgb[i], float(i) / 255.f);
  for (const char *name : {"frame10.png", "frame11.png", "flow10.flo"})
    std::filesystem::remove(root / name);
  std::filesystem::remove(root);
  const auto y = n::arange(0, 96).reshape({1, 96, 1, 1}),
             x = n::arange(0, 96).reshape({1, 1, 96, 1});
  const auto mask =
      n::logical_and(n::greater_equal(y, 60.), n::greater_equal(x, 60.)).to(n::DType::Float32);
  const m::FlowData input{n::zeros({1, 96, 96, 3}),       n::ones({1, 96, 96, 3}),
                          n::ones({1, 96, 96, 2}),        mask,
                          n::zeros({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32)};
  std::vector<std::int64_t> expected;
  for (int oy : {0, 8, 16, 24, 32})
    for (int ox : {0, 8, 16, 24, 32}) {
      int count = 0;
      for (int row = oy; row < oy + 64; ++row)
        for (int col = ox; col < ox + 64; ++col)
          count += row >= 60 && col >= 60;
      if (count > 409 && expected.size() < 16) {
        expected.push_back(oy);
        expected.push_back(ox);
      }
    }
  EXPECT_EQ(m::FlowCropOrigins(input).to_vector<std::int64_t>(), expected);
  const auto crops = m::FlowSceneCrops(input);
  EXPECT_EQ(crops.first.shape(), (n::Shape{8, 64, 64, 3}));
  EXPECT_EQ(crops.flow.shape(), (n::Shape{8, 64, 64, 2}));
}
TEST(FlowNetTensor, EveryNonconstantMaskedPyramidValueIndependentBlockOracle) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<float> flow(2 * 4096 * 2), valid(2 * 4096);
  for (size_t i = 0; i < valid.size(); ++i) {
    valid[i] = i % 3 != 0;
    flow[2 * i] = float(int(i % 31) - 15) * .17f;
    flow[2 * i + 1] = float(int(i % 19) - 9) * .23f;
  }
  m::FlowData data;
  data.flow = n::from_vector(flow, {2, 64, 64, 2});
  data.valid = n::from_vector(valid, {2, 64, 64, 1});
  const auto targets = m::PrepareFlowSupervision(data);
  for (int level = 2; level <= 6; ++level) {
    const int factor = 1 << level, side = 64 / factor;
    const auto values = targets.flow[level - 2].to_vector<float>(),
               mask = targets.valid[level - 2].to_vector<float>();
    for (int b = 0; b < 2; ++b)
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
          double sum[2] = {};
          int count = 0;
          for (int dy = 0; dy < factor; ++dy)
            for (int dx = 0; dx < factor; ++dx) {
              const int i = (b * 64 + y * factor + dy) * 64 + x * factor + dx;
              if (valid[i] > 0) {
                ++count;
                sum[0] += flow[2 * i];
                sum[1] += flow[2 * i + 1];
              }
            }
          const int q = (b * side + y) * side + x;
          EXPECT_EQ(mask[q], count > 0);
          for (int c = 0; c < 2; ++c)
            EXPECT_EQ(values[2 * q + c], count ? float(sum[c] / count / 20) : 0.f);
        }
  }
}
TEST(FlowNetTensor, EqualImageWeightingUnknownTargetsAndEmptySupport) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto truth = n::zeros({3, 1, 3, 2});
  const auto prediction = n::tensor(
      {3, 4, 3, 4, 3, 4, 0, 0, 0, 0, 0, 0, 999, 999, 999, 999, 999, 999}, {3, 1, 3, 2});
  const auto valid = n::tensor({1, 1, 1, 1, 0, 0, 0, 0, 0}, {3, 1, 3, 1});
  const auto score = m::ScoreTensorFlow(prediction, truth, valid);
  EXPECT_EQ(score.epe.item(), 2.5);
  EXPECT_EQ(m::ScoreFlow(prediction.to(n::DType::Float64), truth.to(n::DType::Float64), valid).epe, 2.5);
  EXPECT_EQ(score.images.item<int>(), 2);
  EXPECT_EQ(score.pixels.item<int>(), 4);
  EXPECT_EQ(score.bad3.item(), .5);
  EXPECT_THROW(m::ScoreTensorFlow(prediction, truth, n::zeros_like(valid)),
               std::invalid_argument);
}
TEST(FlowNetTensor, EveryHsvAndSixPanelPreviewPixel) {
  namespace n = dm::nn;
  namespace m = n::models;
  constexpr int b = 2, h = 3, w = 4;
  std::vector<float> raw(b * h * w * 3), flow(b * h * w * 2), pred(flow.size()),
      valid(b * h * w);
  for (size_t i = 0; i < raw.size(); ++i)
    raw[i] = float(int(i % 17) - 2) / 13;
  for (size_t i = 0; i < valid.size(); ++i) {
    flow[2 * i] = float(int(i % 11) - 5) * .7f;
    flow[2 * i + 1] = float(int(i % 7) - 3) * .9f;
    pred[2 * i] = flow[2 * i] + .3f;
    pred[2 * i + 1] = flow[2 * i + 1] - .2f;
    valid[i] = i % 4 != 0;
  }
  const auto first = n::from_vector(raw, {b, h, w, 3}), second = first * .7,
             warped = first * .3, estimate = n::from_vector(pred, {b, h, w, 2});
  const m::FlowData data{first,
                         second,
                         n::from_vector(flow, {b, h, w, 2}),
                         n::from_vector(valid, {b, h, w, 1}),
                         n::zeros({b}, n::DType::Int32),
                         n::zeros({b}, n::DType::Int32)};
  const auto tile = m::FlowPreviewTiles(data, estimate, warped).to_vector<float>();
  const auto sv = second.to_vector<float>(), wv = warped.to_vector<float>();
  auto color = [](double u, double v) {
    const double saturation = std::min(1., std::hypot(u, v) / 4),
                 hue = (std::atan2(v, u) / 6.2831853071795864769 + 1) * 6,
                 part = hue - std::floor(hue), p = 1 - saturation, q = 1 - saturation * part,
                 t = 1 - saturation * (1 - part);
    switch (int(std::floor(hue)) % 6) {
    case 0:
      return std::array<double, 3>{1, t, p};
    case 1:
      return std::array<double, 3>{q, 1, p};
    case 2:
      return std::array<double, 3>{p, 1, t};
    case 3:
      return std::array<double, 3>{p, q, 1};
    case 4:
      return std::array<double, 3>{t, p, 1};
    default:
      return std::array<double, 3>{1, p, q};
    }
  };
  for (int batch = 0; batch < b; ++batch)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const int i = (batch * h + y) * w + x;
        const auto truth_rgb = color(flow[2 * i], flow[2 * i + 1]),
                   pred_rgb = color(pred[2 * i], pred[2 * i + 1]);
        for (int c = 0; c < 3; ++c) {
          const double vals[] = {raw[3 * i + c],
                                 sv[3 * i + c],
                                 valid[i] ? truth_rgb[c] : 0.,
                                 pred_rgb[c],
                                 valid[i] && c == 0
                                     ? std::hypot(double(pred[2 * i]) - flow[2 * i],
                                                  double(pred[2 * i + 1]) - flow[2 * i + 1]) /
                                           5
                                     : 0.,
                                 wv[3 * i + c]};
          for (int k = 0; k < 6; ++k)
            EXPECT_EQ(tile[((batch * h + y) * 6 * w + k * w + x) * 3 + c],
                      float(std::lround(255 * std::clamp(vals[k], 0., 1.))) / 255.f);
        }
      }
}
TEST(FlowNetTensor, PreparedPublishedSceneFamiliesAndAllCropCounts) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto root = std::filesystem::current_path() / "build/cnn-datasets/optical_flow";
  if (!std::filesystem::exists(root / "pairs.npz"))
    GTEST_SKIP() << "Optional prepared official Middlebury cache absent";
  const auto d = m::LoadFlowPairs(root);
  std::set<int> previous;
  for (const auto *split : {&d.fit, &d.validation, &d.test}) {
    std::set<int> current;
    for (auto family : split->family_ids.to_vector<int>()) {
      EXPECT_FALSE(previous.contains(family));
      current.insert(family);
    }
    previous.insert(current.begin(), current.end());
    EXPECT_TRUE(n::all(n::greater(n::sum(split->valid, {1, 2, 3}), 409.)).item<bool>());
  }
  EXPECT_EQ(d.fit.first.size(0), 32);
  EXPECT_EQ(d.validation.first.size(0), 8);
  EXPECT_EQ(d.test.first.size(0), 24);
  EXPECT_EQ(previous.size(), 6);
}
