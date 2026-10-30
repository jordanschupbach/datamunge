#include "common/multilabel.hpp"
#include <dm/dm.hpp>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
namespace nn = dm::nn;
using nn::Tensor;
using namespace multilabel;
namespace {
void LossChecks() {
  const std::vector<float> input = {-1000, 1000, 0, -2, 2, 0, -.5f, .5f},
                           truth = {1, 0, 1, 0, 0, 1, 1, 0}, weights = {2, 3, 4, 5};
  Tensor x = nn::from_vector(input, {2, 4}).requires_grad_();
  const Tensor loss =
      MultilabelObjective(x, nn::from_vector(truth, {2, 4}), nn::from_vector(weights, {4}));
  const auto g = nn::grad(loss, {x})[0].to_vector<float>();
  const auto probabilities = nn::sigmoid(x.detach()).to_vector<float>();
  double expected = 0;
  for (int i = 0; i < 8; ++i) {
    const double z = input[std::size_t(i)], y = truth[std::size_t(i)],
                 weight = y ? weights[std::size_t(i % 4)] : 1,
                 p = z >= 0 ? 1 / (1 + std::exp(-z)) : std::exp(z) / (1 + std::exp(z));
    expected += weight * (std::max(z, 0.) - z * y + std::log1p(std::exp(-std::abs(z)))) / 8;
    check::Near(g[std::size_t(i)], weight * (p - y) / 8, 1e-7,
                "every BCE logit derivative including exactzero/extreme logits and positive "
                "class factors");
  }
  check::Near(loss.item(), expected, 4e-5, "known finite weighted mean BCE");
  check::Require(probabilities[4] + probabilities[7] > 1,
                 "independent sigmoid probabilities do not compete via softmax");
  const Tensor labels =
      nn::from_vector(std::vector<float>{1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0}, {3, 4});
  const auto calculated = MultilabelWeights(labels).to_vector<float>(),
             prior = MultilabelPrior(labels, 1).to_vector<float>();
  const double expected_weight[] = {1, 4, 4, 1.5}, expected_prior[] = {.6, .2, .2, .4};
  for (int c = 0; c < 4; ++c) {
    check::Near(calculated[std::size_t(c)], expected_weight[c], 0,
                "fit-only Laplace/capped class weight formula");
    check::Near(prior[std::size_t(c)], expected_prior[c], 3e-8,
                "fit-only smoothed class prior");
  }
  check::Near(MultilabelWeights(nn::zeros({80, 1})).to_vector<float>()[0], 20, 0,
              "rare/absent class positive weights capped20");
  std::cout
      << " all independent weighted BCE values/every logit gradient at extreme/zero,sigmoid "
         "noncompetition,fit-only class weight/prior/clipping fixtures passed\n";
}

void MetricChecks() {
  check::Near(MultilabelAP({.9, .7, .5}, {1, 0, 1}), 5. / 6, 1e-12,
              "known noninterpolated AP5/6");
  check::Near(MultilabelAP({.5, .5}, {1, 0}), .5, 0,
              "constant-score tie group AP equals prevalence");
  check::Near(MultilabelAP({.5, .5}, {0, 1}), .5, 0,
              "tie AP invariant to positive row ordering");
  check::Near(MultilabelAP({.9, .1}, {0, 0}), -1, 0,
              "absent-positive AP explicitly undefined");
  const Tensor probability =
      nn::from_vector(std::vector<float>{.9f, .8f, .4f, .7f, .6f, .2f, .5f, .3f, .1f}, {3, 3});
  const Tensor truth = nn::from_vector(std::vector<float>{1, 1, 0, 0, 1, 0, 1, 0, 0}, {3, 3});
  const auto score = ScoreMultilabel(probability, truth, .5);
  check::Require(score.tp == 4 && score.fp == 1 && score.fn == 0 && score.tn == 4 &&
                     score.ap_classes == 2 && score.absent_classes == 1,
                 "all known multilabel confusion counts/absent class support");
  check::Near(score.map, 11. / 12, 1e-12, "mean AP over only positive-support classes");
  check::Near(score.micro_ap, .8875, 1e-12, "independent pooled-bit AP");
  check::Near(score.micro_f1, 8. / 9, 1e-12, "micro F1 pooled bits");
  check::Near(score.macro_f1, .6, 1e-12, "macro F1 includes absent class zero contribution");
  check::Near(score.hamming, 1. / 9, 1e-12, "Hamming bit error");
  check::Near(score.exact, 2. / 3, 1e-12, "subset exact-match accuracy");
  check::Near(CalibrateMultilabel(probability, truth), .5, 1e-12,
              "validation global micro-F1 best tie chooses highest threshold");
  const Tensor empty = nn::zeros({3, 3});
  check::Near(CalibrateMultilabel(probability, empty), 1.01, 0,
              "validation with no positives chooses predict-none sentinel on ties");
  const auto allnegative = ScoreMultilabel(empty, empty);
  check::Require(allnegative.exact == 1 && allnegative.micro_f1 == 0 &&
                     allnegative.ap_classes == 0 && allnegative.absent_classes == 3,
                 "all-negative empty-label metric conventions");
  std::cout
      << " known AP5/6/grouped ties/order invariance,absent-support handling,all "
         "confusion/F1/Hamming/exact metrics and validation-only threshold fixtures passed\n";
}

std::pair<std::int64_t, std::int64_t> Census(int divisor) {
  std::int64_t parameters = 0, state = 0;
  auto conv = [&](int in, int out, int kernel) {
    parameters += std::int64_t(kernel) * kernel * in * out + 2 * out;
    state += 2 * out;
  };
  int in = 64 / divisor;
  conv(3, in, 3);
  for (int stage = 0; stage < 4; ++stage) {
    const int width = (64 << stage) / divisor;
    for (int block = 0; block < 2; ++block) {
      conv(in, width, 3);
      conv(width, width, 3);
      if (in != width)
        conv(in, width, 1);
      in = width;
    }
  }
  parameters += std::int64_t(in) * 80 + 80;
  return {parameters, state};
}

void ModelChecks() {
  for (int divisor : {1, 8}) {
    nn::ParameterStore p(251), state(257);
    std::vector<float> input(2 * 64 * 64 * 3);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = float(.5 + .4 * std::sin(i * .17));
    Tensor image = nn::from_vector(input, {2, 64, 64, 3}).requires_grad_();
    const auto net = resnet::ResNet(p, state, image - .5, MultilabelConfig(divisor), true);
    auto wrt = p.parameters();
    wrt.push_back(image);
    const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), wrt);
    const auto census = Census(divisor);
    check::Require(p.count() == census.first && state.count() == census.second &&
                       p.size() == 62 && state.size() == 40 && net.blocks.size() == 8,
                   "independent full/compact ResNet/head/BN census");
    std::cout << " " << (divisor == 1 ? "full" : "compact")
              << " multilabel ResNet:" << p.count()
              << " parameters/62 derivatives,BN=" << state.count()
              << "/40,8blocks (executed)\n";
    if (divisor == 1)
      continue;
    std::vector<Tensor> stages{net.stem};
    for (const auto &s : net.stages)
      stages.push_back(s);
    for (const auto &b : net.blocks)
      stages.push_back(b);
    stages.push_back(net.logits);
    std::vector<nn::Shape> expected{
        {2, 64, 64, 8}, {2, 64, 64, 8}, {2, 32, 32, 16}, {2, 16, 16, 32}, {2, 8, 8, 64}};
    for (int stage = 0; stage < 4; ++stage)
      for (int block = 0; block < 2; ++block)
        expected.push_back({2, 64 / (1 << stage), 64 / (1 << stage), 8 << stage});
    expected.push_back({2, 80});
    for (int i = 0; i < 14; ++i)
      check::Require(stages[std::size_t(i)].shape() == expected[std::size_t(i)],
                     "every residual stage/block and 80-logit geometry");
    double minimum = INFINITY, maximum = 0;
    for (const auto &g : gradients) {
      double norm = 0;
      for (float value : g.to_vector<float>()) {
        check::Require(std::isfinite(value), "finite multi-label model/RGB derivative");
        norm += double(value) * value;
      }
      norm = std::sqrt(norm);
      check::Require(norm > 0, "every62 learned tensor and observedRGB receive gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all62parameter and RGB gradients finite/nonzero,norm range["
              << minimum << "," << maximum << "],14exact residual/head shapes passed\n";
  }
}

} // namespace
TEST(MultilabelModels, IndependentWeightedStableBceEveryLogitDerivative) {
  LossChecks();
}
TEST(MultilabelModels, GroupedTieApConfusionF1AndCalibrationOracles) {
  MetricChecks();
}
TEST(MultilabelModels, FullCompactResNetAll62GradientsAndBnCensus) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(MultilabelMetrics, Float64CutoffAndBinaryDtypeConventions) {
  const auto p = nn::full({1, 1}, .35);
  const auto y = nn::ones({1, 1}, nn::DType::Bool);
  EXPECT_DOUBLE_EQ(nn::models::CalibrateMultilabel(p, y).item(), 6 * .05);
  const auto metric = nn::models::ScoreMultilabel(nn::full({1, 1}, .75), y, .7500000000001);
  EXPECT_EQ(metric.fn.item<std::int64_t>(), 1);
  EXPECT_DOUBLE_EQ(metric.micro_f1.item(), 0);
  const auto weights = nn::models::MultilabelWeights(nn::zeros({0, 3}, nn::DType::Bool));
  EXPECT_EQ(weights.to_vector<float>(), (std::vector<float>{1, 1, 1}));
  EXPECT_EQ(nn::models::MultilabelPrior(nn::zeros({0, 3}), 1).to_vector<float>(),
            (std::vector<float>{.5, .5, .5}));
  EXPECT_DOUBLE_EQ(nn::models::MultilabelAP(nn::zeros({0}), nn::zeros({0})).item(), -1);
}
TEST(MultilabelData, ImageIngressCategoryOrAndGatherGradients) {
  const auto root = std::filesystem::temp_directory_path() / "dm-multilabel-native-image";
  std::filesystem::create_directories(root);
  nn::save_png(nn::ones({2, 3, 1}), (root / "000000000009.jpg").string());
  const auto data = nn::models::ReadMultilabelImages(root, {9, 9}, {{0, 0, 79}, {}});
  EXPECT_DOUBLE_EQ(nn::sum(data.rgb).item(), 2 * 64 * 64 * 3);
  EXPECT_DOUBLE_EQ(nn::sum(data.labels).item(), 2);
  const auto bits = data.labels.to_vector<float>();
  EXPECT_FLOAT_EQ(bits[0], 1);
  EXPECT_FLOAT_EQ(bits[79], 1);
  for (int i = 80; i < 160; ++i)
    EXPECT_FLOAT_EQ(bits[std::size_t(i)], 0);
  auto x = nn::ones({3, 2, 2, 1}).requires_grad_();
  const auto selected = nn::models::MultilabelRows(x, {2, 2, 0});
  const auto gradient = nn::grad(nn::sum(selected), {x})[0].to_vector<float>();
  for (int i = 0; i < 12; ++i)
    EXPECT_FLOAT_EQ(gradient[std::size_t(i)], i < 4 ? 1 : i >= 8 ? 2 : 0);
  EXPECT_THROW(nn::models::MultilabelRows(x, {3}), std::invalid_argument);
  std::filesystem::remove(root / "000000000009.jpg");
  std::filesystem::remove(root);
}
TEST(MultilabelData, EveryAnnotationDecisionAndHalfBytePixel) {
  const int count = 2;
  std::vector<float> rgb(count * 64 * 64 * 3), truth(count * 80), probability(count * 80);
  for (std::size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = float((i * 7) % 256) / 255.f;
  rgb[0] = std::nextafter(.5f / 255.f, 0.f);
  for (int i = 0; i < count * 80; ++i) {
    truth[std::size_t(i)] = i % 3 == 0;
    probability[std::size_t(i)] = i % 4 < 2 ? .75f : .25f;
  }
  nn::models::MultilabelData data{nn::from_vector(rgb, {count, 64, 64, 3}),
                                  nn::from_vector(truth, {count, 80}),
                                  nn::arange(1, count + 1, 1, nn::DType::Int32)};
  const auto tiles =
      nn::models::MultilabelPreviewTiles(data, nn::from_vector(probability, {count, 80}), .5);
  ASSERT_EQ(tiles.shape(), nn::Shape({count, 64, 160, 3}));
  const auto pixels = tiles.to_vector<float>();
  for (int n = 0; n < count; ++n)
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 160; ++x)
        for (int c = 0; c < 3; ++c) {
          float expected = 0;
          if (x < 64)
            expected = float(std::lround(rgb[std::size_t((n * 64 * 64 + y * 64 + x) * 3 + c)] *
                                         255)) /
                       255.f;
          else {
            const bool first = x >= 68 && x < 108, second = x >= 116 && x < 156;
            const int local = first ? x - 68 : second ? x - 116 : -1;
            if (local >= 0 && local % 4 < 3 && y % 8 >= 1 && y % 8 < 7) {
              const int label = y / 8 * 10 + local / 4;
              const bool t = truth[std::size_t(n * 80 + label)] > .5,
                         p = probability[std::size_t(n * 80 + label)] >= .5;
              if (first)
                expected = c == 1 ? (t ? 1.f : 35.f / 255) : 0;
              else if (t && p)
                expected = c == 1 ? 1 : 0;
              else if (p)
                expected = c == 0 ? 1 : 0;
              else if (t)
                expected = c < 2 ? 1 : 0;
              else
                expected = 20.f / 255;
            }
          }
          EXPECT_FLOAT_EQ(pixels[std::size_t(((n * 64 + y) * 160 + x) * 3 + c)], expected);
        }
}
TEST(MultilabelData, PreparedSourceAnnotationOrAndCornerPixels) {
  const std::filesystem::path root = "build/cnn-datasets/multilabel";
  if (!std::filesystem::exists(root / "multilabel.npz"))
    GTEST_SKIP()
        << "Prepare the optional real COCO128 fixture with datamunge-prepare-data multilabel DIRECTORY.";
  const auto data = nn::models::LoadMultilabel(root);
  std::set<std::int32_t> used;
  for (const auto *split : {&data.fit, &data.validation, &data.test}) {
    const auto ids = split->ids.to_vector<std::int32_t>();
    const auto labels = split->labels.to_vector<float>();
    const auto images = split->rgb.to_vector<float>();
    for (std::size_t row = 0; row < ids.size(); ++row) {
      EXPECT_TRUE(used.insert(ids[row]).second);
      std::ostringstream stem;
      stem << std::setfill('0') << std::setw(12) << ids[row];
      std::ifstream file(root / "coco128/labels/train2017" / (stem.str() + ".txt"));
      std::array<float, 80> expected{};
      std::string line;
      while (std::getline(file, line)) {
        std::istringstream record(line);
        int category;
        record >> category;
        ASSERT_GE(category, 0);
        ASSERT_LT(category, 80);
        expected[std::size_t(category)] = 1;
      }
      for (int c = 0; c < 80; ++c)
        EXPECT_FLOAT_EQ(labels[row * 80 + std::size_t(c)], expected[std::size_t(c)]);
      const auto pixel =
          dm::image::read_image(
              (root / "coco128/images/train2017" / (stem.str() + ".jpg")).string())
              .get_pixel(0, 0);
      EXPECT_FLOAT_EQ(images[row * 64 * 64 * 3], pixel.r / 255.f);
      EXPECT_FLOAT_EQ(images[row * 64 * 64 * 3 + 1], pixel.g / 255.f);
      EXPECT_FLOAT_EQ(images[row * 64 * 64 * 3 + 2], pixel.b / 255.f);
    }
  }
  EXPECT_EQ(used.size(), 128);
}
