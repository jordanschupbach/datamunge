#include "common/bilinear.hpp"
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
using nn::Tensor;
using namespace fine;
namespace {
void OuterProductChecks() {
  std::vector<float> a(24), b(36), up(12);
  for (int i = 0; i < 24; ++i)
    a[i] = float((i % 9 - 4) * .17);
  for (int i = 0; i < 36; ++i)
    b[i] = float((i % 11 - 5) * .23);
  for (int i = 0; i < 12; ++i)
    up[i] = float((i % 7 - 3) * .19);
  Tensor first = nn::from_vector(a, {2, 2, 3, 2}).requires_grad_(),
         second = nn::from_vector(b, {2, 2, 3, 3}).requires_grad_();
  const auto output = SpatialBilinear(first, second);
  const auto gradient =
      nn::grad(nn::sum(output * nn::from_vector(up, {2, 2, 3})), {first, second});
  const auto out = output.to_vector<double>(), ga = gradient[0].to_vector<double>(),
             gb = gradient[1].to_vector<double>();
  std::vector<double> da(a.size()), db(b.size());
  double error = 0, gradient_error = 0;
  for (int n = 0; n < 2; ++n)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j) {
        double expected = 0;
        for (int position = 0; position < 6; ++position) {
          const double av = a[(n * 6 + position) * 2 + i], bv = b[(n * 6 + position) * 3 + j],
                       weight = up[(n * 2 + i) * 3 + j];
          expected += av * bv / 6;
          da[(n * 6 + position) * 2 + i] += weight * bv / 6;
          db[(n * 6 + position) * 3 + j] += weight * av / 6;
        }
        error = std::max(error, std::abs(expected - out[(n * 2 + i) * 3 + j]));
      }
  for (std::size_t i = 0; i < a.size(); ++i)
    gradient_error = std::max(gradient_error, std::abs(da[i] - ga[i]));
  for (std::size_t i = 0; i < b.size(); ++i)
    gradient_error = std::max(gradient_error, std::abs(db[i] - gb[i]));
  check::Near(error, 0, 2e-7, "all spatial mean outer-product entries");
  check::Near(gradient_error, 0, 2e-7, "all first AND second feature Jacobian entries");
  std::vector<float> pa(a.size()), pb(b.size());
  const int permutation[] = {5, 2, 1, 4, 0, 3};
  for (int n = 0; n < 2; ++n)
    for (int i = 0; i < 6; ++i) {
      for (int c = 0; c < 2; ++c)
        pa[(n * 6 + i) * 2 + c] = a[(n * 6 + permutation[i]) * 2 + c];
      for (int c = 0; c < 3; ++c)
        pb[(n * 6 + i) * 3 + c] = b[(n * 6 + permutation[i]) * 3 + c];
    }
  const auto permuted =
      SpatialBilinear(nn::from_vector(pa, {2, 2, 3, 2}), nn::from_vector(pb, {2, 2, 3, 3}))
          .to_vector<double>();
  for (std::size_t i = 0; i < out.size(); ++i)
    check::Near(permuted[i], out[i], 2e-7,
                "aligned location order does not change bilinear pooling");
  const auto unpaired =
      SpatialBilinear(nn::from_vector(pa, {2, 2, 3, 2}), nn::from_vector(b, {2, 2, 3, 3}))
          .to_vector<double>();
  double difference = 0;
  for (std::size_t i = 0; i < out.size(); ++i)
    difference += std::abs(unpaired[i] - out[i]);
  check::Require(difference > .1,
                 "bilinear is not an outer product of independently averaged features");
  std::cout << " all signed multi-batch outer-product values/BOTH stream Jacobians max errors="
            << error << "/" << gradient_error
            << ",joint spatial invariance/unpaired interaction sensitivity passed\n";
}

void RootLossChecks() {
  const std::vector<float> input = {-4, 0, 9, -.01f, 1e-8f, 1},
                           up = {.3f, -.7f, .2f, -.4f, .6f, .1f};
  Tensor x = nn::from_vector(input, {2, 3}).requires_grad_();
  const auto output = BilinearUnit(BilinearRoot(x));
  const auto gradient =
      nn::grad(nn::sum(output * nn::from_vector(up, {2, 3})), {x})[0].to_vector<double>();
  const auto out = output.to_vector<double>();
  double error = 0;
  for (int n = 0; n < 2; ++n) {
    std::array<double, 3> root{}, derivative{};
    double norm2 = 0, dot = 0;
    for (int c = 0; c < 3; ++c) {
      const double value = input[n * 3 + c], den = std::abs(value) + double(1e-5f);
      root[c] = value / std::sqrt(den);
      derivative[c] = (.5 * std::abs(value) + double(1e-5f)) / std::pow(den, 1.5);
      norm2 += root[c] * root[c];
      dot += root[c] * up[n * 3 + c];
    }
    const double norm = std::sqrt(norm2);
    for (int c = 0; c < 3; ++c) {
      check::Near(out[n * 3 + c], root[c] / norm, 2e-7,
                  "every signed normalized bilinear value");
      const double expected =
          (up[n * 3 + c] / norm - root[c] * dot / std::pow(norm, 3)) * derivative[c];
      error = std::max(error, std::abs(expected - gradient[n * 3 + c]));
    }
  }
  check::Near(error, 0, 3e-5,
              "all smoothed signed-root and L2 Jacobian coordinates including atzero");
  Tensor zero = nn::zeros({2, 3}).requires_grad_();
  const auto zout = BilinearUnit(BilinearRoot(zero));
  const auto zg =
      nn::grad(nn::sum(zout * nn::from_vector(up, {2, 3})), {zero})[0].to_vector<double>();
  for (double v : zout.to_vector<double>())
    check::Near(v, 0, 0, "zero bilinear descriptor stayszero");
  for (double v : zg)
    check::Require(std::isfinite(v), "zero-root/zero-norm derivative finite");
  const std::vector<float> scores = {1000, 999, 998, 997, 996, 995, -3, 2, .5f, -1, 1, 0};
  Tensor z = nn::from_vector(scores, {2, 6}).requires_grad_();
  const auto loss = FlowerCrossEntropy(z, nn::tensor({0, 4}).to(nn::DType::Int32), 6);
  const auto gz = nn::grad(loss, {z})[0].to_vector<double>();
  double expected_loss = 0;
  for (int n = 0; n < 2; ++n) {
    double maximum = -INFINITY, sum = 0;
    for (int c = 0; c < 6; ++c)
      maximum = std::max(maximum, double(scores[n * 6 + c]));
    for (int c = 0; c < 6; ++c)
      sum += std::exp(scores[n * 6 + c] - maximum);
    const int target = n ? 4 : 0;
    expected_loss += (std::log(sum) + maximum - scores[n * 6 + target]) / 2;
    for (int c = 0; c < 6; ++c)
      check::Near(gz[n * 6 + c],
                  (std::exp(scores[n * 6 + c] - maximum) / sum - (c == target)) / 2, 1e-7,
                  "every stable CE logit derivative");
  }
  check::Near(loss.item(), expected_loss, 1e-7, "known logsumexp NLL");
  const auto metric =
      ScoreFlowers(nn::full({6, 6}, 1.0 / 6), nn::arange(0, 6, 1, nn::DType::Int32));
  check::Near(metric.nll, std::log(6.), 1e-7, "uniform flower NLL");
  check::Near(metric.accuracy, 1. / 6, 1e-12, "deterministic first-class tie accuracy");
  check::Near(metric.macro_accuracy, 1. / 6, 1e-12, "balanced macro accuracy");
  check::Near(metric.brier, 5. / 6, 1e-7, "six-way Brier sum,not channel average");
  std::cout << " signed-root/unit every analytic Jacobian max error=" << error
            << ",zero safety,extreme CE/all derivatives and known NLL/accuracy/Brier fixtures "
               "passed\n";
}

std::int64_t Census(int divisor, int classes) {
  const int widths[] = {64, 128, 256, 512, 512}, layers[] = {2, 2, 3, 3, 3};
  std::int64_t one = 0;
  int in = 3;
  for (int block = 0; block < 5; ++block)
    for (int layer = 0; layer < layers[block]; ++layer) {
      const int out = widths[block] / divisor;
      one += 9LL * in * out + out;
      in = out;
    }
  return 2 * one + std::int64_t(in) * in * classes + classes;
}

void ModelChecks() {
  for (int divisor : {1, 16}) {
    const int classes = divisor == 1 ? 102 : 6;
    nn::ParameterStore p(233);
    std::vector<float> input(2 * 64 * 64 * 3), up(std::size_t(2) * classes);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = float(.5 + .4 * std::sin(i * .13));
    for (std::size_t i = 0; i < up.size(); ++i)
      up[i] = float((int(i) % 7 - 3) * .11);
    Tensor image = nn::from_vector(input, {2, 64, 64, 3}).requires_grad_();
    const auto net = BilinearCNN(p, image, divisor, classes);
    auto vars = p.parameters();
    vars.push_back(image);
    const auto gradients =
        nn::grad(nn::sum(net.logits * nn::from_vector(up, {2, classes})), vars);
    check::Require(p.count() == Census(divisor, classes) && p.size() == 54 &&
                       net.stages.size() == 37,
                   "independent parameter/26conv/two-stream/head census");
    std::cout
        << " " << (divisor == 1 ? "full" : "compact") << " bilinear:" << p.count()
        << " parameters/54 derivatives (executed),37stages,two independent feature stores\n";
    if (divisor == 1)
      continue;
    std::vector<nn::Shape> expected;
    const int widths[] = {4, 8, 16, 32, 32}, layers[] = {2, 2, 3, 3, 3};
    for (int stream = 0; stream < 2; ++stream) {
      int side = 64;
      for (int block = 0; block < 5; ++block) {
        for (int layer = 0; layer < layers[block]; ++layer)
          expected.push_back({2, side, side, widths[block]});
        if (block < 4) {
          side /= 2;
          expected.push_back({2, side, side, widths[block]});
        }
      }
    }
    expected.push_back({2, 32, 32});
    expected.push_back({2, 1024});
    expected.push_back({2, 6});
    for (int i = 0; i < 37; ++i)
      check::Require(net.stages[i].shape() == expected[i],
                     "all exact two-stream/spatial outer/root/unit/head shapes");
    const auto d = net.stages[35].to_vector<double>();
    for (int n = 0; n < 2; ++n) {
      double norm = 0;
      for (int c = 0; c < 1024; ++c)
        norm += d[n * 1024 + c] * d[n * 1024 + c];
      check::Near(norm, 1, 2e-6, "unit bilinear descriptor");
    }
    double minimum = INFINITY, maximum = 0;
    for (const auto &g : gradients) {
      double norm = 0;
      for (float v : g.to_vector<float>()) {
        check::Require(std::isfinite(v), "finite both-stream parameter/RGB derivative");
        norm += double(v) * v;
      }
      norm = std::sqrt(norm);
      check::Require(
          norm > 0,
          "every54 tensor in BOTH independent streams/head and RGB receives gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all54parameter and RGB gradients finite/nonzero,norm range["
              << minimum << "," << maximum
              << "],all37 exact stages and unit descriptors passed\n";
  }
}

} // namespace
TEST(BilinearModels, AlignedOuterEveryValueAndBothStreamJacobians) {
  OuterProductChecks();
}
TEST(BilinearModels, SignedRootUnitAndCrossEntropyIndependentDerivatives) {
  RootLossChecks();
}
TEST(BilinearModels, FullAndCompactAllStagesAndGradients) {
  nn::set_num_threads(2);
  ModelChecks();
}

TEST(FlowerData, IndependentProbabilityMetrics) {
  std::vector<float> values(36);
  std::vector<std::int32_t> labels{0, 1, 2, 3, 4, 5};
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      values[std::size_t(i * 6 + j)] = i == j ? .5f : .1f;
  const auto probability = nn::from_vector(values, {6, 6});
  const auto score = nn::models::ScoreFlowers(probability, nn::from_vector(labels));
  double nll = 0, brier = 0;
  for (int i = 0; i < 6; ++i) {
    nll -= std::log(double(values[std::size_t(i * 6 + i)])) / 6;
    for (int j = 0; j < 6; ++j)
      brier += std::pow(double(values[std::size_t(i * 6 + j)]) - (i == j), 2) / 6;
  }
  EXPECT_NEAR(score.nll.item(), nll, 1e-12);
  EXPECT_NEAR(score.brier.item(), brier, 1e-12);
  EXPECT_DOUBLE_EQ(score.accuracy.item(), 1);
  EXPECT_DOUBLE_EQ(score.macro_accuracy.item(), 1);
  const auto counts = score.confusion.to_vector<std::int64_t>();
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      EXPECT_EQ(counts[std::size_t(i * 6 + j)], i == j);
  const auto subset = nn::models::ScoreFlowers(nn::slice_axis(probability, 0, 0, 1),
                                               nn::from_vector(std::vector<std::int32_t>{0}));
  EXPECT_DOUBLE_EQ(subset.macro_accuracy.item(), 1);
  EXPECT_THROW(nn::models::ScoreFlowers(probability * .9, nn::from_vector(labels)),
               std::invalid_argument);
}
TEST(FlowerData, IndependentMeanColorAndPrototypeOracle) {
  std::vector<float> rgb(6 * 64 * 64 * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = float((i % 12288 * 7) % 128) / 255.f + .04f * float(i / 12288);
  nn::models::FlowerData data{nn::from_vector(rgb, {6, 64, 64, 3}),
                              nn::arange(0, 6, 1, nn::DType::Int32),
                              nn::arange(1, 7, 1, nn::DType::Int32)};
  std::vector<float> means(18);
  for (int row = 0; row < 6; ++row)
    for (int c = 0; c < 3; ++c) {
      double sum = 0;
      for (int px = 0; px < 4096; ++px)
        sum += rgb[std::size_t((row * 4096 + px) * 3 + c)];
      means[std::size_t(row * 3 + c)] = float(sum / 4096);
    }
  const auto observed = nn::models::FlowerColorMeans(data).to_vector<float>();
  EXPECT_EQ(observed, means);
  const auto probability = nn::models::FlowerColorBaseline(data, data).to_vector<float>();
  for (int row = 0; row < 6; ++row) {
    double weights[6], total = 0;
    for (int other = 0; other < 6; ++other) {
      double distance = 0;
      for (int c = 0; c < 3; ++c)
        distance += std::pow(
            double(means[std::size_t(row * 3 + c)]) - means[std::size_t(other * 3 + c)], 2);
      weights[other] = std::exp(-distance / .02);
      total += weights[other];
    }
    for (int other = 0; other < 6; ++other)
      EXPECT_NEAR(probability[std::size_t(row * 6 + other)], weights[other] / total, 1e-7);
  }
}
TEST(FlowerData, EveryPredictionTilePixelAndProbabilityBar) {
  std::vector<float> rgb(6 * 64 * 64 * 3), probability(36, 0);
  for (std::size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = float((i * 7) % 256) / 255.f;
  rgb[std::size_t((2 * 64 + 2) * 3)] = std::nextafter(.5f / 255.f, 0.f);
  for (int row = 0; row < 6; ++row)
    probability[std::size_t(row * 6 + (row == 1 ? 2 : row))] = 1;
  nn::models::FlowerData data{nn::from_vector(rgb, {6, 64, 64, 3}),
                              nn::arange(0, 6, 1, nn::DType::Int32),
                              nn::arange(1, 7, 1, nn::DType::Int32)};
  const auto tiles =
      nn::models::FlowerPredictionTiles(data, nn::from_vector(probability, {6, 6}));
  ASSERT_EQ(tiles.shape(), nn::Shape({6, 80, 64, 3}));
  const auto values = tiles.to_vector<float>();
  for (int row = 0; row < 6; ++row)
    for (int y = 0; y < 80; ++y)
      for (int x = 0; x < 64; ++x)
        for (int c = 0; c < 3; ++c) {
          float expected = 0;
          if (y < 64) {
            const bool border = x < 2 || y < 2 || x >= 62 || y >= 62;
            expected = border
                           ? (c == (row == 1 ? 0 : 1) ? 1.f : 0.f)
                           : float(std::lround(
                                 rgb[std::size_t((row * 4096 + y * 64 + x) * 3 + c)] * 255)) /
                                 255.f;
          } else if (x < 60 && x % 10 < 8) {
            const int klass = x / 10;
            const int height =
                int(std::lround(probability[std::size_t(row * 6 + klass)] * 14));
            if (y >= 80 - height)
              expected = c == (klass == row ? 1 : 2) ? 1 : 0;
          }
          EXPECT_FLOAT_EQ(values[std::size_t(((row * 80 + y) * 64 + x) * 3 + c)], expected);
        }
}
TEST(FlowerData, ImageIngressAndRowGradients) {
  const auto root = std::filesystem::temp_directory_path() / "dm-flower-native-image";
  std::filesystem::create_directories(root / "jpg");
  nn::save_png(nn::ones({2, 3, 3}), (root / "jpg/image_00001.jpg").string());
  const auto data = nn::models::ReadFlowerImageRows(root, {1}, {0});
  EXPECT_EQ(data.rgb.shape(), nn::Shape({1, 64, 64, 3}));
  EXPECT_DOUBLE_EQ(nn::sum(data.rgb).item(), 64 * 64 * 3);
  auto pixels = nn::ones({3, 2, 2, 1}).requires_grad_();
  const auto selected = nn::models::FlowerRows(pixels, {2, 2, 0});
  const auto gradient = nn::grad(nn::sum(selected), {pixels})[0].to_vector<float>();
  for (int i = 0; i < 12; ++i)
    EXPECT_FLOAT_EQ(gradient[std::size_t(i)], i < 4 ? 1 : i >= 8 ? 2 : 0);
  EXPECT_THROW(nn::models::FlowerRows(pixels, {3}), std::invalid_argument);
  std::filesystem::remove(root / "jpg/image_00001.jpg");
  std::filesystem::remove(root / "jpg");
  std::filesystem::remove(root);
}
