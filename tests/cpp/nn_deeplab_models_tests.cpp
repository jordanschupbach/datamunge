#include <dm/dm.hpp>
#include <functional>
#include <gtest/gtest.h>
namespace nn = dm::nn;
namespace m = nn::models;
using nn::Tensor;
namespace check {
template <typename A, typename B>
void Near(A a, B b, double tolerance, const std::string &message) {
  EXPECT_NEAR(double(a), double(b), tolerance) << message;
}
inline void Require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
} // namespace check
namespace {

std::vector<float> Wave(std::size_t n, const std::function<float(int)> &f) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = f(int(i));
  return v;
}

std::pair<double, double> GradientNorms(const nn::ParameterStore &p, const Tensor &objective,
                                        bool nonzero, const std::string &what) {
  const auto params = p.parameters();
  const auto grads = nn::grad(objective, params);
  double low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm =
        std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(std::isfinite(norm) && (!nonzero || norm > 0),
                   "finite nonzero " + what + " derivative: " + p.names()[k]);
    low = std::min(low, norm);
    high = std::max(high, norm);
  }
  return {low, high};
}

void AspChecks() {
  nn::ParameterStore p(17);
  const std::array<int, 4> rates = {1, 2, 3, 4};
  const auto value = Wave(13 * 15, [](int i) {
    return std::sin(.13f * i) - .3f;
  });
  const auto weights = Wave(13 * 15, [](int i) {
    return std::cos(.21f * i);
  });
  Tensor image = nn::from_vector(value, {1, 13, 15, 1});
  image.requires_grad_(true);
  {
    nn::NoGradGuard guard;
    m::AtrousScorePyramid(p, "aspp", image, 1, rates);
  }
  for (int i = 0; i < 4; ++i) {
    const auto name = "aspp/rate" + std::to_string(rates[std::size_t(i)]);
    p.set(name + "/W", nn::full({3, 3, 1, 1}, .1f * (i + 1)));
    p.set(name + "/b", nn::full({1}, .01f * (i + 1)));
  }
  const auto branches = m::AtrousScorePyramid(p, "aspp", image, 1, rates);
  Tensor output = branches[0];
  for (int i = 1; i < 4; ++i)
    output = output + branches[std::size_t(i)];
  const auto gradient =
      nn::grad(nn::sum(output * nn::from_vector(weights, {1, 13, 15, 1})), {image})[0]
          .to_vector<float>();
  const auto out = output.to_vector<float>();
  std::vector<double> expected(13 * 15), derivative(13 * 15);
  for (int r = 0; r < 13; ++r)
    for (int c = 0; c < 15; ++c)
      for (int branch = 0; branch < 4; ++branch) {
        expected[std::size_t(r * 15 + c)] += .01f * (branch + 1);
        const double coefficient = .1f * (branch + 1);
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int y = r + dy * rates[std::size_t(branch)],
                      x = c + dx * rates[std::size_t(branch)];
            if (y < 0 || x < 0 || y >= 13 || x >= 15)
              continue;
            expected[std::size_t(r * 15 + c)] += coefficient * value[std::size_t(y * 15 + x)];
            derivative[std::size_t(y * 15 + x)] +=
                coefficient * weights[std::size_t(r * 15 + c)];
          }
      }
  double error = 0;
  for (int i = 0; i < 13 * 15; ++i) {
    error = std::max(error, std::abs(out[std::size_t(i)] - expected[std::size_t(i)]));
    error = std::max(error, std::abs(gradient[std::size_t(i)] - derivative[std::size_t(i)]));
  }
  check::Near(error, 0, 3e-6,
              "independent four-rate atrous score sum and all input derivatives");
  std::cout << "  ASPP rates1/2/3/4 sum and input derivatives match host taps; error=" << error
            << std::endl;
}

std::vector<double> CrfReference(const std::vector<double> &logits,
                                 const std::vector<float> &kernel, int iterations) {
  auto softmax = [](const std::vector<double> &x) {
    std::vector<double> p(x.size());
    for (std::size_t i = 0; i < 3; ++i) {
      const double maximum = std::max(x[2 * i], x[2 * i + 1]),
                   a = std::exp(x[2 * i] - maximum), b = std::exp(x[2 * i + 1] - maximum);
      p[2 * i] = a / (a + b);
      p[2 * i + 1] = b / (a + b);
    }
    return p;
  };
  auto q = softmax(logits);
  for (int step = 0; step < iterations; ++step) {
    auto next = logits;
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t c = 0; c < 2; ++c)
          next[2 * i + c] += kernel[i * 3 + j] * q[2 * j + c];
    q = softmax(next);
  }
  return q;
}

void CrfChecks() {
  std::vector<float> pixels(9);
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c)
      pixels[std::size_t(i * 3 + c)] = i == 2 ? .9f : .1f;
  m::DenseCrfConfig cfg;
  cfg.spatial_weight = .1f;
  cfg.appearance_weight = 4;
  cfg.appearance_sigma = 2;
  const auto kernel = m::DenseCrfKernel(nn::from_vector(pixels, {1, 3, 3}), cfg);
  const auto k = kernel.to_vector<float>();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const float v = k[std::size_t(i * 3 + j)];
      check::Require(std::isfinite(v) && v >= 0, "nonnegative finite CRF kernel");
      check::Near(v, k[std::size_t(j * 3 + i)], 0, "symmetric normalization");
      if (i == j)
        check::Near(v, 0, 0, "no self interaction");
    }
  const std::vector<float> unary = {-3, 3, .2f, 0, 3, -3},
                           weights = {.1f, .9f, .3f, -.4f, .8f, -.2f};
  Tensor logits = nn::from_vector(unary, {3, 2});
  logits.requires_grad_(true);
  const Tensor result = m::DenseCrfMeanField(logits, kernel, 5);
  const auto gradient =
      nn::grad(nn::sum(result * nn::from_vector(weights, {3, 2})), {logits})[0]
          .to_vector<float>();
  const auto out = result.to_vector<float>();
  std::vector<double> raw(unary.begin(), unary.end());
  const auto reference = CrfReference(raw, k, 5);
  auto objective = [&](const std::vector<double> &values) {
    const auto probabilities = CrfReference(values, k, 5);
    double total = 0;
    for (std::size_t i = 0; i < 6; ++i)
      total += probabilities[i] * weights[i];
    return total;
  };
  double error = 0, gradient_error = 0;
  for (std::size_t i = 0; i < 6; ++i) {
    error = std::max(error, std::abs(out[i] - reference[i]));
    const double original = raw[i], step = 1e-4;
    raw[i] = original + step;
    const double plus = objective(raw);
    raw[i] = original - step;
    const double minus = objective(raw);
    raw[i] = original;
    gradient_error =
        std::max(gradient_error, std::abs(gradient[i] - (plus - minus) / (2 * step)));
  }
  check::Near(error, 0, 2e-7, "independent host CRF updates");
  check::Near(gradient_error, 0, 2e-6, "CRF input central differences");
  check::Require(out[1] > out[0] && out[3] > out[2] && out[4] > out[5],
                 "bilateral affinity corrects ambiguous same-color middle pixel and preserves "
                 "color boundary");
  std::cout << "  Gaussian/bilateral Potts mean field matches host; forward error=" << error
            << ", unary derivative error=" << gradient_error << std::endl;
}

m::DeepLabOutputs Build(nn::ParameterStore &p, nn::ParameterStore &state, const Tensor &images,
                        bool v2, bool compact, bool training) {
  if (v2) {
    m::DeepLabV2Config c;
    if (compact) {
      c.classes = 4;
      c.backbone_depth = 18;
      c.width_divisor = 8;
      c.rates = {1, 2, 3, 4};
    }
    return m::DeepLabV2(p, state, images, c, training);
  }
  m::DeepLabV1Config c;
  if (compact) {
    c.classes = 4;
    c.width_divisor = 8;
    c.fc_width = 128;
    c.fc_rate = 4;
    c.keep_probability = 1;
  }
  return m::DeepLabV1(p, images, c, training);
}

void BackboneChecks() {
  for (int stride : {32, 16, 8}) {
    nn::ParameterStore p(17), state(18);
    m::ResNetConfig c;
    c.depth = 18;
    c.width_divisor = 8;
    c.include_classifier = false;
    const auto net = m::DenseResNet(p, state, nn::full({1, 96, 96, 3}, .5), c, stride, true);
    check::Require(
        p.count() == 176712 && state.count() == 1200,
        "output-stride change preserves basic ResNet parameters and default counts");
    for (int stage = 0; stage < 4; ++stage) {
      const std::int64_t side = 96 / std::min(stride, 4 << stage);
      check::Require(net.stages[std::size_t(stage)].shape()[1] == side &&
                         net.stages[std::size_t(stage)].shape()[2] == side,
                     "unchanged standard strides or requested dense stages");
    }
  }
  std::cout
      << "  default stride32 and atrous stride16/8 backbones preserve parameter/state counts "
         "and expected stage geometry"
      << std::endl;
}

void ModelChecks() {
  for (bool v2 : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const auto net = Build(p, state,
                           nn::from_vector(Wave(33 * 33 * 3,
                                                [](int i) {
                                                  return std::sin(.13f * i);
                                                }),
                                           {1, 33, 33, 3}),
                           v2, false, true);
    check::Require(p.count() == (v2 ? 44048532 : 20505429) &&
                       state.count() == (v2 ? 105344 : 0),
                   "full DeepLab trainable/state counts");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full DeepLab");
    check::Require(p.size() == std::size_t(v2 ? 320 : 32),
                   "every full DeepLab parameter connected");
    std::cout << "  full DeepLab v" << (v2 ? 2 : 1) << ": " << p.count() << " parameters, "
              << p.size() << " derivatives (executed on a 33x33 image)" << std::endl;
  }
  for (bool v2 : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const auto net = Build(p, state,
                           nn::from_vector(Wave(2 * 96 * 96 * 3,
                                                [](int i) {
                                                  return std::sin(.13f * i);
                                                }),
                                           {2, 96, 96, 3}),
                           v2, true, true);
    check::Require(p.count() == (v2 ? 185944 : 321452), "compact DeepLab parameter counts");
    check::Require(net.logits.shape() == nn::Shape({2, 96, 96, 4}) &&
                       net.features.shape()[1] == 12 && net.features.shape()[2] == 12,
                   "stride-eight features and resized logits");
    const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "DeepLab");
    std::cout << "  compact DeepLab v" << (v2 ? 2 : 1) << ": " << p.count()
              << " parameters; all derivative norms [" << norms.first << ',' << norms.second
              << "]" << std::endl;
  }
}

} // namespace
TEST(NnDeepLab, IndependentFourRateAtrousValuesEveryInputDerivative) {
  AspChecks();
}
TEST(NnDeepLab, IndependentPottsUpdatesEveryUnaryFiniteDifference) {
  CrfChecks();
}
TEST(NnDeepLab, OutputStridesPreserveWeightsStateAndStageShapes) {
  BackboneChecks();
}
TEST(NnDeepLab, FullV1V2EveryGradientCountsAndCompactEveryNonzeroGradient) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(NnDeepLab, ExactTensorGaussianAndBilateralKernelMatchesIndependentAllPairs) {
  const int H = 3, W = 4, P = H * W;
  std::vector<float> image(P * 3);
  for (int i = 0; i < P * 3; ++i)
    image[i] = float(std::sin(.17 * i) * .4 + .5);
  m::DenseCrfConfig cfg;
  cfg.spatial_sigma = 1.7f;
  cfg.appearance_sigma = 2.3f;
  cfg.color_sigma = .3f;
  cfg.spatial_weight = .4f;
  cfg.appearance_weight = 1.7f;
  const auto kernel =
      m::DenseCrfKernel(nn::from_vector(image, {H, W, 3}).requires_grad_(), cfg)
          .to_vector<float>();
  std::vector<float> expected(P * P, 0);
  for (int comp = 0; comp < 2; ++comp) {
    std::vector<double> weights(P * P), degree(P);
    const double sigma = comp ? cfg.appearance_sigma : cfg.spatial_sigma,
                 scale = comp ? cfg.appearance_weight : cfg.spatial_weight;
    for (int i = 0; i < P; ++i)
      for (int j = i + 1; j < P; ++j) {
        const double dy = i / W - j / W, dx = i % W - j % W;
        double exponent = (dy * dy + dx * dx) / (2 * sigma * sigma);
        if (comp)
          for (int c = 0; c < 3; ++c) {
            const double delta = image[i * 3 + c] - image[j * 3 + c];
            exponent += delta * delta / (2 * double(cfg.color_sigma) * cfg.color_sigma);
          }
        const double v = std::exp(-exponent);
        weights[i * P + j] = weights[j * P + i] = v;
        degree[i] += v;
        degree[j] += v;
      }
    for (int i = 0; i < P; ++i)
      for (int j = 0; j < P; ++j)
        if (i != j)
          expected[i * P + j] += float(scale * weights[i * P + j] /
                                       std::sqrt(std::max(degree[i] * degree[j], 1e-30)));
  }
  for (int i = 0; i < P * P; ++i)
    EXPECT_NEAR(kernel[i], expected[i], 3e-8);
  EXPECT_FALSE(m::DenseCrfKernel(nn::from_vector(image, {H, W, 3}).requires_grad_(), cfg)
                   .requires_grad());
  EXPECT_EQ(m::DenseCrfKernel(nn::zeros({1, 1, 3})).item(), 0);
}
TEST(NnDeepLab, PixelCentreSamplingSelectsOneFourSevenAndPreservesGradients) {
  auto x = nn::arange(0, 96 * 96).reshape({1, 96, 96, 1}).requires_grad_();
  const auto y = m::SampleSegmentationGrid(x);
  EXPECT_EQ(y.shape(), nn::Shape({1, 32, 32, 1}));
  EXPECT_EQ(y.at({0, 0, 0, 0}), 97);
  EXPECT_EQ(y.at({0, 31, 31, 0}), 94 * 96 + 94);
  const auto g = nn::grad(nn::sum(y), {x})[0];
  EXPECT_EQ(nn::count_nonzero(g).item(), 1024);
  EXPECT_EQ(g.at({0, 0, 0, 0}), 0);
  EXPECT_EQ(g.at({0, 1, 1, 0}), 1);
}
TEST(NnDeepLab, InvalidRatesShapesBandWidthsAndIterationsReject) {
  nn::ParameterStore p(17);
  EXPECT_THROW(m::AtrousScorePyramid(p, "bad", nn::ones({1, 4, 4, 1}), 2, {1, 1, 2, 3}),
               std::invalid_argument);
  m::DenseCrfConfig c;
  c.color_sigma = NAN;
  EXPECT_THROW(m::DenseCrfKernel(nn::zeros({2, 2, 3}), c), std::invalid_argument);
  EXPECT_THROW(m::DenseCrfKernel(nn::zeros({65, 65, 3})), std::invalid_argument);
  EXPECT_THROW(m::DenseCrfMeanField(nn::zeros({3, 2}), nn::zeros({3, 3}), 0),
               std::invalid_argument);
}
