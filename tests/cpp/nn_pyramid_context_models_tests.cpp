#include <dm/dm.hpp>
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
namespace v3_checks {

std::vector<float> Wave(std::size_t n, double a, double b, double scale) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = float(a + b * std::sin(scale * double(i)));
  return v;
}

void ImagePoolingChecks() {
  nn::ParameterStore p(17), state(18);
  m::ConvNormConfig cfg;
  cfg.scale = true;
  cfg.epsilon = 1e-5;
  const auto value = Wave(2 * 7 * 9 * 2, .7, .4, .13);
  Tensor input = nn::from_vector(value, {2, 7, 9, 2});
  {
    nn::NoGradGuard guard;
    m::ImagePoolBranch(p, state, "pool", input, 2, cfg);
  }
  p.set("pool/project/W", nn::from_vector(std::vector<float>{1, 0, 0, 1}, {1, 1, 2, 2}));
  p.set("pool/project/bn/gamma", nn::full({2}, std::sqrt(1.f + float(cfg.epsilon))));
  input.requires_grad_(true);
  const Tensor output = m::ImagePoolBranch(p, state, "pool", input, 2, cfg);
  const auto gradient = nn::grad(nn::sum(output), {input})[0].to_vector<float>();
  const auto out = output.to_vector<float>();
  double error = 0;
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 2; ++c) {
      double mean = 0;
      for (int i = 0; i < 63; ++i)
        mean += value[std::size_t((n * 63 + i) * 2 + c)] / 63.;
      for (int i = 0; i < 63; ++i) {
        const auto index = std::size_t((n * 63 + i) * 2 + c);
        error = std::max(error, std::abs(out[index] - mean));
        error = std::max(error, std::abs(gradient[index] - 1.));
      }
    }
  check::Near(
      error, 0, 5e-6,
      "image-pooling projection broadcasts independent global means and accumulates all pixel "
      "derivatives");
  std::cout << "  image-level ASPP branch mean, constant broadcast and every input derivative "
               "passed; error="
            << error << std::endl;
}

m::DeepLabV3Config Config(bool plus, bool compact) {
  m::DeepLabV3Config c;
  c.decoder = plus;
  if (compact) {
    c.classes = 4;
    c.backbone_depth = 18;
    c.width_divisor = 8;
    c.head_width = 32;
    c.decoder_width = 32;
    c.low_width = 8;
    c.rates = {1, 2, 3};
    c.keep_probability = 1;
  }
  return c;
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

void ModelChecks() {
  for (bool plus : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const auto net =
        m::DeepLabV3(p, state, nn::from_vector(Wave(33 * 33 * 3, 0, 1, .13), {1, 33, 33, 3}),
                     Config(plus, false), true);
    check::Require(p.count() == (plus ? 26696133 : 39048533) &&
                       state.count() == (plus ? 70720 : 56192),
                   "full ResNet50 DeepLab v3/plus parameter/state counts");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full DeepLab v3");
    check::Require(p.size() == std::size_t(plus ? 203 : 179) && net.branches.size() == 5,
                   "all five branches and decoder parameters connected");
    std::cout << "  full DeepLab " << (plus ? "v3+" : "v3") << ": " << p.count()
              << " parameters, " << p.size() << " derivatives (executed on a 33x33 image)"
              << std::endl;
  }
  for (bool plus : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const auto net = m::DeepLabV3(
        p, state, nn::from_vector(Wave(3 * 96 * 96 * 3, 0, 1, .13), {3, 96, 96, 3}),
        Config(plus, true), true);
    std::cout << "  compact parameter/state counts:" << p.count() << '/' << state.count()
              << std::endl;
    check::Require(p.count() == (plus ? 198004 : 241740) &&
                       state.count() == (plus ? 2256 : 1584),
                   "compact v3/plus parameter/state counts");
    check::Require(p.at("aspp/project/W").shape()[2] == 160,
                   "five ASPP branches concatenate before projection");
    check::Require(net.logits.shape() == nn::Shape({3, 96, 96, 4}) &&
                       net.features.shape()[1] == 6 &&
                       (plus ? net.low : net.coarse).shape()[1] == (plus ? 24 : 6),
                   "stride16 features, optional stride4 decoder and full logits");
    const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "DeepLab v3");
    std::cout << "  compact " << (plus ? "v3+" : "v3") << ": " << p.count()
              << " parameters; every derivative norm [" << norms.first << ',' << norms.second
              << "]" << std::endl;
  }
}

} // namespace v3_checks
TEST(NnV3, IndependentContextMeansAndEveryInputDerivative) {
  v3_checks::ImagePoolingChecks();
}
TEST(NnV3, FullEveryGradientCountsCompactNonzeroAndDynamicGrids) {
  nn::set_num_threads(2);
  v3_checks::ModelChecks();
}
namespace psp_checks {

Tensor Wave(const nn::Shape &shape, double scale, bool cosine = false) {
  std::int64_t n = 1;
  for (auto d : shape)
    n *= d;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = cosine ? std::cos(float(scale) * float(i)) : std::sin(float(scale) * float(i));
  return nn::from_vector(v, shape);
}

std::pair<double, double> GradientNorms(const nn::ParameterStore &p, const Tensor &objective,
                                        bool nonzero, const std::string &what) {
  const auto params = p.parameters();
  const auto grads = nn::grad(objective, params);
  check::Require(grads.size() == params.size(), what + ": every parameter has a derivative");
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

void PoolChecks() {
  for (const auto shape :
       std::vector<std::array<int, 3>>{{7, 9, 3}, {2, 3, 6}, {12, 12, 2}, {1, 1, 1}}) {
    const int h = shape[0], w = shape[1], bins = shape[2];
    Tensor input = Wave({2, h, w, 2}, .17);
    input.requires_grad_(true);
    const Tensor weights = Wave({2, bins, bins, 2}, .23, true);
    const Tensor pool = m::AdaptiveAveragePool(input, bins);
    const auto gradient = nn::grad(nn::sum(pool * weights), {input})[0].to_vector<float>();
    const auto value = input.to_vector<float>(), out = pool.to_vector<float>(),
               weight = weights.to_vector<float>();
    std::vector<double> expected_gradient(value.size(), 0);
    double error = 0;
    for (int n = 0; n < 2; ++n)
      for (int by = 0; by < bins; ++by)
        for (int bx = 0; bx < bins; ++bx)
          for (int c = 0; c < 2; ++c) {
            const int y0 = by * h / bins, y1 = ((by + 1) * h + bins - 1) / bins,
                      x0 = bx * w / bins, x1 = ((bx + 1) * w + bins - 1) / bins,
                      index = ((n * bins + by) * bins + bx) * 2 + c;
            double mean = 0;
            for (int y = y0; y < y1; ++y)
              for (int x = x0; x < x1; ++x) {
                const int i = ((n * h + y) * w + x) * 2 + c;
                mean += value[std::size_t(i)] / double((y1 - y0) * (x1 - x0));
                expected_gradient[std::size_t(i)] +=
                    weight[std::size_t(index)] / double((y1 - y0) * (x1 - x0));
              }
            error = std::max(error, std::abs(mean - out[std::size_t(index)]));
          }
    for (std::size_t i = 0; i < value.size(); ++i)
      error = std::max(error, std::abs(expected_gradient[i] - gradient[i]));
    check::Near(error, 0, 2e-6,
                "every independent adaptive-pool mean and overlapping pixel gradient");
    std::cout << "  adaptive pool " << h << 'x' << w << " -> " << bins << 'x' << bins
              << ": host value/gradient error=" << error << std::endl;
  }
}

m::PSPNetConfig Config(bool compact) {
  m::PSPNetConfig c;
  if (compact) {
    c.classes = 4;
    c.depth = 18;
    c.width_divisor = 8;
    c.pool_width = 16;
    c.head_width = 32;
    c.aux_width = 16;
    c.keep_probability = 1;
  }
  return c;
}

Tensor Objective(const m::PSPNetOutputs &net) {
  return nn::mean(nn::square(net.logits)) + .4 * nn::mean(nn::square(net.auxiliary));
}

void ModelChecks() {
  {
    nn::ParameterStore p(17), state(17);
    const auto net = m::PSPNet(p, state, Wave({1, 33, 33, 3}, .13), Config(false), true);
    GradientNorms(p, Objective(net), false, "full PSPNet");
    check::Require(net.pools.size() == 4, "full PSPNet every parameter connected");
    check::Require(p.at("head/W").shape() == nn::Shape({3, 3, 4096, 512}),
                   "full pyramid channels concatenate to 4096");
    check::Require(
        p.count() == 67949930 && state.count() == 110976 && p.size() == 334,
        "full PSPNet101 census (TF: 67949930 parameters, 110976 BN state, 334 gradients)");
    std::cout << "  full PSPNet101: " << p.count() << " parameters, " << state.count()
              << " BN state, " << p.size() << " gradients (executed on 33x33)" << std::endl;
  }
  for (int size : {96, 97}) {
    nn::ParameterStore p(17), state(17);
    const auto net = m::PSPNet(p, state, Wave({2, size, size, 3}, .13), Config(true), true);
    check::Require(net.logits.shape() == nn::Shape({2, size, size, 4}) &&
                       net.auxiliary.shape() == net.logits.shape(),
                   "main and auxiliary logits match input grid");
    const auto norms = GradientNorms(p, Objective(net), true, "PSPNet");
    const int bins[] = {1, 2, 3, 6};
    for (int k = 0; k < 4; ++k)
      check::Require(net.pools[std::size_t(k)].shape()[1] == bins[k] &&
                         net.pools[std::size_t(k)].shape()[2] == bins[k],
                     "four distinct pyramid sizes");
    std::cout << "  compact PSPNet " << size << "px: " << p.count()
              << " parameters; every gradient norm [" << norms.first << ',' << norms.second
              << "]" << std::endl;
  }
}

} // namespace psp_checks
TEST(NnPsp, IndependentContextMeansAndEveryInputDerivative) {
  psp_checks::PoolChecks();
}
TEST(NnPsp, FullEveryGradientCountsCompactNonzeroAndDynamicGrids) {
  nn::set_num_threads(2);
  psp_checks::ModelChecks();
}
TEST(NnPyramidContext, AdaptiveBinsLargerThanGridAccumulateEveryReuse) {
  auto x = nn::full({1, 1, 1, 2}, 3.).requires_grad_();
  const auto y = m::AdaptiveAveragePool(x, 6);
  EXPECT_EQ(y.shape(), nn::Shape({1, 6, 6, 2}));
  EXPECT_EQ(nn::min(y).item(), 3);
  const auto g = nn::grad(nn::sum(y), {x})[0];
  EXPECT_EQ(g.to_vector<float>(), std::vector<float>({36, 36}));
}
TEST(NnPyramidContext, V3StrideEightDoublesDenseFeatureResolutionAndPspCanDisableAuxiliary) {
  nn::ParameterStore p(1), state(1);
  m::DeepLabV3Config c;
  c.classes = 4;
  c.backbone_depth = 18;
  c.width_divisor = 8;
  c.head_width = 8;
  c.decoder_width = 8;
  c.low_width = 4;
  c.rates = {1, 2, 3};
  c.keep_probability = 1;
  const auto first = m::DeepLabV3(p, state, nn::ones({1, 64, 64, 3}), c);
  const auto count = p.count();
  c.output_stride = 8;
  const auto second = m::DeepLabV3(p, state, nn::ones({1, 64, 64, 3}), c);
  EXPECT_EQ(first.features.shape()[1], 4);
  EXPECT_EQ(second.features.shape()[1], 8);
  EXPECT_EQ(p.count(), count);
  nn::ParameterStore q(1), qs(1);
  m::PSPNetConfig cfg;
  cfg.classes = 4;
  cfg.depth = 18;
  cfg.width_divisor = 8;
  cfg.pool_width = 8;
  cfg.head_width = 8;
  cfg.aux_width = 8;
  cfg.auxiliary = false;
  const auto net = m::PSPNet(q, qs, nn::ones({1, 33, 33, 3}), cfg);
  EXPECT_FALSE(net.auxiliary.defined());
  EXPECT_FALSE(q.contains("aux/conv/W"));
  EXPECT_EQ(net.pools.size(), 4);
}
TEST(NnPyramidContext, InvalidBinsAndDuplicateOrOverflowingAtrousRatesReject) {
  EXPECT_THROW(m::AdaptiveAveragePool(nn::ones({1, 3, 4, 1}), 0), std::invalid_argument);
  nn::ParameterStore p(1), state(1);
  m::DeepLabV3Config c;
  c.rates = {1, 1, 2};
  EXPECT_THROW(m::DeepLabV3(p, state, nn::ones({1, 32, 32, 3}), c), std::invalid_argument);
  c.output_stride = 8;
  c.rates = {1, 2, std::numeric_limits<int>::max()};
  EXPECT_THROW(m::DeepLabV3(p, state, nn::ones({1, 32, 32, 3}), c), std::invalid_argument);
}
