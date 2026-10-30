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
namespace {

void FusionChecks() {
  nn::ParameterStore p(17), state(18);
  m::ResNetNormConfig norm;
  norm.scale = true;
  norm.epsilon = 1e-5;
  std::vector<float> h(35), l(12);
  for (int i = 0; i < 35; ++i)
    h[std::size_t(i)] = 1 + .03f * i;
  for (int i = 0; i < 12; ++i)
    l[std::size_t(i)] = .5f + .07f * i;
  Tensor high = nn::from_vector(h, {1, 5, 7, 1}), low = nn::from_vector(l, {1, 3, 4, 1});
  {
    nn::NoGradGuard guard;
    m::HRFuse(p, state, "fixture", {high, low}, {1, 1}, norm);
  }
  for (const auto &name : p.names()) {
    const auto shape = p.at(name).shape();
    std::vector<float> value(std::size_t(nn::shape_numel(shape)), 0.f);
    if (name.size() >= 2 && name.substr(name.size() - 2) == "/W")
      value[value.size() / 2] = 1;
    else if (name.find("/gamma") != std::string::npos)
      std::fill(value.begin(), value.end(), std::sqrt(1.f + float(norm.epsilon)));
    p.set(name, nn::from_vector(value, shape));
  }
  high.requires_grad_(true);
  low.requires_grad_(true);
  const auto fused = m::HRFuse(p, state, "fixture", {high, low}, {1, 1}, norm);
  const auto gradient = nn::grad(nn::sum(fused[0]) + nn::sum(fused[1]), {high, low});
  const auto o0 = fused[0].to_vector<float>(), o1 = fused[1].to_vector<float>();
  const auto g0 = gradient[0].to_vector<float>(), g1 = gradient[1].to_vector<float>();
  std::vector<double> gh(35, 1), gl(12, 1);
  double error = 0;
  for (int y = 0; y < 5; ++y)
    for (int x = 0; x < 7; ++x) {
      const double sy = y * .5, sx = x * .5;
      const int y0 = int(sy), x0 = int(sx), y1 = std::min(y0 + 1, 2), x1 = std::min(x0 + 1, 3);
      double up = 0;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
          const int index = (dy ? y1 : y0) * 4 + (dx ? x1 : x0);
          const double weight =
              (dy ? sy - y0 : 1 - (sy - y0)) * (dx ? sx - x0 : 1 - (sx - x0));
          up += weight * l[std::size_t(index)];
          gl[std::size_t(index)] += weight;
        }
      error = std::max(error,
                       std::abs(o0[std::size_t(y * 7 + x)] - h[std::size_t(y * 7 + x)] - up));
    }
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 4; ++x) {
      const int index = y * 14 + x * 2;
      gh[std::size_t(index)] += 1;
      error =
          std::max(error, std::abs(double(o1[std::size_t(y * 4 + x)] -
                                          l[std::size_t(y * 4 + x)] - h[std::size_t(index)])));
    }
  for (int i = 0; i < 35; ++i)
    error = std::max(error, std::abs(gh[std::size_t(i)] - g0[std::size_t(i)]));
  for (int i = 0; i < 12; ++i)
    error = std::max(error, std::abs(gl[std::size_t(i)] - g1[std::size_t(i)]));
  check::Near(error, 0, 2e-6, "every high/low exchange value and input gradient");
  std::cout << "  odd-grid high/low fusion independent bilinear/subsample and gradient error="
            << error << std::endl;
}

m::HRNetConfig Config(bool compact) {
  m::HRNetConfig c;
  if (compact) {
    c.classes = 4;
    c.width = 8;
    c.stem_width = 16;
    c.stage1_width = 16;
    c.stage1_blocks = 2;
    c.blocks = 2;
    c.modules = {1, 1, 1};
  }
  return c;
}

std::vector<float> Sine(std::size_t n) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = std::sin(.13f * float(i));
  return v;
}

void ModelChecks() {
  {
    nn::ParameterStore p(17), state(18);
    const auto net = m::HRNet(p, state, nn::from_vector(Sine(33 * 33 * 3), {1, 33, 33, 3}),
                              Config(false), true);
    const auto params = p.parameters();
    const auto grads = nn::grad(nn::mean(nn::square(net.logits)), params);
    for (std::size_t k = 0; k < grads.size(); ++k)
      check::Require(std::isfinite(nn::sum(nn::square(grads[k])).item<double>()),
                     "finite full HRNet derivative: " + p.names()[k]);
    check::Require(net.fusion_modules == 8 && net.branches.size() == 4 &&
                       grads.size() == p.size(),
                   "full W32 eight exchange modules/four branches/all parameters");
    check::Require(p.at("head/project/W").shape() == nn::Shape({1, 1, 480, 480}),
                   "full four streams concatenate480channels");
    std::cout << "  full HRNetV2 W32: " << p.count() << " parameters, " << state.count()
              << " BN state, " << grads.size() << " gradients (executed on a 33x33 image)"
              << std::endl;
  }
  for (int size : {96, 97}) {
    nn::ParameterStore p(17), state(18);
    const auto net = m::HRNet(
        p, state, nn::from_vector(Sine(std::size_t(2 * size * size * 3)), {2, size, size, 3}),
        Config(true), true);
    check::Require(net.logits.shape() == nn::Shape({2, size, size, 4}),
                   "HRNet logits restore odd/even input grid");
    std::int64_t resolution = (size + 3) / 4;
    for (int k = 0; k < 4; ++k) {
      const auto &shape = net.branches[std::size_t(k)].shape();
      check::Require(shape[1] == resolution && shape[2] == resolution &&
                         shape[3] == 8 * (1 << k),
                     "HRNet persistent branch dimensions");
      resolution = (resolution + 1) / 2;
    }
    const auto params = p.parameters();
    const auto grads = nn::grad(nn::mean(nn::square(net.logits)), params);
    double low = INFINITY, high = 0;
    for (std::size_t k = 0; k < grads.size(); ++k) {
      const double norm =
          std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
      check::Require(std::isfinite(norm) && norm > 0,
                     "finite nonzero HRNet derivative: " + p.names()[k]);
      low = std::min(low, norm);
      high = std::max(high, norm);
    }
    std::cout << "  compact HRNet " << size << "px: " << p.count()
              << " parameters; all gradient norms [" << low << ',' << high << "]" << std::endl;
  }
}

} // namespace
TEST(NnHrnet, IndependentOddGridProjectionInterpolationSubsamplingEveryInputGradient) {
  FusionChecks();
}
TEST(NnHrnet, FullEveryGradientCompactEveryNonzeroGradientAndDynamicFourBranches) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(NnHrnet, InvalidWidthsResolutionsAndImageGeometryReject) {
  nn::ParameterStore p(17), state(18);
  m::ResNetNormConfig norm;
  EXPECT_THROW(m::HRFuse(p, state, "bad", {nn::ones({1, 5, 7, 1}), nn::ones({1, 2, 3, 2})},
                         {1, 2}, norm),
               std::invalid_argument);
  m::HRNetConfig c;
  c.width = std::numeric_limits<int>::max();
  EXPECT_THROW(m::HRNet(p, state, nn::ones({1, 32, 32, 3}), c), std::invalid_argument);
  EXPECT_THROW(m::HRNet(p, state, nn::ones({1, 32, 32, 1})), std::invalid_argument);
}
