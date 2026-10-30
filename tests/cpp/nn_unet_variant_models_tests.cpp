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
struct TestSegmentationModel {
  Tensor logits;
  std::vector<Tensor> auxiliary;
};
namespace {

Tensor Wave(const nn::Shape &shape, double scale) {
  std::int64_t n = 1;
  for (auto d : shape)
    n *= d;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = std::sin(float(scale) * float(i));
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

void GateChecks() {
  nn::ParameterStore p(17);
  p.set("gate/x/W", nn::from_vector(std::vector<float>{.5f, -.2f}, {1, 1, 2, 1}));
  p.set("gate/g/W", nn::from_vector(std::vector<float>{.3f, .4f}, {1, 1, 2, 1}));
  p.set("gate/x/b", nn::from_vector(std::vector<float>{.1f}, {1}));
  p.set("gate/g/b", nn::from_vector(std::vector<float>{-.1f}, {1}));
  p.set("gate/psi/W", nn::from_vector(std::vector<float>{.7f}, {1, 1, 1, 1}));
  p.set("gate/psi/b", nn::from_vector(std::vector<float>{-.2f}, {1}));
  std::vector<float> a(12), b(12);
  for (int i = 0; i < 6; ++i) {
    a[std::size_t(2 * i)] = float(.2 + .1 * i);
    a[std::size_t(2 * i + 1)] = float(-.3 + .15 * i);
    b[std::size_t(2 * i)] = float(-.2 + .05 * i);
    b[std::size_t(2 * i + 1)] = i ? float(.4 - .02 * i) : -2.f;
  }
  auto run = [&](std::vector<float> *coefficients, std::vector<float> *output) {
    Tensor x = nn::from_vector(a, {1, 2, 3, 2}), g = nn::from_vector(b, {1, 2, 3, 2});
    x.requires_grad_(true);
    g.requires_grad_(true);
    const auto net = m::AttentionGate(p, "gate", x, g, 1);
    const auto gradients = nn::grad(nn::sum(net.output), {x, g});
    if (coefficients)
      *coefficients = net.coefficients.to_vector<float>();
    if (output)
      *output = net.output.to_vector<float>();
    return std::make_pair(gradients[0].to_vector<float>(), gradients[1].to_vector<float>());
  };
  std::vector<float> coefficients, output;
  const auto grads = run(&coefficients, &output);
  double error = 0;
  const double wx[] = {.5, -.2}, wg[] = {.3, .4};
  for (int i = 0; i < 6; ++i) {
    const double z = .5 * a[std::size_t(2 * i)] - .2 * a[std::size_t(2 * i + 1)] +
                     .3 * b[std::size_t(2 * i)] + .4 * b[std::size_t(2 * i + 1)];
    const double coefficient = 1 / (1 + std::exp(-(.7 * std::max(z, 0.) - .2)));
    const double sum = a[std::size_t(2 * i)] + a[std::size_t(2 * i + 1)];
    const double chain = z > 0 ? sum * coefficient * (1 - coefficient) * .7 : 0;
    error = std::max(error, std::abs(coefficients[std::size_t(i)] - coefficient));
    for (int c = 0; c < 2; ++c) {
      const std::size_t k = std::size_t(2 * i + c);
      error = std::max(error, std::abs(output[k] - a[k] * coefficient));
      error = std::max(error, std::abs(grads.first[k] - (coefficient + chain * wx[c])));
      error = std::max(error, std::abs(grads.second[k] - chain * wg[c]));
    }
  }
  check::Near(error, 0, 2e-7, "independent additive-gate forward and chain rule");
  p.set("gate/psi/b", nn::from_vector(std::vector<float>{100.f}, {1}));
  const auto saturated = run(nullptr, nullptr);
  for (int i = 0; i < 12; ++i) {
    check::Near(saturated.first[std::size_t(i)], 1, 0,
                "saturated open gate identity derivative");
    check::Near(saturated.second[std::size_t(i)], 0, 0, "saturated gate query derivative");
  }
  std::cout
      << "  additive attention values/input derivatives, negative ReLU and saturated sigmoid "
         "passed; error="
      << error << std::endl;
}

TestSegmentationModel Build(nn::ParameterStore &p, const Tensor &images, bool nested,
                            bool compact) {
  if (nested) {
    m::NestedUnetConfig c;
    c.classes = compact ? 4 : 2;
    c.base_width = compact ? 8 : 32;
    const auto net = m::NestedUnet(p, images, c);
    return {net.logits, std::vector<Tensor>(net.heads.begin(), net.heads.end() - 1)};
  }
  m::AttentionUnetConfig c;
  c.classes = compact ? 4 : 2;
  c.base_width = compact ? 8 : 64;
  return {m::AttentionUnet(p, images, c).logits, {}};
}

Tensor Objective(const TestSegmentationModel &net) {
  Tensor sum = nn::mean(nn::square(net.logits));
  for (const auto &output : net.auxiliary)
    sum = sum + nn::mean(nn::square(output));
  return sum;
}

void ModelChecks() {
  for (bool nested : {true, false}) {
    nn::ParameterStore p(17);
    const auto net = Build(p, Wave({1, 32, 32, 3}, .13), nested, false);
    GradientNorms(p, Objective(net), false, "full U-Net variant");
    check::Require(p.count() == (nested ? 9156264 : 31381414),
                   "full no-BN nested/attention U-Net parameter counts");
    check::Require(p.size() == (nested ? 68u : 70u),
                   "full U-Net variant parameter connections (TF: 68/70)");
    if (nested)
      check::Require(
          p.at("x0_4/conv1/W").shape()[2] == 192 && net.auxiliary.size() == 3,
          "deepest dense node concatenates all four prior nodes and two-width lower feature");
    std::cout << "  full " << (nested ? "UNet++" : "attention U-Net") << ": " << p.count()
              << " parameters, all " << p.size() << " derivatives (executed on 32x32)"
              << std::endl;
  }
  for (bool nested : {true, false}) {
    nn::ParameterStore p(17);
    const auto net = Build(p, Wave({2, 96, 96, 3}, .13), nested, true);
    check::Require(net.logits.shape() == nn::Shape({2, 96, 96, 4}),
                   "full-resolution variant logits");
    const auto norms = GradientNorms(p, Objective(net), true, "variant");
    std::cout << "  compact " << (nested ? "UNet++" : "attention U-Net") << ": " << p.count()
              << " parameters, every derivative norm [" << norms.first << ',' << norms.second
              << "]" << std::endl;
  }
}

} // namespace
TEST(NnUnetVariants, IndependentGateValuesAllInputDerivativesReluAndSaturatedSigmoid) {
  GateChecks();
}
TEST(NnUnetVariants, Full68And70GradientsCountsCompactEveryGradientAndDenseWidths) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(NnUnetVariants, NestedOddAlignmentAndAttentionRoundedGeometry) {
  nn::ParameterStore p(17), q(17);
  m::NestedUnetConfig a;
  a.base_width = 2;
  a.classes = 4;
  const auto nested = m::NestedUnet(p, nn::ones({1, 95, 97, 3}), a);
  EXPECT_EQ(nested.logits.shape(), nn::Shape({1, 95, 97, 4}));
  EXPECT_EQ(nested.heads.size(), 4);
  EXPECT_EQ(nested.nodes.size(), 15);
  m::AttentionUnetConfig b;
  b.base_width = 3;
  b.classes = 4;
  const auto attention = m::AttentionUnet(q, nn::ones({1, 95, 97, 3}), b);
  EXPECT_EQ(attention.logits.shape(), nn::Shape({1, 80, 96, 4}));
  EXPECT_EQ(attention.coefficients.size(), 4);
}
TEST(NnUnetVariants, InvalidGateShapeAndTooSmallInputReject) {
  nn::ParameterStore p(1);
  EXPECT_THROW(m::AttentionGate(p, "bad", nn::ones({1, 4, 4, 2}), nn::ones({1, 3, 4, 2}), 1),
               std::invalid_argument);
  EXPECT_THROW(m::NestedUnet(p, nn::ones({1, 8, 8, 3})), std::invalid_argument);
  EXPECT_THROW(m::AttentionUnet(p, nn::ones({1, 8, 8, 3})), std::invalid_argument);
}
