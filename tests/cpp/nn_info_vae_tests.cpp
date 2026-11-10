#include <gtest/gtest.h>

#include <datamunge/nn/info_variational_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnInfoVae, HandUnequalKernelsUnitDiagonalsSignedMmdAndAllTerms) {
  const auto t = n::DType::Float64;
  auto       x = n::tensor({0., 1.}, {2, 1}, t), y = n::tensor({0., 2., 3.}, {3, 1}, t);
  auto       out    = m::ImqMmd(x, y);
  auto       kernel = [](double d) {
    return (1 / (1 + d * d) + 4 / (4 + d * d) + 16 / (16 + d * d)) / 3;
  };
  double cross = 0, yy = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      double value = kernel(x.at({i, 0}) - y.at({j, 0}));
      EXPECT_NEAR(out.xy.at({i, j}), value, 1e-15);
      cross += value / 6;
    }
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(out.yy.at({i, i}), 1);
    for (int j = 0; j < 3; ++j)
      if (i != j)
        yy += kernel(y.at({i, 0}) - y.at({j, 0})) / 6;
  }
  EXPECT_NEAR(out.within_x.item(), kernel(1), 1e-15);
  EXPECT_NEAR(out.within_y.item(), yy, 1e-15);
  EXPECT_NEAR(out.cross.item(), cross, 1e-15);
  EXPECT_NEAR(out.value.item(), kernel(1) + yy - 2 * cross, 1e-15);
  EXPECT_LT(m::ImqMmd(x, x).value.item(), 0);
  EXPECT_EQ(m::ImqMmd(n::zeros({3, 2}, t), n::zeros({4, 2}, t)).value.item(), 0);
  EXPECT_NEAR(out.value.item(), m::ImqMmd(x + 7, y + 7).value.item(), 1e-15);
  EXPECT_NEAR(
      out.value.item(),
      m::ImqMmd(n::take_rows(x, {1, 0}), n::take_rows(y, {2, 0, 1})).value.item(),
      1e-15);
}
TEST(NnInfoVae, BothKernelAndUnbiasedMmdFirstSecondDerivativesAndStrides) {
  const auto t        = n::DType::Float64;
  auto       x        = n::tensor({.1, .4, -.3, .6, .2, .7}, {3, 2}, t).requires_grad_(),
       y              = n::tensor({.3, -.2, .5, .8}, {2, 2}, t).requires_grad_();
  n::TensorFunction f = [](const auto& a) {
    return m::ImqMmd(a[0], a[1]).value + .2 * n::mean(n::square(m::ImqKernel(a[0], a[1])));
  };
  EXPECT_TRUE(n::gradcheck(f, {x, y}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, y}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  const auto sx = n::transpose(n::transpose(x).contiguous()),
             sy = n::transpose(n::transpose(y).contiguous());
  EXPECT_FALSE(sx.is_contiguous());
  EXPECT_FALSE(sy.is_contiguous());
  EXPECT_EQ(m::ImqKernel(sx, sy).to_vector<double>(), m::ImqKernel(x, y).to_vector<double>());
  EXPECT_EQ(m::ImqMmd(sx, sy).value.item(), m::ImqMmd(x, y).value.item());
}
TEST(NnInfoVae, ExactObjectiveCoefficientsThreeInputPathsAndFloat32Rounding) {
  const auto t = n::DType::Float64;
  auto       a = n::tensor({4., 6.}, {2}, t).requires_grad_(),
       b = n::tensor({2., 4.}, {2}, t).requires_grad_(), c = n::scalar(.25, t).requires_grad_();
  const auto out = m::InfoVaeObjective(a, b, c);
  EXPECT_EQ(out.item(), 8.875);
  EXPECT_EQ(m::InfoVaeObjective(a, b, c, 1).item(), 7.5);
  EXPECT_EQ(m::InfoVaeObjective(a, b, c, 0, 1).item(), 8);
  const auto g = n::grad(out, {a, b, c});
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>{.5, .5}));
  EXPECT_EQ(g[1].to_vector<double>(), (std::vector<double>{.25, .25}));
  EXPECT_EQ(g[2].item(), 9.5);
  EXPECT_EQ(m::InfoVaeObjective(a, b, -c).item(), 4.125);
  n::TensorFunction f = [](const auto& v) { return m::InfoVaeObjective(v[0], v[1], v[2]); };
  EXPECT_TRUE(n::gradcheck(f, {a, b, c}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {a, b, c}).ok);
  float alpha = .234567891f, lambda = 1.19876543f;
  auto  kl = n::ones({2}).requires_grad_(), mm = n::scalar(.3).requires_grad_();
  auto  gradients = n::grad(m::InfoVaeObjective(n::zeros({2}), kl, mm, alpha, lambda), {kl, mm});
  EXPECT_EQ(gradients[0].at({0}), float((1.f - alpha) * .5f));
  EXPECT_EQ(gradients[1].item(), float((alpha + lambda) - 1.f));
}
TEST(NnInfoVae, EveryTinyParameterImageNoiseAndIndependentPriorFirstSecondDerivatives) {
  const auto t = n::DType::Float64;
  for (double alpha : {.5, 1.}) {
    n::ParameterStore p(3901, t);
    auto              x = n::tensor({.2, .4, .3, .7, .5, .6, .3, .6, .2, .4, .7, .5}, {2, 6}, t),
         epsilon        = n::tensor({.2, -.3, .4, .1}, {2, 2}, t),
         prior          = n::tensor({.5, -.2, .1, .7, -.3, .2}, {3, 2}, t);
    auto value          = [&]() {
      return m::InfoVariationalAutoencoder(p, x, epsilon, prior, alpha, 10, 6, 2, 3).loss;
    };
    value();
    auto                gradients = n::grad(value(), p.parameters());
    std::vector<double> analytic;
    for (const auto& g : gradients) {
      auto v = g.to_vector<double>();
      analytic.insert(analytic.end(), v.begin(), v.end());
    }
    EXPECT_EQ(p.size(), 10);
    EXPECT_EQ(p.count(), 70);
    auto original = p.flatten();
    for (std::size_t j = 0; j < original.size(); ++j) {
      auto a = original;
      a[j] += 1e-6;
      p.unflatten(a);
      double plus = value().item();
      a[j] -= 2e-6;
      p.unflatten(a);
      double minus = value().item();
      EXPECT_NEAR((plus - minus) / 2e-6, analytic[j], 2e-8);
    }
    p.unflatten(original);
    auto image = x.clone().requires_grad_(), noise = epsilon.clone().requires_grad_(),
         reference      = prior.clone().requires_grad_();
    n::TensorFunction f = [&](const auto& a) {
      return m::InfoVariationalAutoencoder(p, a[0], a[1], a[2], alpha, 10, 6, 2, 3).loss;
    };
    EXPECT_TRUE(n::gradcheck(f, {image, noise, reference}, {.atol = 1e-8, .rtol = 1e-5}).ok);
    EXPECT_TRUE(n::gradgradcheck(f, {image, noise, reference}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  }
}
TEST(NnInfoVae, PriorAndNoiseAffectOnlyApplicablePathsAndUnequalBatchReconstructions) {
  n::ParameterStore p(3901);
  auto              x = n::full({3, 784}, .3), e = n::ones({3, 4}), pr = n::zeros({4, 4});
  auto              a = m::InfoVariationalAutoencoder(p, x, e, pr),
       b              = m::InfoVariationalAutoencoder(p, x, e, pr + 2),
       c              = m::InfoVariationalAutoencoder(p, x, e * .2, pr);
  EXPECT_EQ(p.count(), 26104);
  EXPECT_EQ(p.size(), 10);
  EXPECT_EQ(a.probability.to_vector<float>(), b.probability.to_vector<float>());
  EXPECT_NE(a.mmd.item(), b.mmd.item());
  EXPECT_NE(a.loss.item(), b.loss.item());
  EXPECT_EQ(a.mu.to_vector<float>(), c.mu.to_vector<float>());
  EXPECT_NE(a.z.to_vector<float>(), c.z.to_vector<float>());
  EXPECT_NE(a.probability.to_vector<float>(), c.probability.to_vector<float>());
  auto small = m::InfoVariationalAutoencoder(
      p, n::take_rows(x, {0, 1}), n::take_rows(e, {0, 1}), n::ones({5, 4}));
  EXPECT_NEAR(
      n::max(n::abs(small.probability - n::take_rows(a.probability, {0, 1}))).item(), 0, 2e-6);
  EXPECT_EQ(small.mmd_xy.shape(), (n::Shape{2, 5}));
  EXPECT_EQ(
      a.probability.to_vector<float>(), n::sigmoid(m::InfoVaeDecode(p, a.z)).to_vector<float>());
}
TEST(NnInfoVae, OwningMmdAndForwardEndpointsCodeOnlyDecoderAndVisibleAdam) {
  n::Tensor xx, xy, within, value, probability, z, loss;
  {
    auto out = m::ImqMmd(n::zeros({2, 2}), n::ones({3, 2}));
    xx       = out.xx;
    xy       = out.xy;
    within   = out.within_x;
    value    = out.value;
  }
  EXPECT_EQ(xx.shape(), (n::Shape{2, 2}));
  EXPECT_EQ(xy.shape(), (n::Shape{2, 3}));
  EXPECT_EQ(within.item(), 1);
  EXPECT_GT(value.item(), 0);
  n::ParameterStore p(3901);
  auto              x = n::full({2, 784}, .3), e = n::ones({2, 4}), pr = n::zeros({3, 4});
  {
    auto out    = m::InfoVariationalAutoencoder(p, x, e, pr);
    probability = out.probability;
    z           = out.z;
    loss        = out.loss;
  }
  const auto before = p.flatten();
  EXPECT_EQ(probability.to_vector<float>(), n::sigmoid(m::InfoVaeDecode(p, z)).to_vector<float>());
  EXPECT_EQ(p.flatten(), before);
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer optimizer(p, .003, config);
  optimizer.step(n::grad(loss, p.parameters()));
  EXPECT_EQ(optimizer.iterations(), 1);
  EXPECT_NE(p.flatten(), before);
  n::ParameterStore decoder(1);
  m::InfoVaeDecode(decoder, n::zeros({1, 4}));
  EXPECT_EQ(decoder.size(), 4);
  EXPECT_EQ(decoder.count(), 13408);
}
TEST(NnInfoVae, AlphaZeroLambdaOneHasExactElboAndZeroPriorGradient) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(3901, t);
  auto              pr  = n::full({3, 2}, .4, t).requires_grad_();
  auto              out = m::InfoVariationalAutoencoder(
      p, n::full({2, 6}, .3, t), n::ones({2, 2}, t), pr, 0, 1, 6, 2, 3);
  EXPECT_EQ(out.loss.item(), out.negative_elbo.item());
  auto g = n::grad(out.loss, {pr}, {.retain_graph=true});
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>(6, 0)));
  auto mmd = n::grad(out.mmd, {pr});
  EXPECT_GT(n::max(n::abs(mmd[0])).item(), 0);
}
TEST(NnInfoVae, InvalidInputsAndProductsFailBeforeLeavesAndStridesAreAccepted) {
  n::ParameterStore p(1);
  auto              x = n::full({2, 6}, .3), e = n::zeros({2, 2}), pr = n::zeros({3, 2});
  for (auto controls : std::vector<std::pair<double, double>>{
           {-.1, 10}, {1.1, 10}, {.5, .9}, {NAN, 10}, {.5, INFINITY}, {1e-100, 10}, {.5, 1e100}})
    EXPECT_THROW(
        m::InfoVariationalAutoencoder(p, x, e, pr, controls.first, controls.second, 6, 2, 3),
        std::invalid_argument);
  EXPECT_THROW(
      m::InfoVariationalAutoencoder(p, n::full({2, 6}, 1.1), e, pr, .5, 10, 6, 2, 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::InfoVariationalAutoencoder(p, x, n::ones({3, 2}), pr, .5, 10, 6, 2, 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::InfoVariationalAutoencoder(p, x, e, n::ones({1, 2}), .5, 10, 6, 2, 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::InfoVariationalAutoencoder(p, x, e, n::ones({3, 2}, n::DType::Float64), .5, 10, 6, 2, 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::InfoVariationalAutoencoder(
          p, x, e, pr, .5, 10, 6, 2, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  EXPECT_THROW(m::ImqMmd(n::ones({1, 2}), pr), std::invalid_argument);
  EXPECT_THROW(m::ImqKernel(e, n::full({2, 2}, INFINITY)), std::invalid_argument);
  EXPECT_THROW(
      m::InfoVaeObjective(n::ones({2}), n::ones({3}), n::scalar(0)), std::invalid_argument);
  auto sx = n::transpose(n::transpose(x).contiguous()),
       se = n::transpose(n::transpose(e).contiguous()),
       sp = n::transpose(n::transpose(pr).contiguous());
  EXPECT_FALSE(sx.is_contiguous());
  EXPECT_EQ(
      m::InfoVariationalAutoencoder(p, sx, se, sp, .5, 10, 6, 2, 3).probability.to_vector<float>(),
      m::InfoVariationalAutoencoder(p, x, e, pr, .5, 10, 6, 2, 3).probability.to_vector<float>());
}
