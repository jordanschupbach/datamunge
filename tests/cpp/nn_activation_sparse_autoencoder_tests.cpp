#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>

namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnActivationSparsity, OriginalFloatFixtureMeansAndAllTwelvePartials) {
  const std::vector<float> values{0, .05f, .2f, 1, 0, .05f, .6f, 1, 0, .05f, 1, 1};
  const auto               h  = n::from_vector(values, {3, 4}).requires_grad_();
  const auto               p  = m::ActivationSparsity(h, .1f, .2f);
  const auto               g  = n::grad(p.total, {h})[0].to_vector<float>();
  double                   l1 = 0, kl = 0;
  for (float v : values)
    l1 += std::abs(double(v)) / 12;
  for (int j = 0; j < 4; ++j) {
    double q = 0;
    for (int b = 0; b < 3; ++b)
      q += values[b * 4 + j] / 3.;
    const double c = std::clamp(q, double(1e-5f), double(1.f - 1e-5f)), r = .05f, s = 1.f - .05f;
    kl += (r * std::log(r / c) + s * std::log(s / (1 - c))) / 4;
    for (int b = 0; b < 3; ++b) {
      double expected = values[b * 4 + j] == 0 ? 0 : double(.1f) / 12;
      if (q > 1e-5f && q < 1.f - 1e-5f)
        expected += double(.2f) * (-r / q + s / (1 - q)) / 12;
      EXPECT_NEAR(g[b * 4 + j], expected, 3e-7);
    }
  }
  EXPECT_NEAR(p.l1.item(), l1, 4e-8);
  EXPECT_NEAR(p.kl.item(), kl, 4e-6);
  EXPECT_NEAR(p.total.item(), double(.1f) * l1 + double(.2f) * kl, 1e-6);
}

TEST(NnActivationSparsity, DoublePrecisionSignedValuesClampEndpointsAndZeroConventions) {
  const auto type = n::DType::Float64;
  const auto h    = n::tensor({0., .1, .2, 1., 0., .1, .2, 1.}, {2, 4}, type).requires_grad_();
  const auto p    = m::ActivationSparsity(h, .1, .2, .2, .1);
  const auto g    = n::grad(p.total, {h})[0];
  for (int b = 0; b < 2; ++b) {
    EXPECT_EQ(g.at({b, 0}), 0);
    EXPECT_NEAR(g.at({b, 1}), .1 / 8 + .2 * (-.2 / .1 + .8 / .9) / 8, 1e-14);
    EXPECT_NEAR(g.at({b, 2}), .1 / 8, 1e-14);
    EXPECT_NEAR(g.at({b, 3}), .1 / 8, 1e-14);
  }
  const auto high = n::full({1, 1}, .9, type).requires_grad_();
  EXPECT_NEAR(
      n::grad(m::ActivationSparsity(high, 0., 1., .2, .1).kl, {high})[0].item(),
      -.2 / .9 + .8 / .1,
      1e-12);
  const auto signed_code    = n::tensor({-.2, .3}, {1, 2}, type).requires_grad_();
  const auto signed_penalty = m::ActivationSparsity(signed_code, .1, 0);
  EXPECT_EQ(
      n::grad(signed_penalty.total, {signed_code})[0].to_vector<double>(),
      (std::vector<double>{-.05, .05}));
  const auto at_target = m::ActivationSparsity(n::full({3, 4}, .2, type), 0, 1, .2);
  EXPECT_NEAR(at_target.kl.item(), 0, 1e-15);
}

TEST(NnActivationSparsity, BatchAggregationChangesKlButDoesNotEnforcePerExampleSparsity) {
  const auto type   = n::DType::Float64;
  const auto h      = n::tensor({.01, .99, .99, .01}, {2, 2}, type);
  const auto all    = m::ActivationSparsity(h, .1, .2, .2);
  const auto first  = m::ActivationSparsity(n::take_rows(h, {0}), .1, .2, .2);
  const auto second = m::ActivationSparsity(n::take_rows(h, {1}), .1, .2, .2);
  EXPECT_LT(all.kl.item(), (first.kl.item() + second.kl.item()) / 2);
  EXPECT_EQ(all.l1.item(), (first.l1.item() + second.l1.item()) / 2);
  const auto constant = m::ActivationSparsity(n::full({2, 2}, .5, type), .1, .2, .2);
  EXPECT_EQ(all.kl.item(), constant.kl.item());
  EXPECT_EQ(all.l1.item(), constant.l1.item());
}

TEST(NnActivationSparsity, FullModelAllParametersInputsAndSecondDerivatives) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(1601, type);
  const auto        x   = n::tensor({.1, .6, .8, .3, .2, .7}, {2, 3}, type).requires_grad_();
  const auto        net = m::SparseAutoencoder(p, x, .1, .2, 3, 2, .2);
  EXPECT_EQ(p.count(), 17);
  EXPECT_EQ(p.size(), 4u);
  EXPECT_NEAR(
      net.loss.item(),
      net.mse.item() + .1 * net.penalty.l1.item() + .2 * net.penalty.kl.item(),
      1e-14);
  const auto          gradients = n::grad(net.loss, p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gradients) {
    const auto v = g.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  const auto initial = p.flatten();
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto altered = initial;
    altered[i] += 1e-6;
    p.unflatten(altered);
    const double plus = m::SparseAutoencoder(p, x, .1, .2, 3, 2, .2).loss.item();
    altered[i] -= 2e-6;
    p.unflatten(altered);
    const double minus = m::SparseAutoencoder(p, x, .1, .2, 3, 2, .2).loss.item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(initial);
  const n::TensorFunction objective = [&](const auto& args) {
    return m::SparseAutoencoder(p, args[0], .1, .2, 3, 2, .2).loss;
  };
  EXPECT_TRUE(n::gradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  const auto     before = p.flatten();
  n::NoGradGuard guard;
  EXPECT_EQ(
      n::dense(p, "decoder", net.code, 3, n::Activation::Sigmoid).to_vector<double>(),
      net.reconstruction.to_vector<double>());
  EXPECT_EQ(before, p.flatten());
  n::ParameterStore original(1601);
  m::SparseAutoencoder(original, n::zeros({1, 784}), .05, .05);
  EXPECT_EQ(original.count(), 101200);
}

TEST(NnActivationSparsity, RejectsInvalidInputsOptionsAndRoundedEndpointsBeforeCreation) {
  n::ParameterStore p(1);
  const auto        image = n::zeros({1, 3});
  for (double weight :
       {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(m::SparseAutoencoder(p, image, weight, .1, 3, 2), std::invalid_argument);
    EXPECT_THROW(m::SparseAutoencoder(p, image, .1, weight, 3, 2), std::invalid_argument);
  }
  EXPECT_THROW(m::SparseAutoencoder(p, image, .1, .2, 3, 2, 1.), std::invalid_argument);
  EXPECT_THROW(m::SparseAutoencoder(p, image, .1, .2, 3, 2, .2, 1e-100), std::invalid_argument);
  EXPECT_THROW(m::SparseAutoencoder(p, image, .1, .2, 3, 2, 1. - 1e-12), std::invalid_argument);
  EXPECT_THROW(
      m::SparseAutoencoder(p, n::zeros({1, 3}, n::DType::Float64), .1, .2, 3, 2),
      std::invalid_argument);
  EXPECT_THROW(m::ActivationSparsity(n::zeros({0, 3}), .1, .2), std::invalid_argument);
  EXPECT_THROW(
      m::ActivationSparsity(n::full({1, 3}, std::numeric_limits<double>::quiet_NaN()), .1, .2),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0u);
}
