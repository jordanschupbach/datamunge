#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>

namespace n = datamunge::nn;

TEST(NnCompetitiveAutoencoders, SignedTiesMasksAndEveryWinnerDerivative) {
  const auto x = n::tensor(
                     {3., 1., 1., 0., 2., 4., 0., 0., -1., -2., 5., 5., -4., -1., -2., -3.},
                     {4, 4},
                     n::DType::Float64)
                     .requires_grad_();
  const auto                row      = n::models::CompetitiveMask(x, false, 2);
  const auto                lifetime = n::models::CompetitiveMask(x, true);
  const std::vector<double> r{1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0};
  const std::vector<double> l{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0};
  EXPECT_EQ(row.to_vector<double>(), r);
  EXPECT_EQ(lifetime.to_vector<double>(), l);
  EXPECT_FALSE(row.requires_grad());
  EXPECT_EQ(n::grad(n::sum(x * row), {x})[0].to_vector<double>(), r);
  EXPECT_EQ(n::grad(n::sum(x * lifetime), {x})[0].to_vector<double>(), l);
  const auto strided = n::transpose(n::transpose(x).contiguous());
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_EQ(n::models::CompetitiveMask(strided, false, 2).to_vector<double>(), r);
  EXPECT_EQ(n::models::CompetitiveMask(strided, true).to_vector<double>(), l);
  EXPECT_EQ(n::sum(n::models::CompetitiveMask(x, false, 0)).item(), 0);
  EXPECT_EQ(n::sum(n::models::CompetitiveMask(x, false, 20)).item(), 16);
  EXPECT_EQ(
      n::models::CompetitiveMask(n::zeros({1, 4}), false, 2).to_vector<float>(),
      (std::vector<float>{1, 1, 0, 0}));
}

TEST(NnCompetitiveAutoencoders, LifetimePartialBatchesAndEmptyAxes) {
  for (int batch : {1, 7, 8, 9, 15, 16, 17, 32}) {
    std::vector<double> values;
    for (int row = 0; row < batch; ++row) {
      values.push_back(row);
      values.push_back(-row);
    }
    const auto mask =
        n::models::CompetitiveMask(n::from_vector(values, {batch, 2}), true).to_vector<double>();
    const int kept = (batch + 7) / 8;
    for (int row = 0; row < batch; ++row) {
      EXPECT_EQ(mask[2 * row], row >= batch - kept ? 1 : 0);
      EXPECT_EQ(mask[2 * row + 1], row < kept ? 1 : 0);
    }
  }
  for (const auto& shape : std::vector<n::Shape>{{0, 4}, {3, 0}, {0, 0}})
    for (bool lifetime : {false, true}) {
      const auto mask = n::models::CompetitiveMask(n::zeros(shape), lifetime);
      EXPECT_EQ(mask.shape(), shape);
      EXPECT_EQ(mask.numel(), 0);
      EXPECT_FALSE(mask.requires_grad());
    }
}

TEST(NnCompetitiveAutoencoders, TrainingAndDeploymentPathsAndOriginalParameterLayout) {
  n::ParameterStore p(1931, n::DType::Float64);
  const auto        x = n::tensor({.1, .2, .8, .7, .3, .4, .2, .8, .6}, {3, 3}, n::DType::Float64);
  auto              net = n::models::CompetitiveAutoencoder(p, x, false, false, 3, 4, 2);
  EXPECT_EQ(p.count(), 31);
  EXPECT_EQ(
      p.names(), (std::vector<std::string>{"encoder/W", "encoder/b", "decoder/W", "decoder/b"}));
  const auto before = p.flatten();
  EXPECT_EQ(net.code.to_vector<double>(), net.training_code.to_vector<double>());
  EXPECT_EQ(
      net.reconstruction.to_vector<double>(), net.training_reconstruction.to_vector<double>());
  EXPECT_EQ(n::sum(net.mask).item(), 6);
  const auto plain = n::models::CompetitiveAutoencoder(p, x, false, true, 3, 4, 2);
  EXPECT_EQ(plain.raw.to_vector<double>(), net.raw.to_vector<double>());
  EXPECT_EQ(plain.code.to_vector<double>(), plain.raw.to_vector<double>());
  EXPECT_EQ(n::sum(plain.mask).item(), 12);
  p.set("encoder/W", n::full({3, 4}, .2, n::DType::Float64));
  p.set("encoder/b", n::full({4}, .1, n::DType::Float64));
  const auto wta = n::models::CompetitiveAutoencoder(p, x, true, false, 3, 4);
  EXPECT_EQ(wta.code.to_vector<double>(), wta.raw.to_vector<double>());
  EXPECT_EQ(n::sum(wta.mask).item(), 4);
  EXPECT_GT(n::max(n::abs(wta.training_code - wta.code)).item(), 0);
  const auto wta_plain = n::models::CompetitiveAutoencoder(p, x, true, true, 3, 4);
  EXPECT_EQ(
      wta_plain.training_reconstruction.to_vector<double>(),
      wta_plain.reconstruction.to_vector<double>());
  EXPECT_EQ(p.count(), 31);
  n::ParameterStore original(1931);
  n::models::CompetitiveAutoencoder(original, n::zeros({1, 784}), false);
  EXPECT_EQ(original.count(), 101200);
  EXPECT_EQ(original.size(), 4u);
}

TEST(NnCompetitiveAutoencoders, EveryParameterAndInputDerivativeOnFixedSupport) {
  const auto          type = n::DType::Float64;
  n::ParameterStore   p(1931, type);
  auto                x   = n::tensor({.1, .2, .8, .7, .3, .4}, {2, 3}, type).requires_grad_();
  const auto          net = n::models::CompetitiveAutoencoder(p, x, false, false, 3, 4, 2);
  const auto          gradients = n::grad(n::mse(net.training_reconstruction, x), p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gradients) {
    const auto v = g.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  const auto initial = p.flatten();
  const auto score   = [&]() {
    return n::mse(
               n::models::CompetitiveAutoencoder(p, x, false, false, 3, 4, 2)
                   .training_reconstruction,
               x)
        .item();
  };
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto altered = initial;
    altered[i] += 1e-6;
    p.unflatten(altered);
    const double plus = score();
    altered[i] -= 2e-6;
    p.unflatten(altered);
    const double minus = score();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(initial);
  const n::TensorFunction objective = [&](const auto& args) {
    return n::mse(
        n::models::CompetitiveAutoencoder(p, args[0], false, false, 3, 4, 2)
            .training_reconstruction,
        args[0]);
  };
  EXPECT_TRUE(n::gradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}

TEST(NnCompetitiveAutoencoders, RejectsMalformedSelectorsAndModelsBeforeCreatingParameters) {
  EXPECT_THROW(n::models::CompetitiveMask(n::Tensor{}, false), std::invalid_argument);
  EXPECT_THROW(n::models::CompetitiveMask(n::zeros({3}), false), std::invalid_argument);
  EXPECT_THROW(
      n::models::CompetitiveMask(n::zeros({2, 3}, n::DType::Int32), false), std::invalid_argument);
  EXPECT_THROW(n::models::CompetitiveMask(n::zeros({2, 3}), false, -1), std::invalid_argument);
  for (double bad :
       {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(n::models::CompetitiveMask(n::full({2, 3}, bad), true), std::invalid_argument);
  n::ParameterStore p(1);
  EXPECT_THROW(n::models::CompetitiveAutoencoder(p, n::Tensor{}, false), std::invalid_argument);
  EXPECT_THROW(
      n::models::CompetitiveAutoencoder(p, n::zeros({0, 3}), false, false, 3, 4, 2),
      std::invalid_argument);
  EXPECT_THROW(
      n::models::CompetitiveAutoencoder(
          p, n::zeros({2, 3}, n::DType::Float64), false, false, 3, 4, 2),
      std::invalid_argument);
  EXPECT_THROW(
      n::models::CompetitiveAutoencoder(p, n::zeros({2, 3}), false, false, 3, 0, 2),
      std::invalid_argument);
  EXPECT_THROW(
      n::models::CompetitiveAutoencoder(p, n::zeros({2, 3}), false, false, 3, 4, -1),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0u);
}

TEST(NnCompetitiveAutoencoders, LifetimeAllParameterAndInputDerivativesOnPositiveSeparatedSupport) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(1931, type);
  auto              x =
      n::tensor({.1, .2, .3, .3, .4, .5, .5, .6, .7, .7, .8, .9}, {4, 3}, type).requires_grad_();
  n::models::CompetitiveAutoencoder(p, x, true, false, 3, 4);
  p.set("encoder/W", n::full({3, 4}, .2, type));
  p.set("encoder/b", n::full({4}, .1, type));
  const auto out = n::models::CompetitiveAutoencoder(p, x, true, false, 3, 4);
  EXPECT_EQ(n::sum(out.mask).item(), 4);
  EXPECT_EQ(out.code.to_vector<double>(), out.raw.to_vector<double>());
  const auto          gs = n::grad(n::mse(out.training_reconstruction, x), p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gs) {
    auto v = g.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  const auto initial = p.flatten();
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto changed = initial;
    changed[i] += 1e-6;
    p.unflatten(changed);
    double plus =
        n::mse(
            n::models::CompetitiveAutoencoder(p, x, true, false, 3, 4).training_reconstruction, x)
            .item();
    changed[i] -= 2e-6;
    p.unflatten(changed);
    double minus =
        n::mse(
            n::models::CompetitiveAutoencoder(p, x, true, false, 3, 4).training_reconstruction, x)
            .item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(initial);
  const n::TensorFunction objective = [&](const auto& args) {
    return n::mse(
        n::models::CompetitiveAutoencoder(p, args[0], true, false, 3, 4).training_reconstruction,
        x.detach());
  };
  EXPECT_TRUE(n::gradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}

TEST(NnCompetitiveAutoencoders, LifetimeTrainingDependsOnCompanionsButDeploymentDoesNot) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(1931, type);
  auto              sample = n::ones({1, 3}, type);
  n::models::CompetitiveAutoencoder(p, sample, true, false, 3, 2);
  p.set("encoder/W", n::full({3, 2}, .2, type));
  p.set("encoder/b", n::full({2}, .1, type));
  auto low = n::models::CompetitiveAutoencoder(
      p, n::concat({sample, n::zeros({7, 3}, type)}, 0), true, false, 3, 2);
  auto high = n::models::CompetitiveAutoencoder(
      p, n::concat({sample, n::full({7, 3}, 2, type)}, 0), true, false, 3, 2);
  EXPECT_EQ(n::take_rows(low.mask, {0}).to_vector<double>(), (std::vector<double>{1, 1}));
  EXPECT_EQ(n::take_rows(high.mask, {0}).to_vector<double>(), (std::vector<double>{0, 0}));
  EXPECT_EQ(
      n::take_rows(low.code, {0}).to_vector<double>(),
      n::take_rows(high.code, {0}).to_vector<double>());
  EXPECT_EQ(
      n::take_rows(low.reconstruction, {0}).to_vector<double>(),
      n::take_rows(high.reconstruction, {0}).to_vector<double>());
  EXPECT_GT(
      n::max(
          n::abs(
              n::take_rows(low.training_reconstruction, {0}) -
              n::take_rows(high.training_reconstruction, {0})))
          .item(),
      0);
  auto alone = n::models::CompetitiveAutoencoder(p, sample, true, false, 3, 2);
  EXPECT_EQ(
      alone.reconstruction.to_vector<double>(),
      n::take_rows(low.reconstruction, {0}).to_vector<double>());
  auto zeros = n::zeros({8, 2}, type).requires_grad_();
  auto mask  = n::models::CompetitiveMask(zeros, true);
  EXPECT_EQ(n::sum(mask).item(), 2);
  EXPECT_EQ(n::sum(zeros * mask).item(), 0);
  EXPECT_EQ(n::take_rows(mask, {0}).to_vector<double>(), (std::vector<double>{1, 1}));
  EXPECT_EQ(
      n::grad(n::sum(zeros * mask), {zeros})[0].to_vector<double>(), mask.to_vector<double>());
}
