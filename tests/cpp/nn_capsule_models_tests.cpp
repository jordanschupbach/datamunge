
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnCapsuleModels, SquashLengthsAndMarginHaveExpectedRegularizedValuesAndDerivatives) {
  const auto type = n::DType::Float64;
  const auto zero = n::zeros({2, 3, 4}, type).requires_grad_();
  const auto squashed = n::models::Squash(zero);
  EXPECT_EQ(n::max(n::abs(squashed)).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::grad(n::sum(squashed), {zero})[0])).item(), 0);
  EXPECT_NEAR(n::models::CapsuleLengths(zero).at({0, 0}), std::sqrt(1e-9), 1e-15);
  const auto lengths =
                 n::from_vector(std::vector<double>{.8, .2, .05}, {1, 3}).requires_grad_(),
             target = n::from_vector(std::vector<double>{1, 0, 0}, {1, 3});
  const auto margin = n::models::MarginLoss(lengths, target);
  EXPECT_NEAR(margin.item(), .015, 1e-15);
  const auto gradient = n::grad(margin, {lengths})[0];
  EXPECT_NEAR(gradient.at({0, 0}), -.2, 1e-15);
  EXPECT_NEAR(gradient.at({0, 1}), .1, 1e-15);
  EXPECT_EQ(gradient.at({0, 2}), 0);
}
TEST(NnCapsuleModels, ThreeRoutingIterationsMatchIndependentStableHostOracle) {
  const auto type = n::DType::Float64;
  const auto u = (1.3 * n::sin(.37 * n::arange(0, 12).to(type) + .2)).reshape({1, 6, 2}),
             weights = n::sin(.37 * n::arange(0, 108).to(type) + .7).reshape({6, 9, 2});
  const auto actual = n::models::DynamicRouting(u, weights, 6, 3, 2, 3, 3);
  const auto input = u.to_vector<double>(), w = weights.to_vector<double>();
  std::vector<double> votes(6 * 3 * 3), logits(6 * 3), couplings(6 * 3), capsules(3 * 3);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 3; ++j)
      for (int d = 0; d < 3; ++d)
        for (int k = 0; k < 2; ++k)
          votes[(i * 3 + j) * 3 + d] += w[(i * 9 + j * 3 + d) * 2 + k] * input[i * 2 + k];
  for (int t = 0; t < 3; ++t) {
    for (int i = 0; i < 6; ++i) {
      double high = -std::numeric_limits<double>::infinity(), total = 0;
      for (int j = 0; j < 3; ++j)
        high = std::max(high, logits[i * 3 + j]);
      for (int j = 0; j < 3; ++j) {
        couplings[i * 3 + j] = std::exp(logits[i * 3 + j] - high);
        total += couplings[i * 3 + j];
      }
      for (int j = 0; j < 3; ++j)
        couplings[i * 3 + j] /= total;
    }
    for (int i = 0; i < 18; ++i)
      EXPECT_NEAR(actual.history[t].reshape({18}).at({i}), couplings[i], 1e-12);
    std::fill(capsules.begin(), capsules.end(), 0);
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 3; ++j)
        for (int d = 0; d < 3; ++d)
          capsules[j * 3 + d] += couplings[i * 3 + j] * votes[(i * 3 + j) * 3 + d];
    for (int j = 0; j < 3; ++j) {
      double norm2 = 0;
      for (int d = 0; d < 3; ++d)
        norm2 += capsules[j * 3 + d] * capsules[j * 3 + d];
      const double scale = norm2 / (1 + norm2) / std::sqrt(norm2 + 1e-9);
      for (int d = 0; d < 3; ++d)
        capsules[j * 3 + d] *= scale;
    }
    if (t < 2)
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 3; ++j)
          for (int d = 0; d < 3; ++d)
            logits[i * 3 + j] += votes[(i * 3 + j) * 3 + d] * capsules[j * 3 + d];
  }
  for (int i = 0; i < 9; ++i)
    EXPECT_NEAR(actual.capsules.reshape({9}).at({i}), capsules[i], 1e-12);
  EXPECT_EQ(actual.history.size(), 3);
  EXPECT_LT(n::max(n::abs(n::sum(actual.couplings, {2}) - 1)).item(), 1e-12);
}
TEST(NnCapsuleModels, RoutingDifferentiatesThroughAgreementIterations) {
  const auto type = n::DType::Float64;
  const auto u = (.7 * n::sin(.37 * n::arange(0, 12).to(type) + .2))
                     .reshape({1, 6, 2})
                     .detach()
                     .requires_grad_(),
             weights = (.6 * n::sin(.31 * n::arange(0, 108).to(type) + .7))
                           .reshape({6, 9, 2})
                           .detach()
                           .requires_grad_();
  const auto out = n::models::DynamicRouting(u, weights, 6, 3, 2, 3, 3);
  const auto gradients = n::grad(n::sum(n::square(out.capsules)), {u, weights});
  n::NoGradGuard guard;
  const double h = 1e-6;
  for (int variable = 0; variable < 2; ++variable) {
    const auto base = variable ? weights : u;
    const auto direction =
        n::one_hot(n::tensor({0}).to(n::DType::Int64), base.numel(), 1, 0, type)
            .reshape(base.shape());
    auto loss = [&](double sign) {
      const auto result = n::models::DynamicRouting(
          variable ? u : u + sign * h * direction,
          variable ? weights + sign * h * direction : weights, 6, 3, 2, 3, 3);
      return n::sum(n::square(result.capsules)).item();
    };
    EXPECT_NEAR(gradients[variable].reshape({base.numel()}).at({0}),
                (loss(1) - loss(-1)) / (2 * h), 1e-7);
  }
}
TEST(NnCapsuleModels, CompleteCountsDecoderOwnershipAndEveryParameterGradient) {
  n::set_num_threads(2);
  const auto x = (.5 + .5 * n::sin(.37 * n::arange(0, 2 * 28 * 28).to(n::DType::Float32)))
                     .reshape({2, 28, 28, 1});
  for (bool decoder : {false, true}) {
    n::ParameterStore p(17);
    const auto out = n::models::CapsNet(p, x, 3, decoder);
    EXPECT_EQ(p.count(), decoder ? 8215568 : 6804224);
    EXPECT_EQ(out.primary.shape(), n::Shape({2, 1152, 8}));
    EXPECT_EQ(out.digits.shape(), n::Shape({2, 10, 16}));
    EXPECT_EQ(bool(out.reconstruct), decoder);
    const auto mask = n::one_hot(n::tensor({1, 2}).to(n::DType::Int64), 10);
    auto objective = n::models::MarginLoss(out.lengths, mask);
    if (decoder) {
      const auto reconstructed = out.reconstruct(mask);
      EXPECT_EQ(reconstructed.shape(), n::Shape({2, 784}));
      EXPECT_GE(n::min(reconstructed).item(), 0);
      EXPECT_LE(n::max(reconstructed).item(), 1);
      objective = objective +
                  .0005 * n::mean(n::sum(n::square(reconstructed - x.reshape({2, 784})), {1}));
      EXPECT_THROW(out.reconstruct(n::ones({3, 10})), std::invalid_argument);
    }
    for (const auto &g : n::grad(objective, p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  }
}
TEST(NnCapsuleModels, RejectsInvalidRoutingShapesDtypesAndMarginDomains) {
  EXPECT_THROW(n::models::Squash(n::ones({2, 3}), 0), std::invalid_argument);
  EXPECT_THROW(
      n::models::DynamicRouting(n::ones({1, 6, 2}), n::ones({6, 9, 2}), 6, 3, 2, 3, 0),
      std::invalid_argument);
  EXPECT_THROW(
      n::models::DynamicRouting(n::ones({1, 5, 2}), n::ones({6, 9, 2}), 6, 3, 2, 3, 3),
      std::invalid_argument);
  EXPECT_THROW(n::models::MarginLoss(n::ones({1, 3}), n::ones({1, 3}), .1, .2),
               std::invalid_argument);
  n::ParameterStore p(17);
  EXPECT_THROW(n::models::CapsNet(p, n::ones({1, 32, 32, 1})), std::invalid_argument);
}
