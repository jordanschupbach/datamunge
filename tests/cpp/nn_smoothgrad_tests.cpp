
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_smooth_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/smoothgrad.cpp"
#undef main
#include "../fixtures/nn_smoothgrad_reference.hpp"
TEST(SmoothGrad, EveryLinearQuadraticFiniteBankMeanVarianceAndBatchingControl) {
  EstimatorChecks();
}
TEST(SmoothGrad, EveryGaussianDrawPointDerivativeAndMomentMatchesHostReference) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto input = (n::sin(n::arange(0, 18) * .21) * .7 + .4).reshape({3, 2, 3}),
             targets = n::zeros({3}, n::DType::Int32);
  auto f = [](const n::Tensor &p, const n::Tensor &) {
    return p * p * 3. - .4;
  };
  for (int samples : {1, 7, 32}) {
    const auto bank = m::SmoothGradNoise(input.shape(), samples, 1171),
               host = ae::GaussianNoise(input.shape(), samples, 1171);
    EXPECT_LE(n::max(n::abs(bank - host)).item(), 1e-14);
    for (bool clipped : {false, true})
      for (int batch : {1, 11}) {
        const auto a = m::SmoothGrad(input, targets, bank, .15f, clipped, batch, f);
        const auto b =
            smooth_reference::SmoothGrad(input, targets, bank, .15f, clipped, batch, f);
        const std::vector<n::Tensor> actual{
            a.mean,           a.mean_absolute, a.mean_square,    a.sample_variance,
            a.standard_error, a.noisy_input,   a.sample_gradient},
            expected{b.mean,           b.mean_absolute, b.mean_square,    b.sample_variance,
                     b.standard_error, b.noisy_input,   b.sample_gradient};
        for (int i = 0; i < 7; ++i)
          EXPECT_LE(n::max(n::abs(actual[i] - expected[i])).item(), 1e-7);
      }
  }
}
TEST(SmoothGrad, InvalidNoiseDtypeShapeSigmaAndCallbackContracts) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = n::ones({2, 3}), t = n::zeros({2}, n::DType::Int32),
             noise = m::SmoothGradNoise(x.shape(), 8, 1);
  auto f = [](const n::Tensor &p, const n::Tensor &) {
    return n::ones(p.shape());
  };
  EXPECT_THROW(m::SmoothGradNoise({2, 0}, 8, 1), std::invalid_argument);
  EXPECT_THROW(m::SmoothGrad(x, t, noise, -.1f, false, 4, f), std::invalid_argument);
  EXPECT_THROW(m::SmoothGrad(x, t, noise, NAN, false, 4, f), std::invalid_argument);
  EXPECT_THROW(m::SmoothGrad(x, t, noise, .1f, false, 4,
                             [](const n::Tensor &p, const n::Tensor &) {
                               return n::full(p.shape(), NAN);
                             }),
               std::invalid_argument);
}
