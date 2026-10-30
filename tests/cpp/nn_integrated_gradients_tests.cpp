
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_ig_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/integrated_gradients.cpp"
#undef main
#include "../fixtures/nn_integrated_reference.hpp"
TEST(IntegratedGradients,
     EveryAnalyticSaturationDummyImplementationInvariantAndBatchingControl) {
  AxiomChecks();
}
TEST(IntegratedGradients,
     EveryActualPathDerivativeNodeAndAttributionMatchesIndependentHostArithmetic) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = (n::sin(n::arange(0, 30) * .37) * .4 + .5).reshape({3, 2, 5}),
             base = n::full({3, 2, 5}, .2);
  const auto classes = n::zeros({3}, n::DType::Int32);
  auto gradient = [](const n::Tensor &p, const n::Tensor &) {
    return p * p * 3. + .1;
  };
  for (int steps : {1, 7, 32})
    for (int batch : {1, 5, 32}) {
      const auto a = m::IntegratedGradients(x, base, classes, steps, batch, gradient);
      const auto b =
          ig_reference::IntegratedGradients(x, base, classes, steps, batch, gradient);
      EXPECT_EQ(a.nodes.to_vector<double>(), b.nodes.to_vector<double>());
      EXPECT_EQ(a.path_gradient.to_vector<float>(), b.path_gradient.to_vector<float>());
      EXPECT_EQ(a.mean_gradient.to_vector<float>(), b.mean_gradient.to_vector<float>());
      EXPECT_EQ(a.attribution.to_vector<float>(), b.attribution.to_vector<float>());
    }
}
TEST(IntegratedGradients, InvalidBaselineTargetQuadratureAndDerivativeContracts) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = n::ones({2, 3}), t = n::zeros({2}, n::DType::Int32);
  auto f = [](const n::Tensor &p, const n::Tensor &) {
    return n::ones(p.shape());
  };
  EXPECT_THROW(m::IntegratedGradients(x, x, t, 0, 2, f), std::invalid_argument);
  EXPECT_THROW(m::IntegratedGradients(x, n::ones({2, 2}), t, 2, 2, f), std::invalid_argument);
  EXPECT_THROW(m::IntegratedGradients(x, x, t, 2, 2,
                                      [](const n::Tensor &, const n::Tensor &) {
                                        return n::full({2, 3}, NAN);
                                      }),
               std::invalid_argument);
}
