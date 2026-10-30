
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_tcn_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.9_1d_convolutions/tcn.cpp"
#undef main
TEST(Tcn, IndependentWeightNormEveryCoordinateFiniteDifference) {
  HostChecks();
}
TEST(Tcn, ExactReceptiveFieldInputJacobianFutureZeroAndPrefixes) {
  CausalityChecks();
}
TEST(Tcn, FullAllStagesAndEveryLearnedGradient) {
  nn::set_num_threads(2);
  FullChecks();
}
TEST(Tcn, TensorGlobalNormClippingMatchesLiteralFloatPolicy) {
  namespace n = dm::nn;
  const std::vector<n::Tensor> gradients = {n::tensor({3., 4.}), n::tensor({12.})};
  std::vector<n::Tensor> terms;
  double squared = 0;
  for (const auto &g : gradients) {
    terms.push_back(n::sum(n::square(g)).to(n::DType::Float64));
    for (float x : g.to_vector<float>())
      squared += double(x) * x;
  }
  const auto norm = n::sqrt(n::sum(n::stack(terms, 0))).to(n::DType::Float32);
  const auto scale = n::minimum(n::ones({}), 1. / (norm + 1e-6));
  const float expected = std::min(1.f, 1.f / (float(std::sqrt(squared)) + 1e-6f));
  EXPECT_EQ(scale.item<float>(), expected);
  for (const auto &g : gradients) {
    const auto actual = (g * scale).to_vector<float>(), raw = g.to_vector<float>();
    for (std::size_t i = 0; i < raw.size(); ++i)
      EXPECT_EQ(actual[i], raw[i] * expected);
  }
}
TEST(Tcn, EveryRealPrefixGradientAndFutureIsolationWithSavedNormalization) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  const auto data = m::LoadSeries("build/cnn-datasets/time_series");
  n::ParameterStore p(461), state(463);
  const m::TcnConfig c{8, 6, 3, data.config.mean, data.config.standard_deviation};
  const auto raw = n::slice_axis(data.pools[2].signals, 0, 0, 2).requires_grad_();
  const auto full = m::Tcn(p, state, raw, c);
  const auto g = n::grad(n::sum(n::select(full.sequence, 1, 74)), {raw})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(g, 1, 75, 150))).item(), 0.);
  n::NoGradGuard guard;
  for (int length : {1, 17, 75, 150}) {
    const auto prefix = m::Tcn(p, state, n::slice_axis(raw, 1, 0, length), c);
    EXPECT_LE(n::max(n::abs(prefix.logits - n::select(full.sequence, 1, length - 1))).item(),
              1e-5);
  }
  const auto time = n::arange(0, 150).reshape({1, 150, 1}),
             changed = raw + n::where(n::greater_equal(time, 75.), 3. * n::sin(time * .3), 0.);
  const auto alternate = m::Tcn(p, state, changed, c);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(alternate.sequence - full.sequence, 1, 0, 75))).item(),
            0.);
  EXPECT_GT(
      n::max(n::abs(n::slice_axis(alternate.sequence - full.sequence, 1, 75, 150))).item(),
      1e-5);
}
TEST(Tcn, RejectInvalidDropoutAndOverflowBeforeLeafCreation) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::ParameterStore p(461), state(463);
  EXPECT_THROW(
      m::Tcn(p, state, n::ones({1, 10, 1}), m::TcnConfig{8, 2, 3, 0, 1}, true, .8, nullptr),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  EXPECT_EQ(state.size(), 0);
  m::TcnConfig c;
  c.kernel = std::numeric_limits<int>::max();
  EXPECT_THROW(m::TcnReceptiveField(c), std::invalid_argument);
  EXPECT_THROW(
      m::TcnPrefix(m::SeriesData{n::zeros({1, 5, 1}), n::zeros({1}), n::zeros({1})}, 6),
      std::invalid_argument);
}
