
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_i3d_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/i3d.cpp"
#undef main
TEST(I3d, Every196InflationKernelInputDerivativeAndStaticTemporalBorders) {
  InflationChecks();
}
TEST(I3d, Every14SonnetPopulationUnitDerivativeAndMovingMomentOracle) {
  UnitChecks();
}
TEST(I3d, ActualFull64Frame224pxAndCompactEveryEndpointAndLearnedGradient) {
  nn::set_num_threads(2);
  GraphChecks(true);
}
TEST(I3d, Every57ActualInitializedKernelHasEqualInflatedTemporalSlices) {
  namespace m = dm::nn::models;
  nn::ParameterStore p(677), s(679);
  nn::NoGradGuard guard;
  m::I3d(p, s, nn::zeros({1, 16, 32, 32, 3}), {8, 3, 16, 32});
  int units = 0;
  for (const auto &name : p.names())
    if (name.ends_with("/W") && name != "Logits/W") {
      const auto w = p.at(name);
      EXPECT_EQ(nn::max(nn::abs(w - nn::expand_dims(nn::select(w, 0, 0), 0))).item(), 0);
      ++units;
    }
  EXPECT_EQ(units, 57);
  EXPECT_EQ(p.size(), 116);
  EXPECT_EQ(s.size(), 114);
}
TEST(I3d, PopulationMovingVarianceAndExplicitAllStateMutationPolicy) {
  namespace m = dm::nn::models;
  nn::ParameterStore p(1), s(2);
  const auto x = nn::arange(1, 5).reshape({2, 1, 1, 2, 1});
  m::I3dUnit(p, s, "u", x, 1, 1, 1, false, .5, false);
  {
    nn::NoGradGuard guard;
    p.set("u/W", nn::ones({1, 1, 1, 1, 1}));
  }
  const auto y = m::I3dUnit(p, s, "u", x, 1, 1, 1, true, .5, true);
  EXPECT_NEAR(s.at("u/mean").item(), 1.25, 1e-7);
  EXPECT_NEAR(s.at("u/variance").item(), 1.125, 1e-7);
  const auto raw = y.to_vector<double>();
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(raw[i], std::max(0., (i + 1 - 2.5) / std::sqrt(1.25 + .001)), 2e-7);
  nn::ParameterStore model(677), state(679);
  nn::Generator gen(673);
  const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
  const auto input = nn::slice_axis(pools[0].rgb, 0, 0, 2);
  const m::I3dConfig c{8, 3, 16, 32};
  m::I3d(model, state, input, c);
  const auto initial = state.flatten();
  m::I3d(model, state, input, c, true, false, &gen);
  EXPECT_EQ(state.flatten(), initial);
  m::I3d(model, state, input, c, true, true, &gen);
  EXPECT_NE(state.flatten(), initial);
  const auto saved = state.flatten(), learned = model.flatten();
  const auto z = m::I3d(model, state, input, c).logits;
  EXPECT_EQ(nn::max(nn::abs(z - m::I3d(model, state, input, c).logits)).item(), 0);
  EXPECT_EQ(state.flatten(), saved);
  EXPECT_EQ(model.flatten(), learned);
}
TEST(I3d, InvalidInflationGeometryDecayDtypeAndDropoutContracts) {
  namespace m = dm::nn::models;
  EXPECT_THROW(m::InflateI3d(nn::ones({3, 3, 1, 1}), 0), std::invalid_argument);
  EXPECT_THROW(m::InflateI3d(nn::ones({3, 3, 1, 1}, nn::DType::Float64), 3),
               std::invalid_argument);
  nn::Generator gen(1);
  EXPECT_THROW(m::InflatedGlorot()({3, 3}, nn::DType::Float32, gen), std::invalid_argument);
  nn::ParameterStore p, s;
  EXPECT_THROW(
      m::I3d(p, s, nn::zeros({1, 16, 32, 32, 3}), {8, 3, 16, 32}, true, false, nullptr),
      std::invalid_argument);
  EXPECT_THROW(m::I3d(p, s, nn::zeros({1, 16, 32, 32, 3}), {8, 3, 16, 32, NAN}),
               std::invalid_argument);
}
