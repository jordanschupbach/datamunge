
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_point_models_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_models_checks.cpp"
#undef main
TEST(PointModels, Every9TransformPenaltyHostDerivativeAndOrthogonalZero) {
  PenaltyChecks();
}
TEST(PointModels, BothFullAndCompactEveryEndpointInventoryAndConnectedGradient) {
  nn::set_num_threads(2);
  GraphChecks();
}
TEST(PointModels, BothFullCanonicalGraphsAllFiniteGradientsOnGenuineMesh) {
  nn::set_num_threads(2);
  const auto arrays = LoadClouds("build/cnn-datasets/modelnet");
  FullRuntime(arrays);
  RealForward(arrays);
}
TEST(PointModels, IndependentAll18SharedDensePopulationBnInputDerivatives) {
  namespace m = dm::nn::models;
  nn::ParameterStore p(1, nn::DType::Float64), state(2, nn::DType::Float64);
  nn::Generator gen(3);
  m::PointLayers layer(p, state, true, false, 4, &gen);
  auto x = nn::from_vector(std::vector<double>{-1, .2, .4, .9, 1.5, -.5, 2.2, .7}, {2, 2, 2});
  layer.Unit("u", x, 2);
  {
    nn::NoGradGuard guard;
    p.set("u/W", nn::from_vector(std::vector<double>{1, .2, -.3, .8}, {2, 2}));
    p.set("u/b", nn::from_vector(std::vector<double>{.1, -.2}));
    p.set("u/bn/gamma", nn::from_vector(std::vector<double>{.7, 1.1}));
    p.set("u/bn/beta", nn::full({2}, 1.5, nn::DType::Float64));
  }
  x.requires_grad_(true);
  const auto coefficients =
      nn::sin(nn::arange(0, 8, 1, nn::DType::Float64) * .37).reshape({2, 2, 2});
  const auto out = layer.Unit("u", x, 2);
  auto leaves = p.parameters();
  leaves.push_back(x);
  const auto grads = nn::grad(nn::sum(out * coefficients), leaves);
  std::vector<std::vector<double>> values;
  for (const auto &v : leaves)
    values.push_back(v.to_vector<double>());
  const auto coeff = coefficients.to_vector<double>();
  auto host = [&](const auto &v) {
    double result = 0;
    for (int c = 0; c < 2; ++c) {
      std::array<double, 4> y{};
      double mean = 0, var = 0;
      for (int row = 0; row < 4; ++row) {
        y[row] = v[1][c] + v[4][row * 2] * v[0][c] + v[4][row * 2 + 1] * v[0][2 + c];
        mean += y[row] / 4.;
      }
      for (double q : y)
        var += (q - mean) * (q - mean) / 4.;
      for (int row = 0; row < 4; ++row)
        result += std::max(0., (y[row] - mean) * v[2][c] / std::sqrt(var + .001) + v[3][c]) *
                  coeff[row * 2 + c];
    }
    return result;
  };
  int count = 0;
  for (std::size_t leaf = 0; leaf < values.size(); ++leaf) {
    const auto actual = grads[leaf].to_vector<double>();
    for (std::size_t i = 0; i < actual.size(); ++i) {
      auto plus = values, minus = values;
      plus[leaf][i] += 1e-5;
      minus[leaf][i] -= 1e-5;
      EXPECT_NEAR(actual[i], (host(plus) - host(minus)) / 2e-5, 1e-7);
      ++count;
    }
  }
  EXPECT_EQ(count, 18);
}
TEST(PointModels, EverySourceMetadataAndRetained256SamplesAlign) {
  namespace m = dm::nn::models;
  const auto pools = m::LoadPoints("build/cnn-datasets/modelnet");
  std::map<std::string, nn::Tensor> a;
  for (const auto &[key, value] : nn::load_npz("build/cnn-datasets/modelnet/point_clouds.npz"))
    a[key] = value;
  for (int i = 0; i < 3; ++i) {
    const std::string name = i == 0 ? "fit" : i == 1 ? "validation" : "test";
    EXPECT_EQ(pools[i].xyz.to_vector<float>(),
              nn::slice_axis(a.at(name + "_xyz"), 1, 0, 256).to_vector<float>());
    EXPECT_EQ(pools[i].labels.to_vector<int>(), a.at(name + "_label").to_vector<int>());
    EXPECT_EQ(pools[i].source.to_vector<int>(), a.at(name + "_source_id").to_vector<int>());
    EXPECT_TRUE(
        nn::all(nn::less_equal(nn::sqrt(nn::sum(nn::square(pools[i].xyz), {2})), 1.00001))
            .item<bool>());
  }
}
TEST(PointModels, AllMovingStateModesAndInvalidPointContracts) {
  namespace m = dm::nn::models;
  const auto data = m::LoadPoints("build/cnn-datasets/modelnet");
  for (bool plus : {false, true}) {
    nn::ParameterStore p(971), s(977);
    nn::Generator gen(971);
    const m::PointConfig c{4, 3, 256, 64, 16, 16, 32};
    auto forward = [&](bool train, bool update) {
      return plus ? m::PointNetPlusPlus(p, s, nn::slice_axis(data[0].xyz, 0, 0, 4), c, train,
                                        update, &gen)
                  : m::PointNet(p, s, nn::slice_axis(data[0].xyz, 0, 0, 4), c, train, update,
                                &gen);
    };
    forward(false, false);
    const auto initial = s.flatten();
    forward(true, false);
    EXPECT_EQ(s.flatten(), initial);
    forward(true, true);
    EXPECT_NE(s.flatten(), initial);
    const auto state = s.flatten(), learned = p.flatten();
    const auto z = forward(false, false).logits;
    EXPECT_EQ(nn::max(nn::abs(z - forward(false, false).logits)).item(), 0);
    EXPECT_EQ(s.flatten(), state);
    EXPECT_EQ(p.flatten(), learned);
  }
  nn::ParameterStore p, s;
  EXPECT_THROW(m::PointNet(p, s, nn::zeros({1, 8, 3}), {4, 3, 256, 64, 16, 16, 32}),
               std::invalid_argument);
  EXPECT_THROW(m::PointTransformPenalty(nn::zeros({1, 2, 3}), 3), std::invalid_argument);
  EXPECT_THROW(m::PointLayers(p, s, true, false, 4, nullptr), std::invalid_argument);
}
