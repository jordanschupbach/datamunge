
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_voxnet_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/voxnet.cpp"
#undef main
TEST(Voxnet, EveryIndependentHostConvolutionValueAndDerivative) {
  ConvChecks();
}
TEST(Voxnet, AllRotationCoordinatesAndPoolAdjoints) {
  RotationPoolChecks();
}
TEST(Voxnet, EveryWeightOnlyPenaltyHostFiniteDifference) {
  RegularizerChecks();
}
TEST(Voxnet, ActualFull40ClassAndThreeClassGeometryEveryGradient) {
  GraphChecks();
}
TEST(Voxnet, EveryVoxelDescriptorAndMemoryPredictionMatchesRetainedHostOracle) {
  namespace n = dm::nn;
  namespace m = n::models;
  auto pools = m::LoadVoxnet("build/cnn-datasets/voxnet");
  for (const auto &d : pools) {
    auto features = m::VoxnetFeatures(d).to_vector<float>(), v = d.voxels.to_vector<float>();
    std::vector<float> expected(d.voxels.size(0) * 64);
    for (int k = 0; k < 4; ++k) {
      auto rotated = m::VoxnetRotate(d.voxels, k).to_vector<float>();
      for (int i = 0; i < d.voxels.size(0); ++i)
        for (int x = 0; x < 4; ++x)
          for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z) {
              double sum = 0;
              for (int a = 0; a < 8; ++a)
                for (int b = 0; b < 8; ++b)
                  for (int c = 0; c < 8; ++c)
                    sum +=
                        rotated[i * 32768 + ((x * 8 + a) * 32 + y * 8 + b) * 32 + z * 8 + c];
              expected[i * 64 + (x * 4 + y) * 4 + z] += sum / 2048.;
            }
    }
    EXPECT_EQ(features, expected);
  }
  const auto a = m::VoxnetFeatures(pools[0]).to_vector<float>(),
             b = m::VoxnetFeatures(pools[2]).to_vector<float>();
  auto labels = pools[0].labels.to_vector<int>();
  std::vector<double> scale(64);
  for (int c = 0; c < 64; ++c) {
    double mean = 0, var = 0;
    for (int i = 0; i < 24; ++i)
      mean += a[i * 64 + c] / 24.;
    for (int i = 0; i < 24; ++i)
      var += std::pow(a[i * 64 + c] - mean, 2) / 24.;
    scale[c] = std::max(1e-6, std::sqrt(var));
  }
  auto actual = m::VoxnetNearest(pools[0], pools[2]).to_vector<float>();
  for (int row = 0; row < 12; ++row) {
    double best = INFINITY;
    int label = -1;
    for (int i = 0; i < 24; ++i) {
      double d = 0;
      for (int c = 0; c < 64; ++c)
        d += std::pow((a[i * 64 + c] - b[row * 64 + c]) / scale[c], 2);
      if (d < best) {
        best = d;
        label = labels[i];
      }
    }
    for (int c = 0; c < 3; ++c)
      EXPECT_EQ(actual[row * 3 + c], c == label ? 1 : 0);
  }
}
TEST(Voxnet, SurfaceBinningDuplicateCountsBorderAndInvalidContracts) {
  namespace n = dm::nn;
  namespace m = n::models;
  auto x =
      n::from_vector(std::vector<float>{-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0}, {1, 4, 3});
  const auto y = m::VoxnetOccupancy(x);
  EXPECT_EQ(n::sum(y).item(), 3);
  EXPECT_EQ(y.at({0, 1, 1, 1, 0}), 1);
  EXPECT_EQ(y.at({0, 15, 15, 15, 0}), 1);
  EXPECT_EQ(y.at({0, 30, 30, 30, 0}), 1);
  EXPECT_THROW(m::VoxnetOccupancy(n::full({1, 1, 3}, NAN)), std::invalid_argument);
  n::ParameterStore p;
  EXPECT_THROW(m::Voxnet(p, n::zeros({1, 31, 32, 32, 1}), 3), std::invalid_argument);
  EXPECT_THROW(m::Voxnet(p, n::zeros({1, 32, 32, 32, 1}), 3, true), std::invalid_argument);
  EXPECT_THROW(m::VoxnetRegularized(p, {}, NAN), std::invalid_argument);
}
