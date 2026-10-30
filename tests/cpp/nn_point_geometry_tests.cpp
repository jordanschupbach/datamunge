
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_point_geometry_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_geometry.cpp"
#undef main
#include "../fixtures/nn_point_geometry_reference.hpp"
TEST(PointGeometry, OriginalAll60CoordinateDerivativesTiesBoundariesAndTranslation) {
  EXPECT_EQ(original_point_geometry_main(), 0);
}
TEST(PointGeometry, EveryHostOracleCoordinateAcrossBatchedCloudsAndOversizedPadding) {
  namespace m = dm::nn::models;
  for (int seed = 0; seed < 12; ++seed) {
    const auto xyz = (nn::sin(nn::arange(0, 90) * (.23 + seed * .1)) * 2.).reshape({3, 10, 3});
    const auto ids = m::PointFps(xyz, 7),
               expected = point_geometry_reference::PointFps(xyz, 7);
    EXPECT_EQ(ids.to_vector<int>(), expected.to_vector<int>());
    const auto centers = m::PointGather(xyz, ids);
    for (float radius : {.1f, 1.f, 10.f})
      for (int slots : {1, 4, 12}) {
        const auto a = m::PointBall(xyz, centers, radius, slots);
        const auto b = point_geometry_reference::PointBall(xyz, centers, radius, slots);
        EXPECT_EQ(a.indices.to_vector<int>(), b.indices.to_vector<int>());
        EXPECT_EQ(a.counts.to_vector<int>(), b.counts.to_vector<int>());
        EXPECT_EQ(m::PointGather(xyz, a.indices).to_vector<float>(),
                  point_geometry_reference::PointGather(xyz, b.indices).to_vector<float>());
      }
  }
}
TEST(PointGeometry, DiscreteOutputsDetachedAndFeatureDuplicateScatterExact) {
  namespace m = dm::nn::models;
  auto xyz = nn::tensor({0., 0., 0., 1., 0., 0., 0., 1., 0.}, {1, 3, 3});
  xyz.requires_grad_(true);
  const auto group = m::PointGroup(xyz, 2, .1, 4);
  EXPECT_FALSE(group.indices.requires_grad());
  EXPECT_FALSE(group.counts.requires_grad());
  const auto repeated =
      m::PointGather(xyz, nn::tensor({1, 1, 2}, {1, 3}).to(nn::DType::Int64));
  const auto g = nn::grad(nn::sum(repeated), {xyz})[0];
  EXPECT_EQ(g.to_vector<float>(), (std::vector<float>{0, 0, 0, 2, 2, 2, 1, 1, 1}));
  EXPECT_EQ(nn::sum(nn::abs(group.relative)).item(), 0);
  EXPECT_EQ(nn::sum(nn::abs(nn::grad(nn::sum(group.relative), {xyz})[0])).item(), 0);
}
TEST(PointGeometry, InvalidFiniteShapeIndexRadiusAndEmptyNeighborhoodContracts) {
  namespace m = dm::nn::models;
  const auto x = nn::zeros({2, 6, 3});
  EXPECT_THROW(m::PointFps(x, 7), std::invalid_argument);
  EXPECT_THROW(m::PointFps(nn::full({1, 3, 3}, NAN), 1), std::invalid_argument);
  EXPECT_THROW(m::PointBall(x, nn::ones({2, 1, 3}), .01, 4), std::invalid_argument);
  EXPECT_THROW(m::PointBall(x, nn::zeros({2, 1, 3}), 0., 4), std::invalid_argument);
  EXPECT_THROW(m::PointGather(x, nn::full({2, 1}, 6., nn::DType::Int32)),
               std::invalid_argument);
  EXPECT_THROW(m::PointGather(x, nn::zeros({2, 1})), std::invalid_argument);
  EXPECT_THROW(m::PointGather(x, nn::zeros({1, 1}, nn::DType::Int32)), std::invalid_argument);
}
