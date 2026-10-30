#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include "../fixtures/nn_point_geometry_reference.hpp"

TEST(PointGeometrySource, EveryActualCompactCadGroupAtBothSetAbstractionLevels) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::NoGradGuard guard;
  const auto pools = m::LoadPoints("build/cnn-datasets/modelnet");
  int objects = 0;
  for (const auto &pool : pools)
    for (std::int64_t row = 0; row < pool.xyz.size(0); ++row) {
      const auto xyz = n::slice_axis(pool.xyz, 0, row, row + 1);
      const auto first = m::PointGroup(xyz, 64, .2f, 16);
      const auto expected_first = point_geometry_reference::PointGroup(xyz, 64, .2f, 16);
      EXPECT_EQ(first.centers.to_vector<float>(), expected_first.centers.to_vector<float>());
      EXPECT_EQ(first.relative.to_vector<float>(), expected_first.relative.to_vector<float>());
      EXPECT_EQ(first.indices.to_vector<int>(), expected_first.indices.to_vector<int>());
      EXPECT_EQ(first.counts.to_vector<int>(), expected_first.counts.to_vector<int>());
      const auto second = m::PointGroup(first.centers, 16, .4f, 32);
      const auto expected_second =
          point_geometry_reference::PointGroup(expected_first.centers, 16, .4f, 32);
      EXPECT_EQ(second.centers.to_vector<float>(), expected_second.centers.to_vector<float>());
      EXPECT_EQ(second.relative.to_vector<float>(), expected_second.relative.to_vector<float>());
      EXPECT_EQ(second.indices.to_vector<int>(), expected_second.indices.to_vector<int>());
      EXPECT_EQ(second.counts.to_vector<int>(), expected_second.counts.to_vector<int>());
      ++objects;
    }
  EXPECT_EQ(objects, 42);
}
