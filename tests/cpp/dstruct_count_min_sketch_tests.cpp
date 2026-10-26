#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(CountMinSketch, EstimateNeverUndercountsInsertedValue) { CountMinSketch<int> sketch(128, 4); sketch.add(3, 4); sketch.add(7, 2); EXPECT_GE(sketch.estimate(3), 4U); EXPECT_GE(sketch.estimate(7), 2U); }
