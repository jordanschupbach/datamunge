#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(VariableLengthArray, ResizesAndPreservesPrefix) { datamunge::dstruct::VariableLengthArray<int> values(2); values.at(0) = 4; values.resize(4); EXPECT_EQ(values.size(), 4U); EXPECT_EQ(values.at(0), 4); }
