#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(FreeList, ReusesReleasedObjects) { datamunge::dstruct::FreeList<int> values; values.release(4); EXPECT_EQ(values.acquire(),4); EXPECT_FALSE(values.acquire().has_value()); }
