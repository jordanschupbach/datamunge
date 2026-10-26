#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(BloomFilter, NeverHasFalseNegativeForInsertedValues) { BloomFilter<int> filter(128, 3); filter.insert(7); filter.insert(42); EXPECT_TRUE(filter.may_contain(7)); EXPECT_TRUE(filter.may_contain(42)); EXPECT_FALSE(filter.may_contain(8)); }
