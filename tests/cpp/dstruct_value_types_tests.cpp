#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(ValueTypes, ProductTupleAndTaggedUnion) { datamunge::dstruct::ProductType<int,int> pair{1,2}; EXPECT_EQ(pair.second,2); datamunge::dstruct::Tuple<int,int> tuple(3,4); EXPECT_EQ(tuple.get<0>(),3); datamunge::dstruct::TaggedUnion<int,double> tagged; tagged.set(2.5); EXPECT_TRUE(tagged.is<double>()); }
TEST(ValueTypes, UnionStoresAnOptionalValue) { datamunge::dstruct::Union<int> value; EXPECT_FALSE(value.value().has_value()); value.set(9); EXPECT_EQ(value.value(), 9); }
