#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(DopeVector, MapsCoordinates){datamunge::dstruct::DopeVector<int> values(2,2);values.at(1,1)=8;EXPECT_EQ(values.at(1,1),8);}
