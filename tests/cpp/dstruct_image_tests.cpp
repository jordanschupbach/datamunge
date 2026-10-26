#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(Image, StoresTypedPixels){datamunge::dstruct::Image<int> image(2,2);image.at(1,0)=9;EXPECT_EQ(image.at(1,0),9);}
