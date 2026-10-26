#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(Bitmap, StoresPixelsByCoordinates){datamunge::dstruct::Bitmap image(3,2);image.set(2,1);EXPECT_TRUE(image.at(2,1));EXPECT_FALSE(image.at(0,0));}
