#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(DaryHeap, SupportsDaryAndTernaryPriorityOrder) { DaryHeap<int, 4> fourary; for (int value : {2, 8, 1, 7, 3}) fourary.push(value); EXPECT_EQ(fourary.pop(), 8); EXPECT_EQ(fourary.pop(), 7); TernaryHeap<int> ternary; ternary.push(1); ternary.push(4); ternary.push(2); EXPECT_EQ(ternary.top(), 4); }
