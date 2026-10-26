#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(ArrayVariants, SortedArrayAndLookupTable) { SortedArray<int> sorted; sorted.insert(3); sorted.insert(1); sorted.insert(2); EXPECT_EQ(sorted.at(0), 1); EXPECT_TRUE(sorted.contains(2)); LookupTable<int> table(3); table.set(2, 8); EXPECT_TRUE(table.contains(2)); EXPECT_EQ(table.at(2), 8); }
TEST(ArrayVariants, BitFieldAndBitboard) { BitField field(16); field.set(3, 5, 21); EXPECT_EQ(field.get(3, 5), 21U); Bitboard board; board.set(63); EXPECT_TRUE(board.test(63)); board.clear(63); EXPECT_FALSE(board.test(63)); }
