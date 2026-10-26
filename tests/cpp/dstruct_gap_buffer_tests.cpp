#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(GapBuffer, EditsAtCursor) { datamunge::dstruct::GapBuffer<char> text; text.insert('a'); text.insert('c'); text.move_cursor(1); text.insert('b'); auto values=text.values(); EXPECT_EQ(values[0],'a'); EXPECT_EQ(values[1],'b'); EXPECT_EQ(values[2],'c'); }
