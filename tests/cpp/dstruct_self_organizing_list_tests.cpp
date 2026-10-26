#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(SelfOrganizingList, MovesFoundValueToFront) { datamunge::dstruct::SelfOrganizingList<int> list; list.push_front(1); list.push_front(2); list.push_front(3); EXPECT_EQ(list.find(1),1); EXPECT_EQ(list.front(),1); }
