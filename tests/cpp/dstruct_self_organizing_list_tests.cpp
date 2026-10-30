#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(SelfOrganizingList, MovesFoundValueToFront) { datamunge::dstruct::SelfOrganizingList<int> list; list.push_front(1); list.push_front(2); list.push_front(3); EXPECT_EQ(list.find(1),1); EXPECT_EQ(list.front(),1); }

TEST(SelfOrganizingList, MoveToFrontKeepsTheOtherElementsInOrder) {
    datamunge::dstruct::SelfOrganizingList<int> list;
    for (int value : {4, 3, 2, 1}) list.push_front(value);  // [1, 2, 3, 4]
    ASSERT_TRUE(list.find(3).has_value());
    EXPECT_EQ(list.to_vector(), (std::vector<int>{3, 1, 2, 4}));
    EXPECT_FALSE(list.find(9).has_value());  // a miss leaves the order unchanged
    EXPECT_EQ(list.to_vector(), (std::vector<int>{3, 1, 2, 4}));
}
