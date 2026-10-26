#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(AssociationList, FindsMostRecentAssociation) { datamunge::dstruct::AssociationList<int,int> list; list.prepend(1,2); list.prepend(1,3); EXPECT_EQ(list.find(1),3); EXPECT_FALSE(list.find(9).has_value()); }
