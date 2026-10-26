#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
TEST(ControlTable, ConfiguresAndTogglesEntries){datamunge::dstruct::ControlTable<int> table(2);table.configure(1,7);EXPECT_TRUE(table.enabled(1));table.enable(1,false);EXPECT_FALSE(table.enabled(1));EXPECT_EQ(table.at(1),7);}
