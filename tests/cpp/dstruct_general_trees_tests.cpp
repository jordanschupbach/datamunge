#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(GeneralTrees, TernaryAndKaryTreesOwnChildren) { TernaryTree<int> ternary(1); ternary.set_middle(ternary.root(), 2); EXPECT_EQ(ternary.root().middle->value, 2); KaryTree<int, 4> kary(1); kary.set_child(kary.root(), 3, 4); EXPECT_EQ(kary.root().children[3]->value, 4); }
TEST(GeneralTrees, RoseTreeSupportsArbitraryChildren) { RoseTree<int> tree(1); tree.add_child(tree.root(), 2); tree.add_child(tree.root(), 3); EXPECT_EQ(tree.root().children.size(), 2U); EXPECT_EQ(tree.root().children[1]->value, 3); }
