#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(TreeIndexes, BinarySearchTreeAndTrie) { BinarySearchTree<int> tree; EXPECT_TRUE(tree.insert(4)); EXPECT_TRUE(tree.insert(2)); EXPECT_TRUE(tree.insert(6)); EXPECT_FALSE(tree.insert(4)); EXPECT_TRUE(tree.erase(4)); EXPECT_TRUE(tree.contains(6)); Trie trie; trie.insert("tree"); EXPECT_TRUE(trie.contains("tree")); EXPECT_TRUE(trie.starts_with("tr")); EXPECT_FALSE(trie.contains("tr")); }
TEST(TreeIndexes, FenwickAndSegmentTree) { FenwickTree<int> fenwick(5); fenwick.add(0, 2); fenwick.add(3, 4); EXPECT_EQ(fenwick.range_sum(0, 4), 6); SegmentTree<int> segment(5); segment.set(0, 2); segment.set(3, 4); EXPECT_EQ(segment.query(0, 4), 6); }
