#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(AssociativeStructures, MapsAndSets) { AssociativeArray<int, int> map; map.insert_or_assign(2, 4); map.insert_or_assign(2, 5); EXPECT_EQ(map.at(2), 5); EXPECT_TRUE(map.erase(2)); Multimap<int, int> multi; multi.insert(1, 2); multi.insert(1, 3); EXPECT_EQ(multi.count(1), 2U); Set<int> set; EXPECT_TRUE(set.insert(3)); EXPECT_FALSE(set.insert(3)); Multiset<int> bag; bag.insert(3); bag.insert(3); EXPECT_EQ(bag.count(3), 2U); }
TEST(AssociativeStructures, HeapAndDisjointSet) { BinaryHeap<int> heap{3, 1, 4, 2}; EXPECT_EQ(heap.pop(), 4); EXPECT_EQ(heap.pop(), 3); DisjointSet sets(4); EXPECT_TRUE(sets.unite(0, 1)); EXPECT_TRUE(sets.connected(0, 1)); EXPECT_FALSE(sets.connected(0, 2)); }
