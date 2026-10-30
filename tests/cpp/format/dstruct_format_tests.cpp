// std::format / std::print support for datamunge::dstruct types.
// (DataFrame is covered by tests/cpp/cpp23_format_tests.cpp.)
#include <datamunge/format/dstruct.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace ds = datamunge::dstruct;

// Ranges: already formattable through the standard range formatter.
static_assert(std::formattable<ds::Array<int, 3>, char>);
static_assert(std::formattable<ds::DynamicArray<int>, char>);

static_assert(std::formattable<ds::DataFrame, char>);
static_assert(std::formattable<ds::LinkedList<int>, char>);
static_assert(std::formattable<ds::DoublyLinkedList<int>, char>);
static_assert(std::formattable<ds::Stack<int>, char>);
static_assert(std::formattable<ds::Queue<int>, char>);
static_assert(std::formattable<ds::DoubleEndedQueue<int>, char>);
static_assert(std::formattable<ds::CircularBuffer<int>, char>);
static_assert(std::formattable<ds::BitArray, char>);
static_assert(std::formattable<ds::SortedArray<int>, char>);
static_assert(std::formattable<ds::LookupTable<int>, char>);
static_assert(std::formattable<ds::BitField, char>);
static_assert(std::formattable<ds::Bitboard, char>);
static_assert(std::formattable<ds::AssociationList<int, int>, char>);
static_assert(std::formattable<ds::AssociativeArray<int, int>, char>);
static_assert(std::formattable<ds::Multimap<int, int>, char>);
static_assert(std::formattable<ds::Set<int>, char>);
static_assert(std::formattable<ds::Multiset<int>, char>);
static_assert(std::formattable<ds::BinaryHeap<int>, char>);
static_assert(std::formattable<ds::DisjointSet, char>);
static_assert(std::formattable<ds::Bitmap, char>);
static_assert(std::formattable<ds::BloomFilter<int>, char>);
static_assert(std::formattable<ds::CountMinSketch<int>, char>);
static_assert(std::formattable<ds::ControlTable<int>, char>);
static_assert(std::formattable<ds::DopeVector<int>, char>);
static_assert(std::formattable<ds::DaryHeap<int, 4>, char>);
static_assert(std::formattable<ds::TernaryHeap<int>, char>);
static_assert(std::formattable<ds::FreeList<int>, char>);
static_assert(std::formattable<ds::GapBuffer<char>, char>);
static_assert(std::formattable<ds::SelfOrganizingList<int>, char>);
static_assert(std::formattable<ds::VariableLengthArray<int>, char>);
static_assert(std::formattable<ds::ThreadSafeVector<int>, char>);
static_assert(std::formattable<ds::TernaryTree<int>, char>);
static_assert(std::formattable<ds::KaryTree<int, 4>, char>);
static_assert(std::formattable<ds::RoseTree<int>, char>);
static_assert(std::formattable<ds::BinarySearchTree<int>, char>);
static_assert(std::formattable<ds::Trie, char>);
static_assert(std::formattable<ds::FenwickTree<int>, char>);
static_assert(std::formattable<ds::SegmentTree<int>, char>);
static_assert(std::formattable<ds::Matrix<int>, char>);
static_assert(std::formattable<ds::SparseMatrix<double>, char>);
static_assert(std::formattable<ds::ParallelArray<std::string, int>, char>);
static_assert(std::formattable<ds::Image<int>, char>);
static_assert(std::formattable<ds::ProductType<int, std::string>, char>);
static_assert(std::formattable<ds::Tuple<int, double, std::string>, char>);
static_assert(std::formattable<ds::TaggedUnion<int, double>, char>);
static_assert(std::formattable<ds::Union<int>, char>);
static_assert(std::formattable<ds::DirectedGraph<int>, char>);
static_assert(std::formattable<ds::DirectedAcyclicGraph<std::string>, char>);
static_assert(std::formattable<ds::UndirectedGraph<int>, char>);
static_assert(std::formattable<ds::WeightedGraph<std::string>, char>);

namespace {
struct Opaque {};
}  // namespace
// Unformattable element types still print (size-only).
static_assert(std::formattable<ds::LinkedList<Opaque>, char>);
static_assert(std::formattable<ds::Matrix<Opaque>, char>);

TEST(DstructFormat, Lists) {
  ds::LinkedList<int> list;
  for (int i : {1, 2, 3}) list.push_back(i);
  EXPECT_EQ(std::format("{}", list), "LinkedList[1, 2, 3]");
  EXPECT_EQ(std::format("{}", ds::LinkedList<Opaque>()), "LinkedList(size=0)");
  ds::DoublyLinkedList<int> dl;
  dl.push_back(1);
  dl.push_front(0);
  EXPECT_EQ(std::format("{}", dl), "DoublyLinkedList[0, 1]");
  EXPECT_EQ(std::format("{}", ds::DoublyLinkedList<Opaque>()), "DoublyLinkedList(size=0)");
  ds::Stack<int> s;
  s.push(1);
  s.push(2);
  EXPECT_EQ(std::format("{}", s), "Stack[2, 1]");
  EXPECT_EQ(s.size(), 2u);
  ds::Queue<int> q;
  for (int i = 1; i <= 8; ++i) q.push(i);
  EXPECT_EQ(std::format("{}", q), "Queue[1, 2, 3, 4, 5, ..., 8] (n=8)");
  EXPECT_EQ(std::format("{}", ds::Queue<int>()), "Queue[]");
  ds::DoubleEndedQueue<int> dq;
  dq.push_back(2);
  dq.push_front(1);
  dq.push_back(3);
  EXPECT_EQ(std::format("{}", dq), "DoubleEndedQueue[1, 2, 3]");
  EXPECT_EQ(std::format("{}", ds::Stack<Opaque>()), "Stack(size=0)");
  ds::CircularBuffer<int> cb(4);
  cb.push(1);
  EXPECT_EQ(std::format("{}", cb), "CircularBuffer(size=1, capacity=4)");
  ds::FreeList<int> fl;
  fl.release(3);
  EXPECT_EQ(std::format("{}", fl), "FreeList(available=1)");
  ds::GapBuffer<char> gb;
  for (char c : {'a', 'b', 'c'}) gb.insert(c);
  gb.move_cursor(1);
  EXPECT_EQ(std::format("{}", gb), "GapBuffer(cursor=1)[a, b, c]");
  ds::SelfOrganizingList<int> sol;
  for (int i : {3, 2, 1}) sol.push_front(i);
  EXPECT_EQ(std::format("{}", sol), "SelfOrganizingList[1, 2, 3]");
  ASSERT_TRUE(sol.find(3).has_value());
  EXPECT_EQ(std::format("{}", sol), "SelfOrganizingList[3, 1, 2]");
  EXPECT_EQ(std::format("{}", sol), "SelfOrganizingList[3, 1, 2]");  // printing does not reorganize
  ds::VariableLengthArray<int> vla(2);
  vla.at(1) = 5;
  EXPECT_EQ(std::format("{}", vla), "VariableLengthArray[0, 5]");
  EXPECT_EQ(std::format("{}", ds::ThreadSafeVector<int>{1, 2}), "ThreadSafeVector[1, 2]");
  ds::ParallelArray<std::string, int> pa;
  pa.push_back("a", 1);
  pa.push_back("b", 2);
  EXPECT_EQ(std::format("{}", pa), "ParallelArray{a: 1, b: 2}");
}

TEST(DstructFormat, ArraysAndBits) {
  ds::BitArray bits(6);
  bits.set(1);
  bits.set(4);
  EXPECT_EQ(std::format("{}", bits), "BitArray(n=6) 010010");
  EXPECT_EQ(std::format("{}", ds::BitArray(100)), "BitArray(n=100) " + std::string(64, '0') + "...");
  ds::SortedArray<int> sa;
  for (int i : {3, 1, 2}) sa.insert(i);
  EXPECT_EQ(std::format("{}", sa), "SortedArray[1, 2, 3]");
  ds::LookupTable<std::string> lt(5);
  lt.set(1, "x");
  lt.set(3, "y");
  EXPECT_EQ(std::format("{}", lt), "LookupTable(size=5, {1: x, 3: y})");
  ds::BitField bf(8);
  bf.set(0, 3, 5);
  EXPECT_EQ(std::format("{}", bf), "BitField(width=8, bits=0b00000101)");
  EXPECT_EQ(std::format("{}", ds::BitField(100)), "BitField(width=100)");
  ds::Bitboard bb;
  bb.set(0);
  bb.set(8);
  EXPECT_EQ(std::format("{}", bb), "Bitboard(0x0000000000000101, squares=2)");
  ds::Bitmap bm(3, 2);
  bm.set(0, 0);
  bm.set(2, 1);
  EXPECT_EQ(std::format("{}", bm), "Bitmap 3x2\n#..\n..#");
  EXPECT_EQ(std::format("{}", ds::BloomFilter<int>(64, 3)), "BloomFilter(bits=64, hashes=3)");
  EXPECT_EQ(std::format("{}", ds::CountMinSketch<int>(8, 2)), "CountMinSketch(width=8, depth=2)");
  ds::ControlTable<int> ct(3);
  ct.configure(0, 10);
  ct.configure(2, 30, false);
  EXPECT_EQ(std::format("{}", ct), "ControlTable(size=3, enabled=1)[10, 0, 30]");
  EXPECT_EQ(std::format("{}", ds::ControlTable<Opaque>(2)), "ControlTable(size=2, enabled=0)");
  ds::DopeVector<int> dv(2, 2);
  dv.at(1, 0) = 4;
  EXPECT_EQ(std::format("{}", dv), "DopeVector<int> 2 x 2\n     [,0] [,1]\n[0,]    0    0\n[1,]    4    0");
}

TEST(DstructFormat, Associative) {
  ds::AssociationList<std::string, int> al;
  al.prepend("a", 1);
  al.prepend("b", 2);
  al.prepend("a", 3);
  EXPECT_EQ(std::format("{}", al), "AssociationList{a: 3, b: 2, a: 1}");
  ds::AssociativeArray<std::string, int> aa;
  aa.insert_or_assign("a", 1);
  EXPECT_EQ(std::format("{}", aa), "AssociativeArray{a: 1}");
  aa.insert_or_assign("b", 2);
  const std::string two = std::format("{}", aa);  // bucket order is hash-dependent
  EXPECT_TRUE(two == "AssociativeArray{a: 1, b: 2}" || two == "AssociativeArray{b: 2, a: 1}") << two;
  EXPECT_EQ(aa.bucket_count(), 17u);
  ds::AssociativeArray<int, int> big;
  for (int i = 0; i < 10; ++i) big.insert_or_assign(i, i * i);  // std::hash<int> is identity: buckets 0..9
  EXPECT_EQ(std::format("{}", big), "AssociativeArray{0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, ...} (n=10)");
  EXPECT_EQ(std::format("{}", ds::AssociativeArray<int, Opaque>()), "AssociativeArray(size=0)");
  EXPECT_EQ(std::format("{}", ds::Multimap<int, int>()), "Multimap{}");
  ds::Multimap<int, std::string> mm;
  mm.insert(1, "x");
  mm.insert(1, "y");
  EXPECT_EQ(std::format("{}", mm), "Multimap{1: y, 1: x}");
  ds::Set<int> set;
  set.insert(1);
  EXPECT_EQ(std::format("{}", set), "Set[1]");
  set.insert(3);
  EXPECT_EQ(std::format("{}", set), "Set[1, 3]");  // identity hash: bucket order
  EXPECT_EQ(std::format("{}", ds::Multiset<int>()), "Multiset[]");
  ds::Multiset<int> ms;
  for (int i : {2, 5, 2}) ms.insert(i);
  EXPECT_EQ(std::format("{}", ms), "Multiset[2, 2, 5]");
  EXPECT_EQ(std::format("{}", ds::BinaryHeap<int>{3, 9, 1}), "BinaryHeap(size=3, top=9)");
  EXPECT_EQ(std::format("{}", ds::BinaryHeap<int>()), "BinaryHeap(size=0)");
  ds::TernaryHeap<int> th;
  th.push(4);
  EXPECT_EQ(std::format("{}", th), "DaryHeap<3>(size=1, top=4)");
  ds::DisjointSet dsu(5);
  EXPECT_EQ(std::format("{}", dsu), "DisjointSet(size=5, sets=5)");
  dsu.unite(0, 1);
  dsu.unite(3, 4);
  dsu.unite(1, 4);
  EXPECT_EQ(std::format("{}", dsu), "DisjointSet(size=5, sets=2)");
}

TEST(DstructFormat, Trees) {
  EXPECT_EQ(std::format("{}", ds::TernaryTree<int>(1)), "TernaryTree(nodes=1)\n  1");
  ds::TernaryTree<int> tt(1);
  auto& left = tt.set_left(tt.root(), 2);
  tt.set_right(tt.root(), 3);
  tt.set_middle(left, 4);
  EXPECT_EQ(std::format("{}", tt), "TernaryTree(nodes=4)\n  1\n    left: 2\n      middle: 4\n    right: 3");
  ds::KaryTree<int, 4> kt(1);
  kt.set_child(kt.root(), 2, 7);
  EXPECT_EQ(std::format("{}", kt), "KaryTree<4>(nodes=2)\n  1\n    [2] 7");
  ds::RoseTree<std::string> rt("root");
  auto& a = rt.add_child(rt.root(), "a");
  rt.add_child(a, "a1");
  rt.add_child(rt.root(), "b");
  EXPECT_EQ(std::format("{}", rt), "RoseTree(nodes=4)\n  root\n    a\n      a1\n    b");
  ds::RoseTree<int> wide(0);
  for (int i = 1; i <= 24; ++i) wide.add_child(wide.root(), i);
  const std::string capped = std::format("{}", wide);
  EXPECT_TRUE(capped.starts_with("RoseTree(nodes=25)\n  0\n    1\n")) << capped;
  EXPECT_TRUE(capped.ends_with("\n    19\n  ... (5 more nodes)")) << capped;
  EXPECT_EQ(std::format("{}", ds::RoseTree<Opaque>(Opaque{})), "RoseTree(nodes=1)");
  ds::BinarySearchTree<int> bst;
  for (int i : {5, 2, 8}) bst.insert(i);
  EXPECT_EQ(std::format("{}", bst), "BinarySearchTree(size=3, min=2)");
  EXPECT_EQ(std::format("{}", ds::BinarySearchTree<int>()), "BinarySearchTree(size=0)");
  ds::Trie trie;
  trie.insert("hi");
  EXPECT_EQ(std::format("{}", trie), "Trie(words=1)");
  ds::FenwickTree<int> ft(3);
  ft.add(0, 2);
  ft.add(2, 7);
  EXPECT_EQ(std::format("{}", ft), "FenwickTree[2, 0, 7]");
  ds::SegmentTree<int> st(4);
  st.set(1, 5);
  st.set(3, 2);
  EXPECT_EQ(std::format("{}", st), "SegmentTree(size=4, total=7)[0, 5, 0, 2]");
  EXPECT_EQ(st.identity(), 0);
}

TEST(DstructFormat, MatricesAndImages) {
  ds::Matrix<int> m(2, 2, 0);
  m.at(0, 1) = 3;
  EXPECT_EQ(std::format("{}", m), "Matrix<int> 2 x 2\n     [,0] [,1]\n[0,]    0    3\n[1,]    0    0");
  EXPECT_EQ(std::format("{}", ds::Matrix<Opaque>(1, 1)), "Matrix<object> 1 x 1");
  ds::SparseMatrix<double> sm(2, 2);
  sm.set(1, 0, 2.5);
  EXPECT_EQ(std::format("{}", sm), "SparseMatrix<double> 2 x 2, nnz=1\n     [,0] [,1]\n[0,]    .    .\n[1,]  2.5    .");
  ds::Image<int> img(3, 1, 7);
  EXPECT_EQ(std::format("{}", img), "Image 3x1\n     [,0] [,1] [,2]\n[0,]    7    7    7");
}

TEST(DstructFormat, ValueTypes) {
  EXPECT_EQ(std::format("{}", ds::ProductType<int, std::string>{1, "a"}), "ProductType(1, a)");
  EXPECT_EQ(std::format("{}", ds::Tuple<int, double, std::string>(1, 2.5, "x")), "Tuple(1, 2.5, x)");
  EXPECT_EQ(std::format("{}", ds::TaggedUnion<int, double>()), "TaggedUnion(index=0, value=0)");
  ds::TaggedUnion<int, std::string> tu;
  tu.set(std::string("hi"));
  EXPECT_EQ(std::format("{}", tu), "TaggedUnion(index=1, value=hi)");
  const auto& ctu = tu;
  EXPECT_EQ(ctu.get<std::string>(), "hi");
  EXPECT_EQ(std::format("{}", ds::TaggedUnion<int, Opaque>()), "TaggedUnion(index=0)");
  ds::Union<int> u;
  EXPECT_EQ(std::format("{}", u), "Union(empty)");
  u.set(4);
  EXPECT_EQ(std::format("{}", u), "Union(4)");
}

TEST(DstructFormat, Graphs) {
  ds::DirectedGraph<int> g{1, 2, 3};
  g.add_edge(1, 2);
  g.add_edge(1, 3);
  EXPECT_EQ(std::format("{}", g), "DirectedGraph(vertices=3, edges=2)\n  1 -> [2, 3]\n  2 -> []\n  3 -> []");
  ds::DirectedAcyclicGraph<std::string> dag;
  dag.add_vertex("a");
  dag.add_vertex("b");
  dag.add_edge("a", "b");
  EXPECT_EQ(std::format("{}", dag), "DirectedAcyclicGraph(vertices=2, edges=1, order=[a, b])");
  ds::UndirectedGraph<int> ug{1, 2};
  ug.add_edge(1, 2);
  EXPECT_EQ(std::format("{}", ug), "UndirectedGraph(vertices=2, edges=1)\n  1 -- [2]\n  2 -- [1]");
  ds::WeightedGraph<std::string> wg({"a", "b"}, false);
  wg.add_edge("a", "b", 1.5);
  EXPECT_EQ(std::format("{}", wg), "WeightedGraph(undirected, vertices=2, edges=1)\n  a -- [b (1.5)]\n  b -- [a (1.5)]");
  ds::DirectedGraph<int> big;
  for (int i = 0; i < 12; ++i) big.add_vertex(i);
  EXPECT_TRUE(std::format("{}", big).ends_with("\n  9 -> []\n  ... (2 more vertices)"));
}
