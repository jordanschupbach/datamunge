#include <gtest/gtest.h>
#include <datamunge/algorithms/burstsort.hpp>
#include <datamunge/algorithms/sorting_dist.hpp>
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::algorithms;
bool byte_less(const std::string& a,const std::string& b) {
  return std::lexicographical_compare(a.begin(),a.end(),b.begin(),b.end(),[](unsigned char x,unsigned char y){return x<y;});
}
bool begins(const std::string& text,const std::string& prefix) {return text.size()>=prefix.size() && text.compare(0,prefix.size(),prefix)==0;}
struct ByteLess {bool operator()(const std::string& a,const std::string& b) const {return byte_less(a,b);}};
struct ReferenceNode {std::string prefix;std::vector<std::size_t> items,terminal;};

// Offline partition from final prefix frequencies, not an online burst insertion replay.
std::vector<ReferenceNode> static_partition(const std::vector<std::string>& input,std::size_t capacity) {
  std::vector<ReferenceNode> nodes(1);nodes[0].items.resize(input.size());std::iota(nodes[0].items.begin(),nodes[0].items.end(),std::size_t{0});
  for (std::size_t i=0;i<nodes.size();++i) {
    const auto prefix=nodes[i].prefix;const auto items=nodes[i].items;
    std::map<unsigned char,std::vector<std::size_t>> groups;
    for (const auto id:items) {
      if (input[id].size()==prefix.size()) nodes[i].terminal.push_back(id);
      else groups[static_cast<unsigned char>(input[id][prefix.size()])].push_back(id);
    }
    for (const auto& [byte,ids]:groups) if (ids.size()>capacity)
      nodes.push_back({prefix+static_cast<char>(byte),ids,{}});
  }
  return nodes;
}
void check_input(const std::vector<std::string>& input,std::size_t capacity) {
  BurstSortOptions options;options.bucket_capacity=capacity;const auto saved=input;
  const auto trace=burstsort_trace(input,options);EXPECT_EQ(input,saved);EXPECT_EQ(trace.input,input);EXPECT_EQ(trace.bucket_capacity,capacity);
  std::vector<std::size_t> order(input.size());std::iota(order.begin(),order.end(),std::size_t{0});
  std::stable_sort(order.begin(),order.end(),[&](std::size_t a,std::size_t b){return byte_less(input[a],input[b]);});
  std::vector<std::string> expected;for (const auto id:order) expected.push_back(input[id]);
  ASSERT_EQ(trace.output,expected);ASSERT_EQ(trace.output_input_indices,order);
  auto plain=input;burstsort_with_options(plain,options);EXPECT_EQ(plain,expected);
  if (capacity==8) {plain=input;burstsort(plain);EXPECT_EQ(plain,expected);}
  EXPECT_EQ(trace.shortcut,input.size()<2);
  if (input.size()<2) {
    EXPECT_TRUE(trace.nodes.empty());EXPECT_TRUE(trace.slots.empty());EXPECT_TRUE(trace.bursts.empty());
    EXPECT_EQ(trace.insertion_visits,0);EXPECT_EQ(trace.byte_routes,0);EXPECT_EQ(trace.redistributed_strings,0);
    EXPECT_EQ(trace.leaf_comparisons,0);EXPECT_EQ(trace.collection_slot_scans,0);EXPECT_EQ(trace.max_depth,0);return;
  }
  const auto reference=static_partition(input,capacity);ASSERT_EQ(trace.nodes.size(),reference.size());
  std::map<std::string,const ReferenceNode*,ByteLess> by_prefix;
  for (const auto& node:reference) by_prefix.emplace(node.prefix,&node);
  std::map<std::string,std::size_t,ByteLess> actual_ids;
  std::size_t visits=0,routes=0,depth=0,comparisons=0;
  for (std::size_t id=0;id<trace.nodes.size();++id) {
    const auto prefix=burstsort_prefix(trace,id);ASSERT_TRUE(actual_ids.emplace(prefix,id).second);
    const auto found=by_prefix.find(prefix);ASSERT_NE(found,by_prefix.end());const auto& want=*found->second;
    const auto& got=trace.nodes[id];EXPECT_EQ(got.depth,prefix.size());EXPECT_EQ(got.strings,want.items.size());
    EXPECT_EQ(got.terminal_input_indices,want.terminal);visits+=want.items.size();routes+=want.items.size()-want.terminal.size();depth=std::max(depth,prefix.size());
    if (id) {EXPECT_LT(got.parent,id);EXPECT_EQ(prefix,burstsort_prefix(trace,got.parent)+static_cast<char>(got.incoming_byte));}
  }
  EXPECT_EQ(trace.insertion_visits,visits);EXPECT_EQ(trace.byte_routes,routes);EXPECT_EQ(trace.max_depth,depth);
  EXPECT_EQ(trace.collection_slot_scans,256*trace.nodes.size());
  std::map<std::pair<std::size_t,unsigned char>,const BurstSortSlot*> slots;
  for (const auto& got:trace.slots) EXPECT_TRUE(slots.emplace(std::make_pair(got.node,got.value),&got).second);
  std::size_t occupied=0;
  for (const auto& want:reference) {
    const auto node=actual_ids.at(want.prefix);std::map<unsigned char,std::vector<std::size_t>> groups;
    for (const auto id:want.items) if (input[id].size()>want.prefix.size()) groups[static_cast<unsigned char>(input[id][want.prefix.size()])].push_back(id);
    for (const auto& [byte,ids]:groups) {
      ++occupied;const auto found=slots.find({node,byte});ASSERT_NE(found,slots.end());const auto& got=*found->second;
      EXPECT_EQ(got.strings,ids.size());
      if (ids.size()>capacity) {
        EXPECT_EQ(got.child,actual_ids.at(want.prefix+static_cast<char>(byte)));
        EXPECT_TRUE(got.input_indices.empty());EXPECT_TRUE(got.sorted_indices.empty());EXPECT_EQ(got.sort_comparisons,0);
      } else {
        EXPECT_EQ(got.child,0);EXPECT_EQ(got.input_indices,ids);auto sorted=ids;std::size_t calls=0;
        std::sort(sorted.begin(),sorted.end(),[&](std::size_t a,std::size_t b){++calls;if (input[a]==input[b]) return a<b;return byte_less(input[a],input[b]);});
        EXPECT_EQ(got.sorted_indices,sorted);EXPECT_EQ(got.sort_comparisons,calls);comparisons+=calls;
        EXPECT_LE(got.strings,capacity);
      }
    }
  }
  EXPECT_EQ(trace.slots.size(),occupied);EXPECT_EQ(trace.leaf_comparisons,comparisons);
  // The (B+1)-th arrival sharing a prefix independently determines its burst time.
  std::vector<const ReferenceNode*> bursts;for (const auto& node:reference) if (!node.prefix.empty()) bursts.push_back(&node);
  std::sort(bursts.begin(),bursts.end(),[&](const auto* a,const auto* b){
    if (a->items[capacity]!=b->items[capacity]) return a->items[capacity]<b->items[capacity];return a->prefix.size()<b->prefix.size();
  });
  ASSERT_EQ(trace.bursts.size(),bursts.size());
  for (std::size_t k=0;k<bursts.size();++k) {
    const auto& got=trace.bursts[k];const auto& want=*bursts[k];
    EXPECT_EQ(got.child,k+1);EXPECT_EQ(got.child,actual_ids.at(want.prefix));EXPECT_EQ(got.child_depth,want.prefix.size());
    EXPECT_EQ(got.value,static_cast<unsigned char>(want.prefix.back()));EXPECT_EQ(got.trigger_input_index,want.items[capacity]);
    EXPECT_EQ(got.strings,capacity+1);EXPECT_EQ(got.node,trace.nodes[got.child].parent);
  }
  EXPECT_EQ(trace.redistributed_strings,trace.bursts.size()*(capacity+1));
}
}

TEST(BurstSortTrace, OriginalWordsUseActualBucketsAndOneBurst) {
  const std::function<void(std::vector<std::string>&)> legacy=burstsort;
  const std::vector<std::string> words{"banana","apple","apricot","avocado","blueberry","cherry","cranberry","coconut",
      "date","elderberry","fig","grape","guava","apple","apex","april","apply","arc","almond"};
  auto plain=words;legacy(plain);auto expected=words;std::sort(expected.begin(),expected.end());EXPECT_EQ(plain,expected);
  const auto t=burstsort_trace(words);ASSERT_EQ(t.nodes.size(),2);ASSERT_EQ(t.bursts.size(),1);
  EXPECT_EQ(burstsort_prefix(t,1),"a");EXPECT_EQ(t.nodes[1].strings,9);EXPECT_EQ(t.bursts[0].strings,9);
  EXPECT_EQ(t.bursts[0].trigger_input_index,18);EXPECT_EQ(t.slots.size(),11);check_input(words,8);
}

TEST(BurstSortTrace, ExhaustiveSmallArraysMatchStaticPrefixPartitionAndStableOrder) {
  const std::vector<std::string> alphabet{"","a","aa","ab","b"};std::size_t arrays=0;
  for (std::size_t n=0,power=1;n<=6;++n,power*=alphabet.size()) for (std::size_t word=0;word<power;++word) {
    auto digits=word;std::vector<std::string> input;
    for (std::size_t i=0;i<n;++i) {input.push_back(alphabet[digits%alphabet.size()]);digits/=alphabet.size();}
    for (const std::size_t capacity:{1,2,8}) check_input(input,capacity);
    ++arrays;
  }
  EXPECT_EQ(arrays,19531);
}

TEST(BurstSortTrace, RandomBinaryPrefixesAndDuplicatesMatchStaticOracle) {
  std::mt19937 rng(200403);const std::array<unsigned char,6> alphabet{0,1,97,98,128,255};
  for (int trial=0;trial<1000;++trial) {
    std::vector<std::string> input(rng()%61);
    for (auto& text:input) {const auto length=rng()%17;for (std::size_t i=0;i<length;++i) text.push_back(static_cast<char>(alphabet[rng()%alphabet.size()]));}
    if (trial%5==0) for (auto& text:input) text=std::string(trial%17,'q');
    check_input(input,1+rng()%17);
  }
}

TEST(BurstSortTrace, ThresholdIsStrictAndTerminalsAreNotByteZero) {
  const std::vector<std::string> eight(8,"abcd"),nine(9,"abcd");
  const auto a=burstsort_trace(eight);EXPECT_EQ(a.nodes.size(),1);EXPECT_TRUE(a.bursts.empty());EXPECT_EQ(a.byte_routes,8);
  const auto b=burstsort_trace(nine);EXPECT_EQ(b.nodes.size(),5);EXPECT_EQ(b.bursts.size(),4);EXPECT_EQ(b.byte_routes,36);
  EXPECT_EQ(b.nodes.back().terminal_input_indices.size(),9);EXPECT_EQ(b.leaf_comparisons,0);
  const std::vector<std::string> binary{"",std::string(1,'\0'),std::string{'\0','a'},"",std::string(1,static_cast<char>(255)),"a"};
  check_input(binary,1);check_input(binary,8);check_input(std::vector<std::string>(20,""),1);
}

TEST(BurstSortTrace, AllByteSlotsCollectInUnsignedOrder) {
  std::vector<std::string> input;
  for (int byte=255;byte>=0;--byte) input.push_back(std::string(1,static_cast<char>(byte)));
  input.push_back("");check_input(input,8);
  const auto t=burstsort_trace(input);EXPECT_EQ(t.nodes.size(),1);EXPECT_EQ(t.slots.size(),256);
  EXPECT_EQ(t.output[0],"");for (std::size_t byte=0;byte<256;++byte) EXPECT_EQ(static_cast<unsigned char>(t.output[byte+1][0]),byte);
}

TEST(BurstSortTrace, DeepDuplicatePrefixDoesNotUseRecursiveInsertionCollectionOrDestruction) {
  constexpr std::size_t length=10000;const std::vector<std::string> input(9,std::string(length,'q'));
  const auto t=burstsort_trace(input);ASSERT_EQ(t.nodes.size(),length+1);EXPECT_EQ(t.bursts.size(),length);
  EXPECT_EQ(t.max_depth,length);EXPECT_EQ(t.byte_routes,9*length);EXPECT_EQ(t.insertion_visits,9*(length+1));
  EXPECT_EQ(t.redistributed_strings,9*length);EXPECT_EQ(t.collection_slot_scans,256*(length+1));
  EXPECT_EQ(t.output,input);EXPECT_EQ(burstsort_prefix(t,length),std::string(length,'q'));
  auto plain=input;burstsort(plain);EXPECT_EQ(plain,input);
}

TEST(BurstSortTrace, CompleteNodeBudgetsPreserveCallerOnFailure) {
  const std::vector<std::string> input(9,"abcd");
  for (std::size_t cap=1;cap<5;++cap) {
    BurstSortOptions options;options.max_nodes=cap;auto mutable_input=input;
    EXPECT_THROW(burstsort_with_options(mutable_input,options),std::length_error);EXPECT_EQ(mutable_input,input);
    EXPECT_THROW(burstsort_trace(input,options),std::length_error);
  }
  BurstSortOptions exact;exact.max_nodes=5;auto a=input;burstsort_with_options(a,exact);EXPECT_EQ(a,input);
  EXPECT_EQ(burstsort_trace(input,exact).nodes.size(),5);
  BurstSortOptions small;small.max_nodes=64;
  auto deep=std::vector<std::string>(9,std::string(10000,'x'));const auto saved=deep;
  EXPECT_THROW(burstsort_with_options(deep,small),std::length_error);EXPECT_EQ(deep,saved);
}

TEST(BurstSortTrace, EmptySingletonAndHugeCapacityUseCorrectShortcuts) {
  check_input({},8);check_input({"a"},1);check_input({""},8);
  const auto huge=std::numeric_limits<std::size_t>::max();check_input({"b","a","a",""},huge);
  BurstSortOptions options;options.max_nodes=1;EXPECT_TRUE(burstsort_trace({},options).nodes.empty());
  EXPECT_EQ(burstsort_trace({"a","b"},options).nodes.size(),1);
}

TEST(BurstSortTrace, PrefixHelperOwnsBytesAndRejectsMalformedPaths) {
  BurstSortOptions options;options.bucket_capacity=1;
  const auto t=burstsort_trace({"aa","aa"},options);EXPECT_EQ(burstsort_prefix(t,0),"");EXPECT_EQ(burstsort_prefix(t,2),"aa");
  EXPECT_THROW(burstsort_prefix(t,t.nodes.size()),std::out_of_range);
  EXPECT_THROW(burstsort_prefix(burstsort_trace({}),0),std::out_of_range);
  auto bad=t;bad.nodes[1].parent=1;EXPECT_THROW(burstsort_prefix(bad,1),std::invalid_argument);
  bad=t;bad.nodes[0].depth=1;EXPECT_THROW(burstsort_prefix(bad,1),std::invalid_argument);
  bad=t;bad.nodes[1].depth=2;EXPECT_THROW(burstsort_prefix(bad,1),std::invalid_argument);
  bad=t;bad.nodes[1].parent=std::numeric_limits<std::size_t>::max();EXPECT_THROW(burstsort_prefix(bad,1),std::invalid_argument);
  bad=t;bad.nodes[1].depth=std::numeric_limits<std::size_t>::max();bad.nodes[2].depth=0;
  EXPECT_THROW(burstsort_prefix(bad,2),std::invalid_argument);
}

TEST(BurstSortTrace, ZeroCapacityIsInvalidBeforeMutationEvenForShortcuts) {
  BurstSortOptions options;options.bucket_capacity=0;
  for (auto input:std::vector<std::vector<std::string>>{{},{"a"},{"b","a"}}) {
    const auto saved=input;EXPECT_THROW(burstsort_with_options(input,options),std::invalid_argument);EXPECT_EQ(input,saved);
    EXPECT_THROW(burstsort_trace(input,options),std::invalid_argument);
  }
}
