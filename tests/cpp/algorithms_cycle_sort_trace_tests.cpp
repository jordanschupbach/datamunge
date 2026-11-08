#include <gtest/gtest.h>
#include <datamunge/algorithms/cycle_sort.hpp>
#include <algorithm>
#include <climits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
void same_work(const CycleSortWork& a,const CycleSortWork& b) {
  EXPECT_EQ(a.output,b.output);
  EXPECT_EQ(a.array_writes,b.array_writes);
  EXPECT_EQ(a.rank_comparisons,b.rank_comparisons);
  EXPECT_EQ(a.equality_comparisons,b.equality_comparisons);
  EXPECT_EQ(a.duplicate_skips,b.duplicate_skips);
  EXPECT_EQ(a.rank_scans,b.rank_scans);
  EXPECT_EQ(a.outer_starts,b.outer_starts);
  EXPECT_EQ(a.cycle_count,b.cycle_count);
}
// Independent std::sort target and event replay; no second rank-scan sort.
void verify(const std::vector<int>& input) {
  const auto t=cycle_sort_trace(input,0);const auto w=cycle_sort_work(input);same_work(t.work,w);
  auto sorted=input;std::sort(sorted.begin(),sorted.end());
  EXPECT_EQ(t.input,input);
  EXPECT_EQ(w.output,sorted);
  EXPECT_EQ(cycle_sort(input),sorted);
  std::size_t lower=0;for(std::size_t i=0;i<input.size();++i) lower+=input[i]!=sorted[i];
  EXPECT_EQ(cycle_sort_write_lower_bound(input),lower);
  ASSERT_EQ(w.array_writes,lower);
  ASSERT_EQ(t.placements.size(),lower);
  auto current=input;std::vector<std::size_t> ids(input.size());std::iota(ids.begin(),ids.end(),std::size_t{0});
  std::vector<bool> written(input.size());std::size_t cycles=0,extra_comparisons=0,skips=0;
  bool open=false;int held=0;std::size_t held_id=0,start=0;
  for(const auto& e:t.placements) {
    ASSERT_LT(e.target,input.size());
  ASSERT_LT(e.cycle_start,input.size());
    if(!open) {++cycles;start=e.cycle_start;held=current[start];held_id=ids[start];open=true;}
    else extra_comparisons+=input.size()-start-1;
    ASSERT_EQ(e.cycle_start,start);
  EXPECT_EQ(e.value,held);
  EXPECT_EQ(e.original_index,held_id);
    EXPECT_EQ(e.displaced_value,current[e.target]);
  EXPECT_EQ(e.displaced_original_index,ids[e.target]);
    // Derive the target block from the independent sorted sequence, not a rank scan.
    const auto first=std::lower_bound(sorted.begin()+static_cast<std::ptrdiff_t>(start),sorted.end(),held)-sorted.begin();
    EXPECT_EQ(e.rank_begin,static_cast<std::size_t>(first));
  EXPECT_EQ(e.target,e.rank_begin+e.duplicate_skips);
    for(std::size_t i=e.rank_begin;i<e.target;++i) EXPECT_EQ(current[i],held);
    EXPECT_NE(current[e.target],held);
  EXPECT_EQ(sorted[e.target],held);
  EXPECT_FALSE(written[e.target]);
  EXPECT_NE(input[e.target],sorted[e.target]);written[e.target]=true;
    EXPECT_EQ(e.rank_comparisons,input.size()-start-1);skips+=e.duplicate_skips;
    current[e.target]=held;ids[e.target]=held_id;held=e.displaced_value;held_id=e.displaced_original_index;
    EXPECT_EQ(e.closes_cycle,e.target==start);if(e.closes_cycle) open=false;
  }
  EXPECT_FALSE(open);
  EXPECT_EQ(current,sorted);
  EXPECT_EQ(ids,t.original_indices);
  EXPECT_EQ(w.cycle_count,cycles);
  const auto n=input.size(),base=n>1?n*(n-1)/2:0;
  EXPECT_EQ(w.outer_starts,n>1?n-1:0);
  EXPECT_EQ(w.rank_comparisons,base+extra_comparisons);
  EXPECT_EQ(w.rank_scans,w.outer_starts+w.array_writes-w.cycle_count);
  EXPECT_EQ(w.equality_comparisons,w.array_writes+skips);
  EXPECT_EQ(w.duplicate_skips,skips);
  EXPECT_LE(2*cycles,w.array_writes);
  EXPECT_LE(w.rank_comparisons,base+(n?n-1:0)*(n?n-1:0));
  auto identity=ids;std::sort(identity.begin(),identity.end());
  for(std::size_t i=0;i<n;++i) {EXPECT_EQ(identity[i],i);
  EXPECT_EQ(input[ids[i]],sorted[i]);
  EXPECT_EQ(written[i],input[i]!=sorted[i]);}
}
}
TEST(CycleSortTrace, TinyCycleAndOriginalSevenIntegers) {
  const auto t=cycle_sort_trace({3,1,2});
  ASSERT_EQ(t.placements.size(),3);
  EXPECT_EQ(t.placements[0].target,2);
  EXPECT_EQ(t.placements[1].target,1);
  EXPECT_EQ(t.placements[2].target,0);
  EXPECT_EQ(t.original_indices,(std::vector<std::size_t>{1,2,0}));
  EXPECT_EQ(t.work.rank_comparisons,7);
  EXPECT_EQ(t.work.equality_comparisons,3);
  EXPECT_EQ(t.work.cycle_count,1);
  const auto w=cycle_sort_work({5,3,8,1,9,2,7});
  EXPECT_EQ(w.array_writes,7);
  EXPECT_EQ(w.rank_comparisons,39);
  EXPECT_EQ(w.equality_comparisons,7);
  EXPECT_EQ(w.cycle_count,3);
  verify({3,1,2});verify({5,3,8,1,9,2,7});
}
TEST(CycleSortTrace, EveryTernarySequenceThroughTenAttainsIndependentWriteBound) {
  std::size_t cases=0,power=1;
  for(std::size_t n=0;n<=10;++n,power*=3) for(std::size_t code=0;code<power;++code) {
    std::vector<int> a(n);auto x=code;for(auto& v:a) {v=static_cast<int>(x%3)-1;x/=3;}verify(a);++cases;
  }
  EXPECT_EQ(cases,88573);
}
TEST(CycleSortTrace, EveryDistinctPermutationThroughEightHasExactCyclePlacements) {
  std::size_t cases=0;
  for(int n=0;n<=8;++n) {
    std::vector<int> p(n);std::iota(p.begin(),p.end(),0);
    do {verify(p);++cases;}while(std::next_permutation(p.begin(),p.end()));
  }
  EXPECT_EQ(cases,46234);
}
TEST(CycleSortTrace, EqualKeysCanReverseIdentitiesWhileCorrectCellsStayUntouched) {
  const auto t=cycle_sort_trace({2,2,1});
  EXPECT_EQ(t.work.output,(std::vector<int>{1,2,2}));
  EXPECT_EQ(t.original_indices,(std::vector<std::size_t>{2,1,0}));
  EXPECT_EQ(t.work.array_writes,2);
  EXPECT_EQ(t.work.duplicate_skips,1);
  EXPECT_EQ(t.work.rank_comparisons,5);
  EXPECT_EQ(t.work.equality_comparisons,3);
  ASSERT_EQ(t.placements.size(),2);
  EXPECT_EQ(t.placements[0].rank_begin,1);
  EXPECT_EQ(t.placements[0].target,2);
  EXPECT_EQ(t.placements[1].target,0);
  const std::vector<int> a{2,2,1};std::vector<std::size_t> stable{0,1,2};std::stable_sort(stable.begin(),stable.end(),[&](auto i,auto j){return a[i]<a[j];});
  EXPECT_EQ(stable,(std::vector<std::size_t>{2,0,1}));
  EXPECT_NE(stable,t.original_indices);
  verify(a);verify({3,1,3,2,1});
}
TEST(CycleSortTrace, SortedEqualReverseAndLongCycleHaveExactQuadraticCounts) {
  for(std::size_t n=0;n<=200;++n) {
    std::vector<int> sorted(n);std::iota(sorted.begin(),sorted.end(),0);const std::size_t base=n>1?n*(n-1)/2:0;
    const auto a=cycle_sort_work(sorted),e=cycle_sort_work(std::vector<int>(n,7));
    EXPECT_EQ(a.rank_comparisons,base);
  EXPECT_EQ(e.rank_comparisons,base);
  EXPECT_EQ(a.array_writes,0);
  EXPECT_EQ(e.equality_comparisons,0);
    auto reversed=sorted;std::reverse(reversed.begin(),reversed.end());const auto r=cycle_sort_work(reversed);
    EXPECT_EQ(r.array_writes,n-n%2);
  EXPECT_EQ(r.cycle_count,n/2);
  EXPECT_EQ(r.duplicate_skips,0);
    std::size_t expected=base;for(std::size_t start=0;start<n/2;++start) expected+=n-start-1;
    EXPECT_EQ(r.rank_comparisons,expected);
    if(n>=2) {auto rotated=sorted;std::rotate(rotated.begin(),rotated.begin()+1,rotated.end());const auto c=cycle_sort_work(rotated);
      EXPECT_EQ(c.array_writes,n);
  EXPECT_EQ(c.cycle_count,1);
  EXPECT_EQ(c.rank_comparisons,base+(n-1)*(n-1));}
  }
}
TEST(CycleSortTrace, SeededMixedKeysAndSignedExtremesMatchStdSortAndEventReplay) {
  std::mt19937 rng(771);std::uniform_int_distribution<int> full(INT_MIN,INT_MAX);
  for(int trial=0;trial<1000;++trial) {
    std::vector<int> a(rng()%150);for(int& v:a) v=trial%2?full(rng):static_cast<int>(rng()%9)-4;verify(a);
  }
  verify({INT_MAX,INT_MIN,0,INT_MAX,INT_MIN});verify({INT_MIN,0,INT_MAX});
}
TEST(CycleSortTrace, EmptySingletonAndOwnedResults) {
  for(const auto& a:std::vector<std::vector<int>>{{},{7},{7,7},{1,2,3}}) verify(a);
  std::vector<int> a{3,1,2};const auto t=cycle_sort_trace(a);const auto w=cycle_sort_work(a);a.assign(100,-99);
  EXPECT_EQ(t.input,(std::vector<int>{3,1,2}));
  EXPECT_EQ(t.work.output,(std::vector<int>{1,2,3}));
  EXPECT_EQ(w.output,t.work.output);
  auto moved=cycle_sort(std::vector<int>{2,1});
  EXPECT_EQ(moved,(std::vector<int>{1,2}));
}
TEST(CycleSortTrace, CompleteTraceCapRejectsRatherThanTruncatingEvenSortedInputs) {
  const std::vector<int> a{3,1,2};
  EXPECT_THROW(cycle_sort_trace(a,2),std::length_error);
  EXPECT_NO_THROW(cycle_sort_trace(a,3));
  EXPECT_NO_THROW(cycle_sort_trace(a,0));
  EXPECT_THROW(cycle_sort_trace(std::vector<int>(4097,1)),std::length_error);
  EXPECT_EQ(cycle_sort_trace({}).placements.size(),0);
  EXPECT_EQ(cycle_sort_trace({1},1).original_indices,(std::vector<std::size_t>{0}));
  EXPECT_EQ(a,(std::vector<int>{3,1,2}));
}
TEST(CycleSortTrace, OriginalSeedThreeSweepRetainsWriteAndStrictComparisonResults) {
  std::mt19937 rng(3);CycleSortWork last;
  for(int n=5;n<=200;n+=5) {
    std::vector<int> a(n);for(int& v:a) v=static_cast<int>(rng()%100000);verify(a);last=cycle_sort_work(a);
  }
  EXPECT_EQ(last.array_writes,199);
  EXPECT_EQ(last.rank_comparisons,57301);
}
