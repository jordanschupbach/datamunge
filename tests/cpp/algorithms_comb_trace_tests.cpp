#include <gtest/gtest.h>
#include <datamunge/algorithms/comparison_sorts.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
using namespace datamunge::algorithms;
std::size_t inversions(const std::vector<int>& a) {
  std::size_t result=0;
  for(std::size_t i=0;i<a.size();++i)for(std::size_t j=i+1;j<a.size();++j)result+=a[i]>a[j];
  return result;
}
void check(const std::vector<int>& input) {
  const auto t=comb_sort_trace(input);const auto w=comb_sort_work(input);const auto b=bubble_sort_work(input);
  auto sorted=input;std::stable_sort(sorted.begin(),sorted.end());
  ASSERT_EQ(comb_sort(input),sorted);ASSERT_EQ(t.work.output,sorted);ASSERT_EQ(w.output,sorted);ASSERT_EQ(b.output,sorted);
  ASSERT_EQ(t.input,input);ASSERT_EQ(t.work.key_comparisons,w.key_comparisons);ASSERT_EQ(t.work.swaps,w.swaps);
  ASSERT_EQ(t.work.pass_count,w.pass_count);ASSERT_EQ(t.work.gap_one_passes,w.gap_one_passes);
  ASSERT_EQ(t.passes.size(),w.pass_count);ASSERT_LE(w.swaps,inversions(input));ASSERT_EQ(b.swaps,inversions(input));
  ASSERT_EQ(b.gap_one_passes,b.pass_count);
  if(input.size()<2) {ASSERT_TRUE(t.passes.empty());ASSERT_EQ(w.key_comparisons,0);ASSERT_EQ(b.key_comparisons,0);return;}
  std::vector<std::size_t> stable_ids(input.size());std::iota(stable_ids.begin(),stable_ids.end(),0);
  std::stable_sort(stable_ids.begin(),stable_ids.end(),[&](auto i,auto j){return input[i]<input[j];});
  std::size_t max_drop=0;
  for(std::size_t i=0;i<stable_ids.size();++i)if(stable_ids[i]>i)max_drop=std::max(max_drop,stable_ids[i]-i);
  ASSERT_EQ(b.pass_count,max_drop+1);ASSERT_EQ(b.key_comparisons,(input.size()-1)*(max_drop+1));
  std::vector<int> state=input;std::vector<std::size_t> ids(input.size());std::iota(ids.begin(),ids.end(),0);
  std::size_t gap=input.size(),comparisons=0,swaps=0,unit_passes=0;
  for(std::size_t k=0;k<t.passes.size();++k) {
    const auto& p=t.passes[k];const auto previous=inversions(state);
    gap=std::max(std::size_t{1},gap*10/13);ASSERT_EQ(p.gap,gap);
    ASSERT_EQ(p.key_comparisons,input.size()-gap);
    std::size_t pass_swaps=0;
    for(std::size_t i=0;i+gap<state.size();++i)if(state[i]>state[i+gap]) {
      std::swap(state[i],state[i+gap]);std::swap(ids[i],ids[i+gap]);++pass_swaps;
    }
    ASSERT_EQ(p.values,state);ASSERT_EQ(p.original_indices,ids);ASSERT_EQ(p.swaps,pass_swaps);
    ASSERT_EQ(p.stopped,gap==1 && pass_swaps==0);ASSERT_EQ(p.stopped,k+1==t.passes.size());
    ASSERT_LE(inversions(state)+pass_swaps,previous);
    if(gap==1){++unit_passes;ASSERT_EQ(inversions(state)+pass_swaps,previous);}
    for(std::size_t i=0;i<state.size();++i)ASSERT_EQ(state[i],input[ids[i]]);
    comparisons+=p.key_comparisons;swaps+=p.swaps;
  }
  ASSERT_EQ(t.original_indices,ids);ASSERT_EQ(w.key_comparisons,comparisons);ASSERT_EQ(w.swaps,swaps);
  ASSERT_EQ(w.gap_one_passes,unit_passes);ASSERT_TRUE(t.passes.back().stopped);
}
}

TEST(CombTrace, OriginalSevenValuePassesAndCounters) {
  const auto t=comb_sort_trace({5,3,8,1,9,2,7});
  const std::vector<std::vector<int>> expected{{2,3,8,1,9,5,7},{1,3,5,2,9,8,7},
    {1,2,5,3,7,8,9},{1,2,3,5,7,8,9},{1,2,3,5,7,8,9}};
  const std::vector<std::size_t> gaps{5,3,2,1,1},swaps{1,2,2,1,0};
  ASSERT_EQ(t.passes.size(),5);
  for(std::size_t i=0;i<5;++i) {EXPECT_EQ(t.passes[i].values,expected[i]);EXPECT_EQ(t.passes[i].gap,gaps[i]);EXPECT_EQ(t.passes[i].swaps,swaps[i]);}
  EXPECT_EQ(t.work.key_comparisons,23);EXPECT_EQ(t.work.swaps,6);EXPECT_EQ(t.work.gap_one_passes,2);
}

TEST(CombTrace, EmptySingletonDuplicatesAndIntegerExtremes) {
  for(const auto& a:std::vector<std::vector<int>>{{},{42},{1,2},{2,1},{0,0,0},
      {INT_MAX,INT_MIN,0,INT_MAX,INT_MIN},{1,2,3,4}})check(a);
}

TEST(CombTrace, ExactNextGapAcrossSizeTRange) {
  const auto maximum=std::numeric_limits<std::size_t>::max();
  const auto exact=[](std::size_t gap) {
    return std::max(std::size_t{1},static_cast<std::size_t>((static_cast<unsigned __int128>(gap)*10)/13));
  };
  for(std::size_t gap=0;gap<10000;++gap)EXPECT_EQ(comb_sort_next_gap(gap),exact(gap));
  for(const auto gap:{maximum,maximum-1,maximum/10,maximum/10+1,maximum/13,maximum/13+1})
    EXPECT_EQ(comb_sort_next_gap(gap),exact(gap));
  std::mt19937_64 engine(42);
  for(int i=0;i<10000;++i){const auto gap=static_cast<std::size_t>(engine());EXPECT_EQ(comb_sort_next_gap(gap),exact(gap));}
  EXPECT_EQ(comb_sort_next_gap(13),10);EXPECT_EQ(comb_sort_next_gap(12),9);
}

TEST(CombTrace, EveryTernaryInputThroughLengthEightMatchesIndependentInversionAndSortOracles) {
  for(std::size_t n=0,power=1;n<=8;++n,power*=3)
    for(std::size_t code=0;code<power;++code) {
      auto x=code;std::vector<int> a(n);
      for(auto& v:a){v=static_cast<int>(x%3)-1;x/=3;}check(a);
      if(::testing::Test::HasFatalFailure())return;
    }
}

TEST(CombTrace, EveryEightElementPermutationRespectsActualPassAndIdentityInvariants) {
  std::vector<int> a(8);std::iota(a.begin(),a.end(),0);
  do{check(a);if(::testing::Test::HasFatalFailure())return;}while(std::next_permutation(a.begin(),a.end()));
}

TEST(CombTrace, SeededDuplicateArraysAndBubbleDisplacementOracle) {
  std::mt19937 engine(9981);
  for(int trial=0;trial<1000;++trial) {
    std::vector<int> a(engine()%50);for(auto& x:a)x=static_cast<int>(engine()%21)-10;
    check(a);if(::testing::Test::HasFatalFailure())return;
  }
}

TEST(CombTrace, NonadjacentStrictSwapsAreNotStableAndRemoveMultipleInversions) {
  const auto t=comb_sort_trace({2,2,1});
  EXPECT_EQ(t.original_indices,(std::vector<std::size_t>{2,1,0}));
  EXPECT_EQ(t.work.swaps,1);EXPECT_EQ(inversions(t.input),2);
  EXPECT_EQ(t.passes[0].original_indices,(std::vector<std::size_t>{2,1,0}));
}

TEST(CombTrace, SortedInputStillExecutesAllWideGapsAndTurtleBubbleCountsAreQuadratic) {
  for(std::size_t n=2;n<=400;++n) {
    std::vector<int> a(n);std::iota(a.begin(),a.end(),0);
    std::size_t gap=n,count=0,passes=0;
    do{gap=std::max(std::size_t{1},gap*10/13);count+=n-gap;++passes;}while(gap>1);
    const auto sorted=comb_sort_work(a);EXPECT_EQ(sorted.key_comparisons,count);EXPECT_EQ(sorted.pass_count,passes);
    EXPECT_EQ(sorted.gap_one_passes,1);EXPECT_EQ(sorted.swaps,0);
    std::iota(a.begin(),a.end()-1,1);a.back()=0;const auto turtle=bubble_sort_work(a);
    EXPECT_EQ(turtle.pass_count,n);EXPECT_EQ(turtle.key_comparisons,n*(n-1));EXPECT_EQ(turtle.swaps,n-1);
  }
}

TEST(CombTrace, PositiveOnlinePassCapIncludesTerminalScan) {
  EXPECT_THROW(comb_sort_trace({},0),std::invalid_argument);
  EXPECT_THROW(comb_sort_trace({1},0),std::invalid_argument);
  EXPECT_NO_THROW(comb_sort_trace({},1));
  EXPECT_NO_THROW(comb_sort_trace({1},1));
  EXPECT_NO_THROW(comb_sort_trace({1,2},1));
  EXPECT_THROW(comb_sort_trace({2,1},1),std::length_error);
  EXPECT_NO_THROW(comb_sort_trace({2,1},2));
  EXPECT_NO_THROW(comb_sort_trace({5,3,8,1,9,2,7},5));
  EXPECT_THROW(comb_sort_trace({5,3,8,1,9,2,7},4),std::length_error);
  EXPECT_NO_THROW(comb_sort_trace({3,2,1},std::numeric_limits<std::size_t>::max()));
}

TEST(CombTrace, OriginalSeedFourFortyShuffleWorkloadsMatchFormerCountingLoops) {
  std::mt19937 engine(4);
  for(std::size_t n=10;n<=400;n+=10) {
    std::vector<int> input(n);std::iota(input.begin(),input.end(),0);std::shuffle(input.begin(),input.end(),engine);
    auto a=input;std::size_t gap=n,comb_count=0;bool swapped=true;
    while(gap>1 || swapped) {
      gap=std::max(std::size_t{1},gap*10/13);swapped=false;
      for(std::size_t i=0;i+gap<n;++i){++comb_count;if(a[i]>a[i+gap]){std::swap(a[i],a[i+gap]);swapped=true;}}
    }
    a=input;std::size_t bubble_count=0;swapped=true;
    while(swapped) {
      swapped=false;
      for(std::size_t i=0;i+1<n;++i){++bubble_count;if(a[i]>a[i+1]){std::swap(a[i],a[i+1]);swapped=true;}}
    }
    EXPECT_EQ(comb_sort_work(input).key_comparisons,comb_count);
    EXPECT_EQ(bubble_sort_work(input).key_comparisons,bubble_count);
  }
}

TEST(CombTrace, CallerAndPassSnapshotsOwnIndependentStorage) {
  std::vector<int> a{3,2,1};auto t=comb_sort_trace(a);EXPECT_EQ(a,(std::vector<int>{3,2,1}));
  a[0]=90;t.passes[0].values[0]=80;t.passes[0].original_indices[0]=70;
  EXPECT_EQ(t.input,(std::vector<int>{3,2,1}));EXPECT_EQ(t.work.output,(std::vector<int>{1,2,3}));
  EXPECT_EQ(t.original_indices,(std::vector<std::size_t>{2,1,0}));
  EXPECT_EQ(t.passes[1].values,(std::vector<int>{1,2,3}));EXPECT_EQ(t.passes[1].original_indices,(std::vector<std::size_t>{2,1,0}));
}
