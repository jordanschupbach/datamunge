#include <gtest/gtest.h>
#include <datamunge/algorithms/exchange_sorts.hpp>
#include <algorithm>
#include <climits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
using namespace datamunge::algorithms;
std::size_t inversions(const std::vector<int>& a) {
  std::size_t result=0;
  for(std::size_t i=0;i<a.size();++i)
    for(std::size_t j=i+1;j<a.size();++j)result+=a[i]>a[j];
  return result;
}

void check(const std::vector<int>& input) {
  const auto t=cocktail_shaker_sort_trace(input);
  auto sorted=input;std::stable_sort(sorted.begin(),sorted.end());
  ASSERT_EQ(t.input,input);ASSERT_EQ(t.output,sorted);
  ASSERT_EQ(cocktail_shaker_sort(input),sorted);
  std::vector<std::size_t> ids(input.size());std::iota(ids.begin(),ids.end(),0);
  std::stable_sort(ids.begin(),ids.end(),[&](auto i,auto j){return input[i]<input[j];});
  ASSERT_EQ(t.original_indices,ids);
  ASSERT_EQ(t.swaps,inversions(input));
  ASSERT_LE(t.passes.size(),input.size());
  if(input.size()<2) {ASSERT_TRUE(t.passes.empty());ASSERT_EQ(t.key_comparisons,0);return;}
  ASSERT_FALSE(t.passes.empty());ASSERT_TRUE(t.passes.back().stopped);
  std::size_t lo=0,hi=input.size()-1,comparisons=0,swaps=0;
  auto before=input;
  std::vector<std::size_t> before_ids(input.size());std::iota(before_ids.begin(),before_ids.end(),0);
  for(std::size_t k=0;k<t.passes.size();++k) {
    const auto& p=t.passes[k];
    ASSERT_EQ(p.forward,k%2==0);ASSERT_EQ(p.lo,lo);ASSERT_EQ(p.hi,hi);
    ASSERT_EQ(p.key_comparisons,hi-lo);ASSERT_EQ(p.values.size(),input.size());
    ASSERT_EQ(p.original_indices.size(),input.size());
    // Independently replay the actual adjacent comparator sequence and its identities.
    std::vector<std::size_t> pairs;
    for(std::size_t j=lo;j<hi;++j)pairs.push_back(j);
    if(!p.forward)std::reverse(pairs.begin(),pairs.end());
    std::size_t pass_swaps=0;
    for(const auto j:pairs)if(before[j]>before[j+1]) {
      std::swap(before[j],before[j+1]);std::swap(before_ids[j],before_ids[j+1]);++pass_swaps;
    }
    ASSERT_EQ(p.values,before);ASSERT_EQ(p.original_indices,before_ids);ASSERT_EQ(p.swaps,pass_swaps);
    ASSERT_EQ(p.stopped,p.swaps==0);ASSERT_EQ(p.stopped,k+1==t.passes.size());
    for(std::size_t j=0;j<input.size();++j) {
      ASSERT_EQ(p.values[j],input[p.original_indices[j]]);
      // Equal values never cross, even in an intermediate unsorted state.
      for(std::size_t q=j+1;q<input.size();++q)
        if(p.values[j]==p.values[q])ASSERT_LT(p.original_indices[j],p.original_indices[q]);
    }
    if(p.forward && p.swaps)--hi;
    if(!p.forward)++lo;
    ASSERT_EQ(p.next_lo,lo);ASSERT_EQ(p.next_hi,hi);
    // Sorted prefix/suffix are in their final global positions after every pass.
    for(std::size_t j=0;j<lo;++j)ASSERT_EQ(p.values[j],sorted[j]);
    for(std::size_t j=hi+1;j<input.size();++j)ASSERT_EQ(p.values[j],sorted[j]);
    comparisons+=p.key_comparisons;swaps+=p.swaps;
  }
  ASSERT_EQ(t.key_comparisons,comparisons);ASSERT_EQ(t.swaps,swaps);
  ASSERT_LE(comparisons,input.size()*(input.size()-1)/2);
}
}

TEST(CocktailTrace, OriginalSevenValuePassesAreActualAndIncludeStoppingScan) {
  const auto t=cocktail_shaker_sort_trace({5,3,8,1,9,2,7});
  ASSERT_EQ(t.passes.size(),5);
  const std::vector<std::vector<int>> expected={{3,5,1,8,2,7,9},{1,3,5,2,8,7,9},
    {1,3,2,5,7,8,9},{1,2,3,5,7,8,9},{1,2,3,5,7,8,9}};
  for(std::size_t i=0;i<5;++i) {EXPECT_EQ(t.passes[i].values,expected[i]);EXPECT_EQ(t.passes[i].key_comparisons,6-i);}
  EXPECT_EQ(t.key_comparisons,20);EXPECT_EQ(t.swaps,10);
}

TEST(CocktailTrace, EmptySingletonSortedEqualAndIntegerExtremes) {
  for(const auto& a:std::vector<std::vector<int>>{{},{42},{1,2},{2,1},{0,0,0},
      {INT_MAX,INT_MIN,0,INT_MAX,INT_MIN},{1,2,3,4}})check(a);
}

TEST(CocktailTrace, EveryTernaryInputThroughLengthEightMatchesStableOracle) {
  for(std::size_t n=0,power=1;n<=8;++n,power*=3)
    for(std::size_t code=0;code<power;++code) {
      auto x=code;std::vector<int> a(n);
      for(auto& v:a){v=static_cast<int>(x%3)-1;x/=3;}check(a);
      if(::testing::Test::HasFatalFailure())return;
    }
}

TEST(CocktailTrace, AllEightElementPermutationsRespectPassInvariants) {
  std::vector<int> a(8);std::iota(a.begin(),a.end(),0);
  do {check(a);if(::testing::Test::HasFatalFailure())return;}while(std::next_permutation(a.begin(),a.end()));
}

TEST(CocktailTrace, SeededDuplicateInputsAndStableIdentities) {
  std::mt19937 engine(7751);
  for(int trial=0;trial<1000;++trial) {
    std::vector<int> a(engine()%50);
    for(auto& x:a)x=static_cast<int>(engine()%21)-10;
    check(a);if(::testing::Test::HasFatalFailure())return;
  }
}

TEST(CocktailTrace, SortedAndReverseHaveExactBestAndWorstCounts) {
  for(std::size_t n=0;n<=128;++n) {
    std::vector<int> a(n);std::iota(a.begin(),a.end(),0);
    const auto best=cocktail_shaker_sort_trace(a);
    EXPECT_EQ(best.key_comparisons,n?n-1:0);EXPECT_EQ(best.swaps,0);
    std::reverse(a.begin(),a.end());const auto worst=cocktail_shaker_sort_trace(a);
    EXPECT_EQ(worst.key_comparisons,n*(n? n-1:0)/2);EXPECT_EQ(worst.swaps,worst.key_comparisons);
    EXPECT_EQ(worst.passes.size(),n<2?0:n);
    if(n>=2){EXPECT_EQ(worst.passes.back().key_comparisons,0);EXPECT_TRUE(worst.passes.back().stopped);}
  }
}

TEST(CocktailTrace, TurtleAndRabbitFamiliesDoNotImplyUniformHalfWork) {
  for(std::size_t n=3;n<=100;++n) {
    std::vector<int> turtle(n);std::iota(turtle.begin(),turtle.end()-1,1);turtle.back()=0;
    const auto slow=cocktail_shaker_sort_trace(turtle);
    EXPECT_EQ(slow.key_comparisons,3*n-6);EXPECT_EQ(slow.passes.size(),3);EXPECT_EQ(slow.swaps,n-1);
    auto rabbit=turtle;std::iota(rabbit.begin()+1,rabbit.end(),0);rabbit[0]=static_cast<int>(n-1);
    const auto fast=cocktail_shaker_sort_trace(rabbit);
    EXPECT_EQ(fast.key_comparisons,2*n-3);EXPECT_EQ(fast.passes.size(),2);EXPECT_EQ(fast.swaps,n-1);
  }
}

TEST(CocktailTrace, PositiveOnlineCapCountsEmptyTerminalPass) {
  EXPECT_THROW(cocktail_shaker_sort_trace({},0),std::invalid_argument);
  EXPECT_THROW(cocktail_shaker_sort_trace({1},0),std::invalid_argument);
  EXPECT_NO_THROW(cocktail_shaker_sort_trace({},1));
  EXPECT_NO_THROW(cocktail_shaker_sort_trace({1,2},1));
  EXPECT_THROW(cocktail_shaker_sort_trace({2,1},1),std::length_error);
  const auto pair=cocktail_shaker_sort_trace({2,1},2);
  ASSERT_EQ(pair.passes.size(),2);EXPECT_FALSE(pair.passes.back().forward);
  EXPECT_EQ(pair.passes.back().next_lo,1);EXPECT_EQ(pair.passes.back().next_hi,0);
  EXPECT_EQ(pair.passes.back().key_comparisons,0);
  EXPECT_NO_THROW(cocktail_shaker_sort_trace({5,3,8,1,9,2,7},5));
  EXPECT_THROW(cocktail_shaker_sort_trace({5,3,8,1,9,2,7},4),std::length_error);
  std::vector<int> reverse(1030);std::iota(reverse.rbegin(),reverse.rend(),0);
  EXPECT_THROW(cocktail_shaker_sort_trace(reverse),std::length_error);
  EXPECT_NO_THROW(cocktail_shaker_sort_trace(reverse,reverse.size()));
  const auto legacy=cocktail_shaker_sort(reverse);
  EXPECT_TRUE(std::is_sorted(legacy.begin(),legacy.end()));
}

TEST(CocktailTrace, CallerAndReturnedSnapshotsHaveIndependentOwnership) {
  std::vector<int> a{3,2,1};auto t=cocktail_shaker_sort_trace(a);EXPECT_EQ(a,(std::vector<int>{3,2,1}));
  a[0]=90;t.passes[0].values[0]=80;t.passes[0].original_indices[0]=70;
  EXPECT_EQ(t.input,(std::vector<int>{3,2,1}));EXPECT_EQ(t.output,(std::vector<int>{1,2,3}));
  EXPECT_EQ(t.original_indices,(std::vector<std::size_t>{2,1,0}));
  EXPECT_EQ(t.passes[1].values,(std::vector<int>{1,2,3}));
  EXPECT_EQ(t.passes[1].original_indices,(std::vector<std::size_t>{2,1,0}));
}
