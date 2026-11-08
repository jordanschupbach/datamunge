#include <gtest/gtest.h>
#include <datamunge/algorithms/counting_sort.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;

TEST(CountingTrace, OriginalWorkedArraysAndActualPlacements) {
  const std::vector<std::int64_t> input{2,5,3,0,2,3,0,3};
  const auto t=counting_sort_trace(input,0,5,6,8);
  const auto& a=t.analysis;
  EXPECT_EQ(a.output,(std::vector<std::int64_t>{0,0,2,2,3,3,3,5}));
  EXPECT_EQ(a.original_indices,(std::vector<std::size_t>{3,6,0,4,2,5,7,1}));
  EXPECT_EQ(a.counts,(std::vector<std::size_t>{2,0,2,3,0,1}));
  EXPECT_EQ(a.starts,(std::vector<std::size_t>{0,2,2,4,7,7}));
  EXPECT_EQ(a.ends,(std::vector<std::size_t>{2,2,4,7,7,8}));
  ASSERT_EQ(t.placements.size(),8u);
  auto cursors=a.ends;
  for(std::size_t e=0;e<input.size();++e) {
    const auto& p=t.placements[e];
    EXPECT_EQ(p.input_index,input.size()-1-e);
    EXPECT_EQ(p.key,input[p.input_index]);EXPECT_EQ(p.bucket,static_cast<std::size_t>(p.key));
    EXPECT_EQ(p.cursor_before,cursors[p.bucket]);
    EXPECT_EQ(p.output_index,--cursors[p.bucket]);
    EXPECT_EQ(a.output[p.output_index],p.key);
    EXPECT_EQ(a.original_indices[p.output_index],p.input_index);
  }
  EXPECT_EQ(cursors,a.starts);
  EXPECT_EQ(a.work.range_items,0u);EXPECT_EQ(a.work.count_increments,8u);
  EXPECT_EQ(a.work.prefix_steps,6u);EXPECT_EQ(a.work.placements,8u);
}
TEST(CountingTrace, OriginalUnpackedRecordsKeepArrivalOrder) {
  const auto a=counting_sort_analyze({1,0,2,1,0,1,2,0});
  EXPECT_EQ(a.original_indices,(std::vector<std::size_t>{1,4,7,0,3,5,2,6}));
  EXPECT_EQ(a.work.range_items,8u);
}
TEST(CountingTrace, ExhaustiveTernarySequencesMatchStableIndexOracle) {
  for(std::size_t n=0,power=1;n<=9;++n,power*=3) {
    for(std::size_t encoded=0;encoded<power;++encoded) {
      auto digits=encoded;std::vector<std::int64_t> input(n);
      for(auto& v:input) { v=static_cast<std::int64_t>(digits%3)-1;digits/=3; }
      std::vector<std::size_t> oracle(n);std::iota(oracle.begin(),oracle.end(),0);
      std::stable_sort(oracle.begin(),oracle.end(),[&](auto i,auto j){return input[i]<input[j];});
      const auto t=counting_sort_trace(input,-1,1,3,n);
      ASSERT_EQ(t.analysis.original_indices,oracle);
      for(std::size_t i=0;i<n;++i) ASSERT_EQ(t.analysis.output[i],input[oracle[i]]);
      auto legacy=input;counting_sort(legacy);ASSERT_EQ(legacy,t.analysis.output);
    }
  }
}
TEST(CountingTrace, RandomExplicitAutoLegacyAndEventReplayAgree) {
  std::mt19937_64 rng(20261107);
  for(int trial=0;trial<1000;++trial) {
    std::vector<std::int64_t> input(rng()%400);
    for(auto& v:input) v=static_cast<std::int64_t>(rng()%81)-40;
    auto original=input;
    const auto a=counting_sort_analyze(input);
    const auto t=counting_sort_trace(input,-50,50);
    std::vector<std::size_t> oracle(input.size());std::iota(oracle.begin(),oracle.end(),0);
    std::stable_sort(oracle.begin(),oracle.end(),[&](auto i,auto j){return input[i]<input[j];});
    ASSERT_EQ(a.original_indices,oracle);ASSERT_EQ(t.analysis.original_indices,oracle);
    auto cursor=t.analysis.ends;
    for(const auto& p:t.placements) {
      ASSERT_EQ(p.key,input[p.input_index]);ASSERT_EQ(p.bucket,static_cast<std::size_t>(p.key+50));
      ASSERT_EQ(p.cursor_before,cursor[p.bucket]);ASSERT_EQ(p.output_index,--cursor[p.bucket]);
      ASSERT_EQ(t.analysis.original_indices[p.output_index],p.input_index);
    }
    ASSERT_EQ(cursor,t.analysis.starts);ASSERT_EQ(input,original);
    counting_sort(input,-50,50);ASSERT_EQ(input,a.output);ASSERT_EQ(input,t.analysis.output);
  }
}
TEST(CountingTrace, NarrowRangesAtBothSignedEndpointsAreSafe) {
  const auto low=std::numeric_limits<std::int64_t>::min(),high=std::numeric_limits<std::int64_t>::max();
  for(const auto lo:{low,high-3}) {
    const std::vector<std::int64_t> input{lo+3,lo,lo+1,lo+3};
    const auto t=counting_sort_trace(input,lo,lo+3,4,4);
    EXPECT_EQ(t.analysis.original_indices,(std::vector<std::size_t>{1,2,0,3}));
    EXPECT_EQ(t.analysis.counts,(std::vector<std::size_t>{1,1,0,2}));
    EXPECT_EQ(counting_sort_analyze(input).output,t.analysis.output);
    auto legacy=input;counting_sort(legacy);EXPECT_EQ(legacy,t.analysis.output);
  }
  EXPECT_THROW(counting_sort_analyze({low,high}),std::length_error);
  EXPECT_THROW(counting_sort_trace({0},low,high),std::length_error);
}
TEST(CountingTrace, CompleteBudgetsAndZeroDisablingAreExact) {
  EXPECT_NO_THROW(counting_sort_trace({1,0,1},0,2,3,3));
  EXPECT_THROW(counting_sort_trace({1,0,1},0,2,2,3),std::length_error);
  EXPECT_THROW(counting_sort_trace({1,0,1},0,2,3,2),std::length_error);
  EXPECT_NO_THROW(counting_sort_trace({1,0,1},0,2,0,0));
  EXPECT_THROW(counting_sort_analyze({0,1},1),std::length_error);
  EXPECT_NO_THROW(counting_sort_analyze({0,1},2));
  EXPECT_THROW(counting_sort_analyze({0},0,1'000'000),std::length_error);
  EXPECT_THROW(counting_sort_trace({0},0,kCountingSortMaxRange,0,0),std::length_error);
}
TEST(CountingTrace, FullValidationPrecedesBudgetsAndLegacyMutation) {
  EXPECT_THROW(counting_sort_trace({0,4},0,3,1,1),std::invalid_argument);
  EXPECT_THROW(counting_sort_analyze({-1},0,kCountingSortMaxRange,1),std::invalid_argument);
  EXPECT_THROW(counting_sort_trace({},2,1,0,0),std::invalid_argument);
  EXPECT_THROW(counting_sort_analyze({},2,1),std::invalid_argument);
  std::vector<std::int64_t> input{3,0,1};const auto copy=input;
  EXPECT_THROW(counting_sort(input,0,2),std::invalid_argument);EXPECT_EQ(input,copy);
  EXPECT_THROW(counting_sort(input,0,kCountingSortMaxRange),std::length_error);EXPECT_EQ(input,copy);
}
TEST(CountingTrace, EmptyAndSingletonConventionsAndOwnedStorage) {
  const auto low=std::numeric_limits<std::int64_t>::min(),high=std::numeric_limits<std::int64_t>::max();
  const auto empty=counting_sort_trace({},low,high,1,1);
  EXPECT_TRUE(empty.analysis.counts.empty());EXPECT_TRUE(empty.placements.empty());
  EXPECT_EQ(empty.analysis.min_value,low);EXPECT_EQ(empty.analysis.max_value,high);
  const auto automatic=counting_sort_analyze({});
  EXPECT_EQ(automatic.min_value,0);EXPECT_EQ(automatic.max_value,0);
  EXPECT_TRUE(automatic.output.empty());EXPECT_EQ(automatic.work.range_items,0u);
  std::vector<std::int64_t> singleton{high};auto a=counting_sort_analyze(singleton);
  EXPECT_EQ(a.counts,(std::vector<std::size_t>{1}));
  EXPECT_EQ(a.original_indices,(std::vector<std::size_t>{0}));
  singleton[0]=low;a.output[0]=0;
  EXPECT_EQ(a.min_value,high);EXPECT_EQ(counting_sort_analyze({high}).output[0],high);
}
