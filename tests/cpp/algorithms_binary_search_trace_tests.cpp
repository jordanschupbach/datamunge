#include <datamunge/algorithms/binary_search.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <random>

using namespace datamunge::algorithms;

namespace {
void check_trace(const std::vector<int>& a,int key) {
  const auto t=binary_search_trace(a,key);
  const auto plain=binary_search(a,key);
  const auto found=std::find(a.begin(),a.end(),key);
  EXPECT_EQ(t.result.found,found!=a.end());
  EXPECT_EQ(t.result.found,plain.found);
  EXPECT_EQ(t.result.index,plain.index);
  EXPECT_EQ(t.result.probes,plain.probes);
  ASSERT_EQ(t.result.probes,t.probes.size());
  EXPECT_LE(t.probes.size(),std::bit_width(a.size()));
  std::ptrdiff_t lo=0,hi=static_cast<std::ptrdiff_t>(a.size())-1;
  for (std::size_t i=0;i<t.probes.size();++i) {
    const auto& step=t.probes[i];
    EXPECT_EQ(step.lo,lo);
    EXPECT_EQ(step.hi,hi);
    ASSERT_GE(step.index,lo);
    ASSERT_LE(step.index,hi);
    EXPECT_EQ(step.index,lo+(hi-lo)/2);
    EXPECT_EQ(step.value,a[static_cast<std::size_t>(step.index)]);
    if (step.value==key) {
      EXPECT_EQ(i+1,t.probes.size());
      EXPECT_EQ(t.result.index,step.index);
    } else if (step.value<key) lo=step.index+1;
    else hi=step.index-1;
  }
  if (t.result.found) {
    ASSERT_GE(t.result.index,0);
    ASSERT_LT(static_cast<std::size_t>(t.result.index),a.size());
    EXPECT_EQ(a[static_cast<std::size_t>(t.result.index)],key);
  } else {
    EXPECT_EQ(t.result.index,-1);
    EXPECT_GT(lo,hi);
  }
}

void enumerate(std::vector<int>& a,int remaining,int minimum) {
  for (int key=-1;key<=4;++key) check_trace(a,key);
  if (!remaining) return;
  for (int value=minimum;value<4;++value) {
    a.push_back(value);
    enumerate(a,remaining-1,value);
    a.pop_back();
  }
}
}

TEST(BinarySearchTrace,OriginalProbeSequenceAndClosedIntervals) {
  const auto t=binary_search_trace({1,3,5,7,9,11,13,15,17,19},13);
  ASSERT_EQ(t.probes.size(),4);
  const std::vector<std::ptrdiff_t> expected{4,7,5,6};
  for (std::size_t i=0;i<4;++i) EXPECT_EQ(t.probes[i].index,expected[i]);
  EXPECT_EQ(t.result.index,6);
  EXPECT_EQ(t.probes.back().lo,6);
  EXPECT_EQ(t.probes.back().hi,6);
}

TEST(BinarySearchTrace,EmptySingletonDuplicateAndExtremeKeys) {
  const int low=std::numeric_limits<int>::min(),high=std::numeric_limits<int>::max();
  for (const auto& a:std::vector<std::vector<int>>{{},{3},{2,2,2,2},{low,0,high}})
    for (int key:{low,-1,0,1,2,3,4,high}) check_trace(a,key);
  EXPECT_EQ(binary_search({2,2,2,2},2).index,1);
}

TEST(BinarySearchTrace,AllSortedFourSymbolArraysThroughLengthNine) {
  std::vector<int> a;
  enumerate(a,9,0);
}

TEST(BinarySearchTrace,RandomArraysMatchIndependentLinearMembership) {
  std::mt19937 rng(20261106);
  for (int trial=0;trial<3000;++trial) {
    std::vector<int> a(rng()%100);
    for (auto& value:a) value=static_cast<int>(rng()%101)-50;
    std::sort(a.begin(),a.end());
    for (int key:{-51,-25,0,25,51}) check_trace(a,key);
  }
}

TEST(BinarySearchTrace,PerfectTreeProbeSumsMatchClosedForm) {
  for (unsigned depth=1;depth<=12;++depth) {
    const std::size_t n=(std::size_t{1}<<depth)-1;
    std::vector<int> a(n);
    for (std::size_t i=0;i<n;++i) a[i]=2*i+1;
    std::size_t total=0;
    for (int value:a) total+=binary_search(a,value).probes;
    EXPECT_EQ(total,(depth-1)*(n+1)+1);
    for (std::size_t i=0;i<=n;++i) {
      const auto absent=binary_search(a,2*i);
      EXPECT_FALSE(absent.found);
      EXPECT_EQ(absent.probes,depth);
    }
  }
}
