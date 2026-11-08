#include <datamunge/algorithms/bitonic_sort.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <numeric>
#include <random>
#include <set>

using namespace datamunge::algorithms;

TEST(BitonicNetwork,PowerOfTwoComparatorAndStageCountsAreExact) {
  for (unsigned p=0;p<=10;++p) {
    const std::size_t n=std::size_t{1}<<p;
    const auto network=bitonic_network(n);
    EXPECT_EQ(network.wires,n);
    EXPECT_EQ(network.comparators.size(),(n/2)*(p*(p+1)/2));
    EXPECT_EQ(network.stages,p*(p+1)/2);
  }
  EXPECT_TRUE(bitonic_network(0).comparators.empty());
  EXPECT_EQ(bitonic_network(0).stages,0);
}

TEST(BitonicNetwork,ArbitraryLengthStagesHaveDisjointInRangeWires) {
  for (std::size_t n=0;n<=128;++n) {
    const auto network=bitonic_network(n);
    std::vector<std::set<std::size_t>> used(network.stages);
    std::size_t last=0;
    for (const auto& c:network.comparators) {
      EXPECT_LT(c.first,c.second);
      EXPECT_LT(c.second,n);
      ASSERT_LT(c.stage,network.stages);
      EXPECT_GE(c.stage,last);last=c.stage;
      EXPECT_TRUE(used[c.stage].insert(c.first).second);
      EXPECT_TRUE(used[c.stage].insert(c.second).second);
    }
    const auto p=n?std::bit_width(n-1):0;
    EXPECT_EQ(network.stages,p*(p+1)/2);
  }
}

TEST(BitonicNetwork,ZeroOnePrincipleExhaustiveThroughSixteenWires) {
  for (std::size_t n=0;n<=16;++n) {
    const auto network=bitonic_network(n);
    for (unsigned mask=0;mask<(1U<<n);++mask) {
      std::vector<int> values(n);
      for (std::size_t j=0;j<n;++j) values[j]=(mask>>j)&1;
      auto expected=values;
      bitonic_sort(values);
      ASSERT_TRUE(std::is_sorted(values.begin(),values.end())) << n << ',' << mask;
      EXPECT_EQ(std::accumulate(values.begin(),values.end(),0),std::accumulate(expected.begin(),expected.end(),0));
      for (const auto& c:network.comparators) {
        const auto low=std::min(expected[c.first],expected[c.second]);
        const auto high=std::max(expected[c.first],expected[c.second]);
        expected[c.first]=c.ascending?low:high;
        expected[c.second]=c.ascending?high:low;
      }
      ASSERT_EQ(expected,values);
    }
  }
}

TEST(BitonicNetwork,TraceHasActualPairsAndMatchesUnrecordedSorting) {
  std::mt19937 rng(20261106);
  for (int trial=0;trial<1000;++trial) {
    std::vector<int> input(rng()%100);
    for (auto& value:input) value=static_cast<int>(rng()%41)-20;
    const auto trace=bitonic_sort_trace(input);
    auto plain=input,reference=input,replay=input;
    bitonic_sort(plain);std::sort(reference.begin(),reference.end());
    EXPECT_EQ(trace.input,input);EXPECT_EQ(trace.output,plain);EXPECT_EQ(plain,reference);
    EXPECT_EQ(trace.steps.size(),bitonic_network(input.size()).comparators.size());
    for (const auto& step:trace.steps) {
      const auto& c=step.comparator;
      EXPECT_EQ(replay[c.first],step.first_before);EXPECT_EQ(replay[c.second],step.second_before);
      EXPECT_EQ(step.swapped,c.ascending==(step.first_before>step.second_before));
      if (step.swapped) std::swap(replay[c.first],replay[c.second]);
      EXPECT_EQ(replay[c.first],step.first_after);EXPECT_EQ(replay[c.second],step.second_after);
    }
    EXPECT_EQ(replay,reference);
  }
}

TEST(BitonicNetwork,TopologyIsIndependentOfValuesButSwapsAreNot) {
  const auto a=bitonic_sort_trace({0,1,2,3,4,5,6,7});
  const auto b=bitonic_sort_trace({7,6,5,4,3,2,1,0});
  ASSERT_EQ(a.steps.size(),b.steps.size());
  bool different=false;
  for (std::size_t i=0;i<a.steps.size();++i) {
    EXPECT_EQ(a.steps[i].comparator.first,b.steps[i].comparator.first);
    EXPECT_EQ(a.steps[i].comparator.second,b.steps[i].comparator.second);
    EXPECT_EQ(a.steps[i].comparator.ascending,b.steps[i].comparator.ascending);
    EXPECT_EQ(a.steps[i].comparator.stage,b.steps[i].comparator.stage);
    different|=a.steps[i].swapped!=b.steps[i].swapped;
  }
  EXPECT_TRUE(different);
}

TEST(BitonicNetwork,GenericTypesAndHistoricalEqualDescendingExchange) {
  std::vector<double> values{1.5,-3.,2.,2.,0.};
  bitonic_sort(values);EXPECT_TRUE(std::is_sorted(values.begin(),values.end()));
  struct Tagged { int key,id;bool operator>(const Tagged& other) const{return key>other.key;} };
  std::vector<Tagged> tagged{{0,0},{0,1},{0,2},{0,3}};
  bitonic_sort(tagged);
  EXPECT_EQ(tagged[0].id,1);EXPECT_EQ(tagged[1].id,0);
  const auto equal=bitonic_sort_trace({1,1,1,1});
  EXPECT_TRUE(equal.steps.front().swapped);
  EXPECT_FALSE(equal.steps.front().comparator.ascending);
}
