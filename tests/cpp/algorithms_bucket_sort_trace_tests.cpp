#include <gtest/gtest.h>
#include <datamunge/algorithms/bucket_sort.hpp>
#include <datamunge/algorithms/distribution_sorts.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::algorithms;

// An independent widened rational oracle; production needs no nonstandard integer type.
std::size_t exact_index(int value,int minimum,int maximum,std::size_t count) {
  const auto offset=static_cast<std::uint64_t>(static_cast<std::int64_t>(value)-minimum);
  const auto span=static_cast<std::uint64_t>(static_cast<std::int64_t>(maximum)-minimum)+1;
  return static_cast<std::size_t>(static_cast<unsigned __int128>(offset)*count/span);
}

void verify_trace(const std::vector<int>& input) {
  const auto trace=bucket_sort_trace(input);
  auto expected=input;
  std::sort(expected.begin(),expected.end());
  ASSERT_EQ(trace.input,input);
  ASSERT_EQ(trace.output,expected);
  ASSERT_EQ(bucket_sort(input),expected);
  const bool shortcut=input.size()<=1 || expected.front()==expected.back();
  ASSERT_EQ(trace.shortcut,shortcut);
  if (input.empty()) {EXPECT_EQ(trace.span,0);EXPECT_EQ(trace.minimum,0);EXPECT_EQ(trace.maximum,0);}
  else {
    EXPECT_EQ(trace.minimum,expected.front());EXPECT_EQ(trace.maximum,expected.back());
    EXPECT_EQ(trace.span,static_cast<std::uint64_t>(static_cast<std::int64_t>(expected.back())-expected.front())+1);
  }
  if (shortcut) {
    EXPECT_EQ(trace.bucket_count,0);EXPECT_TRUE(trace.buckets.empty());EXPECT_TRUE(trace.bucket_of_input.empty());
    EXPECT_EQ(trace.nonempty_buckets,0);EXPECT_EQ(trace.max_occupancy,0);EXPECT_EQ(trace.occupancy_square_sum,0);
    EXPECT_EQ(trace.key_comparisons,0);EXPECT_EQ(trace.shifts,0);EXPECT_EQ(trace.placements,0);
    return;
  }
  ASSERT_EQ(trace.bucket_count,input.size());ASSERT_EQ(trace.buckets.size(),input.size());
  ASSERT_EQ(trace.bucket_of_input.size(),input.size());
  std::vector<std::vector<int>> scatter(input.size());
  for (std::size_t i=0;i<input.size();++i) {
    const auto bucket=exact_index(input[i],trace.minimum,trace.maximum,input.size());
    ASSERT_EQ(trace.bucket_of_input[i],bucket);
    scatter[bucket].push_back(input[i]);
  }
  std::size_t comparisons=0,shifts=0,placements=0,nonempty=0,maximum=0,squares=0;
  std::vector<int> gathered;
  for (std::size_t b=0;b<scatter.size();++b) {
    const auto& record=trace.buckets[b];
    ASSERT_EQ(record.input,scatter[b]);
    auto sorted=scatter[b];std::sort(sorted.begin(),sorted.end());ASSERT_EQ(record.output,sorted);
    std::size_t local_shifts=0,local_comparisons=0;
    // Prefix ranks give actual insertion shifts/tests without running insertion sort.
    for (std::size_t i=1;i<record.input.size();++i) {
      std::size_t larger=0;
      for (std::size_t j=0;j<i;++j) larger+=record.input[j]>record.input[i];
      local_shifts+=larger;local_comparisons+=larger+(larger<i?1:0);
    }
    const auto size=record.input.size();
    const auto local_placements=size?size-1:0;
    EXPECT_EQ(record.shifts,local_shifts);EXPECT_EQ(record.key_comparisons,local_comparisons);
    EXPECT_EQ(record.placements,local_placements);
    shifts+=local_shifts;comparisons+=local_comparisons;placements+=local_placements;
    nonempty+=size>0;maximum=std::max(maximum,size);squares+=size*size;
    gathered.insert(gathered.end(),record.output.begin(),record.output.end());
  }
  EXPECT_EQ(gathered,expected);EXPECT_EQ(trace.shifts,shifts);EXPECT_EQ(trace.key_comparisons,comparisons);
  EXPECT_EQ(trace.placements,placements);EXPECT_EQ(trace.nonempty_buckets,nonempty);
  EXPECT_EQ(trace.max_occupancy,maximum);EXPECT_EQ(trace.occupancy_square_sum,squares);
  EXPECT_EQ(placements,input.size()-nonempty);
}
}

TEST(BucketSortTrace, PreservesLegacyCallAndOriginalExample) {
  const std::function<std::vector<int>(std::vector<int>)> legacy=bucket_sort;
  const std::vector<int> input{5,3,8,1,9,2,7};
  EXPECT_EQ(legacy(input),(std::vector<int>{1,2,3,5,7,8,9}));
  const auto t=bucket_sort_trace(input);
  EXPECT_EQ(t.bucket_of_input,(std::vector<std::size_t>{3,1,5,0,6,0,4}));
  EXPECT_EQ(t.nonempty_buckets,6);EXPECT_EQ(t.max_occupancy,2);EXPECT_EQ(t.occupancy_square_sum,9);
  EXPECT_EQ(t.key_comparisons,1);EXPECT_EQ(t.shifts,0);EXPECT_EQ(t.placements,1);
  verify_trace(input);
}

TEST(BucketSortTrace, ExhaustiveSmallInputsAndActualPrefixRankCounters) {
  std::size_t checked=0;
  for (std::size_t n=0,power=1;n<=7;++n,power*=5) {
    for (std::size_t word=0;word<power;++word) {
      auto digits=word;std::vector<int> input(n);
      for (auto& value:input) {value=static_cast<int>(digits%5)-2;digits/=5;}
      verify_trace(input);++checked;
    }
  }
  EXPECT_EQ(checked,97656);
}

TEST(BucketSortTrace, FullIntegerDomainAndRandomOracle) {
  verify_trace({INT_MAX,INT_MIN,0,-1,1,INT_MIN,INT_MAX});
  verify_trace({INT_MIN,INT_MAX});
  verify_trace({INT_MIN,INT_MIN+1,INT_MAX-1,INT_MAX});
  std::mt19937 rng(20261106);
  std::uniform_int_distribution<int> value(INT_MIN,INT_MAX);
  for (int trial=0;trial<2000;++trial) {
    std::vector<int> input(rng()%151);
    for (auto& x:input) x=value(rng);
    verify_trace(input);
  }
}

TEST(BucketSortTrace, ExactIndexForHugeBucketCountsWithoutAllocation) {
  const auto largest=std::numeric_limits<std::size_t>::max();
  for (const auto count:{std::size_t{1},std::size_t{7},std::size_t{100},largest/2,largest}) {
    for (const auto x:{INT_MIN,INT_MIN+1,-1,0,1,INT_MAX-1,INT_MAX})
      EXPECT_EQ(bucket_sort_index(x,INT_MIN,INT_MAX,count),exact_index(x,INT_MIN,INT_MAX,count));
    EXPECT_EQ(bucket_sort_index(42,42,42,count),0);
  }
  std::mt19937_64 rng(9801);
  std::uniform_int_distribution<int> value(INT_MIN,INT_MAX);
  for (int trial=0;trial<20000;++trial) {
    int low=value(rng),high=value(rng);if (low>high) std::swap(low,high);
    const int x=std::clamp(value(rng),low,high);
    auto count=static_cast<std::size_t>(rng());if (!count) count=1;
    EXPECT_EQ(bucket_sort_index(x,low,high,count),exact_index(x,low,high,count));
  }
}

TEST(BucketSortTrace, IndexBoundariesAndMonotonicity) {
  for (int low=-20;low<=20;++low) for (int high=low;high<=20;++high)
    for (std::size_t count=1;count<=32;++count) {
      std::size_t previous=0;
      for (int x=low;x<=high;++x) {
        const auto index=bucket_sort_index(x,low,high,count);
        ASSERT_EQ(index,exact_index(x,low,high,count));ASSERT_LT(index,count);ASSERT_GE(index,previous);
        previous=index;
      }
    }
}

TEST(BucketSortTrace, EarlyExitsDoNotInventScatterWork) {
  verify_trace({});verify_trace({42});verify_trace({INT_MIN});verify_trace({7,7,7,7});
  const auto t=bucket_sort_trace(std::vector<int>(1000000,INT_MAX));
  EXPECT_TRUE(t.shortcut);EXPECT_TRUE(t.buckets.empty());EXPECT_TRUE(t.bucket_of_input.empty());
  EXPECT_EQ(t.output.size(),1000000);EXPECT_EQ(t.span,1);EXPECT_EQ(t.key_comparisons,0);
}

TEST(BucketSortTrace, EqualOccupancyCanHaveLinearOrQuadraticInsertionWork) {
  for (std::size_t m=0;m<=256;++m) {
    const auto ascending=bucket_sort_trace(bucket_sort_collision_input(m,false));
    const auto descending=bucket_sort_trace(bucket_sort_collision_input(m,true));
    ASSERT_EQ(ascending.output,descending.output);
    EXPECT_EQ(ascending.max_occupancy,m+1);EXPECT_EQ(descending.max_occupancy,m+1);
    EXPECT_EQ(ascending.nonempty_buckets,2);EXPECT_EQ(descending.nonempty_buckets,2);
    EXPECT_EQ(ascending.occupancy_square_sum,(m+1)*(m+1)+1);
    EXPECT_EQ(descending.occupancy_square_sum,ascending.occupancy_square_sum);
    EXPECT_EQ(ascending.key_comparisons,m);EXPECT_EQ(ascending.shifts,0);
    EXPECT_EQ(descending.key_comparisons,m*(m+1)/2);EXPECT_EQ(descending.shifts,m*(m-1)/2);
    EXPECT_EQ(ascending.placements,m);EXPECT_EQ(descending.placements,m);
  }
}

TEST(BucketSortTrace, NarrowIntegerRangeKeepsEqualKeysCheap) {
  std::vector<int> repeated;
  for (int i=0;i<1000;++i) {repeated.push_back(2);repeated.push_back(1);repeated.push_back(0);}
  const auto t=bucket_sort_trace(repeated);
  EXPECT_EQ(t.nonempty_buckets,3);EXPECT_EQ(t.max_occupancy,1000);
  EXPECT_EQ(t.occupancy_square_sum,3000000);EXPECT_EQ(t.shifts,0);EXPECT_EQ(t.key_comparisons,2997);
  EXPECT_EQ(t.bucket_of_input[0],2000);EXPECT_EQ(t.bucket_of_input[1],1000);EXPECT_EQ(t.bucket_of_input[2],0);
  EXPECT_TRUE(std::is_sorted(t.output.begin(),t.output.end()));
}

TEST(BucketSortTrace, WorkloadsOwnTheirRandomStateAndPreserveOutliers) {
  for (const std::size_t n:{0,1,2,100,2048}) {
    const auto a=bucket_sort_workloads(n,1);
    const auto other=bucket_sort_workloads(n,92);
    const auto b=bucket_sort_workloads(n,1);
    EXPECT_EQ(a.uniform,b.uniform);EXPECT_EQ(a.clustered,b.clustered);
    ASSERT_EQ(a.uniform.size(),n);ASSERT_EQ(a.clustered.size(),n);
    if (n>=2) {EXPECT_EQ(a.clustered[0],0);EXPECT_EQ(a.clustered[1],999);}
    for (const auto* sample:{&a.uniform,&a.clustered,&other.uniform,&other.clustered})
      for (const auto x:*sample) {EXPECT_GE(x,0);EXPECT_LE(x,999);}
    verify_trace(a.uniform);verify_trace(a.clustered);
  }
}

TEST(BucketSortTrace, InvalidIndexAndCollisionSizesThrowBeforeAllocation) {
  EXPECT_THROW(bucket_sort_index(0,0,0,0),std::invalid_argument);
  EXPECT_THROW(bucket_sort_index(0,1,0,1),std::invalid_argument);
  EXPECT_THROW(bucket_sort_index(-1,0,9,10),std::invalid_argument);
  EXPECT_THROW(bucket_sort_index(10,0,9,10),std::invalid_argument);
  EXPECT_THROW(bucket_sort_collision_input(std::numeric_limits<std::size_t>::max()),std::length_error);
  EXPECT_THROW(bucket_sort_collision_input(INT_MAX),std::length_error);
  EXPECT_THROW(bucket_sort_collision_input(50000),std::length_error);
  const auto boundary=bucket_sort_collision_input(46339);
  EXPECT_EQ(boundary.size(),46341);EXPECT_EQ(boundary.back(),2147441940);
  EXPECT_THROW(bucket_sort_collision_input(46340),std::length_error);
}
