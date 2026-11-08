#include <datamunge/algorithms/branch_and_bound.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::int64_t brute(const std::vector<int>& values,const std::vector<int>& weights,int capacity,
                  const std::vector<std::size_t>& order,std::size_t level=0,int initial_weight=0,int initial_value=0) {
  std::int64_t best=initial_value;
  for (std::size_t mask=0;mask<(std::size_t{1}<<(order.size()-level));++mask) {
    std::int64_t weight=initial_weight,value=initial_value;
    for (std::size_t bit=0;bit<order.size()-level;++bit) if (mask&(std::size_t{1}<<bit)) {
      const auto i=order[level+bit];weight+=weights[i];value+=values[i];
    }
    if (weight<=capacity) best=std::max(best,value);
  }
  return best;
}
void check_trace(const std::vector<int>& values,const std::vector<int>& weights,int capacity,const KnapsackTrace& trace) {
  const auto plain=knapsack_branch_and_bound(values,weights,capacity);
  ASSERT_EQ(trace.result.value,plain.value);ASSERT_EQ(trace.result.chosen,plain.chosen);
  ASSERT_EQ(trace.result.nodes_explored,plain.nodes_explored);ASSERT_EQ(trace.original_size,values.size());
  std::vector<std::size_t> all(values.size());std::iota(all.begin(),all.end(),0);
  ASSERT_EQ(trace.result.value,brute(values,weights,capacity,all));
  ASSERT_EQ(trace.result.chosen.size(),values.size());std::int64_t final_weight=0,final_value=0;
  for (std::size_t i=0;i<values.size();++i) {
    ASSERT_TRUE(trace.result.chosen[i]==0 || trace.result.chosen[i]==1);
    if (trace.result.chosen[i]) {final_weight+=weights[i];final_value+=values[i];ASSERT_GT(values[i],0);}
  }
  ASSERT_LE(final_weight,capacity);ASSERT_EQ(final_value,trace.result.value);
  ASSERT_EQ(trace.result.nodes_explored,trace.nodes.size());
  std::size_t positive=0;for (int v:values) positive+=v>0;
  ASSERT_EQ(trace.item_order.size(),positive);
  auto sorted=trace.item_order;std::sort(sorted.begin(),sorted.end());ASSERT_EQ(std::unique(sorted.begin(),sorted.end()),sorted.end());
  for (auto index:trace.item_order) {ASSERT_LT(index,values.size());ASSERT_GT(values[index],0);}
  long long pruned=0,leaves=0,infeasible=0;int incumbent=0;
  std::vector<unsigned> children(trace.nodes.size());
  for (std::size_t index=0;index<trace.nodes.size();++index) {
    const auto& node=trace.nodes[index];ASSERT_LE(node.level,positive);
    if (index==0) {ASSERT_EQ(node.parent,std::numeric_limits<std::size_t>::max());ASSERT_EQ(node.level,0);}
    else {
      ASSERT_LT(node.parent,index);const auto& parent=trace.nodes[node.parent];++children[node.parent];
      ASSERT_EQ(parent.state,KnapsackNodeState::branch);ASSERT_EQ(node.level,parent.level+1);
      ASSERT_EQ(node.item_index,trace.item_order[parent.level]);
      ASSERT_EQ(node.weight,parent.weight+(node.took_item?weights[node.item_index]:0));
      ASSERT_EQ(node.value,parent.value+(node.took_item?values[node.item_index]:0));
    }
    ASSERT_GE(node.weight,0);ASSERT_LE(node.weight,capacity);
    ASSERT_EQ(node.incumbent_before,incumbent);incumbent=std::max(incumbent,node.value);ASSERT_EQ(node.incumbent_after,incumbent);
    const auto& bound=node.bound;ASSERT_GE(bound.fractional_denominator,1);ASSERT_GE(bound.fractional_numerator,0);
    ASSERT_EQ(bound.integer_upper,bound.whole_value+bound.fractional_numerator/bound.fractional_denominator);
    const auto completion=brute(values,weights,capacity,trace.item_order,node.level,node.weight,node.value);
    ASSERT_GE(bound.integer_upper,completion);ASSERT_GE(bound.integer_upper,node.value);
    ASSERT_DOUBLE_EQ(bound.relaxation_value,double(bound.whole_value)+double(bound.fractional_numerator)/bound.fractional_denominator);
    if (node.state==KnapsackNodeState::pruned) {
      ++pruned;ASSERT_LT(node.level,positive);ASSERT_LE(bound.integer_upper,incumbent);ASSERT_LE(completion,incumbent);
    } else if (node.state==KnapsackNodeState::leaf) {++leaves;ASSERT_EQ(node.level,positive);}
    else {
      ASSERT_GT(bound.integer_upper,incumbent);ASSERT_LT(node.level,positive);
      if (weights[trace.item_order[node.level]]>capacity-node.weight) ++infeasible;
    }
  }
  for (std::size_t index=0;index<trace.nodes.size();++index) {
    const auto& node=trace.nodes[index];
    const unsigned expected=node.state==KnapsackNodeState::branch?
        1+(weights[trace.item_order[node.level]]<=capacity-node.weight):0;
    ASSERT_EQ(children[index],expected);
  }
  ASSERT_EQ(incumbent,trace.result.value);ASSERT_EQ(pruned,trace.bound_prunes);ASSERT_EQ(leaves,trace.leaves);ASSERT_EQ(infeasible,trace.infeasible_takes);
}
}
TEST(KnapsackTrace, TinyExampleRecordsActualPrunesAndRationalFloor) {
  const auto trace=knapsack_branch_and_bound_trace({10,8,5},{2,3,3},5);check_trace({10,8,5},{2,3,3},5,trace);
  ASSERT_EQ(trace.nodes.size(),5);EXPECT_EQ(trace.result.value,18);EXPECT_EQ(trace.bound_prunes,3);
  const auto& last=trace.nodes.back();EXPECT_EQ(last.parent,0);EXPECT_FALSE(last.took_item);
  EXPECT_EQ(last.bound.whole_value,8);EXPECT_EQ(last.bound.fractional_numerator,10);EXPECT_EQ(last.bound.fractional_denominator,3);
  EXPECT_EQ(last.bound.integer_upper,11);EXPECT_NEAR(last.bound.relaxation_value,11+1.0/3,1e-14);
}
TEST(KnapsackTrace, OriginalExampleKeepsTheOptimalOriginalIndexChoices) {
  const std::vector<int> values{60,100,120,40,70},weights{10,20,30,15,25};
  const auto trace=knapsack_branch_and_bound_trace(values,weights,50);check_trace(values,weights,50,trace);
  EXPECT_EQ(trace.result.value,220);EXPECT_EQ(trace.result.chosen,(std::vector<int>{0,1,1,0,0}));
}
TEST(KnapsackTrace, All55986SmallInstancesVerifyEveryBoundAndPruneAgainstEnumeration) {
  const std::array<std::pair<int,int>,6> choices{{{-1,0},{0,1},{1,0},{1,1},{2,1},{2,2}}};
  std::size_t checked=0,combinations=1;
  for (std::size_t n=0;n<=5;++n) {
    for (std::size_t code=0;code<combinations;++code) {
      std::vector<int> values(n),weights(n);auto rest=code;
      for (std::size_t i=0;i<n;++i) {const auto [v,w]=choices[rest%choices.size()];rest/=choices.size();values[i]=v;weights[i]=w;}
      for (int capacity=0;capacity<=5;++capacity) {
        check_trace(values,weights,capacity,knapsack_branch_and_bound_trace(values,weights,capacity));++checked;
      }
    }
    combinations*=choices.size();
  }
  EXPECT_EQ(checked,55986);
}
TEST(KnapsackTrace, RandomSignedProfitsAndFreeItemsMatchEnumeration) {
  std::mt19937 engine(31);std::uniform_int_distribution<int> profit(-5,100),weight(0,30),capacity(0,60);
  for (int trial=0;trial<1000;++trial) {
    const auto n=engine()%13;std::vector<int> values(n),weights(n);
    for (auto& v:values) v=profit(engine);for (auto& w:weights) w=weight(engine);
    const auto cap=capacity(engine);check_trace(values,weights,cap,knapsack_branch_and_bound_trace(values,weights,cap));
  }
}
TEST(KnapsackTrace, ExactDensityResolvesProductsDifferingByOneNearIntLimit) {
  const int m=std::numeric_limits<int>::max();
  const std::vector<int> values{m-2,m-1},weights{m-1,m};
  const auto trace=knapsack_branch_and_bound_trace(values,weights,m);check_trace(values,weights,m,trace);
  ASSERT_EQ(trace.item_order,(std::vector<std::size_t>{1,0}));EXPECT_EQ(trace.result.value,m-1);
  EXPECT_EQ(trace.nodes.front().bound.integer_upper,m-1);
}
TEST(KnapsackTrace, SignedValuesAreOptionalAndFreePositiveItemsAreTaken) {
  EXPECT_EQ(knapsack_branch_and_bound({1,-100},{1,1},2).value,1);
  EXPECT_EQ(knapsack_branch_and_bound({7,0,-9,4},{0,0,0,2},0).chosen,(std::vector<int>{1,0,0,0}));
  const auto ignored=knapsack_branch_and_bound_trace({0,-1},{0,1},0);check_trace({0,-1},{0,1},0,ignored);
  ASSERT_EQ(ignored.nodes.size(),1);EXPECT_EQ(ignored.nodes[0].state,KnapsackNodeState::leaf);
  EXPECT_EQ(knapsack_branch_and_bound({1,1},{1,1},1).chosen,(std::vector<int>{1,0}));
}
TEST(KnapsackTrace, InvalidInputRejectedBeforeIndexing) {
  EXPECT_THROW(knapsack_branch_and_bound({1,2},{1},1),std::invalid_argument);
  EXPECT_THROW(knapsack_branch_and_bound_trace({},{0},1),std::invalid_argument);
  EXPECT_THROW(knapsack_branch_and_bound({1},{-1},1),std::invalid_argument);
  EXPECT_THROW(knapsack_branch_and_bound_trace({0},{-1},1),std::invalid_argument);
  EXPECT_THROW(knapsack_branch_and_bound({}, {},-1),std::invalid_argument);
}
TEST(KnapsackTrace, WeightSumsAndValueRangeAreCheckedWithoutRejectingLargeBounds) {
  const int m=std::numeric_limits<int>::max();
  const auto trace=knapsack_branch_and_bound_trace({1,1},{m,m},m);check_trace({1,1},{m,m},m,trace);
  EXPECT_EQ(trace.result.value,1);EXPECT_EQ(knapsack_branch_and_bound({m,m},{1,1},1).value,m);
  EXPECT_THROW(knapsack_branch_and_bound({m,1},{0,0},0),std::overflow_error);
  EXPECT_THROW(knapsack_branch_and_bound_trace({m,m},{1,1},2),std::overflow_error);
  EXPECT_EQ(knapsack_branch_and_bound({std::numeric_limits<int>::min()},{0},0).value,0);
}
TEST(KnapsackTrace, ExplicitStackHandlesFiveThousandFreeItems) {
  const std::size_t n=5000;const auto trace=knapsack_branch_and_bound_trace(std::vector<int>(n,1),std::vector<int>(n,0),0);
  EXPECT_EQ(trace.result.value,n);EXPECT_EQ(trace.result.nodes_explored,2*n+1);
  EXPECT_EQ(std::accumulate(trace.result.chosen.begin(),trace.result.chosen.end(),0),n);
  EXPECT_EQ(trace.bound_prunes,n-1);EXPECT_EQ(trace.leaves,2);
}
TEST(KnapsackTrace, OddCapacityParityGapKeepsManyBranchesAlive) {
  for (int n:{4,6,8,10,12,14,16,18}) {
    const auto result=knapsack_branch_and_bound(std::vector<int>(n,2),std::vector<int>(n,2),n-1);
    EXPECT_EQ(result.value,n-2);
    long long maximum_subsets=1;
    for (int i=1;i<=n/2-1;++i) maximum_subsets=maximum_subsets*(n-i+1)/i;
    EXPECT_GE(result.nodes_explored,maximum_subsets);
  }
  const auto trace=knapsack_branch_and_bound_trace({2,2,2,2},{2,2,2,2},3);
  check_trace({2,2,2,2},{2,2,2,2},3,trace);EXPECT_EQ(trace.nodes.front().bound.integer_upper,3);EXPECT_EQ(trace.result.value,2);
}
TEST(KnapsackTrace, SeededInstancesPreserveLegacyDrawOrderAndMatchIndependentDP) {
  const std::vector<std::size_t> sizes{4,6,8,10,12,14,16,18,20,22,24,26,28};
  const auto instances=knapsack_random_instances(sizes);ASSERT_EQ(instances.size(),sizes.size());
  std::mt19937 rng(777);std::uniform_int_distribution<int> value(1,100),weight(1,30);
  for (std::size_t row=0;row<sizes.size();++row) {
    const auto& instance=instances[row];std::vector<int> expected_values(sizes[row]),expected_weights(sizes[row]);
    for (auto& v:expected_values) v=value(rng);for (auto& w:expected_weights) w=weight(rng);
    ASSERT_EQ(instance.values,expected_values);ASSERT_EQ(instance.weights,expected_weights);ASSERT_EQ(instance.capacity,5*sizes[row]);
    std::vector<int> dp(instance.capacity+1);
    for (std::size_t item=0;item<sizes[row];++item)
      for (int c=instance.capacity;c>=instance.weights[item];--c)
        dp[c]=std::max(dp[c],dp[c-instance.weights[item]]+instance.values[item]);
    const auto result=knapsack_branch_and_bound(instance.values,instance.weights,instance.capacity);EXPECT_EQ(result.value,dp.back());
  }
  EXPECT_TRUE(knapsack_random_instances({}).empty());EXPECT_TRUE(knapsack_random_instances({0})[0].values.empty());
}
TEST(KnapsackTrace, GeneratorValidatesBeforeAllocatingOrMultiplying) {
  EXPECT_THROW(knapsack_random_instances({4},-1),std::invalid_argument);
  EXPECT_THROW(knapsack_random_instances({},5,777,0),std::invalid_argument);
  EXPECT_THROW(knapsack_random_instances({},5,777,100,0),std::invalid_argument);
  EXPECT_THROW(knapsack_random_instances({std::numeric_limits<std::size_t>::max()}),std::overflow_error);
  EXPECT_THROW(knapsack_random_instances({std::numeric_limits<std::size_t>::max()},0),std::length_error);
}
