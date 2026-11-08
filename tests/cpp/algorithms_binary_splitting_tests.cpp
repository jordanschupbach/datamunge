#include <datamunge/algorithms/binary_splitting.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace datamunge::algorithms;

namespace {
// Independent sequential accumulation of the normalised subtree fraction.
std::pair<std::uint64_t,std::uint64_t> sequential(int begin,int end) {
  std::uint64_t p=0,q=1;
  for (int k=begin;k<end;++k) {
    const std::uint64_t factor=k==0?1:k;
    p=p*factor+1;
    q*=factor;
  }
  return {p,q};
}
}

TEST(BinarySplitting,AllSupportedRootsMatchSequentialExactArithmetic) {
  for (int n=1;n<=21;++n) {
    const auto [p,q]=sequential(0,n);
    const auto result=binary_splitting_e(n);
    EXPECT_EQ(result.num,p);
    EXPECT_EQ(result.den,q);
    EXPECT_DOUBLE_EQ(result.value,static_cast<double>(p)/static_cast<double>(q));
  }
}

TEST(BinarySplitting,PreservesTheLegacyLowerClamp) {
  for (int n:{std::numeric_limits<int>::min(),-100,-1,0,1}) {
    const auto result=binary_splitting_e(n);
    EXPECT_EQ(result.num,1);
    EXPECT_EQ(result.den,1);
    EXPECT_DOUBLE_EQ(result.value,1);
    const auto trace=binary_splitting_e_trace(n);
    EXPECT_EQ(trace.terms,1);
    EXPECT_EQ(trace.nodes.size(),1);
  }
}

TEST(BinarySplitting,RejectsOutOfRangeCountsBeforeRecursion) {
  for (int n:{22,23,100,1000000000,std::numeric_limits<int>::max()}) {
    EXPECT_THROW(binary_splitting_e(n),std::overflow_error);
    EXPECT_THROW(binary_splitting_e_trace(n),std::overflow_error);
  }
}

TEST(BinarySplitting,RawFractionsAreExactButNotAlwaysReduced) {
  const auto three=binary_splitting_e(3),four=binary_splitting_e(4);
  EXPECT_EQ(three.num,5);
  EXPECT_EQ(three.den,2);
  EXPECT_EQ(four.num,16);
  EXPECT_EQ(four.den,6);
  EXPECT_EQ(std::gcd(four.num,four.den),2);
  EXPECT_EQ(binary_splitting_e(21).den,2432902008176640000ULL);
}

TEST(BinarySplitting,EveryRecordedNodeMatchesIndependentNormalisedFraction) {
  for (int n=1;n<=21;++n) {
    const auto trace=binary_splitting_e_trace(n);
    ASSERT_EQ(trace.nodes.size(),2*n-1);
    const auto plain=binary_splitting_e(n);
    EXPECT_EQ(trace.result.num,plain.num);
    EXPECT_EQ(trace.result.den,plain.den);
    EXPECT_DOUBLE_EQ(trace.result.value,plain.value);
    std::vector<unsigned> height(trace.nodes.size());
    int leaves=0;
    for (std::size_t i=0;i<trace.nodes.size();++i) {
      const auto& node=trace.nodes[i];
      const auto [p,q]=sequential(node.begin,node.end);
      EXPECT_EQ(node.num,p);
      EXPECT_EQ(node.den,q);
      if (node.end-node.begin==1) {
        ++leaves;
        EXPECT_EQ(node.left,-1);
        EXPECT_EQ(node.right,-1);
      } else {
        ASSERT_GE(node.left,0);
        ASSERT_GE(node.right,0);
        ASSERT_LT(node.left,i);
        ASSERT_LT(node.right,i);
        const auto& left=trace.nodes[node.left];
        const auto& right=trace.nodes[node.right];
        EXPECT_EQ(left.begin,node.begin);
        EXPECT_EQ(left.end,right.begin);
        EXPECT_EQ(right.end,node.end);
        EXPECT_LE(std::abs((left.end-left.begin)-(right.end-right.begin)),1);
        height[i]=1+std::max(height[node.left],height[node.right]);
      }
    }
    EXPECT_EQ(leaves,n);
    EXPECT_EQ(trace.nodes.back().begin,0);
    EXPECT_EQ(trace.nodes.back().end,n);
    EXPECT_EQ(height.back(),std::bit_width(static_cast<unsigned>(n-1)));
  }
}

TEST(BinarySplitting,LeafThreeIsNormalisedRatherThanAnOriginalFactorialTerm) {
  const auto trace=binary_splitting_e_trace(4);
  const auto it=std::find_if(trace.nodes.begin(),trace.nodes.end(),[](const auto& node){
    return node.begin==3 && node.end==4;
  });
  ASSERT_NE(it,trace.nodes.end());
  EXPECT_EQ(it->num,1);
  EXPECT_EQ(it->den,3); // The original 1/3! would instead have denominator 6.
}
