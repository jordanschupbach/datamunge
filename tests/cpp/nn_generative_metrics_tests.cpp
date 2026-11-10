#include <gtest/gtest.h>
#include <datamunge/nn/generative_metrics.hpp>
#include <cmath>
namespace n=datamunge::nn;
namespace m=n::models;

TEST(NnGenerativeMetrics, UnequalPoolsMatchOneDimensionalClosedFormAndSymmetry) {
  auto t=n::DType::Float64;
  auto a=n::tensor({0.,2.},{2,1},t).requires_grad_(),b=n::tensor({1.,3.,5.},{3,1},t).requires_grad_();
  auto o=m::GaussianFrechet(a,b),reverse=m::GaussianFrechet(b,a);
  EXPECT_NEAR(o.fid.item(),4+std::pow(std::sqrt(2.)-2,2),1e-13);
  EXPECT_NEAR(o.fid.item(),reverse.fid.item(),1e-13);
  EXPECT_NEAR(o.mean_term.item(),4,1e-14);
  EXPECT_NEAR(o.trace_a.item(),2,1e-14);
  EXPECT_NEAR(o.trace_b.item(),4,1e-14);
  EXPECT_EQ(o.cross_singular_values.shape(),(n::Shape{2}));
  EXPECT_FALSE(o.fid.requires_grad());EXPECT_FALSE(o.cross_singular_values.requires_grad());
}
TEST(NnGenerativeMetrics, DifferentDistributionsWithEqualMomentsHaveZeroDistance) {
  auto t=n::DType::Float64;double r=std::sqrt(2.);
  auto a=n::tensor({-1.,-1.,-1.,1.,1.,-1.,1.,1.},{4,2},t);
  auto b=n::tensor({r,0.,-r,0.,0.,r,0.,-r},{4,2},t);
  auto o=m::GaussianFrechet(a,b);
  EXPECT_NEAR(o.fid.item(),0,1e-12);
  EXPECT_NEAR(o.mean_term.item(),0,1e-15);
  EXPECT_NEAR(o.trace_a.item(),o.trace_b.item(),1e-13);
  auto same=m::GaussianFrechet(a,a);EXPECT_NEAR(same.fid.item(),0,1e-12);
}
TEST(NnGenerativeMetrics, BalancedConfidentUniformAndCollapsedProbabilityControls) {
  auto t=n::DType::Float64;
  auto onehot=n::tensor({1.,0.,0.,1.},{2,2},t).requires_grad_();
  auto collapsed=n::tensor({1.,0.,1.,0.},{2,2},t);
  EXPECT_NEAR(m::InceptionScore(onehot).item(),2,1e-14);
  EXPECT_NEAR(m::InceptionScore(collapsed).item(),1,1e-14);
  EXPECT_NEAR(m::InceptionScore(n::full({5,4},.25,t)).item(),1,1e-14);
  EXPECT_FALSE(m::InceptionScore(onehot).requires_grad());
  auto split=m::InceptionScoreSplits(n::tile(onehot,{2,1}),2);
  EXPECT_EQ(split.scores.shape(),(n::Shape{2}));
  EXPECT_NEAR(split.mean.item(),2,1e-14);EXPECT_EQ(split.population_std.item(),0);
}
TEST(NnGenerativeMetrics, OrderedUnequalSplitsKeepAllRowsAndUsePopulationSpread) {
  auto t=n::DType::Float64;
  auto p=n::tensor({1.,0.,1.,0.,0.,1.,0.,1.,1.,0.},{5,2},t);
  auto o=m::InceptionScoreSplits(p,3);
  // floor(i*5/3) boundaries: [0,1),[1,3),[3,5), scores1,2,2.
  auto scores=o.scores.to_vector<double>();
  EXPECT_NEAR(scores[0],1,1e-14);EXPECT_NEAR(scores[1],2,1e-14);EXPECT_NEAR(scores[2],2,1e-14);
  EXPECT_NEAR(o.mean.item(),5./3,1e-14);
  EXPECT_NEAR(o.population_std.item(),std::sqrt(2./9),1e-14);
  auto permuted=n::tensor({1.,0.,0.,1.,0.,1.,1.,0.,1.,0.},{5,2},t);
  EXPECT_NE(m::InceptionScoreSplits(permuted,3).mean.item(),o.mean.item());
}
TEST(NnGenerativeMetrics, InvalidFeaturesProbabilitiesAndSplitCountsReject) {
  auto x=n::zeros({2,3});
  EXPECT_THROW(m::GaussianFrechet(n::zeros({1,3}),x),std::invalid_argument);
  EXPECT_THROW(m::GaussianFrechet(x,n::zeros({2,4})),std::invalid_argument);
  EXPECT_THROW(m::GaussianFrechet(x,n::full({2,3},INFINITY)),std::invalid_argument);
  EXPECT_THROW(m::InceptionScore(n::tensor({.5,.8},{1,2})),std::invalid_argument);
  EXPECT_THROW(m::InceptionScore(n::tensor({-.1,1.1},{1,2})),std::invalid_argument);
  EXPECT_THROW(m::InceptionScore(n::ones({3,1})),std::invalid_argument);
  EXPECT_THROW(m::InceptionScoreSplits(n::full({2,3},1./3),3),std::invalid_argument);
  EXPECT_THROW(m::InceptionScoreSplits(n::full({2,3},1./3),0),std::invalid_argument);
}
