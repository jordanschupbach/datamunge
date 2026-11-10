#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
TEST(NnFlowStudy, ExplicitTriplesPreserveThreeFamiliesAndFinalDtype) {
  auto base=n::from_vector(std::vector<double>{0,0,0,2,1,-10},{2,3}).requires_grad_();
  auto banana=m::FlowStudyPoints(base,0);
  EXPECT_EQ(banana.points.dtype(),n::DType::Float32);
  EXPECT_EQ(banana.components.to_vector<std::int32_t>(),(std::vector<std::int32_t>{-1,-1}));
  EXPECT_EQ(banana.points.to_vector<float>(),(std::vector<float>{0,-.5,2,2.5}));
  auto ring=m::FlowStudyPoints(base,1,n::DType::Float64);
  auto values=ring.points.to_vector<double>();EXPECT_DOUBLE_EQ(values[0],-2);EXPECT_NEAR(values[1],0,3e-16);EXPECT_DOUBLE_EQ(values[2],2.3);EXPECT_DOUBLE_EQ(values[3],.15);
  EXPECT_EQ(ring.components.to_vector<std::int32_t>(),(std::vector<std::int32_t>{4,0}));
  auto checker=m::FlowStudyPoints(base,2,n::DType::Float64);values=checker.points.to_vector<double>();
  EXPECT_EQ(checker.components.to_vector<std::int32_t>(),(std::vector<std::int32_t>{4,0}));
  EXPECT_DOUBLE_EQ(values[0],-1.5);EXPECT_DOUBLE_EQ(values[1],.5);EXPECT_DOUBLE_EQ(values[2],-1.0227501319481793);EXPECT_DOUBLE_EQ(values[3],-1.1586552539314572);
  EXPECT_FALSE(banana.points.requires_grad());EXPECT_FALSE(ring.points.requires_grad());EXPECT_FALSE(checker.components.requires_grad());
}
TEST(NnFlowStudy, ErfcTailsAndRoundedCheckerBoundariesAreNotClipped) {
  // Phi(-10) remains positive although .5*(1+erf(-10/sqrt(2))) rounds to zero.
  auto base=n::from_vector(std::vector<double>{-10,0,.5,10,10,10},{2,3});
  auto out=m::FlowStudyPoints(base,2,n::DType::Float64);
  EXPECT_EQ(out.components.to_vector<std::int32_t>(),(std::vector<std::int32_t>{5,7}));
  auto v=out.points.to_vector<double>();EXPECT_NEAR(v[0],7.619853024160593e-24,2e-37);EXPECT_DOUBLE_EQ(v[2],2);EXPECT_DOUBLE_EQ(v[3],2);
  EXPECT_TRUE(std::isinf(m::FlowStudyLogProbability(n::slice_axis(out.points,0,1,2),2).item()));
  // Banana ignores component coordinates, but still validates the complete input.
  auto banana=m::FlowStudyPoints(base,0);EXPECT_EQ(banana.components.to_vector<std::int32_t>(),(std::vector<std::int32_t>{-1,-1}));
}
TEST(NnFlowStudy, CheckerTruthUsesHalfOpenParityAndLegitimateZeroDensity) {
  auto x=n::from_vector(std::vector<double>{-2,-2,-1,-2,0,0,2,0,-2,2,-1.5,-1.5,-.5,-1.5,-2.0001,-2},{8,2}).requires_grad_();
  auto out=m::FlowStudyLogProbability(x,2);auto v=out.to_vector<double>();
  for(int row:{0,2,5})EXPECT_DOUBLE_EQ(v[row],-std::log(8.));
  for(int row:{1,3,4,6,7}) {EXPECT_TRUE(std::isinf(v[row]));EXPECT_LT(v[row],0);}
  EXPECT_EQ(out.dtype(),n::DType::Float64);EXPECT_FALSE(out.requires_grad());
  auto diagnostic=m::FlowStudySampleDiagnostics(x);EXPECT_DOUBLE_EQ(diagnostic.checkerboard_support_fraction,3./8);
}
TEST(NnFlowStudy, AnalyticBananaAndStableRingTruthDoNotFitEmpiricalComponents) {
  auto x=n::from_vector(std::vector<double>{0,-.5,2,2.5,-1,0},{3,2});
  EXPECT_LT(n::max(n::abs(m::FlowStudyLogProbability(x,0)-m::BananaLogProbability(x))).item(),1e-15);
  auto origin=m::FlowStudyLogProbability(n::zeros({1,2},n::DType::Float64),1).item();
  EXPECT_NEAR(origin,-std::log(2*std::acos(-1.)*.15*.15)-2/(.15*.15),3e-14);
  // Every Gaussian PDF underflows at this finite point; log-sum-exp stays finite.
  auto far=n::from_vector(std::vector<double>{100,100},{1,2});EXPECT_TRUE(std::isfinite(m::FlowStudyLogProbability(far,1).item()));
}
TEST(NnFlowStudy, ModeCountsUseNearestCentersAndInclusiveThreeSigmaRadius) {
  auto x=n::from_vector(std::vector<double>{2,0,-2,0,2,.45,2,std::nextafter(.45,1.),0,0},{5,2});
  auto out=m::FlowStudySampleDiagnostics(x);
  EXPECT_EQ(out.mode_counts.to_vector<std::int32_t>(),(std::vector<std::int32_t>{2,0,0,0,1,0,0,0}));
  EXPECT_EQ(out.modes_covered,2);EXPECT_DOUBLE_EQ(out.fraction_near_modes,3./5);
  EXPECT_DOUBLE_EQ(out.checkerboard_support_fraction,2./5);
  EXPECT_NEAR(out.banana_curvature_moment,(.45+std::nextafter(.45,1.))*3/5,2e-16);
}
TEST(NnFlowStudy, CompleteRingCoverageCanStillHaveLittleConcentratedMass) {
  std::vector<double> values;for(int j=0;j<8;++j) {values.push_back(2*std::cos(2*std::acos(-1.)*j/8));values.push_back(2*std::sin(2*std::acos(-1.)*j/8));}
  values.resize(200,0.);auto out=m::FlowStudySampleDiagnostics(n::from_vector(values,{100,2}));
  EXPECT_EQ(out.modes_covered,8);EXPECT_DOUBLE_EQ(out.fraction_near_modes,.08);EXPECT_EQ(out.mode_counts.to_vector<std::int32_t>(),std::vector<std::int32_t>(8,1));
}
TEST(NnFlowStudy, BothFloatingInputsStridesOwnershipAndNoMutation) {
  n::Tensor points,components,counts,truth;
  for(auto dtype:{n::DType::Float32,n::DType::Float64}) {
    auto base=n::cast(n::from_vector(std::vector<double>{0,0,0,2,1,-1},{2,3}),dtype).requires_grad_();auto before=base.clone();
    auto strided=n::transpose(n::transpose(base,{1,0}).contiguous(),{1,0});
    for(int kind=0;kind<3;++kind) {auto a=m::FlowStudyPoints(base,kind,dtype),b=m::FlowStudyPoints(strided,kind,dtype);EXPECT_TRUE(n::equal_bits(a.points,b.points));EXPECT_TRUE(n::equal_bits(a.components,b.components));points=a.points;components=a.components;truth=m::FlowStudyLogProbability(points,kind);counts=m::FlowStudySampleDiagnostics(points).mode_counts;}
    EXPECT_TRUE(n::equal_bits(base,before));
  }
  EXPECT_EQ(points.shape(),(n::Shape{2,2}));EXPECT_EQ(components.shape(),(n::Shape{2}));EXPECT_EQ(counts.shape(),(n::Shape{8}));EXPECT_EQ(truth.shape(),(n::Shape{2}));EXPECT_FALSE(truth.requires_grad());
}
TEST(NnFlowStudy, InvalidGeometryKindsDtypesAndNonfiniteInputsReject) {
  for(auto x:{n::Tensor(),n::zeros({3}),n::zeros({0,3}),n::zeros({1,2}),n::zeros({1,3},n::DType::Int32),n::full({1,3},NAN),n::full({1,3},INFINITY)}) {
    EXPECT_THROW(m::FlowStudyPoints(x,0),std::invalid_argument);
  }
  for(auto x:{n::Tensor(),n::zeros({2}),n::zeros({0,2}),n::zeros({1,3}),n::zeros({1,2},n::DType::Int32),n::full({1,2},NAN),n::full({1,2},INFINITY)}) {
    EXPECT_THROW(m::FlowStudyLogProbability(x,2),std::invalid_argument);
    EXPECT_THROW(m::FlowStudySampleDiagnostics(x),std::invalid_argument);
  }
  for(int kind:{-1,3}) {
    EXPECT_THROW(m::FlowStudyPoints(n::zeros({1,3}),kind),std::invalid_argument);
    EXPECT_THROW(m::FlowStudyLogProbability(n::zeros({1,2}),kind),std::invalid_argument);
  }
  EXPECT_THROW(m::FlowStudyPoints(n::zeros({1,3}),0,n::DType::Int32),std::invalid_argument);
}
TEST(NnFlowStudy, UnsafeArithmeticRejectsWithoutClippingOrMutation) {
  auto large=n::full({1,3},1e200,n::DType::Float64);auto before=large.clone();
  EXPECT_THROW(m::FlowStudyPoints(large,0,n::DType::Float64),std::invalid_argument);
  EXPECT_THROW(m::FlowStudyPoints(n::full({1,3},1e40,n::DType::Float64),1),std::invalid_argument);
  auto x=n::slice_axis(large,1,0,2);
  EXPECT_THROW(m::FlowStudyLogProbability(x,0),std::invalid_argument);
  EXPECT_THROW(m::FlowStudyLogProbability(x,1),std::invalid_argument);
  EXPECT_THROW(m::FlowStudySampleDiagnostics(x),std::invalid_argument);
  EXPECT_TRUE(std::isinf(m::FlowStudyLogProbability(x,2).item()));EXPECT_TRUE(n::equal_bits(large,before));
}
