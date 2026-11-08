#include <datamunge/algorithms/birkhoff_interpolation.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

using namespace datamunge::algorithms;

TEST(BirkhoffChecked,InspectableDerivativeMatrix) {
  const auto matrix=birkhoff_matrix({{-1,0,-2},{-1,2,-18},{1,0,10},{1,2,30}});
  EXPECT_EQ(matrix,(std::vector<std::vector<double>>{{1,-1,1,-1},{0,0,2,-6},
                                                   {1,1,1,1},{0,0,2,6}}));
  EXPECT_TRUE(birkhoff_matrix({}).empty());
}

TEST(BirkhoffChecked,OriginalCubicAndDerivativeConstraints) {
  const std::vector<BirkhoffConstraint> conditions{{-1,0,-2},{-1,2,-18},{1,0,10},{1,2,30}};
  const auto c=birkhoff_polynomial(conditions);
  ASSERT_EQ(c.size(),4);
  for (std::size_t k=0;k<4;++k) EXPECT_NEAR(c[k],k+1,1e-13);
  for (const auto& condition:conditions)
    EXPECT_NEAR(birkhoff_eval_derivative(c,condition.node,condition.order),condition.value,1e-12);
}

TEST(BirkhoffChecked,ScalingDoesNotRejectATinyAffineCoordinate) {
  const auto c=birkhoff_polynomial({{0,0,1},{1e-16,0,2}});
  ASSERT_EQ(c.size(),2);
  EXPECT_DOUBLE_EQ(c[0],1);
  EXPECT_NEAR(c[1]/1e16,1,1e-14);
  EXPECT_NEAR(birkhoff_eval(c,1e-16),2,1e-14);
}

TEST(BirkhoffChecked,NonpoisedIncludesBothInconsistentAndUnderdeterminedData) {
  EXPECT_TRUE(birkhoff_polynomial({{-1,0,0},{1,0,0},{0,1,0}}).empty());
  EXPECT_TRUE(birkhoff_polynomial({{-1,0,0},{1,0,0},{0,1,1}}).empty());
  EXPECT_TRUE(birkhoff_polynomial({{0,1,0},{-1,2,0},{1,2,0}}).empty());
  EXPECT_TRUE(birkhoff_polynomial({{0,0,1},{0,0,1}}).empty());
  EXPECT_TRUE(birkhoff_polynomial({{0,0,1},{0,0,2}}).empty());
  EXPECT_TRUE(birkhoff_polynomial({{0,5,0}}).empty());
}

TEST(BirkhoffChecked,AnalyticNearSingularityMatchesQuadraticFamily) {
  for (double t:{1.,.5,.1,.01,.001,1e-6}) {
    const auto c=birkhoff_polynomial({{-1,0,0},{1,0,0},{t,1,1}});
    ASSERT_EQ(c.size(),3);
    EXPECT_NEAR(c[0]*2*t,-1,1e-10);
    EXPECT_NEAR(c[1],0,1e-10);
    EXPECT_NEAR(c[2]*2*t,1,1e-10);
    EXPECT_NEAR(birkhoff_eval_derivative(c,t,1),1,1e-10);
  }
  EXPECT_TRUE(birkhoff_polynomial({{-1,0,0},{1,0,0},{1e-16,1,1}}).empty());
}

TEST(BirkhoffChecked,DerivativeEvaluationAgainstIndependentMonomialFormula) {
  std::mt19937 rng(20261106);
  for (int trial=0;trial<300;++trial) {
    std::vector<double> coefficients(1+rng()%8);
    for (auto& value:coefficients) value=static_cast<int>(rng()%11)-5;
    for (double x:{-1.,-.25,0.,.25,1.}) for (int order=0;order<10;++order) {
      long double expected=0;
      for (std::size_t k=order;k<coefficients.size();++k) {
        long double falling=1;
        for (int j=0;j<order;++j) falling*=k-j;
        expected+=coefficients[k]*falling*std::pow(static_cast<long double>(x),static_cast<int>(k)-order);
      }
      EXPECT_NEAR(birkhoff_eval_derivative(coefficients,x,order),static_cast<double>(expected),1e-9);
      if (order==0) EXPECT_DOUBLE_EQ(birkhoff_eval(coefficients,x),birkhoff_eval_derivative(coefficients,x,0));
    }
  }
  EXPECT_EQ(birkhoff_eval({},0),0);
  EXPECT_EQ(birkhoff_eval_derivative({},0,3),0);
}

TEST(BirkhoffChecked,GeneratedLacunaryCubicSchemesRecoverKnownCoefficients) {
  std::mt19937 rng(17);
  for (int trial=0;trial<300;++trial) {
    std::vector<double> expected(4);
    for (auto& value:expected) value=static_cast<int>(rng()%21)-10;
    const double a=expected[0],b=expected[1],c=expected[2],d=expected[3];
    const auto recovered=birkhoff_polynomial({{-1,0,a-b+c-d},{-1,2,2*c-6*d},
                                             {1,0,a+b+c+d},{1,2,2*c+6*d}});
    ASSERT_EQ(recovered.size(),4);
    for (std::size_t k=0;k<4;++k) EXPECT_NEAR(recovered[k],expected[k],1e-12);
  }
}

TEST(BirkhoffChecked,RejectsInvalidConditionsAndEvaluationInputs) {
  EXPECT_THROW(birkhoff_polynomial({{0,-1,1}}),std::invalid_argument);
  EXPECT_THROW(birkhoff_eval_derivative({1},0,-1),std::invalid_argument);
  for (double bad:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(birkhoff_matrix({{bad,0,1}}),std::invalid_argument);
    EXPECT_THROW(birkhoff_polynomial({{0,0,bad}}),std::invalid_argument);
    EXPECT_THROW(birkhoff_eval({1},bad),std::invalid_argument);
    EXPECT_THROW(birkhoff_eval({bad},0),std::invalid_argument);
    EXPECT_THROW(birkhoff_eval_derivative({bad},0,10),std::invalid_argument);
  }
}

TEST(BirkhoffChecked,RejectsNonrepresentableMatrixAndPolynomialArithmetic) {
  const double huge=std::numeric_limits<double>::max();
  EXPECT_THROW(birkhoff_matrix({{huge,0,0},{0,0,0},{1,0,0}}),std::overflow_error);
  EXPECT_THROW(birkhoff_eval({0,huge},2),std::overflow_error);
  EXPECT_THROW(birkhoff_eval_derivative({0,0,huge},1,1),std::overflow_error);
}
