#include <datamunge/algorithms/borwein_pi.hpp>
#include <gtest/gtest.h>
#include <array>
#include <cfenv>
#include <climits>
#include <numbers>
namespace bp=datamunge::algorithms;
namespace {
// Independent AGM lower bound: BB4(n) equals GL lower(2n) in exact arithmetic.
long double agm_lower(int n) {
    long double a=1,b=1/std::sqrt(2.L),s=.25L,weight=1;
    for (int k=0;k<2*n;++k) {
        const auto next=(a+b)/2;
        s-=weight*(a-next)*(a-next);
        b=std::sqrt(a*b);a=next;weight*=2;
    }
    const auto next_a=(a+b)/2;
    return next_a*next_a/s;
}
}
TEST(BorweinTrace, OriginalSignaturesAndShortDirectValuesAreRetained) {
    auto scalar=bp::borwein_pi;
    auto sequence=bp::borwein_pi_sequence;
    const std::array<double,5> original{2.9142135623730985,3.1415926462135468,
      3.1415926535897918,3.1415926535897918,3.1415926535897918};
    const auto values=sequence(4);
    ASSERT_EQ(values.size(),original.size());
    for (int n=0;n<5;++n) {
        EXPECT_DOUBLE_EQ(scalar(n),original[n]);EXPECT_DOUBLE_EQ(values[n],original[n]);
    }
}
TEST(BorweinTrace, NegativeRequestsKeepTheHistoricalSeedClamp) {
    for (auto mode:{bp::BorweinEvaluation::direct,bp::BorweinEvaluation::rationalised})
      for (int n:{INT_MIN,-100,-1,0}) {
        const auto trace=bp::borwein_pi_trace(n,mode);
        EXPECT_EQ(trace.requested_iterations,0);EXPECT_EQ(trace.iterations_performed,0);
        EXPECT_TRUE(trace.steps.empty());EXPECT_FALSE(trace.stagnated);
        EXPECT_EQ(trace.estimate,bp::borwein_pi(0));
        EXPECT_EQ(bp::borwein_pi_trace(n,mode).estimate,trace.initial_estimate);
        EXPECT_EQ(bp::borwein_pi_sequence(n),(std::vector<double>{trace.initial_estimate}));
      }
}
TEST(BorweinTrace, ScalarSequenceAndActualTraceAgreeAtEveryPrefix) {
    for (auto mode:{bp::BorweinEvaluation::direct,bp::BorweinEvaluation::rationalised})
      for (int n=0;n<=32;++n) {
        const auto trace=bp::borwein_pi_trace(n,mode);
        const auto values=bp::borwein_pi_sequence(n);
        ASSERT_EQ(values.size(),std::size_t(n)+1);
        EXPECT_EQ(values.front(),trace.initial_estimate);
        if (mode==bp::BorweinEvaluation::direct) EXPECT_EQ(values.back(),trace.estimate);
        if (mode==bp::BorweinEvaluation::direct) EXPECT_EQ(trace.estimate,bp::borwein_pi(n));
        EXPECT_EQ(trace.evaluation,mode);EXPECT_EQ(trace.requested_iterations,n);
        EXPECT_EQ(trace.steps.size(),trace.iterations_performed);
        EXPECT_LE(trace.iterations_performed,n);EXPECT_LE(trace.iterations_performed,5);
        for (const auto& step:trace.steps) {
            if (mode==bp::BorweinEvaluation::direct) EXPECT_EQ(values[step.iteration],step.estimate);
            EXPECT_EQ(step.estimate,bp::borwein_pi_trace(step.iteration,mode).estimate);
        }
        if (trace.stagnated) {
            ASSERT_FALSE(trace.steps.empty());EXPECT_TRUE(trace.steps.back().unchanged);
            if (mode==bp::BorweinEvaluation::direct)
                for (std::size_t k=trace.iterations_performed;k<values.size();++k) EXPECT_EQ(values[k],trace.estimate);
        }
      }
}
TEST(BorweinTrace, RecordsContainActualRootCorrectionAndStateTransitions) {
    for (auto mode:{bp::BorweinEvaluation::direct,bp::BorweinEvaluation::rationalised}) {
        const auto trace=bp::borwein_pi_trace(4,mode);
        double y=trace.initial_y,a=trace.initial_inverse_pi;
        int iteration=0;
        for (const auto& row:trace.steps) {
            EXPECT_EQ(row.iteration,++iteration);EXPECT_EQ(row.previous_y,y);
            EXPECT_EQ(row.y_fourth,y*y*y*y);
            EXPECT_EQ(row.fourth_root,std::pow(1-row.y_fourth,.25));
            EXPECT_EQ(row.scale,std::ldexp(1.,2*(iteration-1)+3));
            EXPECT_EQ(row.correction,row.scale*row.y*(1+row.y+row.y*row.y));
            EXPECT_EQ(row.root_rounded_to_one,row.fourth_root==1 && row.y_fourth>0);
            EXPECT_EQ(row.unchanged,row.inverse_pi==a);
            EXPECT_EQ(row.estimate,1/row.inverse_pi);
            EXPECT_TRUE(std::isfinite(row.estimate));EXPECT_GT(row.inverse_pi,0);
            EXPECT_GE(row.y,0);EXPECT_LT(row.y,y);
            y=row.y;a=row.inverse_pi;
        }
        EXPECT_EQ(iteration,trace.iterations_performed);
    }
}
TEST(BorweinTrace, RationalisationPreservesTinyYWithoutPromisingBetterPi) {
    const auto direct=bp::borwein_pi_trace(4);
    const auto rationalised=bp::borwein_pi_trace(4,bp::BorweinEvaluation::rationalised);
    ASSERT_TRUE(direct.stagnated);ASSERT_TRUE(rationalised.stagnated);
    const auto& d=direct.steps.back();const auto& r=rationalised.steps.back();
    EXPECT_TRUE(d.root_rounded_to_one);EXPECT_TRUE(r.root_rounded_to_one);
    EXPECT_EQ(d.fourth_root,1);EXPECT_EQ(r.fourth_root,1);
    EXPECT_EQ(d.y,0);EXPECT_GT(r.y,0);EXPECT_GT(r.correction,0);
    EXPECT_EQ(direct.initial_inverse_pi,rationalised.initial_inverse_pi);
    EXPECT_EQ(direct.initial_y,rationalised.initial_y);
    EXPECT_LT(std::abs(direct.estimate-std::numbers::pi),1e-14);
    EXPECT_LT(std::abs(rationalised.estimate-std::numbers::pi),1e-14);
}
TEST(BorweinTrace, HugeScalarRequestsStayFiniteWithoutOverflowOrInvalidArithmetic) {
    for (auto mode:{bp::BorweinEvaluation::direct,bp::BorweinEvaluation::rationalised}) {
        const auto plateau=bp::borwein_pi_trace(4,mode).estimate;
        for (int n:{512,1024,1000000,INT_MAX}) {
            std::feclearexcept(FE_ALL_EXCEPT);
            const auto result=mode==bp::BorweinEvaluation::direct?bp::borwein_pi(n):bp::borwein_pi_trace(n,mode).estimate;
            EXPECT_EQ(std::fetestexcept(FE_INVALID|FE_OVERFLOW),0);
            EXPECT_TRUE(std::isfinite(result));EXPECT_EQ(result,plateau);
            const auto trace=bp::borwein_pi_trace(n,mode);
            EXPECT_EQ(trace.requested_iterations,n);EXPECT_EQ(trace.estimate,result);
            EXPECT_TRUE(trace.stagnated);EXPECT_LE(trace.steps.size(),5u);
        }
    }
}
TEST(BorweinTrace, LongSequencesPadThePlateauWithoutLateNaNs) {
    {
        const auto trace=bp::borwein_pi_trace(2048);
        const auto values=bp::borwein_pi_sequence(2048);
        ASSERT_EQ(values.size(),2049u);
        for (std::size_t k=trace.iterations_performed;k<values.size();++k) {
            EXPECT_EQ(values[k],trace.estimate);EXPECT_TRUE(std::isfinite(values[k]));
        }
    }
}
TEST(BorweinTrace, InvalidEvaluationModesAreRejectedConsistently) {
    for (auto mode:{static_cast<bp::BorweinEvaluation>(-1),static_cast<bp::BorweinEvaluation>(2),static_cast<bp::BorweinEvaluation>(100)}) {
        EXPECT_THROW(bp::borwein_pi_trace(4,mode),std::invalid_argument);
        EXPECT_THROW(bp::borwein_pi_trace(-1,mode),std::invalid_argument);
    }
}
TEST(BorweinTrace, EarlyApproximantsAgreeWithIndependentLongDoubleAgm) {
    const long double machin=16*std::atan(1.L/5)-4*std::atan(1.L/239);
    EXPECT_NEAR(machin,std::numbers::pi_v<long double>,1e-18L);
    for (int n=0;n<=2;++n) for (auto mode:{bp::BorweinEvaluation::direct,bp::BorweinEvaluation::rationalised})
        EXPECT_NEAR(bp::borwein_pi_trace(n,mode).estimate,agm_lower(n),8e-15L)<<n;
    EXPECT_GT(machin-agm_lower(0),.22L);EXPECT_LT(machin-agm_lower(1),8e-9L);
}
TEST(BorweinTrace, LogarithmicExactArithmeticModelMatchesIndependentDecimalValues) {
    // Decimal(80) evaluation of equation (29); this does not bound rounded iterates.
    const std::array<double,5> decimal{-0.5303329796394902054,-8.114531111362575900,
      -40.25750361223880585,-170.6355735897276128,-693.9540334736667279};
    for (int n=0;n<5;++n) EXPECT_NEAR(bp::borwein_pi_log10_error_bound(n),decimal[n],2e-13);
    EXPECT_EQ(bp::borwein_pi_log10_error_bound(-1),bp::borwein_pi_log10_error_bound(0));
    EXPECT_EQ(bp::borwein_pi_log10_error_bound(INT_MIN),bp::borwein_pi_log10_error_bound(0));
    EXPECT_LT(bp::borwein_pi_log10_error_bound(4),-600);
    EXPECT_GT(std::abs(bp::borwein_pi(4)-std::numbers::pi),1e-16);
}
TEST(BorweinTrace, LogModelStaysFiniteUntilItsNegativeMagnitudeExceedsDouble) {
    double previous=1;
    for (int n=0;n<=511;++n) {
        const auto value=bp::borwein_pi_log10_error_bound(n);
        EXPECT_TRUE(std::isfinite(value))<<n;EXPECT_LT(value,previous);previous=value;
    }
    for (int n:{512,1024,INT_MAX}) {
        EXPECT_EQ(bp::borwein_pi_log10_error_bound(n),-std::numeric_limits<double>::infinity());
    }
}
