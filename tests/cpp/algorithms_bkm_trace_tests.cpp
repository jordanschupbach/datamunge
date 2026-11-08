#include <datamunge/algorithms/bkm.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

using namespace datamunge::algorithms;

TEST(BkmTrace, DomainEndpointsAreActualFiniteAccumulations) {
    const auto domain=bkm_domain();
    EXPECT_EQ(domain.terms,53);
    EXPECT_NEAR(domain.max_argument,4.768462058062743,1e-14);
    EXPECT_NEAR(domain.max_exponent,std::log(domain.max_argument),2e-15);
    EXPECT_EQ(bkm_ln(1),0);EXPECT_EQ(bkm_exp(0),1);EXPECT_EQ(bkm_exp(-0.),1);
    EXPECT_EQ(bkm_ln(domain.max_argument),domain.max_exponent);
    EXPECT_EQ(bkm_exp(domain.max_exponent),domain.max_argument);
    for (const auto& step:bkm_ln_trace(domain.max_argument).steps) EXPECT_TRUE(step.accepted);
    for (const auto& step:bkm_exp_trace(domain.max_exponent).steps) EXPECT_TRUE(step.accepted);
    EXPECT_THROW(bkm_ln(std::nextafter(1.,0.)),std::domain_error);
    EXPECT_THROW(bkm_ln(std::nextafter(domain.max_argument,10.)),std::domain_error);
    EXPECT_THROW(bkm_exp(std::nextafter(0.,-1.)),std::domain_error);
    EXPECT_THROW(bkm_exp(std::nextafter(domain.max_exponent,10.)),std::domain_error);
}

TEST(BkmTrace, InvalidDomainsAndTermCountsFailExplicitly) {
    for (double value:{-1.,0.,.5,5.,std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(bkm_ln(value),std::domain_error);
        EXPECT_THROW(bkm_ln_trace(value),std::domain_error);
    }
    for (double value:{-1.,2.,std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(bkm_exp(value),std::domain_error);
        EXPECT_THROW(bkm_exp_trace(value),std::domain_error);
    }
    for (int terms:{std::numeric_limits<int>::min(),-1,0,54,std::numeric_limits<int>::max()}) {
        EXPECT_THROW(bkm_ln_trace(3,terms),std::invalid_argument);
        EXPECT_THROW(bkm_exp_trace(1,terms),std::invalid_argument);
    }
}

TEST(BkmTrace, DenseSweepMatchesLongDoubleElementaryFunctions) {
    const auto domain=bkm_domain();
    for (int i=0;i<=10000;++i) {
        const double v=1+(domain.max_argument-1)*i/10000;
        const double y=domain.max_exponent*i/10000;
        EXPECT_LE(std::abs(bkm_ln(v)-std::log(static_cast<long double>(v))),4e-15L)<<v;
        EXPECT_LE(std::abs(bkm_exp(y)-std::exp(static_cast<long double>(y))),8e-15L)<<y;
    }
}

TEST(BkmTrace, ActualStatesReplayAndPrefixesShareTheScalarLoop) {
    const auto domain=bkm_domain();
    std::mt19937_64 rng(716);
    for (bool logarithm:{true,false}) for (int sample=0;sample<100;++sample) {
        const double u=std::generate_canonical<double,53>(rng);
        const double input=logarithm?1+u*(domain.max_argument-1):u*domain.max_exponent;
        const auto trace=logarithm?bkm_ln_trace(input):bkm_exp_trace(input);
        EXPECT_EQ(trace.input,input);EXPECT_EQ(trace.logarithm,logarithm);ASSERT_EQ(trace.steps.size(),53);
        EXPECT_EQ(trace.value,logarithm?bkm_ln(input):bkm_exp(input));
        double product=1,sum=0;
        for (int k=0;k<53;++k) {
            const auto& step=trace.steps[k];EXPECT_EQ(step.shift,k);
            EXPECT_EQ(step.candidate_product,product+std::ldexp(product,-k));
            EXPECT_EQ(step.candidate_log,sum+std::log1p(std::ldexp(1.,-k)));
            EXPECT_EQ(step.accepted,logarithm?step.candidate_product<=input:step.candidate_log<=input);
            EXPECT_EQ(step.product,step.accepted?step.candidate_product:product);
            EXPECT_EQ(step.log_sum,step.accepted?step.candidate_log:sum);
            EXPECT_GE(step.product,product);EXPECT_GE(step.log_sum,sum);
            EXPECT_LE(logarithm?step.product:step.log_sum,input);
            product=step.product;sum=step.log_sum;
            const auto prefix=logarithm?bkm_ln_trace(input,k+1):bkm_exp_trace(input,k+1);
            EXPECT_EQ(prefix.value,logarithm?sum:product);
        }
    }
}

TEST(BkmTrace, ExactProductThreePlateausAfterTwoAttempts) {
    const auto trace=bkm_ln_trace(3);
    ASSERT_TRUE(trace.steps[0].accepted);ASSERT_TRUE(trace.steps[1].accepted);
    EXPECT_EQ(trace.steps[0].product,2);EXPECT_EQ(trace.steps[1].product,3);
    for (std::size_t k=2;k<trace.steps.size();++k) {
        EXPECT_FALSE(trace.steps[k].accepted);
        EXPECT_EQ(trace.steps[k].product,3);
        EXPECT_EQ(trace.steps[k].log_sum,trace.steps[1].log_sum);
    }
    const double boundary=std::log1p(1.);
    const auto equal=bkm_exp_trace(boundary);
    EXPECT_TRUE(equal.steps.front().accepted);EXPECT_EQ(equal.value,2);
    const auto below=bkm_exp_trace(std::nextafter(boundary,0.));
    EXPECT_FALSE(below.steps.front().accepted);
}

TEST(BkmTrace, TruncationEnvelopeHoldsUntilDoubleRoundoff) {
    const auto domain=bkm_domain();
    for (int i=0;i<=100;++i) for (int terms=1;terms<=53;++terms) {
        const double v=1+(domain.max_argument-1)*i/100;
        const double y=domain.max_exponent*i/100;
        const auto ln=bkm_ln_trace(v,terms),ex=bkm_exp_trace(y,terms);
        const long double tail=std::ldexp(1.L,1-terms);
        const long double ln_residual=std::log(static_cast<long double>(v))-ln.value;
        const long double ex_residual=y-std::log(static_cast<long double>(ex.value));
        EXPECT_GE(ln_residual,-4e-15L);EXPECT_LE(ln_residual,tail+4e-15L);
        EXPECT_GE(ex_residual,-4e-15L);EXPECT_LE(ex_residual,tail+4e-15L);
    }
}
