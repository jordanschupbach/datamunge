#include <datamunge/algorithms/cycle_detection.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>

namespace ca=datamunge::algorithms;
namespace {
struct Oracle {std::size_t mu,lambda;std::uint64_t entry;};
Oracle first_repeat(std::uint64_t x,const ca::CycleFunction& f) {
    std::unordered_map<std::uint64_t,std::size_t> seen;
    while (!seen.contains(x)) {seen.emplace(x,seen.size());x=f(x);}
    return {seen.at(x),seen.size()-seen.at(x),x};
}
std::uint64_t decimal_oracle(std::uint64_t value) {
    std::uint64_t sum=0;
    for (char c:std::to_string(value)) sum+=(c-'0')*(c-'0');
    return sum;
}
void check_work(const ca::CycleWork& brent,const ca::CycleWork& floyd,const Oracle& expected) {
    EXPECT_EQ(brent.result.mu,expected.mu);
    EXPECT_EQ(brent.result.lambda,expected.lambda);
    EXPECT_EQ(brent.entry,expected.entry);
    EXPECT_EQ(floyd.result.mu,expected.mu);
    EXPECT_EQ(floyd.result.lambda,expected.lambda);
    EXPECT_EQ(floyd.entry,expected.entry);
    const auto p=std::bit_ceil(std::max(expected.mu+1,expected.lambda));
    const auto t=expected.lambda*std::max(std::size_t{1},(expected.mu+expected.lambda-1)/expected.lambda);
    EXPECT_EQ(brent.detection_evaluations,p-1+expected.lambda);
    EXPECT_EQ(brent.alignment_evaluations,expected.lambda);
    EXPECT_EQ(brent.entry_evaluations,2*expected.mu);
    EXPECT_EQ(brent.period_evaluations,0u);
    EXPECT_EQ(brent.evaluations,p-1+2*expected.lambda+2*expected.mu);
    EXPECT_EQ(brent.comparisons,brent.detection_evaluations+expected.mu+1);
    EXPECT_EQ(floyd.detection_evaluations,3*t);
    EXPECT_EQ(floyd.alignment_evaluations,0u);
    EXPECT_EQ(floyd.entry_evaluations,2*expected.mu);
    EXPECT_EQ(floyd.period_evaluations,expected.lambda);
    EXPECT_EQ(floyd.evaluations,3*t+2*expected.mu+expected.lambda);
    EXPECT_EQ(floyd.comparisons,t+expected.mu+1+expected.lambda);
}
}

TEST(CycleWork, ExhaustiveMapsThroughFiveStatesAgreeWithFirstSeenOracle) {
    std::size_t maps=0,starts=0;
    for (std::size_t n=1;n<=5;++n) {
        std::size_t choices=1;
        for (std::size_t i=0;i<n;++i) choices*=n;
        for (std::size_t code=0;code<choices;++code) {
            auto encoded=code;
            std::vector<std::uint64_t> next(n);
            for (auto& v:next) {v=encoded%n;encoded/=n;}
            const ca::CycleFunction f=[&](std::uint64_t x){return next.at(x);};
            ++maps;
            for (std::uint64_t start=0;start<n;++start) {
                ++starts;
                const auto expected=first_repeat(start,f);
                check_work(ca::brent_cycle_work(start,f),ca::floyd_cycle_work(start,f),expected);
                const auto b=ca::brent_cycle_detection(start,f);
                const auto ff=ca::floyd_cycle_detection(start,f);
                EXPECT_EQ(b.mu,expected.mu);EXPECT_EQ(b.lambda,expected.lambda);
                EXPECT_EQ(ff.mu,expected.mu);EXPECT_EQ(ff.lambda,expected.lambda);
                const auto trace=ca::brent_cycle_trace(start,f);
                EXPECT_EQ(trace.work.entry,expected.entry);
                EXPECT_EQ(trace.work.result.mu,expected.mu);
                EXPECT_EQ(trace.work.result.lambda,expected.lambda);
                for (const auto& row:trace.steps) {
                    auto reference=start,runner=start;
                    for (std::size_t k=0;k<row.reference_index;++k) reference=f(reference);
                    for (std::size_t k=0;k<row.runner_index;++k) runner=f(runner);
                    EXPECT_EQ(row.reference,reference);EXPECT_EQ(row.runner,runner);
                    if (row.phase==ca::CyclePhase::detect) {
                        EXPECT_TRUE(std::has_single_bit(row.power));
                        EXPECT_EQ(row.reference_index,row.power-1);
                        EXPECT_EQ(row.runner_index-row.reference_index,row.offset);
                    } else EXPECT_EQ(row.power,0u);
                }
            }
        }
    }
    EXPECT_EQ(maps,3413u);EXPECT_EQ(starts,16739u);
}

TEST(CycleWork, ControlledTailAndPeriodBoundariesHaveExactPhaseCounts) {
    for (std::size_t mu=0;mu<=64;++mu) for (std::size_t lambda=1;lambda<=64;++lambda) {
        const auto next=ca::rho_successors(mu,lambda);
        const ca::CycleFunction f=[&](std::uint64_t x){return next.at(x);};
        check_work(ca::brent_cycle_work(0,f),ca::floyd_cycle_work(0,f),{mu,lambda,mu});
    }
}

TEST(CycleWork, OriginalRhoPreservesLegacyTransitionOrderAndAddsNoTraceCalls) {
    const auto next=ca::rho_successors(3,4);
    std::vector<std::uint64_t> calls;
    const ca::CycleFunction f=[&](std::uint64_t x){calls.push_back(x);return next.at(x);};
    const std::vector<std::uint64_t> brent_calls{0,1,2,3,4,5,6,0,1,2,3,0,4,1,5,2,6};
    const std::vector<std::uint64_t> floyd_calls{0,0,1,1,2,3,2,4,5,3,6,3,0,4,1,5,2,6,3,4,5,6};
    ca::brent_cycle_detection(0,f);EXPECT_EQ(calls,brent_calls);calls.clear();
    const auto b=ca::brent_cycle_work(0,f);EXPECT_EQ(calls,brent_calls);calls.clear();
    const auto trace=ca::brent_cycle_trace(0,f);EXPECT_EQ(calls,brent_calls);calls.clear();
    ca::floyd_cycle_detection(0,f);EXPECT_EQ(calls,floyd_calls);calls.clear();
    const auto ff=ca::floyd_cycle_work(0,f);EXPECT_EQ(calls,floyd_calls);
    EXPECT_EQ(trace.work.evaluations,b.evaluations);EXPECT_EQ(b.evaluations,17u);EXPECT_EQ(ff.evaluations,22u);
}

TEST(CycleWork, CheckpointEqualityIsNotAMeetingOrAnExtraCall) {
    const auto next=ca::rho_successors(3,4);
    const auto trace=ca::brent_cycle_trace(0,[&](std::uint64_t x){return next.at(x);});
    std::vector<std::size_t> anchors{0},powers{1};
    for (std::size_t i=0;i<trace.steps.size();++i) {
        const auto& row=trace.steps[i];
        if (row.checkpoint) {
            ASSERT_GT(i,0u);ASSERT_LT(i+1,trace.steps.size());
            anchors.push_back(row.reference_index);powers.push_back(row.power);
            EXPECT_EQ(row.reference,row.runner);EXPECT_EQ(row.offset,0u);
            EXPECT_EQ(row.evaluations,trace.steps[i-1].evaluations);
            EXPECT_EQ(trace.steps[i+1].evaluations,row.evaluations+1);
            EXPECT_FALSE(trace.steps[i+1].checkpoint);
        }
    }
    EXPECT_EQ(anchors,(std::vector<std::size_t>{0,1,3}));
    EXPECT_EQ(powers,(std::vector<std::size_t>{1,2,4}));
    const auto& last=trace.steps.back();
    EXPECT_EQ(last.phase,ca::CyclePhase::locate);EXPECT_EQ(last.offset,3u);
    EXPECT_EQ(last.reference,3u);EXPECT_EQ(last.runner,3u);
    EXPECT_EQ(last.reference_index,3u);EXPECT_EQ(last.runner_index,7u);
}

TEST(CycleWork, EveryPositiveBudgetStopsBeforeExceedingItInEveryPhase) {
    const auto next=ca::rho_successors(3,4);
    std::size_t calls=0;
    const ca::CycleFunction f=[&](std::uint64_t x){++calls;return next.at(x);};
    for (std::size_t budget=1;budget<17;++budget) {
        calls=0;EXPECT_THROW(ca::brent_cycle_work(0,f,{budget}),std::runtime_error);
        EXPECT_EQ(calls,budget);
        calls=0;EXPECT_THROW(ca::brent_cycle_trace(0,f,{budget}),std::runtime_error);
        EXPECT_EQ(calls,budget);
    }
    for (std::size_t budget=1;budget<22;++budget) {
        calls=0;EXPECT_THROW(ca::floyd_cycle_work(0,f,{budget}),std::runtime_error);
        EXPECT_EQ(calls,budget);
    }
    calls=0;EXPECT_EQ(ca::brent_cycle_work(0,f,{17}).evaluations,17u);EXPECT_EQ(calls,17u);
    calls=0;EXPECT_EQ(ca::brent_cycle_trace(0,f,{17}).work.evaluations,17u);EXPECT_EQ(calls,17u);
    calls=0;EXPECT_EQ(ca::floyd_cycle_work(0,f,{22}).evaluations,22u);EXPECT_EQ(calls,22u);
}

TEST(CycleWork, IncreasingOrbitCanBeBoundedWithoutWaitingForUnsignedWrap) {
    std::size_t calls=0;
    const ca::CycleFunction f=[&](std::uint64_t x){++calls;return x+1;};
    EXPECT_THROW(ca::brent_cycle_work(0,f,{5}),std::runtime_error);
    EXPECT_EQ(calls,5u);calls=0;
    EXPECT_THROW(ca::brent_cycle_trace(0,f,{5}),std::runtime_error);
    EXPECT_EQ(calls,5u);calls=0;
    EXPECT_THROW(ca::floyd_cycle_work(0,f,{5}),std::runtime_error);
    EXPECT_EQ(calls,5u);
}

TEST(CycleWork, EmptyTransitionsAndCallbackExceptionsPropagate) {
    const ca::CycleFunction empty;
    EXPECT_THROW(ca::brent_cycle_detection(0,empty),std::bad_function_call);
    EXPECT_THROW(ca::floyd_cycle_detection(0,empty),std::bad_function_call);
    EXPECT_THROW(ca::brent_cycle_work(0,empty),std::bad_function_call);
    EXPECT_THROW(ca::floyd_cycle_work(0,empty),std::bad_function_call);
    EXPECT_THROW(ca::brent_cycle_trace(0,empty),std::bad_function_call);
    std::size_t calls=0;
    const ca::CycleFunction throws=[&](std::uint64_t x){if (++calls==4) throw std::domain_error("transition");return x+1;};
    EXPECT_THROW(ca::brent_cycle_detection(0,throws),std::domain_error);
    EXPECT_EQ(calls,4u);calls=0;
    EXPECT_THROW(ca::floyd_cycle_detection(0,throws),std::domain_error);
    EXPECT_EQ(calls,4u);calls=0;
    EXPECT_THROW(ca::brent_cycle_trace(0,throws),std::domain_error);
    EXPECT_EQ(calls,4u);
}

TEST(CycleWork, OrbitPrefixMakesOnlyCallsNeededForStoredStates) {
    const ca::CycleFunction empty;
    EXPECT_TRUE(ca::iterate_states(42,empty,0).empty());
    EXPECT_EQ(ca::iterate_states(42,empty,1),(std::vector<std::uint64_t>{42}));
    EXPECT_THROW(ca::iterate_states(42,empty,2),std::bad_function_call);
    std::vector<std::uint64_t> calls;
    const auto prefix=ca::iterate_states(0,[&](std::uint64_t x){calls.push_back(x);return (x+1)%3;},5);
    EXPECT_EQ(prefix,(std::vector<std::uint64_t>{0,1,2,0,1}));
    EXPECT_EQ(calls,(std::vector<std::uint64_t>{0,1,2,0}));
    EXPECT_THROW(ca::iterate_states(0,empty,std::numeric_limits<std::size_t>::max()),std::length_error);
}

TEST(CycleWork, RhoFixtureRejectsZeroPeriodAndUnrepresentableStorage) {
    EXPECT_EQ(ca::rho_successors(3,4),(std::vector<std::uint64_t>{1,2,3,4,5,6,3}));
    EXPECT_EQ(ca::rho_successors(0,1),(std::vector<std::uint64_t>{0}));
    EXPECT_THROW(ca::rho_successors(3,0),std::invalid_argument);
    EXPECT_THROW(ca::rho_successors(std::numeric_limits<std::size_t>::max(),1),std::overflow_error);
    EXPECT_THROW(ca::rho_successors(0,std::numeric_limits<std::size_t>::max()),std::length_error);
}

TEST(CycleWork, DecimalDigitSquaresMatchIndependentStringOracleAndBound) {
    EXPECT_EQ(ca::sum_squared_decimal_digits(0),0u);
    EXPECT_EQ(ca::sum_squared_decimal_digits(19),82u);
    EXPECT_EQ(ca::sum_squared_decimal_digits(std::numeric_limits<std::uint64_t>::max()),decimal_oracle(std::numeric_limits<std::uint64_t>::max()));
    std::mt19937_64 rng(51);
    for (std::size_t i=0;i<10000;++i) {
        const auto n=rng();const auto actual=ca::sum_squared_decimal_digits(n);
        EXPECT_EQ(actual,decimal_oracle(n));EXPECT_LE(actual,1620u);
    }
}

TEST(CycleWork, OriginalHappyExamplesHaveCorrectEntriesAndZeroIsSeparate) {
    struct Case {std::uint64_t n;bool happy;std::size_t mu,lambda;std::uint64_t entry;};
    const std::vector<Case> cases{{7,true,5,1,1},{19,true,4,1,1},{23,true,3,1,1},
        {4,false,0,8,4},{116,false,3,8,58},{999,false,4,8,89},{0,false,0,1,0},{1,true,0,1,1}};
    for (const auto& c:cases) {
        const auto expected=first_repeat(c.n,decimal_oracle);
        EXPECT_EQ(expected.mu,c.mu)<<c.n;EXPECT_EQ(expected.lambda,c.lambda)<<c.n;
        EXPECT_EQ(expected.entry,c.entry)<<c.n;
        const auto w=ca::brent_cycle_work(c.n,ca::sum_squared_decimal_digits);
        check_work(w,ca::floyd_cycle_work(c.n,ca::sum_squared_decimal_digits),expected);
        EXPECT_EQ(ca::is_happy_number(c.n),c.happy)<<c.n;
    }
}

TEST(CycleWork, AllBoundedPositiveDigitOrbitsReachOneOrTheEightCycleContainingFour) {
    for (std::uint64_t n=1;n<=1620;++n) {
        const auto oracle=first_repeat(n,decimal_oracle);
        const bool happy=oracle.lambda==1 && oracle.entry==1;
        if (!happy) {
            ASSERT_EQ(oracle.lambda,8u)<<n;
            auto state=oracle.entry;bool has_four=false;
            for (std::size_t i=0;i<oracle.lambda;++i) {has_four|=state==4;state=decimal_oracle(state);}
            EXPECT_TRUE(has_four)<<n;
        }
        EXPECT_EQ(ca::is_happy_number(n),happy)<<n;
    }
    std::mt19937_64 rng(123);
    for (std::size_t i=0;i<2000;++i) {
        const auto n=rng();const auto oracle=first_repeat(n,decimal_oracle);
        EXPECT_EQ(ca::is_happy_number(n),n!=0 && oracle.entry==1 && oracle.lambda==1);
    }
}
