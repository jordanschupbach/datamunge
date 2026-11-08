#include <datamunge/algorithms/exponential_backoff.hpp>
#include <gtest/gtest.h>
#include <bit>
#include <cfenv>
#include <climits>
#include <cmath>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
constexpr double maximum=std::numeric_limits<double>::max();
double iterative_oracle(int k,double base,double cap){double d=base;for(int j=0;j<k&&d<cap;++j){if(d==0)return d;if(d>maximum/2)return cap;d*=2;}return d<cap?d:cap;}
void exact(double actual,double expected){EXPECT_EQ(std::bit_cast<std::uint64_t>(actual),std::bit_cast<std::uint64_t>(expected));}
void certificate(const BackoffSchedule& s,double base,double cap){ASSERT_EQ(s.steps.size(),s.retry_times.size());double total=0;for(std::size_t i=0;i<s.steps.size();++i){const auto& r=s.steps[i];
    EXPECT_EQ(r.attempt,int(i));exact(r.window,iterative_oracle(i,base,cap));
    EXPECT_GE(r.u,0);
    EXPECT_LE(r.u,1);exact(r.delay,r.u*r.window);
    EXPECT_LE(r.delay,r.window);
    EXPECT_EQ(r.capped,r.window==cap);
    EXPECT_EQ(r.scaled,i>0&&base>0&&r.window<cap);total+=r.delay;exact(s.retry_times[i],total);}exact(s.total_delay,total);}
}
TEST(BackoffKernel, TinyActualWindowsAndCumulativeIdleWaits){
    const auto s=exponential_backoff_schedule(4,1,5,{.25,.5,.75,1});certificate(s,1,5);
    EXPECT_EQ(s.retry_times,(std::vector<double>{.25,1.25,4.25,9.25}));
    EXPECT_EQ(s.total_delay,9.25);
    EXPECT_TRUE(s.jittered);
    EXPECT_FALSE(s.steps[0].scaled);
    EXPECT_TRUE(s.steps[1].scaled);
    EXPECT_TRUE(s.steps[3].capped);
}
TEST(BackoffKernel, WideSeededFiniteCasesMatchIndependentSafeDoubling){
    std::mt19937_64 rng(20150304);for(int t=0;t<12000;++t){const auto base=std::ldexp(.5+double(rng()%1024)/2048,int(rng()%2090)-1073),cap=std::ldexp(.5+double(rng()%1024)/2048,int(rng()%2090)-1073);const int k=t%3?int(rng()%2200):INT_MAX;const double u=double(rng()>>11)*0x1p-53;const auto r=exponential_backoff_step(k,base,cap,u);const auto expected=iterative_oracle(k,base,cap);exact(r.window,expected);exact(r.delay,u*expected);
    EXPECT_EQ(r.capped,expected==cap);
    EXPECT_EQ(r.scaled,k>0&&base>0&&expected<cap);}
}
TEST(BackoffKernel, AllBinaryPowerScalesAndCapNeighbours){
    for(int exponent=-1074;exponent<=1023;++exponent){const auto base=std::ldexp(1.,exponent);for(int k:{0,1,2,31,1024,2098,INT_MAX})for(double cap:{base,std::nextafter(base,0.),std::nextafter(base,maximum),maximum})exact(exponential_backoff_delay(k,base,cap),iterative_oracle(k,base,cap));}
}
TEST(BackoffKernel, InclusiveEndpointAndSignedZeroPreserveLegacyBits){
    for(double base:{0.,-0.,1.})for(double cap:{0.,-0.,30.})for(int k:{0,1,32,INT_MAX}){const auto w=iterative_oracle(k,base,cap);exact(exponential_backoff_delay(k,base,cap),w);for(double u:{0.,-0.,.5,1.})exact(exponential_backoff_jittered(k,base,cap,u),u*w);}
    EXPECT_EQ(exponential_backoff_jittered(3,1,100,1),8);
}
TEST(BackoffKernel, CappingAvoidsIntermediateFloatingOverflow){
    volatile double base=.75*maximum;std::feclearexcept(FE_ALL_EXCEPT);const auto r=exponential_backoff_step(1,base,maximum);const auto overflow=std::fetestexcept(FE_OVERFLOW);
    EXPECT_EQ(r.window,maximum);
    EXPECT_TRUE(r.capped);
    EXPECT_FALSE(r.scaled);
    EXPECT_EQ(overflow,0);
    std::feclearexcept(FE_ALL_EXCEPT);
    EXPECT_EQ(exponential_backoff_delay(INT_MAX,std::numeric_limits<double>::denorm_min(),maximum),maximum);
    EXPECT_EQ(std::fetestexcept(FE_OVERFLOW),0);
}
TEST(BackoffKernel, HugeZeroIndexDoesNotScaleAndImmediateCapsWork){
    auto z=exponential_backoff_step(INT_MAX,0,30);
    EXPECT_EQ(z.window,0);
    EXPECT_FALSE(z.scaled);
    EXPECT_FALSE(z.capped);
    auto c=exponential_backoff_step(INT_MAX,100,30);
    EXPECT_EQ(c.window,30);
    EXPECT_TRUE(c.capped);
    EXPECT_FALSE(c.scaled);
    c=exponential_backoff_step(0,1,0);
    EXPECT_EQ(c.window,0);
    EXPECT_TRUE(c.capped);
}
TEST(BackoffKernel, ScalarAndEmptySequenceDomainsReject){
    const auto nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    for(double invalid:{-1.,nan,inf,-inf}){
        EXPECT_THROW(exponential_backoff_delay(0,invalid,30),std::invalid_argument);
        EXPECT_THROW(exponential_backoff_delay(0,1,invalid),std::invalid_argument);
        EXPECT_THROW(exponential_backoff_sequence(0,invalid,30),std::invalid_argument);
        EXPECT_THROW(exponential_backoff_schedule(0,1,invalid),std::invalid_argument);
        EXPECT_THROW(exponential_backoff_seeded(0,invalid,30),std::invalid_argument);
    }
    for(double invalid:{-1.,1.0000001,nan,inf})EXPECT_THROW(exponential_backoff_jittered(0,1,30,invalid),std::invalid_argument);
    EXPECT_THROW(exponential_backoff_delay(-1,1,30),std::invalid_argument);
    EXPECT_THROW(exponential_backoff_sequence(-1,1,30),std::invalid_argument);
    EXPECT_THROW(exponential_backoff_schedule(-1,1,30),std::invalid_argument);
    EXPECT_THROW(exponential_backoff_seeded(-1,1,30),std::invalid_argument);
    EXPECT_THROW(exponential_backoff_seeded(0,1,30,0),std::invalid_argument);
}
TEST(BackoffSchedule, OriginalSeededValuesAndActualUniformBits){
    const auto s=exponential_backoff_seeded(8,1,30);certificate(s,1,30);
    const std::vector<double> windows{1,2,4,8,16,30,30,30};std::uint64_t state=88172645463325252ULL;
    for(int i=0;i<8;++i){state^=state<<13;state^=state>>7;state^=state<<17;exact(s.steps[i].u,(state>>11)*0x1p-53);exact(s.steps[i].delay,windows[i]*((state>>11)*0x1p-53));}
    EXPECT_NEAR(s.steps[0].delay,.4742589867636229,1e-15);
}
TEST(BackoffSchedule, EmptyModeAndExplicitUniformValidation){
    const auto a=exponential_backoff_schedule(0,1,30),b=exponential_backoff_seeded(0,1,30);
    EXPECT_TRUE(a.steps.empty());
    EXPECT_TRUE(a.retry_times.empty());
    EXPECT_EQ(a.total_delay,0);
    EXPECT_FALSE(a.jittered);
    EXPECT_TRUE(b.jittered);
    EXPECT_THROW(exponential_backoff_schedule(0,1,30,{.5}),std::invalid_argument);
    EXPECT_THROW(exponential_backoff_schedule(2,1,30,{.5}),std::invalid_argument);
    EXPECT_THROW(exponential_backoff_schedule(2,1,30,{.5,-1}),std::invalid_argument);
    const auto zeros=exponential_backoff_schedule(4,1,30,{0,0,0,0});
    EXPECT_EQ(zeros.retry_times,(std::vector<double>{0,0,0,0}));
}
TEST(BackoffSchedule, PrefixCapacityDoesNotRestrictPureWindowOutput){
    const auto windows=exponential_backoff_sequence(2,maximum,maximum);
    EXPECT_EQ(windows,(std::vector<double>{maximum,maximum}));
    EXPECT_THROW(exponential_backoff_schedule(2,maximum,maximum),std::overflow_error);
    EXPECT_THROW(exponential_backoff_schedule(2,maximum,maximum,{1,1}),std::overflow_error);
    const auto s=exponential_backoff_schedule(2,maximum,maximum,{.5,.5});
    EXPECT_EQ(s.total_delay,maximum);certificate(s,maximum,maximum);
}
TEST(BackoffSchedule, OwnedInputsResultsAndSameSeedControls){
    std::vector<double> u{.25,.5,.75};auto a=exponential_backoff_schedule(3,1,30,u);u.assign(3,0);
    EXPECT_EQ(a.steps[2].u,.75);auto b=exponential_backoff_schedule(3,1,30,u);b.steps[0].window=99;
    EXPECT_EQ(a.steps[0].window,1);
    const auto first=exponential_backoff_seeded(12,1,30,99),again=exponential_backoff_seeded(12,1,30,99),prefix=exponential_backoff_seeded(3,1,30,99),other=exponential_backoff_seeded(12,1,30,100);
    EXPECT_EQ(first.retry_times,again.retry_times);
    EXPECT_NE(first.retry_times,other.retry_times);
    EXPECT_EQ(std::vector<double>(first.retry_times.begin(),first.retry_times.begin()+3),prefix.retry_times);
}
TEST(BackoffSchedule, CappedTimelineIdentityAndChangeOfTimeUnits){
    const auto s=exponential_backoff_schedule(8,1,30);certificate(s,1,30);
    EXPECT_EQ(s.retry_times,(std::vector<double>{1,3,7,15,31,61,91,121}));
    EXPECT_FALSE(s.jittered);
    const auto a=exponential_backoff_seeded(10,.25,7.5,123),b=exponential_backoff_seeded(10,1,30,123);for(int i=0;i<10;++i){exact(4*a.steps[i].window,b.steps[i].window);exact(4*a.steps[i].delay,b.steps[i].delay);exact(4*a.retry_times[i],b.retry_times[i]);}
    for(int n=0;n<=256;++n)certificate(exponential_backoff_seeded(n,1,30,123),1,30);
}
