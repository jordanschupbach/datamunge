#include <datamunge/algorithms/extended_euclidean.hpp>
#include <gtest/gtest.h>
#include <bit>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
using namespace datamunge::algorithms;
namespace {
using I=std::int64_t;using U=std::uint64_t;using W=__int128;
constexpr I minimum=std::numeric_limits<I>::min(),maximum=std::numeric_limits<I>::max();
U magnitude(I a){return a<0?U{0}-static_cast<U>(a):static_cast<U>(a);}
W absolute(W a){return a<0?-a:a;}
void certificate(I a,I b){
    const auto t=extended_gcd_trace(a,b,0);const auto& c=t.outcome.certificate;const auto ordinary=extended_gcd_work(a,b);
    ASSERT_EQ(c.gcd,std::gcd(magnitude(a),magnitude(b)));
    EXPECT_TRUE(bezout_certificate_valid(a,b,c));
    EXPECT_EQ(static_cast<W>(a)*c.x+static_cast<W>(b)*c.y,static_cast<W>(c.gcd));
    EXPECT_EQ(c.gcd,ordinary.certificate.gcd);
    EXPECT_EQ(c.x,ordinary.certificate.x);
    EXPECT_EQ(c.y,ordinary.certificate.y);
    const auto D=t.outcome.work.divisions;
    EXPECT_EQ(t.outcome.work.coefficient_updates,2*D);
    ASSERT_EQ(t.rows.size(),D+2);
    EXPECT_EQ(t.captured_entries,D+2);
    EXPECT_EQ(t.rows[0].remainder,static_cast<W>(a));
    EXPECT_EQ(t.rows[0].s,1);
    EXPECT_EQ(t.rows[0].t,0);
    EXPECT_FALSE(t.rows[0].divided);
    EXPECT_EQ(t.rows[1].remainder,static_cast<W>(b));
    EXPECT_EQ(t.rows[1].s,0);
    EXPECT_EQ(t.rows[1].t,1);
    EXPECT_FALSE(t.rows[1].divided);
    for(std::size_t i=0;i<t.rows.size();++i){const auto& r=t.rows[i];
    EXPECT_EQ(static_cast<W>(a)*r.s+static_cast<W>(b)*r.t,r.remainder);
        EXPECT_LE(absolute(r.s),W{1}<<63);
    EXPECT_LE(absolute(r.t),W{1}<<63);
    EXPECT_LE(absolute(r.remainder),W{1}<<63);
        if(i){const auto& previous=t.rows[i-1];
    EXPECT_EQ(previous.s*r.t-r.s*previous.t,(i%2)?W{1}:W{-1});}
        if(i>=2){const auto& first=t.rows[i-2];const auto& second=t.rows[i-1];
    EXPECT_TRUE(r.divided);
    ASSERT_NE(second.remainder,0);
            EXPECT_EQ(r.quotient,first.remainder/second.remainder);
    EXPECT_EQ(r.remainder,first.remainder-r.quotient*second.remainder);
    EXPECT_EQ(r.s,first.s-r.quotient*second.s);
    EXPECT_EQ(r.t,first.t-r.quotient*second.t);
    EXPECT_LT(absolute(r.remainder),absolute(second.remainder));}
    }
    EXPECT_EQ(t.rows.back().remainder,0);const auto& final=t.rows[t.rows.size()-2];
    EXPECT_EQ(t.outcome.work.sign_normalized,final.remainder<0);
    EXPECT_EQ(static_cast<W>(c.x),final.remainder<0?-final.s:final.s);
    EXPECT_EQ(static_cast<W>(c.y),final.remainder<0?-final.t:final.t);
    if(c.gcd<=U(maximum)){const auto legacy=extended_gcd(a,b);
    EXPECT_EQ(legacy.gcd,I(c.gcd));
    EXPECT_EQ(legacy.x,c.x);
    EXPECT_EQ(legacy.y,c.y);}
    else EXPECT_THROW(extended_gcd(a,b),std::overflow_error);
}
}
TEST(EuclidTrace, OriginalWorkedRowsAndActualCounts){
    const auto t=extended_gcd_trace(240,46);
    EXPECT_EQ(t.outcome.certificate.gcd,2);
    EXPECT_EQ(t.outcome.certificate.x,-9);
    EXPECT_EQ(t.outcome.certificate.y,47);
    EXPECT_EQ(t.outcome.work.divisions,5);
    EXPECT_EQ(t.captured_entries,7);
    const std::vector<W> remainders{240,46,10,6,4,2,0},s{1,0,1,-4,5,-9,23},u{0,1,-5,21,-26,47,-120},q{0,0,5,4,1,1,2};
    ASSERT_EQ(t.rows.size(),7);for(std::size_t i=0;i<7;++i){EXPECT_EQ(t.rows[i].remainder,remainders[i]);
    EXPECT_EQ(t.rows[i].s,s[i]);
    EXPECT_EQ(t.rows[i].t,u[i]);
    EXPECT_EQ(t.rows[i].quotient,q[i]);}certificate(240,46);
}
TEST(EuclidTrace, ExhaustiveSmallSignedGridHasExactRowsAndCertificates){for(I a=-128;a<=128;++a)for(I b=-128;b<=128;++b)certificate(a,b);}
TEST(EuclidTrace, SeededFullWidthPairsAndPowerBoundaries){
    std::mt19937_64 rng(1996);for(int i=0;i<10000;++i)certificate(std::bit_cast<I>(rng()),std::bit_cast<I>(rng()));
    for(int bit=0;bit<63;++bit){I p=I{1}<<bit;for(I a:std::vector<I>{minimum,maximum,p,-p,p-1,-p+1})for(I b:std::vector<I>{minimum,maximum,p,-p,p-1,-p+1})certificate(a,b);}
}
TEST(EuclidTrace, MinimumInputsAndWidenedTerminalRowsAvoidAllOldFailures){
    for(const auto& pair:std::vector<std::pair<I,I>>{{minimum,-1},{minimum,0},{minimum,1},{minimum,maximum},{0,minimum},{minimum,minimum}})certificate(pair.first,pair.second);
    const auto div=extended_gcd_trace(minimum,-1);
    EXPECT_EQ(div.rows.back().quotient,W{1}<<63);
    EXPECT_EQ(div.rows.back().t,-(W{1}<<63));
    const auto terminal=extended_gcd_trace(minimum,1);
    EXPECT_EQ(terminal.rows.back().t,W{1}<<63);
    EXPECT_EQ(terminal.outcome.certificate.y,1);
    const auto edge=extended_gcd_trace(minimum,maximum);
    EXPECT_EQ(edge.rows.back().t,W{1}<<63);
    EXPECT_EQ(edge.outcome.certificate.x,-1);
    EXPECT_EQ(edge.outcome.certificate.y,-1);
}
TEST(EuclidTrace, LegacyZeroPairLayoutAndNarrowRangeArePreserved){
    static_assert(sizeof(ExtendedGcd)==3*sizeof(I));static_assert(std::is_standard_layout_v<ExtendedGcd>);static_assert(std::is_trivially_copyable_v<ExtendedGcd>);
    const auto z=extended_gcd(0,0);
    EXPECT_EQ(z.gcd,0);
    EXPECT_EQ(z.x,1);
    EXPECT_EQ(z.y,0);
    EXPECT_EQ(extended_gcd_trace(0,0).rows.size(),2);
    for(const auto& pair:std::vector<std::pair<I,I>>{{minimum,0},{0,minimum},{minimum,minimum}}){EXPECT_EQ(extended_gcd_certificate(pair.first,pair.second).gcd,U{1}<<63);
    EXPECT_THROW(extended_gcd(pair.first,pair.second),std::overflow_error);}
}
TEST(EuclidTrace, FibonacciFamilyAndInitialZeroQuotientCountActualDivisions){
    std::vector<I> f{0,1};while(f.back()<=maximum-f[f.size()-2])f.push_back(f.back()+f[f.size()-2]);
    ASSERT_EQ(f.back(),7540113804746346429LL);for(std::size_t i=3;i<f.size();++i){const auto forward=extended_gcd_work(f[i],f[i-1]),reverse=extended_gcd_work(f[i-1],f[i]);
    EXPECT_EQ(forward.work.divisions,i-2);
    EXPECT_EQ(reverse.work.divisions,i-1);certificate(f[i],f[i-1]);}
    EXPECT_EQ(extended_gcd_work(f.back(),f[f.size()-2]).work.divisions,90);
    EXPECT_EQ(extended_gcd_work(f[f.size()-2],f.back()).work.divisions,91);
}
TEST(EuclidTrace, ExactBaseOneShortDisabledQuotasAndOwnedCopies){
    auto t=extended_gcd_trace(240,46,7);
    EXPECT_EQ(t.captured_entries,7);
    EXPECT_THROW(extended_gcd_trace(240,46,6),std::length_error);
    EXPECT_THROW(extended_gcd_trace(240,46,1),std::length_error);
    EXPECT_EQ(extended_gcd_trace(240,46,0).rows.size(),7);
    EXPECT_EQ(extended_gcd_trace(0,0,2).rows.size(),2);
    EXPECT_THROW(extended_gcd_trace(0,0,1),std::length_error);
    const auto other=extended_gcd_trace(240,46);t.rows[2].s=999;t.outcome.certificate.x=999;
    EXPECT_EQ(other.rows[2].s,1);
    EXPECT_EQ(other.outcome.certificate.x,-9);
}
TEST(EuclidCertificate, GreatestDivisorAndIdentityBothMatterWithOverflowSafeMalformedInput){
    EXPECT_TRUE(bezout_certificate_valid(0,0,{0,1,0}));
    EXPECT_TRUE(bezout_certificate_valid(0,0,{0,minimum,minimum}));
    EXPECT_FALSE(bezout_certificate_valid(3,5,{0,0,0}));
    EXPECT_FALSE(bezout_certificate_valid(6,10,{6,1,0})); // Correct linear combination, not a common divisor.
    EXPECT_FALSE(bezout_certificate_valid(6,10,{2,0,0})); // Common divisor, wrong identity.
    EXPECT_FALSE(bezout_certificate_valid(minimum,minimum,{1,minimum,minimum})); // Sum would be +2^127.
    EXPECT_FALSE(bezout_certificate_valid(0,0,{std::numeric_limits<U>::max(),1,1}));
    EXPECT_TRUE(bezout_certificate_valid(minimum,0,{U{1}<<63,-1,0}));
}
TEST(EuclidInverse, SamePassOriginalSignedAndUnitModulusCases){
    for(const auto& input:std::vector<std::pair<I,I>>{{3,11},{-3,11},{7,26},{0,1},{minimum,1},{minimum,maximum},{maximum,maximum-1}}){const auto t=modular_inverse_trace(input.first,input.second);
    EXPECT_EQ(t.inverse,modular_inverse(input.first,input.second));
    EXPECT_EQ(t.euclid.a,input.first);
    EXPECT_EQ(t.euclid.b,input.second);
    EXPECT_TRUE(bezout_certificate_valid(input.first,input.second,t.euclid.outcome.certificate));
    EXPECT_GE(t.inverse,0);
    EXPECT_LT(t.inverse,input.second);const W residue=((static_cast<W>(input.first)*t.inverse)%input.second+input.second)%input.second;
    EXPECT_EQ(residue,1%input.second);}
    EXPECT_EQ(modular_inverse(minimum,1),0);
    EXPECT_EQ(modular_inverse(0,1),0);
    EXPECT_EQ(modular_inverse_trace(3,11,6).euclid.captured_entries,6);
}
TEST(EuclidInverse, InvalidModuliNonunitsAndTraceQuotaPrecedence){
    for(I m:std::vector<I>{minimum,-11,0}){
        EXPECT_THROW(modular_inverse(minimum,m),std::invalid_argument);
        EXPECT_THROW(modular_inverse_trace(minimum,m,1),std::invalid_argument);
    }
    for(const auto& pair:std::vector<std::pair<I,I>>{{0,5},{6,9},{4,8},{minimum,2}}){EXPECT_THROW(modular_inverse(pair.first,pair.second),std::invalid_argument);
    EXPECT_THROW(modular_inverse_trace(pair.first,pair.second,0),std::invalid_argument);}
    EXPECT_THROW(modular_inverse_trace(3,11,5),std::length_error);
    EXPECT_THROW(modular_inverse_trace(6,9,2),std::length_error);
    EXPECT_EQ(modular_inverse_trace(minimum,1,3).inverse,0);
}
