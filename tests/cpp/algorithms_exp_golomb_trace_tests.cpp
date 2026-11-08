#include <gtest/gtest.h>
#include <datamunge/algorithms/exp_golomb.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
using namespace datamunge::algorithms;
namespace {
using U=std::uint64_t;
constexpr U maximum=std::numeric_limits<U>::max();
// Independent division/string-carry oracle, including the 65-digit q+1 case.
std::string binary(U n){std::string s;do{s.push_back('0'+n%2);n/=2;}while(n);std::reverse(s.begin(),s.end());return s;}
std::string oracle(U n,int k){
    const U divisor=U{1}<<k;auto s=binary(n/divisor);bool carry=true;
    for(auto i=s.size();i>0&&carry;--i){carry=s[i-1]=='1';s[i-1]=carry?'0':'1';}if(carry)s.insert(s.begin(),'1');
    std::string low;auto r=n%divisor;for(int i=0;i<k;++i){low.push_back('0'+r%2);r/=2;}std::reverse(low.begin(),low.end());
    return std::string(s.size()-1,'0')+s+low;
}
std::vector<bool> bits(const std::string& s){std::vector<bool> b;for(char c:s)b.push_back(c=='1');return b;}
void certificate(const std::vector<U>& values,int k){
    std::string expected;for(auto n:values)expected+=oracle(n,k);
    const auto e=exp_golomb_encode_work(values,k);
    ASSERT_EQ(exp_golomb_bit_string(e.bits),expected);
    const auto t=exp_golomb_encode_trace(values,k,0),d=exp_golomb_decode_trace(e.bits,INT_MAX,k,0);
    ASSERT_EQ(t.bits,e.bits);
    ASSERT_EQ(t.values,values);
    ASSERT_EQ(d.values,values);
    ASSERT_EQ(d.bits,e.bits);
    ASSERT_EQ(t.words.size(),values.size());
    ASSERT_EQ(d.words.size(),values.size());
    EXPECT_EQ(t.captured_entries,e.bits.size()+2*values.size());
    EXPECT_EQ(d.captured_entries,t.captured_entries);
    EXPECT_EQ(t.consumed_bits,expected.size());
    EXPECT_EQ(d.consumed_bits,expected.size());
    std::size_t pos=0,prefixes=0,quotients=0,suffixes=0;
    for(std::size_t i=0;i<values.size();++i){
        auto word=oracle(values[i],k);auto m=word.find('1');const auto& w=t.words[i];const auto& v=d.words[i];
        EXPECT_EQ(w.value,values[i]);
    EXPECT_EQ(w.begin,pos);
    EXPECT_EQ(w.payload_begin,pos+m);
        EXPECT_EQ(w.suffix_begin,pos+word.size()-k);
    EXPECT_EQ(w.end,pos+word.size());
        EXPECT_EQ(v.value,w.value);
    EXPECT_EQ(v.begin,w.begin);
    EXPECT_EQ(v.payload_begin,w.payload_begin);
    EXPECT_EQ(v.suffix_begin,w.suffix_begin);
    EXPECT_EQ(v.end,w.end);
        EXPECT_EQ(exp_golomb_code_length(values[i],k),word.size());pos+=word.size();prefixes+=m;quotients+=m+1;suffixes+=k;
    }
    for(const auto* w:{&e.work,&t.work,&d.work}){EXPECT_EQ(w->values,values.size());
    EXPECT_EQ(w->zero_prefix_bits,prefixes);
    EXPECT_EQ(w->quotient_bits,quotients);
    EXPECT_EQ(w->suffix_bits,suffixes);
    EXPECT_EQ(w->bit_visits,expected.size());}
    const auto s=exp_golomb_size(values,k);
    EXPECT_EQ(s.values,values.size());
    EXPECT_EQ(s.bits,expected.size());
    EXPECT_EQ(s.order,k);
    const auto max=values.empty()?0:*std::max_element(values.begin(),values.end());
    EXPECT_EQ(s.maximum,max);
    EXPECT_EQ(s.fixed_width,values.empty()?0:binary(max).size());
    EXPECT_EQ(s.fixed_bits,values.size()*s.fixed_width);
}
}
TEST(ExpGolombOracle, ExhaustiveUint16OrdersZeroThroughSeven){
    std::vector<U> values;for(U n=0;n<65536;++n)values.push_back(n);
    for(int k=0;k<8;++k){std::string expected;for(auto n:values)expected+=oracle(n,k);
        auto e=exp_golomb_encode_work(values,k);
    ASSERT_EQ(exp_golomb_bit_string(e.bits),expected)<<k;
        auto d=exp_golomb_decode_work(e.bits,65536,k);
    EXPECT_EQ(d.values,values);
    EXPECT_EQ(d.consumed_bits,e.bits.size());
    EXPECT_EQ(e.work.bit_visits,expected.size());}
}
TEST(ExpGolombOracle, AllOrdersPowersNeighboursAndMaximum){
    std::vector<U> values{0,1,2,maximum};for(int j=1;j<64;++j){U p=U{1}<<j;values.push_back(p-1);values.push_back(p);values.push_back(p+1);}
    for(int k=0;k<64;++k)certificate(values,k);
}
TEST(ExpGolombOracle, SeededFullWidthActualSpans){std::mt19937_64 rng(0xe901bULL);for(int r=0;r<320;++r){std::vector<U> v;for(int j=0;j<r%31;++j)v.push_back(rng());certificate(v,r%64);}}
TEST(ExpGolombOracle, EveryBitStringThroughLengthSixteen){
    for(int k=0;k<5;++k){std::unordered_map<std::string,U> dictionary;for(U n=0;n<2048;++n){auto s=oracle(n,k);if(s.size()<=16)dictionary.emplace(s,n);}
        for(int length=0;length<=16;++length)for(unsigned pattern=0;pattern<(1u<<length);++pattern){
            std::string s;for(int i=length;i>0;--i)s.push_back((pattern>>(i-1))&1?'1':'0');
            auto input=bits(s);auto it=dictionary.end();std::size_t end=0;
            for(std::size_t j=1;j<=s.size();++j){it=dictionary.find(s.substr(0,j));if(it!=dictionary.end()){end=j;break;}}
            if(s.empty()){EXPECT_TRUE(exp_golomb_decode_work(input,1,k).values.empty());}
            else if(end){auto d=exp_golomb_decode_work(input,1,k);
    ASSERT_EQ(d.values.size(),1);
    EXPECT_EQ(d.values[0],it->second);
    EXPECT_EQ(d.consumed_bits,end);}
            else {EXPECT_THROW(exp_golomb_decode_work(input,1,k),std::invalid_argument)<<k<<':'<<s;}
        }
    }
}
TEST(ExpGolombLimits, NativeMaximumHasRealSixtyFiveBitQuotient){
    auto t=exp_golomb_encode_trace({maximum});
    ASSERT_EQ(t.bits.size(),129);
    EXPECT_EQ(std::count(t.bits.begin(),t.bits.end(),true),1);
    EXPECT_TRUE(t.bits[64]);
    EXPECT_EQ(t.words[0].payload_begin,64);
    EXPECT_EQ(t.words[0].suffix_begin,129);
    EXPECT_EQ(t.captured_entries,131);
    EXPECT_EQ(exp_golomb_decode(t.bits,1),(std::vector<U>{maximum}));
    EXPECT_EQ(exp_golomb_code_length(maximum,63),66);
    EXPECT_EQ(exp_golomb_code_length(0,63),64);
    EXPECT_EQ(exp_golomb_code_length(0),1);
}
TEST(ExpGolombLimits, BoundedPrefixPreservesLegacyFraming){
    auto d=exp_golomb_decode_work({true,false},1);
    EXPECT_EQ(d.values,(std::vector<U>{0}));
    EXPECT_EQ(d.consumed_bits,1);
    EXPECT_THROW(exp_golomb_decode_work({true,false},2),std::invalid_argument);
    EXPECT_TRUE(exp_golomb_decode_work({false},0).values.empty());
    EXPECT_EQ(exp_golomb_decode_work({false},0).consumed_bits,0);
    EXPECT_TRUE(exp_golomb_decode({},INT_MAX).empty());
    EXPECT_EQ(exp_golomb_decode({true},INT_MAX),(std::vector<U>{0}));
}
TEST(ExpGolombLimits, EveryProperSingleWordPrefixIsIncomplete){
    for(int k=0;k<64;++k)for(U n:std::vector<U>{0,10,maximum}){auto b=bits(oracle(n,k));for(std::size_t j=1;j<b.size();++j){std::vector<bool> p(b.begin(),b.begin()+j);
    EXPECT_THROW(exp_golomb_decode(p,1,k),std::invalid_argument);}}
}
TEST(ExpGolombLimits, InvalidParametersEvenOnEmptyInputs){
    for(int k:{-1,64,INT_MIN,INT_MAX}){
        EXPECT_THROW(exp_golomb_code_length(0,k),std::invalid_argument);
    EXPECT_THROW(exp_golomb_size({},k),std::invalid_argument);
        EXPECT_THROW(exp_golomb_encode({},k),std::invalid_argument);
    EXPECT_THROW(exp_golomb_encode_trace({},k),std::invalid_argument);
        EXPECT_THROW(exp_golomb_decode({},0,k),std::invalid_argument);
    EXPECT_THROW(exp_golomb_decode_trace({},0,k),std::invalid_argument);
    }
    EXPECT_THROW(exp_golomb_decode({},-1),std::invalid_argument);
    EXPECT_THROW(exp_golomb_decode_trace({},-1),std::invalid_argument);
}
TEST(ExpGolombLimits, OverwidePrefixesAndNativeOverflow){
    for(int k=0;k<64;++k)EXPECT_THROW(exp_golomb_decode(std::vector<bool>(65-k,false),1,k),std::overflow_error);
    auto b=bits(std::string(64,'0')+'1'+std::string(64,'0'));b.back()=true;
    EXPECT_THROW(exp_golomb_decode(b,1),std::overflow_error);
    b=bits(std::string(63,'0')+'1'+std::string(63,'0')+'0');b[126]=true;
    EXPECT_THROW(exp_golomb_decode(b,1,1),std::overflow_error);
    // An unread invalid-width tail is outside a satisfied prefix request.
    b=std::vector<bool>(100,false);b.insert(b.begin(),true);
    EXPECT_EQ(exp_golomb_decode(b,1),(std::vector<U>{0}));
}
TEST(ExpGolombLimits, AppendKeepsArbitraryPrefixAndCountsOnlyNewWord){
    const std::vector<bool> prefix{false,true,false,false,true};
    for(int k=0;k<64;++k){auto b=prefix;auto w=exp_golomb_append_work(maximum,b,k);
    EXPECT_EQ(std::vector<bool>(b.begin(),b.begin()+prefix.size()),prefix);
    EXPECT_EQ(exp_golomb_bit_string(std::vector<bool>(b.begin()+prefix.size(),b.end())),oracle(maximum,k));
    EXPECT_EQ(w.values,1);
    EXPECT_EQ(w.bit_visits,b.size()-prefix.size());}
    auto b=prefix;
    EXPECT_THROW(exp_golomb_append_work(0,b,64),std::invalid_argument);
    EXPECT_EQ(b,prefix);
    exp_golomb0_encode_value(3,b);
    EXPECT_EQ(exp_golomb_bit_string(b),exp_golomb_bit_string(prefix)+"00100");
}
TEST(ExpGolombCapture, ExactQuotaOneShortAndDisabled){
    for(int k=0;k<64;++k){auto t=exp_golomb_encode_trace({0,10,maximum},k,0);auto c=t.captured_entries;
        EXPECT_NO_THROW(exp_golomb_encode_trace({0,10,maximum},k,c));
    EXPECT_THROW(exp_golomb_encode_trace({0,10,maximum},k,c-1),std::length_error);
        EXPECT_NO_THROW(exp_golomb_decode_trace(t.bits,INT_MAX,k,c));
    EXPECT_THROW(exp_golomb_decode_trace(t.bits,INT_MAX,k,c-1),std::length_error);
    }
    EXPECT_EQ(exp_golomb_encode_trace({},0,1).captured_entries,0);
}
TEST(ExpGolombCapture, WholeInputReservedButOnlyActualValuesCounted){
    auto t=exp_golomb_decode_trace({true,false},1,0,4);
    EXPECT_EQ(t.captured_entries,4);
    EXPECT_EQ(t.consumed_bits,1);
    EXPECT_EQ(t.work.bit_visits,1);
    EXPECT_EQ(t.bits.size(),2);
    EXPECT_THROW(exp_golomb_decode_trace({true,false},1,0,3),std::length_error);
    EXPECT_THROW(exp_golomb_decode_trace({true,false},0,0,1),std::length_error);
    EXPECT_EQ(exp_golomb_decode_trace({true,false},0,0,2).captured_entries,2);
    EXPECT_EQ(exp_golomb_decode_trace({true},INT_MAX,0,3).values.size(),1);
}
TEST(ExpGolombCapture, ReturnedOwnershipIsIndependent){
    std::vector<U> v{10};auto e=exp_golomb_encode_trace(v,2);v[0]=0;
    EXPECT_EQ(e.values[0],10);auto b=e.bits;auto d=exp_golomb_decode_trace(b,1,2);b[0]=true;e.words[0].value=99;e.bits[0]=true;
    EXPECT_EQ(d.values[0],10);
    EXPECT_EQ(d.words[0].value,10);
    EXPECT_FALSE(d.bits[0]);
}
TEST(ExpGolombSize, EmptyZeroConventionAndExactThreshold){
    auto s=exp_golomb_size({});
    EXPECT_EQ(s.bits,0);
    EXPECT_EQ(s.maximum,0);
    EXPECT_EQ(s.fixed_width,0);
    EXPECT_EQ(s.fixed_bits,0);
    s=exp_golomb_size({0,0,0});
    EXPECT_EQ(s.fixed_width,1);
    EXPECT_EQ(s.bits,3);
    EXPECT_EQ(s.fixed_bits,3);
    EXPECT_EQ(exp_golomb_size({0,1,3,7,15,100,1000,65535}).bits,90);
    for(U n=0;n<255;++n)EXPECT_LT(exp_golomb_code_length(n),16);
    EXPECT_EQ(exp_golomb_code_length(255),17);
}
