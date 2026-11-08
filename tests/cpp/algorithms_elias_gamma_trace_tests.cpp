#include <datamunge/algorithms/elias_gamma.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <charconv>
#include <limits>
#include <optional>
#include <random>
#include <unordered_map>
namespace {
using namespace datamunge::algorithms;
std::string binary_reference(std::uint64_t n){std::string bits;do{bits.push_back(n%2?'1':'0');n/=2;}while(n);std::reverse(bits.begin(),bits.end());return bits;}
std::string code_reference(std::uint64_t n){const auto payload=binary_reference(n);return std::string(payload.size()-1,'0')+payload;}
void certificate(const GammaTrace& t){
    ASSERT_EQ(t.words.size(),t.values.size());std::size_t end=0,prefix=0,payload=0;std::uint64_t maximum=0;std::size_t max_width=0;
    for(std::size_t i=0;i<t.words.size();++i){const auto& w=t.words[i];const auto expected=code_reference(t.values[i]),binary=binary_reference(t.values[i]);
        EXPECT_GT(w.value,0u);EXPECT_EQ(w.value,t.values[i]);EXPECT_EQ(w.begin,end);EXPECT_EQ(w.payload_begin,w.begin+binary.size()-1);EXPECT_EQ(w.end,w.begin+expected.size());ASSERT_LE(w.end,t.bits.size());
        EXPECT_EQ(t.bits.substr(w.begin,w.end-w.begin),expected);EXPECT_EQ(t.bits.substr(w.payload_begin,w.end-w.payload_begin),binary);
        std::uint64_t parsed=0;const auto first=t.bits.data()+w.payload_begin,last=t.bits.data()+w.end;const auto result=std::from_chars(first,last,parsed,2);EXPECT_EQ(result.ec,std::errc{});EXPECT_EQ(result.ptr,last);EXPECT_EQ(parsed,w.value);
        prefix+=binary.size()-1;payload+=binary.size();end=w.end;if(w.value>maximum){maximum=w.value;max_width=binary.size();}
    }
    EXPECT_EQ(end,t.bits.size());EXPECT_EQ(t.work.values,t.values.size());EXPECT_EQ(t.work.zero_prefix_bits,prefix);EXPECT_EQ(t.work.payload_bits,payload);EXPECT_EQ(t.work.bit_visits,prefix+payload);EXPECT_EQ(t.work.bit_visits,t.bits.size());EXPECT_EQ(t.captured_entries,t.bits.size()+2*t.values.size());
    EXPECT_EQ(elias_gamma_encode(t.values),t.bits);EXPECT_EQ(elias_gamma_decode(t.bits),t.values);
    const auto s=elias_gamma_size_summary(t.values);EXPECT_EQ(s.values,t.values.size());EXPECT_EQ(s.maximum,maximum);EXPECT_EQ(s.gamma_bits,t.bits.size());EXPECT_EQ(s.fixed_width,max_width);EXPECT_EQ(s.fixed_bits,max_width*t.values.size());
}
std::optional<std::vector<std::uint64_t>> dictionary_parse(const std::string& bits,const std::unordered_map<std::string,std::uint64_t>& dictionary){
    std::vector<std::uint64_t> out;std::size_t begin=0;
    while(begin<bits.size()){bool matched=false;for(std::size_t end=begin+1;end<=bits.size();++end){const auto it=dictionary.find(bits.substr(begin,end-begin));if(it!=dictionary.end()){out.push_back(it->second);begin=end;matched=true;break;}}if(!matched)return std::nullopt;}
    return out;
}
}
TEST(GammaTrace, AllPositive16BitValuesMatchIndependentDivisionCodewords){
    using namespace datamunge::algorithms;
    for(std::uint64_t n=1;n<=65535;++n){const auto expected=code_reference(n);EXPECT_EQ(elias_gamma_encode_one(n),expected);EXPECT_EQ(elias_gamma_code_length(n),expected.size());const auto t=elias_gamma_decode_trace(expected);ASSERT_EQ(t.values.size(),1u);EXPECT_EQ(t.values[0],n);certificate(t);}
}
TEST(GammaTrace, EveryBinaryStringThrough16BitsMatchesIndependentCodewordDictionary){
    using namespace datamunge::algorithms;
    std::unordered_map<std::string,std::uint64_t> dictionary;for(std::uint64_t n=1;n<=255;++n)dictionary.emplace(code_reference(n),n);
    std::size_t cases=0;
    for(unsigned length=0;length<=16;++length)for(std::size_t code=0;code<(std::size_t{1}<<length);++code){
        std::string bits(length,'0');for(unsigned i=0;i<length;++i)if(code&(std::size_t{1}<<i))bits[length-1-i]='1';const auto expected=dictionary_parse(bits,dictionary);
        if(expected){const auto t=elias_gamma_decode_trace(bits);EXPECT_FALSE(t.encoding);EXPECT_EQ(t.values,*expected);certificate(t);}
        else{EXPECT_THROW((void)elias_gamma_decode(bits),std::invalid_argument);}
        ++cases;
    }
    EXPECT_EQ(cases,131071u);
}
TEST(GammaTrace, FullWidthPowersNeighboursAndSeededSequencesKeepActualSpans){
    using namespace datamunge::algorithms;
    for(unsigned k=0;k<64;++k){const auto p=std::uint64_t{1}<<k;std::vector<std::uint64_t> values{p,p+1};if(p>1)values.push_back(p-1);
        const auto t=elias_gamma_encode_trace(values);EXPECT_TRUE(t.encoding);certificate(t);certificate(elias_gamma_decode_trace(t.bits));}
    const auto maximum=std::numeric_limits<std::uint64_t>::max();const auto hi=elias_gamma_encode_trace({std::uint64_t{1}<<63,maximum});certificate(hi);EXPECT_EQ(hi.words[0].end,127u);EXPECT_EQ(hi.words[1].end,254u);EXPECT_EQ(hi.bits.substr(127),std::string(63,'0')+std::string(64,'1'));
    std::mt19937_64 rng(1975);
    for(int trial=0;trial<600;++trial){std::vector<std::uint64_t> values(rng()%25);for(auto& n:values){n=rng();if(!n)n=1;}
        const auto t=elias_gamma_encode_trace(values);certificate(t);const auto d=elias_gamma_decode_trace(t.bits);certificate(d);EXPECT_EQ(d.values,values);}
}
TEST(GammaTrace, ZeroNonbinaryTruncatedAndOverwideWordsFailWithoutPartialResults){
    using namespace datamunge::algorithms;
    EXPECT_THROW((void)elias_gamma_encode_one(0),std::invalid_argument);
    EXPECT_THROW((void)elias_gamma_encode({1,0,2}),std::invalid_argument);
    EXPECT_THROW((void)elias_gamma_size_summary({1,0,2}),std::invalid_argument);
    EXPECT_THROW((void)elias_gamma_code_length(0),std::invalid_argument);
    EXPECT_THROW((void)elias_gamma_encode_trace({1,0,2},1),std::invalid_argument);
    for(const auto& bits:std::vector<std::string>{"x","0x","1x","10x","2"," \t","10\n",std::string("1\0",2),"1\xc3\xa9","0","000","01","001","0100"}){
        EXPECT_THROW((void)elias_gamma_decode(bits),std::invalid_argument);
        EXPECT_THROW((void)elias_gamma_decode_trace(bits,0),std::invalid_argument);
    }
    for(std::size_t zeros=1;zeros<=63;++zeros){
        EXPECT_THROW((void)elias_gamma_decode(std::string(zeros,'0')),std::invalid_argument);
        for(std::size_t payload=0;payload<zeros;++payload){const auto bits=std::string(zeros,'0')+"1"+std::string(payload,'0');
            EXPECT_THROW((void)elias_gamma_decode(bits),std::invalid_argument);
        }
    }
    for(const auto& bits:std::vector<std::string>{std::string(64,'0'),std::string(64,'0')+"1"+std::string(64,'0'),"1"+std::string(64,'0')}){
        EXPECT_THROW((void)elias_gamma_decode(bits),std::overflow_error);
        EXPECT_THROW((void)elias_gamma_decode_trace(bits,0),std::overflow_error);
    }
    certificate(elias_gamma_encode_trace({}));certificate(elias_gamma_decode_trace(""));EXPECT_TRUE(elias_gamma_decode("").empty());
}
TEST(GammaTrace, ExactCompleteAndOnlineQuotasAndIndependentOwnership){
    using namespace datamunge::algorithms;
    const auto t=elias_gamma_encode_trace({13},9);EXPECT_EQ(t.captured_entries,9u);certificate(t);certificate(elias_gamma_decode_trace(t.bits,9));
    EXPECT_THROW((void)elias_gamma_encode_trace({13},8),std::length_error);
    EXPECT_THROW((void)elias_gamma_decode_trace(t.bits,8),std::length_error);
    EXPECT_THROW((void)elias_gamma_decode_trace(t.bits,6),std::length_error);
    const auto full=elias_gamma_encode_trace({1,2,5},15);for(std::size_t cap=1;cap<15;++cap){
        EXPECT_THROW((void)elias_gamma_encode_trace({1,2,5},cap),std::length_error);
        EXPECT_THROW((void)elias_gamma_decode_trace(full.bits,cap),std::length_error);
    }
    EXPECT_EQ(elias_gamma_encode_trace({13},0).captured_entries,9u);EXPECT_EQ(elias_gamma_decode_trace(t.bits,0).captured_entries,9u);EXPECT_EQ(elias_gamma_encode_trace({},1).captured_entries,0u);EXPECT_EQ(elias_gamma_decode_trace("",1).captured_entries,0u);
    std::vector<std::uint64_t> values{1,2,5};const auto e=elias_gamma_encode_trace(values);values.clear();EXPECT_EQ(e.values.size(),3u);
    std::string bits=e.bits;const auto d=elias_gamma_decode_trace(bits);bits.assign("invalid");EXPECT_EQ(d.bits,e.bits);auto copy=d;copy.words[0].end=99;copy.values[0]=99;copy.bits.clear();EXPECT_EQ(d.words[0].end,1u);EXPECT_EQ(d.values[0],1u);EXPECT_FALSE(d.bits.empty());
    // The input-only quota rejects before copying a large all-binary input or parsing it.
    EXPECT_THROW((void)elias_gamma_decode_trace(std::string(200000,'1'),1),std::length_error);
}
TEST(GammaTrace, OriginalSummaryAndExactEmissionParseGrowthFamilies){
    using namespace datamunge::algorithms;
    const std::vector<std::uint64_t> original{1,2,5,3,100,7,42,1000,8,9};const auto s=elias_gamma_size_summary(original);EXPECT_EQ(s.gamma_bits,74u);EXPECT_EQ(s.fixed_width,10u);EXPECT_EQ(s.fixed_bits,100u);certificate(elias_gamma_encode_trace(original));
    for(std::size_t n:{1,2,4,8,16,32,64,256}){const std::vector<std::uint64_t> values(n,13);const auto e=elias_gamma_encode_trace(values);const auto d=elias_gamma_decode_trace(e.bits);
        EXPECT_EQ(e.work.values,n);EXPECT_EQ(e.work.zero_prefix_bits,3*n);EXPECT_EQ(e.work.payload_bits,4*n);EXPECT_EQ(e.work.bit_visits,7*n);EXPECT_EQ(e.captured_entries,9*n);EXPECT_EQ(d.work.bit_visits,7*n);EXPECT_EQ(d.work.values,n);certificate(e);certificate(d);}
}
