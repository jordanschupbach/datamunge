#include <gtest/gtest.h>
#include <datamunge/algorithms/double_dabble.hpp>
#include <algorithm>
#include <charconv>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
std::uint64_t low(std::uint64_t v,int b){return b==64?v:b==0?0:v%(std::uint64_t{1}<<b);}
std::string text(const std::vector<int>& d){std::string s;for(int digit:d){EXPECT_GE(digit,0);EXPECT_LE(digit,9);s+=static_cast<char>('0'+digit);}return s;}
std::vector<int> field(std::uint64_t prefix,std::size_t size){auto s=std::to_string(prefix);std::vector<int>d(size-s.size());for(char c:s)d.push_back(c-'0');return d;}
void verify(std::uint64_t value,int bits){auto t=double_dabble_trace(value,bits);const auto selected=low(value,bits);ASSERT_EQ(t.value,value);ASSERT_EQ(t.retained_value,selected);ASSERT_EQ(t.capacity,std::to_string(low(std::numeric_limits<std::uint64_t>::max(),bits)).size());ASSERT_EQ(t.num_bits,bits);ASSERT_EQ(t.steps.size(),bits);ASSERT_EQ(text(t.digits),std::to_string(selected));ASSERT_EQ(t.digits,double_dabble(value,bits));ASSERT_EQ(text(t.digits),double_dabble_string(value,bits));ASSERT_EQ(bcd_to_uint64(t.digits),selected);
std::uint64_t prefix=0;std::size_t corrections=0,carries=0;for(std::size_t i=0;i<t.steps.size();++i){auto&s=t.steps[i];ASSERT_EQ(s.bit_index,bits-1-static_cast<int>(i));ASSERT_EQ(s.input_bit,(selected>>s.bit_index)&1);auto old=field(prefix,t.capacity);ASSERT_EQ(s.before,old);auto corrected=old;std::size_t count=0;for(auto&d:corrected)if(d>=5){d+=3;++count;}EXPECT_EQ(s.corrected,corrected);EXPECT_EQ(s.corrections,count);corrections+=count;std::vector<int> after(t.capacity),out(t.capacity);int carry=s.input_bit;for(std::size_t j=t.capacity;j-->0;){int doubled=2*old[j]+carry;after[j]=doubled%10;out[j]=doubled/10;carry=out[j];carries+=carry;}EXPECT_EQ(carry,0);EXPECT_EQ(s.after,after);EXPECT_EQ(s.carry_out,out);prefix=prefix*2+s.input_bit;EXPECT_EQ(s.prefix,prefix);EXPECT_EQ(s.after,field(prefix,t.capacity));}
EXPECT_EQ(t.work.bits_processed,bits);EXPECT_EQ(t.work.digit_checks,bits*t.capacity);EXPECT_EQ(t.work.nibble_shifts,bits*t.capacity);EXPECT_EQ(t.work.add3_corrections,corrections);EXPECT_EQ(t.work.decimal_carries,carries);EXPECT_EQ(corrections,carries);auto w=double_dabble_work(value,bits);EXPECT_EQ(w.bits_processed,t.work.bits_processed);EXPECT_EQ(w.digit_checks,t.work.digit_checks);EXPECT_EQ(w.add3_corrections,t.work.add3_corrections);EXPECT_EQ(w.decimal_carries,t.work.decimal_carries);EXPECT_EQ(w.nibble_shifts,t.work.nibble_shifts);}
}
TEST(DoubleDabbleTrace, HandThirteenAndOriginalCases){verify(13,4);auto t=double_dabble_trace(13,4);ASSERT_EQ(t.steps.back().before,(std::vector<int>{0,6}));EXPECT_EQ(t.steps.back().corrected,(std::vector<int>{0,9}));EXPECT_EQ(t.steps.back().after,(std::vector<int>{1,3}));EXPECT_EQ(t.steps.back().carry_out,(std::vector<int>{0,1}));EXPECT_EQ(t.work.add3_corrections,1);for(auto [v,b]:std::vector<std::pair<std::uint64_t,int>>{{243,8},{0,8},{255,8},{1000,16},{65535,16},{123456,20},{1000000,20},{4294967295ULL,32}})verify(v,b);}
TEST(DoubleDabbleTrace, ExhaustiveTracesThroughTwelveBitsMatchDecimalDoublingAndFormatting){for(int b=0;b<=12;++b)for(std::uint64_t v=0;v<(std::uint64_t{1}<<b);++v)verify(v,b);}
TEST(DoubleDabbleTrace, EverySixteenBitValueMatchesIndependentDecimalFormatting){for(std::uint64_t v=0;v<65536;++v){auto d=double_dabble(v,16);EXPECT_EQ(text(d),std::to_string(v));EXPECT_EQ(double_dabble_string(v,16),std::to_string(v));EXPECT_EQ(bcd_to_uint64(d),v);}}
TEST(DoubleDabbleTrace, AllWidthsMaskHighBitsAndFullWidthArithmeticIsSafe){std::mt19937_64 random(24);for(int b=0;b<=64;++b){verify(0,b);verify(std::numeric_limits<std::uint64_t>::max(),b);for(int i=0;i<32;++i)verify(random(),b);EXPECT_EQ(double_dabble_capacity(b),std::to_string(low(std::numeric_limits<std::uint64_t>::max(),b)).size());}EXPECT_EQ(double_dabble_string(std::numeric_limits<std::uint64_t>::max(),64),"18446744073709551615");EXPECT_EQ(double_dabble_string(243,4),"3");EXPECT_EQ(double_dabble_string(243,0),"0");}
TEST(DoubleDabbleTrace, DecimalThresholdsAndPowerBoundariesRoundTrip){for(std::uint64_t p=1;;){for(auto v:{p-1,p,p+1})verify(v,64);if(p>std::numeric_limits<std::uint64_t>::max()/10)break;p*=10;}for(int b=0;b<64;++b){auto p=std::uint64_t{1}<<b;verify(p-1,64);verify(p,64);verify(p+1,64);}}
TEST(DoubleDabbleTrace, WidthAndInverseDomainChecksAndOverflowPriority){for(int b:{std::numeric_limits<int>::min(),-1,65,std::numeric_limits<int>::max()}){EXPECT_THROW(double_dabble(0,b),std::invalid_argument);
EXPECT_THROW(double_dabble_string(0,b),std::invalid_argument);
EXPECT_THROW(double_dabble_trace(0,b),std::invalid_argument);
EXPECT_THROW(double_dabble_work(0,b),std::invalid_argument);
EXPECT_THROW(double_dabble_capacity(b),std::invalid_argument);}
EXPECT_THROW(bcd_to_uint64({}),std::invalid_argument);
EXPECT_THROW(bcd_to_uint64({1,10}),std::invalid_argument);
EXPECT_THROW(bcd_to_uint64({-1}),std::invalid_argument);
std::vector<int> overflow(21,9);EXPECT_THROW(bcd_to_uint64(overflow),std::overflow_error);
overflow.push_back(-1);EXPECT_THROW(bcd_to_uint64(overflow),std::invalid_argument);
EXPECT_EQ(bcd_to_uint64(std::vector<int>(100,0)),0);EXPECT_EQ(bcd_to_uint64({0,0,2,4,3}),243);
std::vector<int> max_digits;for(char c:std::string("18446744073709551615"))max_digits.push_back(c-'0');EXPECT_EQ(bcd_to_uint64(max_digits),std::numeric_limits<std::uint64_t>::max());max_digits.back()=6;EXPECT_THROW(bcd_to_uint64(max_digits),std::overflow_error);}
TEST(DoubleDabbleTrace, SnapshotsOwnValuesAndBoundedHistoryKeepsRequestedPadding){auto t=double_dabble_trace(13,4);auto copy=t;t.steps.back().before[1]=0;t.digits[0]=9;EXPECT_EQ(copy.steps.back().before[1],6);EXPECT_EQ(copy.digits,(std::vector<int>{1,3}));auto zero=double_dabble_trace(0,64);ASSERT_EQ(zero.steps.size(),64);for(auto&s:zero.steps){EXPECT_EQ(s.after,std::vector<int>(20,0));EXPECT_EQ(s.before.size(),20);}EXPECT_EQ(zero.digits,(std::vector<int>{0}));auto empty=double_dabble_trace(123,0);EXPECT_TRUE(empty.steps.empty());EXPECT_EQ(empty.capacity,1);EXPECT_EQ(empty.retained_value,0);}
