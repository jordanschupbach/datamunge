#include <gtest/gtest.h>
#include <datamunge/algorithms/checksums.hpp>
#include <datamunge/image/detail/crc32.hpp>
#include <algorithm>
#include <bitset>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::uint32_t bitwise(std::string_view bytes) {
  std::uint32_t raw=0xFFFFFFFFu;
  for(const unsigned char byte:bytes) {
    raw^=byte;
    for(int k=0;k<8;++k)raw=(raw&1)?(raw>>1)^0xEDB88320u:raw>>1;
  }
  return raw^0xFFFFFFFFu;
}
const std::vector<std::size_t> generator_bits{0,6,9,10,16,20,21,22,24,25,27,28,30,31,32};
}
TEST(Crc32Trace, StandardValuesAndActualFirstLookup) {
  EXPECT_EQ(crc32(""),0u);EXPECT_EQ(crc32("123456789"),0xCBF43926u);
  EXPECT_EQ(crc32("The quick brown fox"),0xB74574DEu);
  const auto t=crc32_trace("1");ASSERT_EQ(t.steps.size(),1u);
  const auto& s=t.steps[0];EXPECT_EQ(s.index,0u);EXPECT_EQ(s.byte,0x31);
  EXPECT_EQ(s.raw_before,0xFFFFFFFFu);EXPECT_EQ(s.lookup_index,0xCE);
  EXPECT_EQ(s.table_value,0x7CDCEFB7u);EXPECT_EQ(s.raw_after,0x7C231048u);
  EXPECT_EQ(t.raw,s.raw_after);EXPECT_EQ(t.checksum,0x83DCEFB7u);
}
TEST(Crc32Trace, ExhaustiveOneAndTwoByteMessagesMatchIndependentBitLoop) {
  for(unsigned a=0;a<256;++a) {
    const std::string one(1,static_cast<char>(a));EXPECT_EQ(crc32(one),bitwise(one));
    for(unsigned b=0;b<256;++b) {
      const auto message=one+static_cast<char>(b);const auto t=crc32_trace(message,2);
      ASSERT_EQ(t.checksum,bitwise(message));ASSERT_EQ(crc32(message),t.checksum);
      ASSERT_EQ(t.steps.size(),2u);
      ASSERT_EQ(t.steps[0].raw_after,bitwise(one)^0xFFFFFFFFu);
      ASSERT_EQ(t.steps[1].raw_before,t.steps[0].raw_after);
      ASSERT_EQ(t.steps[1].lookup_index,static_cast<std::uint8_t>((t.steps[1].raw_before^b)&255));
      ASSERT_EQ(t.steps[1].raw_after,(t.steps[1].raw_before>>8)^t.steps[1].table_value);
    }
  }
}
TEST(Crc32Trace, RandomBinaryInputsAndExistingCodecStreamingAgree) {
  std::mt19937_64 rng(20261107);
  for(int trial=0;trial<1000;++trial) {
    std::string message(rng()%2049,'\0');for(auto& c:message)c=static_cast<char>(rng());
    const auto t=crc32_trace(message);ASSERT_EQ(t.checksum,bitwise(message));
    const auto cut=message.size()/2;
    const auto* data=reinterpret_cast<const unsigned char*>(message.data());
    const auto first=datamunge::image::detail::crc32_update(0,data,cut);
    EXPECT_EQ(datamunge::image::detail::crc32_update(first,data+cut,message.size()-cut),t.checksum);
    if(!message.empty())EXPECT_EQ(t.steps.back().raw_after,t.raw);
  }
}
TEST(Crc32Trace, OriginalCompleteSingleBitCensusAndIndependentPopcounts) {
  const std::string message="The quick brown fox";const auto scan=crc32_flip_scan(message,2907);
  EXPECT_EQ(scan.checksum,0xB74574DEu);EXPECT_EQ(scan.detected,152u);EXPECT_TRUE(scan.has_flips);
  ASSERT_EQ(scan.flips.size(),152u);unsigned minimum=32,maximum=0,total=0;
  std::set<std::uint32_t> syndromes;
  for(std::size_t bit=0;bit<152;++bit) {
    const auto changed=crc32_flip_bits(message,{bit});const auto& f=scan.flips[bit];
    EXPECT_EQ(f.bit,bit);EXPECT_EQ(f.checksum,bitwise(changed));
    EXPECT_EQ(f.syndrome,f.checksum^scan.checksum);
    const auto count=std::bitset<32>(f.syndrome).count();EXPECT_EQ(f.changed_bits,count);
    total+=count;minimum=std::min(minimum,f.changed_bits);maximum=std::max(maximum,f.changed_bits);
    syndromes.insert(f.syndrome);
  }
  EXPECT_EQ(syndromes.size(),152u);EXPECT_EQ(total,2350u);
  EXPECT_DOUBLE_EQ(scan.mean_changed_bits,2350./152);
  EXPECT_EQ(scan.minimum_changed_bits,minimum);EXPECT_EQ(scan.maximum_changed_bits,maximum);
}
TEST(Crc32Trace, EqualLengthSyndromesIgnorePayloadAndComposeLinearly) {
  std::mt19937 rng(4);
  for(int trial=0;trial<300;++trial) {
    const std::size_t n=1+rng()%64;std::string a(n,0),b(n,0);
    for(auto& c:a)c=static_cast<char>(rng());for(auto& c:b)c=static_cast<char>(rng());
    const auto i=rng()%(n*8),j=rng()%(n*8);
    const auto single=crc32(crc32_flip_bits(a,{i}))^crc32(a);
    EXPECT_EQ(single,crc32(crc32_flip_bits(b,{i}))^crc32(b));
    const auto other=crc32(crc32_flip_bits(a,{j}))^crc32(a);
    EXPECT_EQ(single^other,crc32(crc32_flip_bits(a,{i,j}))^crc32(a));
  }
}
TEST(Crc32Trace, AllDoubleBitErrorsOfOriginalMessageAreDetected) {
  const auto scan=crc32_flip_scan("The quick brown fox");
  for(std::size_t i=0;i<scan.flips.size();++i)for(std::size_t j=i+1;j<scan.flips.size();++j) {
    ASSERT_NE(scan.flips[i].syndrome^scan.flips[j].syndrome,0u);
  }
}
TEST(Crc32Trace, ExhaustiveShortBurstsAndSampledWidth32BurstsDetect) {
  const std::string original(8,'\0');const auto baseline=crc32(original);
  for(std::size_t width=1;width<=12;++width) {
    const std::size_t patterns=width<=2?1:std::size_t{1}<<(width-2);
    for(std::size_t start=0;start+width<=64;++start)for(std::size_t pattern=0;pattern<patterns;++pattern) {
      std::vector<std::size_t> bits{start};if(width>1)bits.push_back(start+width-1);
      for(std::size_t i=1;i+1<width;++i)if(pattern>>(i-1)&1)bits.push_back(start+i);
      ASSERT_NE(crc32(crc32_flip_bits(original,bits)),baseline);
    }
  }
  std::mt19937 rng(32);
  for(std::size_t width=13;width<=32;++width)for(int trial=0;trial<100;++trial) {
    const auto start=rng()%(65-width);std::vector<std::size_t> bits{start,start+width-1};
    for(std::size_t i=1;i+1<width;++i)if(rng()&1)bits.push_back(start+i);
    ASSERT_NE(crc32(crc32_flip_bits(original,bits)),baseline);
  }
}
TEST(Crc32Trace, OddWeightSpan33GeneratorWitnessIsUndetectedAtEveryOffset) {
  const std::string original="The quick brown fox";const auto baseline=crc32(original);
  ASSERT_EQ(generator_bits.size(),15u);
  for(std::size_t offset=0;offset+33<=original.size()*8;++offset) {
    auto positions=generator_bits;for(auto& p:positions)p+=offset;
    const auto changed=crc32_flip_bits(original,positions);
    ASSERT_NE(changed,original);ASSERT_EQ(crc32(changed),baseline);
  }
  const auto zero=std::string(5,0);const auto changed=crc32_flip_bits(zero,generator_bits);
  EXPECT_EQ(crc32(zero),crc32(changed));
  EXPECT_EQ(static_cast<unsigned char>(changed[0]),0x41);EXPECT_EQ(static_cast<unsigned char>(changed[4]),1);
}
TEST(Crc32Trace, CompleteCapsEmptyConventionsAndOwnedCopies) {
  EXPECT_NO_THROW(crc32_trace("abc",3));EXPECT_THROW(crc32_trace("abc",2),std::length_error);
  EXPECT_NO_THROW(crc32_trace("abc",0));
  EXPECT_NO_THROW(crc32_flip_scan("A",9));EXPECT_THROW(crc32_flip_scan("A",8),std::length_error);
  EXPECT_NO_THROW(crc32_flip_scan("A",0));EXPECT_THROW(crc32_flip_scan("The quick brown fox",2906),std::length_error);
  EXPECT_THROW(crc32_flip_bits("A",{0,8}),std::invalid_argument);
  EXPECT_THROW(crc32_flip_bits("",{0}),std::invalid_argument);
  EXPECT_EQ(crc32_flip_bits("A",{0,0,1}),"C");
  const auto empty=crc32_trace("");EXPECT_EQ(empty.raw,0xFFFFFFFFu);EXPECT_EQ(empty.checksum,0);EXPECT_TRUE(empty.steps.empty());
  const auto census=crc32_flip_scan("");EXPECT_FALSE(census.has_flips);EXPECT_TRUE(census.flips.empty());
  EXPECT_EQ(census.detected,0);EXPECT_EQ(census.minimum_changed_bits,0);EXPECT_EQ(census.maximum_changed_bits,0);EXPECT_EQ(census.mean_changed_bits,0);
  std::string message="abc";auto trace=crc32_trace(message);auto scan=crc32_flip_scan(message);
  message[0]='z';trace.steps[0].raw_after=0;scan.flips[0].syndrome=0;
  EXPECT_EQ(trace.checksum,crc32("abc"));EXPECT_NE(crc32_trace("abc").steps[0].raw_after,0);
  EXPECT_NE(crc32_flip_scan("abc").flips[0].syndrome,0);
}
