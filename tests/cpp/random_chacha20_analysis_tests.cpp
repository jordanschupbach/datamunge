#include <gtest/gtest.h>
#include <datamunge/random/chacha20_analysis.hpp>
#include <datamunge/algorithms/chacha20.hpp>
#include <datamunge/algorithms/hex_encoding.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
namespace {
namespace r=datamunge::random;
using Words=std::array<std::uint32_t,16>;
using Key=std::array<std::uint8_t,32>;using Nonce=std::array<std::uint8_t,12>;
std::string decode(const std::string& hex) {
  std::string out;for(std::size_t i=0;i<hex.size();i+=2)out.push_back(static_cast<char>(std::stoul(hex.substr(i,2),nullptr,16)));return out;
}
const std::string block_hex="10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
    "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e";
const std::string cipher_hex="6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0b"
    "f91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d8"
    "07ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab7793736"
    "5af90bbf74a35be6b40b8eedf2785e42874d";
Key rfc_key() {Key k{};std::iota(k.begin(),k.end(),std::uint8_t{0});return k;}
std::uint32_t rol(std::uint32_t x,int n) {return static_cast<std::uint32_t>((static_cast<std::uint64_t>(x)<<n)|(x>>(32-n)));}
std::array<std::uint32_t,4> quarter(std::array<std::uint32_t,4> v) {
  // Widened additions and masks form an independent scalar reference.
  v[0]=static_cast<std::uint32_t>((std::uint64_t(v[0])+v[1])&0xffffffffULL);v[3]=rol(v[3]^v[0],16);
  v[2]=static_cast<std::uint32_t>((std::uint64_t(v[2])+v[3])&0xffffffffULL);v[1]=rol(v[1]^v[2],12);
  v[0]=static_cast<std::uint32_t>((std::uint64_t(v[0])+v[1])&0xffffffffULL);v[3]=rol(v[3]^v[0],8);
  v[2]=static_cast<std::uint32_t>((std::uint64_t(v[2])+v[3])&0xffffffffULL);v[1]=rol(v[1]^v[2],7);return v;
}
void column(Words& x) {
  for(std::size_t j=0;j<4;++j) {auto v=quarter({x[j],x[4+j],x[8+j],x[12+j]});for(std::size_t row=0;row<4;++row)x[4*row+j]=v[row];}
}
std::uint32_t read(const std::uint8_t* p) {std::uint64_t v=0,m=1;for(int j=0;j<4;++j){v+=p[j]*m;m*=256;}return static_cast<std::uint32_t>(v);}
std::array<std::uint8_t,64> reference(const Key& key,std::uint32_t counter,const Nonce& nonce,std::vector<Words>* rounds=nullptr) {
  Words initial{0x61707865U,0x3320646eU,0x79622d32U,0x6b206574U};
  for(std::size_t j=0;j<8;++j)initial[4+j]=read(key.data()+4*j);initial[12]=counter;for(std::size_t j=0;j<3;++j)initial[13+j]=read(nonce.data()+4*j);
  auto x=initial;
  // Diagonal rounds are computed by rotating rows, doing columns and restoring rows.
  for(int pair=0;pair<10;++pair) {
    column(x);if(rounds)rounds->push_back(x);
    for(std::size_t row=1;row<4;++row)std::rotate(x.begin()+4*row,x.begin()+5*row,x.begin()+4*row+4);
    column(x);
    for(std::size_t row=1;row<4;++row)std::rotate(x.begin()+4*row,x.begin()+4*row+4-row,x.begin()+4*row+4);
    if(rounds)rounds->push_back(x);
  }
  std::array<std::uint8_t,64> bytes{};
  for(std::size_t j=0;j<16;++j) {std::uint32_t v=x[j]+initial[j];for(std::size_t k=0;k<4;++k){bytes[4*j+k]=v%256;v/=256;}}
  return bytes;
}

std::uint64_t read64(const std::uint8_t* bytes) {
  std::uint64_t result=0;for(int j=7;j>=0;--j)result=result*256+bytes[j];return result;
}
Key key_bytes(const std::array<std::uint32_t,8>& words) {
  Key out{};for(std::size_t i=0;i<32;++i)out[i]=static_cast<std::uint8_t>(words[i/4]>>(8*(i%4)));return out;
}
Nonce nonce_bytes(const std::array<std::uint32_t,3>& words) {
  Nonce out{};for(std::size_t i=0;i<12;++i)out[i]=static_cast<std::uint8_t>(words[i/4]>>(8*(i%4)));return out;
}
}
TEST(ChaChaEngine, CompleteRfcBlockAndWordEndianness) {
  const std::array<std::uint32_t,8> key{0x03020100,0x07060504,0x0b0a0908,0x0f0e0d0c,0x13121110,0x17161514,0x1b1a1918,0x1f1e1d1c};
  r::ChaCha20 rng(key,{0x09000000,0x4a000000,0},1);
  const auto bytes=decode(block_hex);
  for(std::size_t i=0;i<8;++i)EXPECT_EQ(rng.next_u64(),read64(reinterpret_cast<const std::uint8_t*>(bytes.data())+8*i));
  EXPECT_EQ(rng.remaining_u64(),34359738352ULL);
}
TEST(ChaChaEngine, OriginalPublicKeySamplerResultsArePreserved) {
  const std::array<std::uint32_t,8> key{0x11111111,0x22222222,0x33333333,0x44444444,0x55555555,0x66666666,0x77777777,0x88888888};
  r::ChaCha20 rng(key,{7,0,0});
  EXPECT_EQ(rng.next_u64(),6902715600735840086ULL);
  EXPECT_NEAR(rng.uniform01(),0.404146,0.0000005);
  EXPECT_EQ(rng.uniform_u64(1,6),6u);
  EXPECT_EQ(rng.remaining_u64(),34359738365ULL);
}
TEST(ChaChaEngine, ThousandRandomStreamsMatchIndependentRowRotationOracle) {
  std::mt19937 random(8439);
  for(int trial=0;trial<1000;++trial) {
    std::array<std::uint32_t,8> key;std::array<std::uint32_t,3> nonce;
    for(auto& x:key)x=random();for(auto& x:nonce)x=random();
    const std::uint32_t initial=trial%3==0?0xffffffffU-trial%8:random();
    r::ChaCha20 rng(key,nonce,initial);
    const auto available=(std::uint64_t{1}<<32)-initial;
    const auto count=static_cast<std::size_t>(std::min<std::uint64_t>(available*8,1+random()%99));
    // Begin inspection at different offsets inside the buffer.
    const std::size_t skip=std::min<std::size_t>(trial%8,count);
    const auto kb=key_bytes(key);const auto nb=nonce_bytes(nonce);
    for(std::size_t i=0;i<skip;++i) {
      const auto block=reference(kb,initial+static_cast<std::uint32_t>(i/8),nb);
      EXPECT_EQ(rng.next_u64(),read64(block.data()+8*(i%8)));
    }
    const auto trace=r::chacha20_trace(rng,count-skip);
    for(std::size_t j=0;j<trace.size();++j) {
      const auto i=j+skip;const auto block=reference(kb,initial+static_cast<std::uint32_t>(i/8),nb);
      EXPECT_EQ(trace[j].value,read64(block.data()+8*(i%8)));
      EXPECT_EQ(trace[j].counter,initial+i/8);EXPECT_EQ(trace[j].word_index,i%8);
      EXPECT_EQ(trace[j].remaining_before,available*8-i);
    }
    EXPECT_EQ(rng.remaining_u64(),available*8-count);
  }
}
TEST(ChaChaEngine, FinalCounterUsesAllEightWordsThenRepeatedlyThrowsWithoutNonceCarry) {
  std::array<std::uint32_t,8> key{};const std::array<std::uint32_t,3> nonce{0xffffffffU,17,29};
  const auto expected=reference(key_bytes(key),0xffffffffU,nonce_bytes(nonce));
  r::ChaCha20 rng(key,nonce,0xffffffffU);
  for(int i=0;i<8;++i) {
    EXPECT_EQ(rng.remaining_u64(),8-i);EXPECT_EQ(rng.next_u64(),read64(expected.data()+8*i));
  }
  for(int i=0;i<3;++i) {EXPECT_THROW(rng.next_u64(),std::length_error);EXPECT_EQ(rng.remaining_u64(),0u);}
  EXPECT_TRUE(r::chacha20_trace(rng,0).empty());
  EXPECT_THROW(r::chacha20_trace(rng,1),std::length_error);
}
TEST(ChaChaEngine, TwoFinalBlocksAndEveryTailHaveExactCapacity) {
  for(std::size_t skip=0;skip<=16;++skip) {
    r::ChaCha20 rng(std::array<std::uint32_t,8>{},{},0xfffffffeU);
    for(std::size_t i=0;i<skip;++i)(void)rng.next_u64();
    EXPECT_EQ(rng.remaining_u64(),16-skip);
    auto copy=rng;EXPECT_THROW(r::chacha20_trace(rng,17-skip),std::length_error);
    EXPECT_EQ(rng.remaining_u64(),copy.remaining_u64());
    const auto tail=r::chacha20_trace(rng,16-skip);
    for(const auto& record:tail)EXPECT_EQ(record.value,copy.next_u64());
    EXPECT_EQ(rng.remaining_u64(),0u);
    EXPECT_THROW(rng.next_u64(),std::length_error);
  }
}
TEST(ChaChaEngine, OversizedAndEmptyTraceDoNotAdvanceRawOrNormalState) {
  r::ChaCha20 rng(17);(void)rng.normal();auto copy=rng;
  EXPECT_THROW(r::chacha20_trace(rng,std::numeric_limits<std::size_t>::max()),std::length_error);
  EXPECT_TRUE(r::chacha20_trace(rng,0).empty());
  EXPECT_EQ(rng.remaining_u64(),copy.remaining_u64());
  EXPECT_DOUBLE_EQ(rng.normal(),copy.normal());EXPECT_EQ(rng.next_u64(),copy.next_u64());
}
TEST(ChaChaEngine, SeedExpansionMatchesLegacyFourSplitMixDrawsForThousandSeeds) {
  std::mt19937_64 random(2015);
  for(int trial=0;trial<1000;++trial) {
    const auto seed=trial==0?0:random();auto state=seed;std::array<std::uint32_t,8> expected{};
    for(std::size_t i=0;i<8;i+=2) {
      auto x=(state+=0x9e3779b97f4a7c15ULL);
      x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;x^=x>>31;
      expected[i]=static_cast<std::uint32_t>(x);expected[i+1]=static_cast<std::uint32_t>(x>>32);
    }
    EXPECT_EQ(r::chacha20_seed_key(seed),expected);
    r::ChaCha20 scalar(seed),expanded(expected);for(int i=0;i<17;++i)EXPECT_EQ(scalar.next_u64(),expanded.next_u64());
  }
  const auto zero=r::chacha20_seed_key(0);EXPECT_EQ(zero[0],0x7b1dcdafu);EXPECT_EQ(zero[1],0xe220a839u);
}
TEST(ChaChaEngine, CopyAssignmentPreservesPartialBufferAndNormalCache) {
  r::ChaCha20 original(42);(void)original.normal();(void)original.next_u64();
  auto copy=original;r::ChaCha20 assigned(0);assigned=original;
  const auto spare=original.normal();EXPECT_DOUBLE_EQ(spare,copy.normal());EXPECT_DOUBLE_EQ(spare,assigned.normal());
  auto baseline=original;assigned=original;
  for(int i=0;i<100;++i)EXPECT_EQ(original.next_u64(),assigned.next_u64());
  EXPECT_NE(original.remaining_u64(),baseline.remaining_u64());
}
TEST(ChaChaEngine, CachedNormalMayRemainAfterRawExhaustion) {
  r::ChaCha20 rng(std::array<std::uint32_t,8>{},{},0xffffffffU);
  (void)rng.normal();auto cached=rng;
  ASSERT_GT(rng.remaining_u64(),0u);
  (void)r::chacha20_trace(rng,static_cast<std::size_t>(rng.remaining_u64()));
  EXPECT_DOUBLE_EQ(rng.normal(),cached.normal());EXPECT_EQ(rng.remaining_u64(),0u);
  EXPECT_THROW(rng.normal(),std::length_error);
}
TEST(ChaChaEngine, UniformMappingUsesNextRawWordsAndRejectionCanConsumeSeveral) {
  r::ChaCha20 sampled(2026),raw(2026);
  for(int i=0;i<100;++i)EXPECT_DOUBLE_EQ(sampled.uniform01(),std::ldexp(static_cast<double>(raw.next_u64()/2048),-53));
  const std::uint64_t range=(std::uint64_t{1}<<63)+1,threshold=(std::uint64_t{1}<<63)-1;
  std::size_t rejected=0;
  for(int i=0;i<1000;++i) {
    unsigned __int128 product;
    do {product=static_cast<unsigned __int128>(raw.next_u64())*range;if(static_cast<std::uint64_t>(product)<threshold)++rejected;}
    while(static_cast<std::uint64_t>(product)<threshold);
    EXPECT_EQ(sampled.uniform_u64(0,range-1),static_cast<std::uint64_t>(product>>64));
    EXPECT_EQ(sampled.remaining_u64(),raw.remaining_u64());
  }
  EXPECT_GT(rejected,0u);
}
TEST(ChaChaEngine, BaseDispatchAndInvalidSamplerValidationPreserveExpectedState) {
  r::ChaCha20 rng(std::array<std::uint32_t,8>{},{},0xffffffffU);r::RandomGenerator& base=rng;
  EXPECT_THROW(base.uniform_u64(2,1),std::invalid_argument);
  EXPECT_THROW(base.uniform(1,1),std::invalid_argument);
  EXPECT_EQ(rng.remaining_u64(),8u);
  for(int i=0;i<8;++i)(void)base.next_u64();
  EXPECT_THROW(base.uniform01(),std::length_error);
  EXPECT_THROW(base.uniform_u64(0,0),std::length_error);
}
