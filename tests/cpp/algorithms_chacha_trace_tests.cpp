#include <gtest/gtest.h>
#include <datamunge/algorithms/chacha20.hpp>
#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/byte_histogram.hpp>
#include <random>
#include <numeric>
#include <sstream>
#include <iomanip>
namespace {
namespace a=datamunge::algorithms;
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
void verify(const Key& key,std::uint32_t counter,const Nonce& nonce) {
  const auto trace=a::chacha20_block_trace(key,counter,nonce);std::vector<Words> rounds;
  const auto expected=reference(key,counter,nonce,&rounds);EXPECT_EQ(trace.output,expected);EXPECT_EQ(a::chacha20_block(key,counter,nonce),expected);
  ASSERT_EQ(trace.quarter_rounds.size(),80u);auto state=trace.initial;
  for(std::size_t i=0;i<80;++i) {
    const auto& step=trace.quarter_rounds[i];const auto round=i/4,q=i%4;EXPECT_EQ(step.round,round);
    std::array<std::size_t,4> indices{};for(std::size_t row=0;row<4;++row)indices[row]=4*row+(q+(round%2?row:0))%4;
    EXPECT_EQ(step.indices,indices);std::array<std::uint32_t,4> before{};
    for(std::size_t j=0;j<4;++j)before[j]=state[indices[j]];EXPECT_EQ(step.before,before);const auto after=quarter(before);EXPECT_EQ(step.after,after);
    for(std::size_t j=0;j<4;++j)state[indices[j]]=after[j];
    if(q==3){EXPECT_EQ(state,rounds[round]);EXPECT_EQ(state,trace.round_states[round]);}
  }
  for(std::size_t j=0;j<16;++j)EXPECT_EQ(trace.feed_forward[j],static_cast<std::uint32_t>(state[j]+trace.initial[j]));
}
}
TEST(ChaChaTrace, RfcQuarterAndAllActualCompoundStages) {
  const std::array<std::uint32_t,4> input{0x11111111,0x01020304,0x9b8d6f43,0x01234567},expected{0xea2a92f4,0xcb1cf8ce,0x4581472e,0x5881c4bb};
  const auto t=a::chacha20_quarter_trace(input);EXPECT_EQ(t.input,input);EXPECT_EQ(t.output,expected);EXPECT_EQ(t.stages[3],expected);EXPECT_EQ(a::chacha20_quarter_round(input),expected);
  EXPECT_EQ(t.stages[0][0],0x12131415u);EXPECT_EQ(t.stages[0][3],rol(input[3]^0x12131415u,16));
}
TEST(ChaChaTrace, QuarterInverseAndIndependentReferenceOnTenThousandInputs) {
  std::mt19937 rng(8439);
  for(int trial=0;trial<10000;++trial) {
    std::array<std::uint32_t,4> input{};for(auto& v:input)v=rng();const auto t=a::chacha20_quarter_trace(input);EXPECT_EQ(t.output,quarter(input));
    auto [x,y,z,w]=t.output;
    y=rol(y,25)^z;z-=w;w=rol(w,24)^x;x-=y;y=rol(y,20)^z;z-=w;w=rol(w,16)^x;x-=y;
    EXPECT_EQ((std::array<std::uint32_t,4>{x,y,z,w}),input);
  }
  EXPECT_EQ(a::detail::chacha_rotl(0x12345678,0),0x12345678u);EXPECT_EQ(a::detail::chacha_rotl(0x12345678,32),0x12345678u);
  EXPECT_EQ(a::detail::chacha_rotl(0x12345678,-1),0x091a2b3cu);
}
TEST(ChaChaTrace, RfcBlockFullBytesMixedStateAndFeedForward) {
  const Nonce nonce{0,0,0,9,0,0,0,0x4a,0,0,0,0};const auto t=a::chacha20_block_trace(rfc_key(),1,nonce);
  const auto bytes=decode(block_hex);for(std::size_t i=0;i<64;++i)EXPECT_EQ(t.output[i],static_cast<unsigned char>(bytes[i]));
  EXPECT_EQ(t.round_states[19],(Words{0x837778ab,0xe238d763,0xa67ae21e,0x5950bb2f,0xc4f2d0c7,0xfc62bb2f,0x8fa018fc,0x3f5ec7b7,0x335271c2,0xf29489f3,0xeabda8fc,0x82e46ebd,0xd19c12b4,0xb04e16de,0x9e83d0cb,0x4e3c50a2}));
  EXPECT_EQ(t.feed_forward[0],0xe4e7f110u);EXPECT_EQ(t.initial[12],1u);EXPECT_EQ(t.initial[13],0x09000000u);verify(rfc_key(),1,nonce);
}
TEST(ChaChaTrace, RfcFull114ByteCipherAndActualTail) {
  const Nonce nonce{0,0,0,0,0,0,0,0x4a,0,0,0,0};
  const std::string plain="Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
  const auto trace=a::chacha20_encrypt_trace(rfc_key(),1,nonce,plain);EXPECT_EQ(trace.output,decode(cipher_hex));EXPECT_EQ(trace.output.size(),114u);ASSERT_EQ(trace.blocks.size(),2u);
  EXPECT_EQ(trace.blocks[0].counter,1u);EXPECT_EQ(trace.blocks[0].offset,0u);EXPECT_EQ(trace.blocks[0].bytes,64u);
  EXPECT_EQ(trace.blocks[1].counter,2u);EXPECT_EQ(trace.blocks[1].offset,64u);EXPECT_EQ(trace.blocks[1].bytes,50u);
  EXPECT_EQ(a::chacha20_encrypt(rfc_key(),1,nonce,trace.output),plain);
}
TEST(ChaChaTrace, FiveHundredRandomBlocksAgainstRowRotationReference) {
  std::mt19937 rng(2008);
  for(int i=0;i<500;++i) {Key key;Nonce nonce;for(auto& v:key)v=rng();for(auto& v:nonce)v=rng();verify(key,rng(),nonce);}
}
TEST(ChaChaTrace, BinaryMessagesAndActualXorBlocksAgainstIndependentReference) {
  std::mt19937 rng(1);
  for(int trial=0;trial<500;++trial) {
    const std::size_t n=trial<5?std::vector<std::size_t>{0,1,63,64,65}[trial]:rng()%4098;
    Key key;Nonce nonce;for(auto& v:key)v=rng();for(auto& v:nonce)v=rng();std::string input(n,'\0');for(auto& v:input)v=static_cast<char>(rng());
    const auto initial=static_cast<std::uint32_t>(rng()%1000);const auto t=a::chacha20_encrypt_trace(key,initial,nonce,input);
    ASSERT_EQ(t.output.size(),n);ASSERT_EQ(t.blocks.size(),n/64+(n%64!=0));
    for(std::size_t block=0;block<t.blocks.size();++block) {
      const auto expected=reference(key,initial+block,nonce);const auto& actual=t.blocks[block];
      EXPECT_EQ(actual.keystream,expected);EXPECT_EQ(actual.counter,initial+block);EXPECT_EQ(actual.offset,64*block);EXPECT_EQ(actual.bytes,std::min<std::size_t>(64,n-64*block));
      for(std::size_t j=0;j<actual.bytes;++j)EXPECT_EQ(static_cast<unsigned char>(t.output[actual.offset+j]),static_cast<unsigned char>(input[actual.offset+j])^expected[j]);
    }
    EXPECT_EQ(a::chacha20_encrypt(key,initial,nonce,input),t.output);EXPECT_EQ(a::chacha20_encrypt(key,initial,nonce,t.output),input);
  }
}
TEST(ChaChaTrace, CounterCapacityAndNoAllocationRangeBoundaries) {
  constexpr auto last=std::numeric_limits<std::uint32_t>::max();
  EXPECT_EQ(a::chacha20_capacity_bytes(0),274877906944ULL);EXPECT_EQ(a::chacha20_capacity_bytes(1),274877906880ULL);EXPECT_EQ(a::chacha20_capacity_bytes(last),64u);
  EXPECT_EQ(a::chacha20_blocks_required(0,274877906944ULL),4294967296ULL);EXPECT_EQ(a::chacha20_blocks_required(last,0),0u);EXPECT_EQ(a::chacha20_blocks_required(last,64),1u);
  EXPECT_THROW(a::chacha20_blocks_required(last,65),std::length_error);
  EXPECT_THROW(a::chacha20_blocks_required(0,274877906945ULL),std::length_error);
  EXPECT_THROW(a::chacha20_blocks_required(0,std::numeric_limits<std::uint64_t>::max()),std::length_error);
  EXPECT_EQ(a::chacha20_blocks_required(last-1,128),2u);
  EXPECT_THROW(a::chacha20_blocks_required(last-1,129),std::length_error);
  EXPECT_EQ(a::chacha20_encrypt({},last,{},std::string(64,'x')).size(),64u);
  EXPECT_THROW(a::chacha20_encrypt({},last,{},std::string(65,'x')),std::length_error);
  EXPECT_THROW(a::chacha20_encrypt_trace({},last,{},std::string(65,'x')),std::length_error);
  const auto t=a::chacha20_encrypt_trace({},last-1,{},std::string(128,'x'));EXPECT_EQ(t.blocks.back().counter,last);
}
TEST(ChaChaTrace, WholeCallBlockCapAndEmptyInput) {
  EXPECT_THROW(a::chacha20_encrypt_trace({},0,{},std::string(65,'x'),1),std::length_error);
  EXPECT_EQ(a::chacha20_encrypt_trace({},0,{},std::string(65,'x'),2).blocks.size(),2u);
  EXPECT_TRUE(a::chacha20_encrypt_trace({},0xffffffff,{},"",1).blocks.empty());EXPECT_TRUE(a::chacha20_encrypt({},0xffffffff,{},"").empty());
}
TEST(ChaChaTrace, OneShotChunkAlignmentAndUnauthenticatedBitFlip) {
  const std::string input(100,'L');const auto whole=a::chacha20_encrypt({},1,{},input);
  const auto aligned=a::chacha20_encrypt({},1,{},input.substr(0,64))+a::chacha20_encrypt({},2,{},input.substr(64));EXPECT_EQ(aligned,whole);
  const auto misaligned=a::chacha20_encrypt({},1,{},input.substr(0,63))+a::chacha20_encrypt({},2,{},input.substr(63));EXPECT_NE(misaligned,whole);
  auto changed=whole;changed[0]^=1;auto plain=a::chacha20_encrypt({},1,{},changed);EXPECT_EQ(plain[0],'M');EXPECT_EQ(plain.substr(1),input.substr(1));
}
TEST(ChaChaTrace, OwnedSnapshotsAndInputsStayUnchanged) {
  const auto key=rfc_key();const Nonce nonce{};const auto t=a::chacha20_block_trace(key,1,nonce);auto copy=t;copy.initial[4]=0;copy.quarter_rounds[0].before[0]=0;copy.round_states[0][0]=0;
  EXPECT_EQ(t.initial[4],0x03020100u);EXPECT_NE(t.quarter_rounds[0].before[0],0u);EXPECT_NE(t.round_states[0][0],0u);
  EXPECT_EQ(key,rfc_key());EXPECT_EQ(nonce,Nonce{});
  const auto e=a::chacha20_encrypt_trace(key,1,nonce,std::string("\0\xff",2));auto other=e;other.output.clear();other.blocks[0].keystream[0]^=1;EXPECT_EQ(e.output.size(),2u);EXPECT_NE(other.blocks[0].keystream,e.blocks[0].keystream);
}
TEST(ByteAnalysis, HexAndEveryByteHistogramBinAgainstIndependentFormatting) {
  std::string input;std::ostringstream expected;expected<<std::hex<<std::setfill('0');
  for(int i=0;i<256;++i){input.push_back(static_cast<char>(i));expected<<std::setw(2)<<i;}
  EXPECT_EQ(a::to_hex(std::string_view(input)),expected.str());EXPECT_EQ(a::to_hex(std::array<std::uint8_t,0>{}),"");EXPECT_EQ(a::to_hex(std::string_view{}),"");
  EXPECT_EQ(a::to_hex(std::array<std::uint8_t,3>{0,128,255}),"0080ff");
  for(std::size_t bins=1;bins<=256;bins*=2) {const auto counts=a::byte_histogram(input,bins);ASSERT_EQ(counts.size(),bins);for(auto count:counts)EXPECT_EQ(count,256/bins);EXPECT_EQ(std::accumulate(counts.begin(),counts.end(),std::size_t{0}),256u);}
  const auto zero=a::byte_histogram(std::string(4096,'\0'),16);EXPECT_EQ(zero[0],4096u);EXPECT_EQ(std::accumulate(zero.begin(),zero.end(),std::size_t{0})/16,256u);
}
TEST(ByteAnalysis, EmptyHistogramsAndInvalidBins) {
  EXPECT_EQ(a::byte_histogram({},16),(std::vector<std::size_t>(16,0)));
  for(std::size_t bins:{0u,3u,5u,255u,257u})EXPECT_THROW(a::byte_histogram({},bins),std::invalid_argument);
  EXPECT_THROW(a::byte_histogram({},std::numeric_limits<std::size_t>::max()),std::invalid_argument);
}
