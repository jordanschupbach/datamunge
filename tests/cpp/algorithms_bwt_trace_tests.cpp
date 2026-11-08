#include <gtest/gtest.h>
#include <datamunge/algorithms/burrows_wheeler.hpp>
#include <datamunge/algorithms/move_to_front.hpp>
#include <datamunge/algorithms/run_length_encoding.hpp>
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::algorithms;
bool byte_less(const std::string& a,const std::string& b) {
  return std::lexicographical_compare(a.begin(),a.end(),b.begin(),b.end(),
      [](unsigned char x,unsigned char y){return x<y;});
}
struct MatrixReference {
  BWTResult result;
  std::vector<std::size_t> starts;
  std::size_t comparisons=0,bytes=0,ties=0;
};
MatrixReference materialized_reference(const std::string& input) {
  MatrixReference reference;const auto n=input.size();
  std::vector<std::string> rotations;
  for (std::size_t start=0;start<n;++start) rotations.push_back(input.substr(start)+input.substr(0,start));
  reference.starts.resize(n);std::iota(reference.starts.begin(),reference.starts.end(),std::size_t{0});
  std::sort(reference.starts.begin(),reference.starts.end(),[&](std::size_t a,std::size_t b) {
    ++reference.comparisons;
    const auto mismatch=std::mismatch(rotations[a].begin(),rotations[a].end(),rotations[b].begin());
    const auto prefix=static_cast<std::size_t>(mismatch.first-rotations[a].begin());
    reference.bytes+=prefix<n?prefix+1:n;
    if (prefix==n) {++reference.ties;return a<b;}
    return byte_less(rotations[a],rotations[b]);
  });
  for (std::size_t row=0;row<n;++row) {
    reference.result.transformed.push_back(rotations[reference.starts[row]].back());
    if (reference.starts[row]==0) reference.result.index=row;
  }
  return reference;
}
std::string matrix_inverse(const BWTResult& encoded) {
  std::vector<std::string> rows(encoded.transformed.size());
  for (std::size_t step=0;step<rows.size();++step) {
    for (std::size_t row=0;row<rows.size();++row) rows[row].insert(rows[row].begin(),encoded.transformed[row]);
    std::stable_sort(rows.begin(),rows.end(),byte_less);
  }
  return rows.empty()?std::string{}:rows[encoded.index];
}
void check_mapping(const std::string& last,const BWTLFMapping& mapping) {
  const auto n=last.size();ASSERT_EQ(mapping.lf.size(),n);ASSERT_EQ(mapping.first_column.size(),n);ASSERT_EQ(mapping.occurrence_rank.size(),n);
  std::array<std::size_t,256> counts{};for (unsigned char value:last) ++counts[value];
  EXPECT_EQ(mapping.counts,counts);
  std::size_t prefix=0;for (std::size_t value=0;value<256;++value) {EXPECT_EQ(mapping.starts[value],prefix);prefix+=counts[value];}
  std::vector<std::size_t> paired(n);std::iota(paired.begin(),paired.end(),std::size_t{0});
  std::stable_sort(paired.begin(),paired.end(),[&](std::size_t a,std::size_t b){return static_cast<unsigned char>(last[a])<static_cast<unsigned char>(last[b]);});
  std::string first;std::vector<std::size_t> inverse(n);
  for (std::size_t row=0;row<n;++row) {first.push_back(last[paired[row]]);inverse[paired[row]]=row;}
  EXPECT_EQ(mapping.first_column,first);EXPECT_EQ(mapping.lf,inverse);
  for (std::size_t row=0;row<n;++row) {
    const auto rank=static_cast<std::size_t>(std::count(last.begin(),last.begin()+row,last[row]));
    EXPECT_EQ(mapping.occurrence_rank[row],rank);EXPECT_LT(mapping.lf[row],n);
    EXPECT_EQ(mapping.first_column[mapping.lf[row]],last[row]);
  }
}
void check_input(const std::string& input,bool matrix_decode=true) {
  const auto saved=input;const auto oracle=materialized_reference(input);const auto trace=bwt_transform_trace(input);
  const auto encoded=bwt_transform(input);ASSERT_EQ(input,saved);
  ASSERT_EQ(encoded.transformed,oracle.result.transformed);ASSERT_EQ(encoded.index,oracle.result.index);
  EXPECT_EQ(trace.input,input);EXPECT_EQ(trace.result.transformed,encoded.transformed);EXPECT_EQ(trace.result.index,encoded.index);
  EXPECT_EQ(trace.rotation_starts,oracle.starts);EXPECT_EQ(trace.rotation_comparisons,oracle.comparisons);
  EXPECT_EQ(trace.byte_comparisons,oracle.bytes);EXPECT_EQ(trace.equal_rotation_comparisons,oracle.ties);
  const auto inverse=bwt_inverse_trace(encoded);EXPECT_EQ(inverse.output,input);EXPECT_EQ(bwt_inverse(encoded),input);
  if (matrix_decode) EXPECT_EQ(matrix_inverse(encoded),input);
  EXPECT_EQ(inverse.input.transformed,encoded.transformed);EXPECT_EQ(inverse.input.index,encoded.index);
  check_mapping(encoded.transformed,inverse.mapping);check_mapping(encoded.transformed,bwt_lf_mapping(encoded.transformed));
  ASSERT_EQ(inverse.steps.size(),input.size());
  auto row=encoded.index;std::string recovered(input.size(),'\0');
  for (std::size_t k=0;k<input.size();++k) {
    const auto& step=inverse.steps[k];EXPECT_EQ(step.row,row);EXPECT_EQ(step.next_row,inverse.mapping.lf[row]);
    EXPECT_EQ(step.output_index,input.size()-1-k);EXPECT_EQ(step.value,static_cast<unsigned char>(encoded.transformed[row]));
    recovered[step.output_index]=static_cast<char>(step.value);row=step.next_row;
  }
  EXPECT_EQ(recovered,input);
  std::array<std::size_t,256> input_counts{};for (unsigned char c:input) ++input_counts[c];
  EXPECT_EQ(inverse.mapping.counts,input_counts);
}
}

TEST(BWTTrace, PreservesLegacySignaturesAndBananaOutputs) {
  const std::function<BWTResult(const std::string&)> forward=bwt_transform;
  const std::function<std::string(const BWTResult&)> inverse=bwt_inverse;
  const auto encoded=forward("banana");EXPECT_EQ(encoded.transformed,"nnbaaa");EXPECT_EQ(encoded.index,3);EXPECT_EQ(inverse(encoded),"banana");
  const auto trace=bwt_transform_trace("banana");EXPECT_EQ(trace.rotation_starts,(std::vector<std::size_t>{5,3,1,0,4,2}));
  const auto decoded=bwt_inverse_trace(encoded);EXPECT_EQ(decoded.mapping.lf,(std::vector<std::size_t>{4,5,3,0,1,2}));
  EXPECT_EQ(decoded.mapping.first_column,"aaabnn");EXPECT_EQ(decoded.steps[0].row,3);EXPECT_EQ(decoded.steps[0].output_index,5);
  check_input("banana");
}

TEST(BWTTrace, ExhaustiveSmallBinaryAlphabetIncludingHighBitBytes) {
  const std::array<unsigned char,3> alphabet{0,128,255};std::size_t checked=0;
  for (std::size_t n=0,power=1;n<=8;++n,power*=3) for (std::size_t word=0;word<power;++word) {
    auto digits=word;std::string input;
    for (std::size_t i=0;i<n;++i) {input.push_back(static_cast<char>(alphabet[digits%3]));digits/=3;}
    check_input(input);++checked;
  }
  EXPECT_EQ(checked,9841);
}

TEST(BWTTrace, FullByteAlphabetUsesUnsignedOrderAndRoundTrips) {
  std::string bytes;for (unsigned value=0;value<256;++value) bytes.push_back(static_cast<char>(value));
  const auto encoded=bwt_transform(bytes);EXPECT_EQ(encoded.index,0);
  EXPECT_EQ(encoded.transformed,bytes.substr(255)+bytes.substr(0,255));check_input(bytes,false);
  std::reverse(bytes.begin(),bytes.end());check_input(bytes,false);
  std::mt19937 rng(9821);std::shuffle(bytes.begin(),bytes.end(),rng);check_input(bytes,false);
  check_input(std::string{'\0',static_cast<char>(255),static_cast<char>(128),'a','\0'});
}

TEST(BWTTrace, RandomByteAndPeriodicInputsMatchIndependentRotationMatrix) {
  std::mt19937 rng(19940510);
  for (int trial=0;trial<1000;++trial) {
    const auto n=rng()%129;std::string input(n,'\0');
    for (auto& c:input) c=static_cast<char>(rng()%256);
    if (trial%4==0 && !input.empty()) for (std::size_t i=0;i<n;++i) input[i]=input[i%std::min<std::size_t>(n,3)];
    check_input(input,n<=10);
  }
}

TEST(BWTTrace, DuplicateRotationsAndPeriodicWalksStillWriteExactlyNBytes) {
  check_input("abab");check_input("baba");check_input("aaaa");check_input("abcabcabc");
  const auto encoded=bwt_transform("abab");EXPECT_EQ(encoded.transformed,"bbaa");EXPECT_EQ(encoded.index,0);
  const auto trace=bwt_inverse_trace(encoded);ASSERT_EQ(trace.steps.size(),4);
  EXPECT_EQ(trace.steps[0].row,trace.steps[2].row);EXPECT_NE(trace.steps[0].row,trace.steps[1].row);
  const BWTResult long_equal{std::string(200000,'x'),0};
  const auto long_trace=bwt_inverse_trace(long_equal);EXPECT_EQ(long_trace.output,long_equal.transformed);
  ASSERT_EQ(long_trace.steps.size(),200000);EXPECT_EQ(long_trace.mapping.counts['x'],200000);
  for (const auto& step:long_trace.steps) {EXPECT_EQ(step.row,0);EXPECT_EQ(step.next_row,0);EXPECT_EQ(step.value,'x');}
  EXPECT_EQ(bwt_inverse(long_equal),long_equal.transformed);
}

TEST(BWTTrace, MappingIsStableForAnyLastColumnWithoutClaimingImageValidity) {
  for (const auto& last:std::vector<std::string>{"","ab","aba","banana",std::string{'\0',static_cast<char>(255),'\0'}})
    check_mapping(last,bwt_lf_mapping(last));
  const BWTResult nonimage{"ab",0};EXPECT_EQ(bwt_inverse(nonimage),"aa");
  const auto rerun=bwt_transform(bwt_inverse(nonimage));EXPECT_NE(rerun.transformed,nonimage.transformed);
  EXPECT_EQ(bwt_inverse_trace(nonimage).output,"aa");
}

TEST(BWTTrace, ExactByteBudgetsThrowBeforeExcessAndLeaveInputUnchanged) {
  const std::string input="banana";const auto baseline=bwt_transform_trace(input);ASSERT_GT(baseline.byte_comparisons,1);
  for (std::size_t budget=1;budget<baseline.byte_comparisons;++budget) {
    BWTForwardOptions options;options.max_byte_comparisons=budget;
    EXPECT_THROW(bwt_transform_trace(input,options),std::runtime_error);
    EXPECT_EQ(input,"banana");
  }
  BWTForwardOptions exact;exact.max_byte_comparisons=baseline.byte_comparisons;
  const auto complete=bwt_transform_trace(input,exact);EXPECT_EQ(complete.byte_comparisons,baseline.byte_comparisons);
  EXPECT_EQ(complete.rotation_starts,baseline.rotation_starts);EXPECT_EQ(complete.result.transformed,baseline.result.transformed);
  exact.max_byte_comparisons=1;EXPECT_TRUE(bwt_transform_trace("",exact).result.transformed.empty());
  EXPECT_EQ(bwt_transform_trace("x",exact).result.transformed,"x");
}

TEST(BWTTrace, ConstantInputInspectsNBytesForEveryActualRotationComparison) {
  for (std::size_t n:{2,4,8,16,32,64,128,256,512}) {
    const auto trace=bwt_transform_trace(std::string(n,'a'));
    EXPECT_EQ(trace.byte_comparisons,n*trace.rotation_comparisons);
    EXPECT_EQ(trace.equal_rotation_comparisons,trace.rotation_comparisons);
    EXPECT_EQ(trace.result.index,0);EXPECT_EQ(trace.result.transformed,std::string(n,'a'));
    EXPECT_TRUE(std::is_sorted(trace.rotation_starts.begin(),trace.rotation_starts.end()));
  }
}

TEST(BWTTrace, RunAndMoveToFrontControlsDoNotPromiseUniversalCompression) {
  const std::string text="the quick brown fox the quick brown fox the quick brown fox";
  const auto encoded=bwt_transform(text);EXPECT_EQ(text.size(),59);
  EXPECT_EQ(rle_encode(text).size(),59);EXPECT_EQ(rle_encode(encoded.transformed).size(),20);
  const auto codes=mtf_encode(encoded.transformed);EXPECT_EQ(std::count(codes.begin(),codes.end(),0),39);
  EXPECT_EQ(mtf_decode(codes),encoded.transformed);
  const auto counter=bwt_transform("aabb");EXPECT_EQ(counter.transformed,"baba");
  EXPECT_EQ(rle_encode(std::string("aabb")).size(),2);EXPECT_EQ(rle_encode(counter.transformed).size(),4);
  for (const auto& input:std::vector<std::string>{"",text,"aabb","aaa",std::string{'\0','\0','a','a','\0'}}) {
    const auto ranks=mtf_encode(input);const auto runs=rle_encode(input).size();
    const auto expected=input.size()-runs+(!input.empty() && input[0]=='\0'?1:0);
    EXPECT_EQ(static_cast<std::size_t>(std::count(ranks.begin(),ranks.end(),0)),expected);
  }
}

TEST(BWTTrace, DefaultEmptyIsSafeAndInvalidPrimaryIndicesThrowBeforeAccess) {
  BWTResult empty;EXPECT_EQ(empty.index,0);EXPECT_TRUE(bwt_inverse(empty).empty());EXPECT_TRUE(bwt_inverse_trace(empty).steps.empty());
  EXPECT_THROW(bwt_inverse(BWTResult{"",1}),std::invalid_argument);
  EXPECT_THROW(bwt_inverse_trace(BWTResult{"",1}),std::invalid_argument);
  EXPECT_THROW(bwt_inverse(BWTResult{"a",1}),std::invalid_argument);
  EXPECT_THROW(bwt_inverse_trace(BWTResult{"ab",2}),std::invalid_argument);
  EXPECT_THROW(bwt_inverse(BWTResult{"banana",std::numeric_limits<std::size_t>::max()}),std::invalid_argument);
  EXPECT_THROW(bwt_inverse_trace(BWTResult{"x",std::numeric_limits<std::size_t>::max()}),std::invalid_argument);
}
