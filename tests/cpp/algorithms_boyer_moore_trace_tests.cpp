#include <datamunge/algorithms/boyer_moore.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
using namespace datamunge::algorithms;
namespace {
std::vector<std::size_t> reference(const std::string& text,const std::string& pattern) {
  std::vector<std::size_t> result;if (pattern.empty()) return result;
  for (auto pos=text.find(pattern);pos!=std::string::npos;pos=text.find(pattern,pos+1)) result.push_back(pos);
  return result;
}
std::string binary(std::size_t n,std::size_t bits) {
  std::string out(n,'A');for (std::size_t i=0;i<n;++i) if (bits&(std::size_t{1}<<i)) out[i]='B';return out;
}
// Direct overlap constraints, independent of the two-pass border construction.
std::ptrdiff_t suffix_oracle(const std::string& pattern,std::size_t k) {
  for (std::size_t distance=1;distance<=pattern.size();++distance) {
    bool compatible=true;
    for (std::size_t old=k;old<pattern.size();++old)
      if (old>=distance && pattern[old]!=pattern[old-distance]) compatible=false;
    if (k>distance && pattern[k-1]==pattern[k-1-distance]) compatible=false;
    if (compatible) return static_cast<std::ptrdiff_t>(distance);
  }
  return 1; // empty pattern
}
void check_trace(const std::string& text,const std::string& pattern,const BoyerMooreTrace& trace) {
  ASSERT_EQ(trace.matches,reference(text,pattern));ASSERT_EQ(trace.matches,boyer_moore_search(text,pattern));
  ASSERT_EQ(trace.text_size,text.size());ASSERT_EQ(trace.pattern_size,pattern.size());
  std::size_t expected_position=0,comparisons=0;std::vector<std::size_t> found;
  for (const auto& step:trace.alignments) {
    ASSERT_EQ(step.position,expected_position);ASSERT_LE(step.position+pattern.size(),text.size());
    ASSERT_GT(step.shift,0);ASSERT_LE(step.shift,pattern.size());
    const auto suffix=step.matched_suffix_length;ASSERT_LE(suffix,pattern.size());
    ASSERT_EQ(pattern.substr(pattern.size()-suffix),text.substr(step.position+pattern.size()-suffix,suffix));
    ASSERT_EQ(step.comparisons,suffix+(step.matched?0:1));comparisons+=step.comparisons;
    if (step.matched) {
      ASSERT_EQ(suffix,pattern.size());ASSERT_EQ(step.mismatch_index,pattern.size());
      ASSERT_EQ(step.bad_character_shift,0);ASSERT_EQ(step.text_byte,0);found.push_back(step.position);
      ASSERT_EQ(step.shift,static_cast<std::size_t>(trace.tables.good_suffix[0]));
    } else {
      ASSERT_EQ(step.mismatch_index,pattern.size()-suffix-1);const auto j=step.mismatch_index;
      ASSERT_NE(pattern[j],text[step.position+j]);ASSERT_EQ(step.text_byte,static_cast<unsigned char>(text[step.position+j]));
      ASSERT_EQ(step.bad_character_shift,static_cast<std::ptrdiff_t>(j)-trace.tables.last[step.text_byte]);
      ASSERT_EQ(step.good_suffix_shift,trace.tables.good_suffix[j+1]);
      ASSERT_EQ(step.shift,static_cast<std::size_t>(std::max<std::ptrdiff_t>({1,step.bad_character_shift,step.good_suffix_shift})));
    }
    expected_position+=step.shift;
  }
  ASSERT_EQ(found,trace.matches);ASSERT_EQ(comparisons,trace.comparisons);
  if (!pattern.empty() && pattern.size()<=text.size()) ASSERT_GT(expected_position,text.size()-pattern.size());
  else ASSERT_TRUE(trace.alignments.empty());
}
}
TEST(BoyerMooreTrace, OriginalDnaTraceAndExactComparisonCounts) {
  const std::string text="GTTATAGCTGATCGCGGCGTAGCGGCGAA",pattern="GCGGCG";
  const auto trace=boyer_moore_trace(text,pattern);check_trace(text,pattern,trace);
  ASSERT_EQ(trace.matches,(std::vector<std::size_t>{13,21}));ASSERT_EQ(trace.alignments.size(),6);
  const std::vector<std::size_t> positions{0,6,12,13,16,21},shifts{6,6,1,3,5,3},counts{1,1,1,6,2,6};
  for (std::size_t i=0;i<positions.size();++i) {
    EXPECT_EQ(trace.alignments[i].position,positions[i]);EXPECT_EQ(trace.alignments[i].shift,shifts[i]);
    EXPECT_EQ(trace.alignments[i].comparisons,counts[i]);
  }
  EXPECT_EQ(trace.comparisons,17);EXPECT_EQ(trace.tables.good_suffix,(std::vector<std::ptrdiff_t>{3,3,3,3,5,2,1}));
  EXPECT_EQ(trace.tables.last['A'],-1);EXPECT_EQ(trace.tables.last['C'],4);EXPECT_EQ(trace.tables.last['G'],5);
}
TEST(BoyerMooreTrace, StrongSuffixTablesMatchDirectOverlapOracleFor2047Patterns) {
  std::size_t checked=0;
  for (std::size_t n=0;n<=10;++n) for (std::size_t bits=0;bits<(std::size_t{1}<<n);++bits) {
    const auto pattern=binary(n,bits);const auto tables=boyer_moore_tables(pattern);
    ASSERT_EQ(tables.good_suffix.size(),n+1);
    for (std::size_t k=0;k<=n;++k) ASSERT_EQ(tables.good_suffix[k],suffix_oracle(pattern,k))<<pattern<<" k="<<k;
    for (unsigned byte=0;byte<256;++byte) {
      const auto position=pattern.rfind(static_cast<char>(byte));
      ASSERT_EQ(tables.last[byte],position==std::string::npos?-1:static_cast<std::ptrdiff_t>(position));
    }
    ++checked;
  }
  EXPECT_EQ(checked,2047);
}
TEST(BoyerMooreTrace, All129921SmallBinaryTextPatternPairsMatchStringFind) {
  std::size_t checked=0;
  for (std::size_t n=0;n<=9;++n) for (std::size_t t=0;t<(std::size_t{1}<<n);++t)
    for (std::size_t m=0;m<=6;++m) for (std::size_t p=0;p<(std::size_t{1}<<m);++p) {
      const auto text=binary(n,t),pattern=binary(m,p);check_trace(text,pattern,boyer_moore_trace(text,pattern));++checked;
    }
  EXPECT_EQ(checked,129921);
}
TEST(BoyerMooreTrace, NegativeBadShiftAndStrongSuffixChoiceAreExposed) {
  const auto trace=boyer_moore_trace("AABB","ABAB");check_trace("AABB","ABAB",trace);
  ASSERT_EQ(trace.alignments.size(),1);const auto& step=trace.alignments[0];
  EXPECT_EQ(step.bad_character_shift,-1);EXPECT_EQ(step.good_suffix_shift,4);EXPECT_EQ(step.shift,4);
  EXPECT_EQ(step.matched_suffix_length,1);EXPECT_EQ(step.comparisons,2);
}
TEST(BoyerMooreTrace, AllByteValuesEmbeddedNulAndUnsignedLookupWork) {
  std::string all;for (unsigned i=0;i<256;++i) all.push_back(static_cast<char>(i));
  const auto tables=boyer_moore_tables(all+all);
  for (unsigned i=0;i<256;++i) EXPECT_EQ(tables.last[i],256+i);
  const std::string text=all+all+all;
  for (std::size_t start=0;start<256;start+=7) {
    const auto pattern=text.substr(start,37);check_trace(text,pattern,boyer_moore_trace(text,pattern));
  }
  const std::string nul_text("A\0B\0B",5),nul_pattern("\0B",2);
  EXPECT_EQ(boyer_moore_trace(nul_text,nul_pattern).matches,(std::vector<std::size_t>{1,3}));
}
TEST(BoyerMooreTrace, EmptyAndOversizedPatternsHaveTablesButNoSearch) {
  const auto empty=boyer_moore_trace("abc","");EXPECT_EQ(empty.tables.good_suffix,(std::vector<std::ptrdiff_t>{1}));
  for (auto index:empty.tables.last) EXPECT_EQ(index,-1);
  EXPECT_TRUE(empty.alignments.empty());EXPECT_EQ(empty.comparisons,0);
  const auto oversized=boyer_moore_trace("A","ABCD");EXPECT_EQ(oversized.tables.good_suffix.size(),5);
  EXPECT_TRUE(oversized.alignments.empty());EXPECT_TRUE(oversized.matches.empty());
}
TEST(BoyerMooreTrace, RepeatedMatchesHaveExactQuadraticComparisonFamily) {
  for (std::size_t n:{32,128,512}) for (std::size_t m:{1,4,16,32}) {
    const auto trace=boyer_moore_trace(std::string(n,'A'),std::string(m,'A'));
    ASSERT_EQ(trace.matches.size(),n-m+1);EXPECT_EQ(trace.comparisons,m*(n-m+1));
    for (auto step:trace.alignments) {EXPECT_TRUE(step.matched);EXPECT_EQ(step.shift,1);}
    const auto skips=boyer_moore_trace(std::string(n,'A'),std::string(m,'B'));
    EXPECT_TRUE(skips.matches.empty());EXPECT_EQ(skips.comparisons,n/m);
    const auto unit=boyer_moore_trace(std::string(n,'A'),std::string(m-1,'A')+'B');
    EXPECT_EQ(unit.comparisons,n-m+1);EXPECT_EQ(unit.alignments.size(),n-m+1);
  }
}
TEST(BoyerMooreTrace, RandomFullByteStringsMatchIndependentFind) {
  std::mt19937_64 engine(723);
  for (int trial=0;trial<2000;++trial) {
    std::string text(1+engine()%200,'\0'),pattern(1+engine()%30,'\0');
    for (char& c:text) c=static_cast<char>(engine()&255);
    for (char& c:pattern) c=static_cast<char>(engine()&255);
    if (trial%2==0 && pattern.size()<=text.size()) text.replace(engine()%(text.size()-pattern.size()+1),pattern.size(),pattern);
    check_trace(text,pattern,boyer_moore_trace(text,pattern));
  }
}
