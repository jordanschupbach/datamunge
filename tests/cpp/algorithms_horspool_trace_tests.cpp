#include <datamunge/algorithms/string_search_extra.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
using namespace datamunge::algorithms;
namespace {
std::vector<std::size_t> find_all(const std::string& text,const std::string& pattern) {
  std::vector<std::size_t> out;if (pattern.empty()) return out;
  for (auto at=text.find(pattern);at!=std::string::npos;at=text.find(pattern,at+1)) out.push_back(at);
  return out;
}
std::string binary(std::size_t size,std::size_t bits) {
  std::string out(size,'A');for (std::size_t i=0;i<size;++i) if (bits&(std::size_t{1}<<i)) out[i]='B';return out;
}
void check_trace(const std::string& text,const std::string& pattern,const HorspoolTrace& trace) {
  ASSERT_EQ(trace.scan.positions,find_all(text,pattern));ASSERT_EQ(trace.scan.positions,boyer_moore_horspool_search(text,pattern));
  ASSERT_EQ(trace.text_size,text.size());ASSERT_EQ(trace.pattern_size,pattern.size());
  ASSERT_EQ(trace.scan.alignments,trace.alignments.size());ASSERT_EQ(trace.shifts,horspool_shifts(pattern));
  std::size_t next=0,total=0;std::vector<std::size_t> matches;
  for (const auto& step:trace.alignments) {
    ASSERT_EQ(step.position,next);ASSERT_LE(step.position+pattern.size(),text.size());
    ASSERT_EQ(step.ending_byte,static_cast<unsigned char>(text[step.position+pattern.size()-1]));
    ASSERT_EQ(step.shift,trace.shifts[step.ending_byte]);ASSERT_GT(step.shift,0);ASSERT_LE(step.shift,pattern.size());
    const auto suffix=step.matched_suffix_length;ASSERT_LE(suffix,pattern.size());
    ASSERT_EQ(pattern.substr(pattern.size()-suffix),text.substr(step.position+pattern.size()-suffix,suffix));
    ASSERT_EQ(step.comparisons,suffix+(step.matched?0:1));total+=step.comparisons;
    if (step.matched) {ASSERT_EQ(step.mismatch_index,pattern.size());ASSERT_EQ(suffix,pattern.size());matches.push_back(step.position);}
    else {
      ASSERT_EQ(step.mismatch_index,pattern.size()-suffix-1);
      ASSERT_NE(pattern[step.mismatch_index],text[step.position+step.mismatch_index]);
    }
    next+=step.shift;
  }
  ASSERT_EQ(total,trace.scan.comparisons);ASSERT_EQ(matches,trace.scan.positions);
  if (pattern.empty() || pattern.size()>text.size()) ASSERT_TRUE(trace.alignments.empty());
  else ASSERT_GT(next,text.size()-pattern.size());
}
}
TEST(HorspoolTrace, WindowEndByteControlsShiftInsteadOfMismatchByte) {
  const auto trace=boyer_moore_horspool_trace("AXCABCA","ABCA");check_trace("AXCABCA","ABCA",trace);
  ASSERT_EQ(trace.alignments.size(),2);const auto& first=trace.alignments[0];
  EXPECT_EQ(first.mismatch_index,1);EXPECT_EQ(first.matched_suffix_length,2);EXPECT_EQ(first.comparisons,3);
  EXPECT_EQ(first.ending_byte,'A');EXPECT_EQ(first.shift,3);EXPECT_EQ(trace.shifts['X'],4);
  EXPECT_EQ(trace.scan.positions,(std::vector<std::size_t>{3}));EXPECT_EQ(trace.scan.comparisons,7);
}
TEST(HorspoolTrace, TablesExcludeFinalPatternByteAndUseRightmostPrefixOccurrence) {
  const auto table=horspool_shifts("abra");EXPECT_EQ(table['a'],3);EXPECT_EQ(table['b'],2);EXPECT_EQ(table['r'],1);EXPECT_EQ(table['z'],4);
  const auto terminal=horspool_shifts("ABC");EXPECT_EQ(terminal['A'],2);EXPECT_EQ(terminal['B'],1);EXPECT_EQ(terminal['C'],3);
  const auto repeated=horspool_shifts("ababa");EXPECT_EQ(repeated['a'],2);EXPECT_EQ(repeated['b'],1);
  for (auto shift:horspool_shifts("")) EXPECT_EQ(shift,0);
  for (auto shift:horspool_shifts("X")) EXPECT_EQ(shift,1);
}
TEST(HorspoolTrace, All2047BinaryPrefixTablesMatchIndependentReverseFind) {
  std::size_t checked=0;
  for (std::size_t n=0;n<=10;++n) for (std::size_t bits=0;bits<(std::size_t{1}<<n);++bits) {
    const auto pattern=binary(n,bits),prefix=pattern.empty()?std::string{}:pattern.substr(0,n-1);
    const auto shifts=horspool_shifts(pattern);
    for (unsigned byte=0;byte<256;++byte) {
      const auto last=prefix.rfind(static_cast<char>(byte));
      EXPECT_EQ(shifts[byte],last==std::string::npos?n:n-1-last);
    }
    ++checked;
  }
  EXPECT_EQ(checked,2047);
}
TEST(HorspoolTrace, OriginalExampleRetainsFourOccurrences) {
  const std::string text="abracadabra_abra_cadabra",pattern="abra";
  const auto trace=boyer_moore_horspool_trace(text,pattern);check_trace(text,pattern,trace);
  EXPECT_EQ(trace.scan.positions,(std::vector<std::size_t>{0,7,12,20}));
}
TEST(HorspoolTrace, All129921SmallBinaryPairsMatchStringFind) {
  std::size_t checked=0;
  for (std::size_t n=0;n<=9;++n) for (std::size_t t=0;t<(std::size_t{1}<<n);++t)
    for (std::size_t m=0;m<=6;++m) for (std::size_t p=0;p<(std::size_t{1}<<m);++p) {
      const auto text=binary(n,t),pattern=binary(m,p);check_trace(text,pattern,boyer_moore_horspool_trace(text,pattern));++checked;
    }
  EXPECT_EQ(checked,129921);
}
TEST(HorspoolTrace, FullBytesNulAndSeededBinarySearchesWork) {
  std::string bytes;for (unsigned i=0;i<256;++i) bytes.push_back(static_cast<char>(i));
  const auto shifts=horspool_shifts(bytes);
  for (unsigned i=0;i<255;++i) EXPECT_EQ(shifts[i],255-i);
  EXPECT_EQ(shifts[255],256);
  const auto text=bytes+bytes+bytes;
  for (std::size_t start=0;start<256;start+=7) {
    const auto pattern=text.substr(start,37);check_trace(text,pattern,boyer_moore_horspool_trace(text,pattern));
  }
  std::mt19937_64 engine(123);
  for (int trial=0;trial<2000;++trial) {
    std::string t(1+engine()%200,'\0'),p(1+engine()%30,'\0');
    for (auto& c:t) c=static_cast<char>(engine()&255);
    for (auto& c:p) c=static_cast<char>(engine()&255);
    if (trial%2==0 && p.size()<=t.size()) t.replace(engine()%(t.size()-p.size()+1),p.size(),p);
    check_trace(t,p,boyer_moore_horspool_trace(t,p));
  }
}
TEST(HorspoolTrace, EmptySingleByteAndOversizedPatternsHaveCorrectMetadata) {
  for (const auto& [text,pattern]:std::vector<std::pair<std::string,std::string>>{{"abc",""},{"",""},{"ab","abc"},{"banana","a"},{"aaaa","aa"}})
    check_trace(text,pattern,boyer_moore_horspool_trace(text,pattern));
  const auto single=boyer_moore_horspool_trace("banana","a");EXPECT_EQ(single.scan.comparisons,6);EXPECT_EQ(single.scan.alignments,6);
  const auto over=boyer_moore_horspool_trace("ab","abc");EXPECT_EQ(over.shifts['a'],2);EXPECT_TRUE(over.scan.positions.empty());
}
TEST(HorspoolTrace, RepeatedMatchAndLateMismatchFamiliesReachQuadraticCounts) {
  for (std::size_t n:{32,128,512}) for (std::size_t m:{1,4,16,32}) {
    const std::string text(n,'A');const auto repeated=boyer_moore_horspool_trace(text,std::string(m,'A'));
    EXPECT_EQ(repeated.scan.comparisons,m*(n-m+1));EXPECT_EQ(repeated.scan.positions.size(),n-m+1);
    const auto misses=boyer_moore_horspool_trace(text,std::string("B")+std::string(m-1,'A'));
    EXPECT_EQ(misses.scan.comparisons,m*(n-m+1));EXPECT_TRUE(misses.scan.positions.empty());
    const auto skips=boyer_moore_horspool_trace(text,std::string(m,'B'));
    EXPECT_EQ(skips.scan.comparisons,n/m);EXPECT_TRUE(skips.scan.positions.empty());
  }
}
TEST(HorspoolTrace, CorpusCountersAndPerRecordPositionsUseSameKernel) {
  const std::vector<std::string> texts{"abracadabra","AXCABCA","",std::string("A\0B\0B",5)};
  datamunge::dstruct::DataFrame frame;frame.add_column("text",texts);
  const std::vector<std::string> patterns{"abra","ABCA","a",std::string("\0B",2),""};
  const auto table=string_search_comparison(frame,"text",patterns);
  for (std::size_t p=0;p<patterns.size();++p) {
    std::size_t matches=0,comparisons=0,alignments=0;
    for (const auto& text:texts) {
      const auto trace=boyer_moore_horspool_trace(text,patterns[p]);
      matches+=trace.scan.positions.size();comparisons+=trace.scan.comparisons;alignments+=trace.scan.alignments;
    }
    const auto row=p*3+1;EXPECT_EQ(table.double_at("matches",row),matches);
    EXPECT_EQ(table.double_at("comparisons",row),comparisons);EXPECT_EQ(table.double_at("alignments",row),alignments);
    EXPECT_EQ(table.double_at("agrees_with_naive",row),1);
  }
}
