#include <datamunge/algorithms/bitap.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <climits>
#include <random>
#include <stdexcept>

using namespace datamunge::algorithms;

namespace {
std::vector<BitapMatch> oracle(const std::string& text,const std::string& pattern,int budget) {
  std::vector<BitapMatch> matches;
  budget=std::max(0,budget);
  if (pattern.size()>text.size()) return matches;
  for (std::size_t start=0;start<=text.size()-pattern.size();++start) {
    int errors=0;
    for (std::size_t j=0;j<pattern.size();++j) errors+=text[start+j]!=pattern[j];
    if (errors<=budget) matches.push_back({start,start+pattern.size(),errors});
  }
  return matches;
}

void same_matches(const std::vector<BitapMatch>& actual,const std::vector<BitapMatch>& expected) {
  ASSERT_EQ(actual.size(),expected.size());
  for (std::size_t i=0;i<expected.size();++i) {
    EXPECT_EQ(actual[i].begin,expected[i].begin);
    EXPECT_EQ(actual[i].end,expected[i].end);
    EXPECT_EQ(actual[i].substitutions,expected[i].substitutions);
  }
}

std::vector<std::string> binary_words(int length) {
  std::vector<std::string> out;
  for (int n=0;n<=length;++n) for (unsigned value=0;value<(1U<<n);++value) {
    std::string word(n,'a');
    for (int j=0;j<n;++j) if (value&(1U<<j)) word[j]='b';
    out.push_back(word);
  }
  return out;
}
}

TEST(BitapTrace,HandCheckedParallelRestartStates) {
  const auto t=bitap_trace("AABA","ABA");
  ASSERT_EQ(t.steps.size(),4);
  const std::vector<std::uint64_t> expected{1,1,2,5};
  for (std::size_t i=0;i<4;++i) EXPECT_EQ(t.steps[i].states[0],expected[i]);
  same_matches(t.matches,{{1,4,0}});
  EXPECT_EQ(t.bytes_scanned,4);
  EXPECT_EQ(t.word_updates,4);
}

TEST(BitapTrace,ExhaustiveSmallBinaryStringsMatchWindowHammingOracle) {
  const auto texts=binary_words(7),patterns=binary_words(4);
  for (const auto& text:texts) for (const auto& pattern:patterns)
    for (int budget=-1;budget<=static_cast<int>(pattern.size())+1;++budget) {
      const auto expected=oracle(text,pattern,budget);
      same_matches(bitap_matches(text,pattern,budget),expected);
      const long start=expected.empty()?-1:static_cast<long>(expected.front().begin);
      const long end=pattern.empty()?0:expected.empty()?-1:static_cast<long>(expected.front().end-1);
      if (budget<=0) EXPECT_EQ(bitap_search(text,pattern),start);
      EXPECT_EQ(bitap_fuzzy_search(text,pattern,budget),end);
    }
}

TEST(BitapTrace,EveryErrorLevelAndPrefixBitHasTheDocumentedSuffixMeaning) {
  std::mt19937 rng(20261106);
  for (int trial=0;trial<1200;++trial) {
    std::string text(1+rng()%12,'a'),pattern(1+rng()%6,'a');
    for (char& c:text) c='a'+rng()%3;
    for (char& c:pattern) c='a'+rng()%3;
    const int requested=rng()%8;
    const auto t=bitap_trace(text,pattern,requested);
    const int budget=std::min(requested,static_cast<int>(pattern.size()));
    EXPECT_EQ(t.effective_budget,budget);
    EXPECT_EQ(t.bytes_scanned,text.size());
    EXPECT_EQ(t.word_updates,text.size()*(budget+1));
    same_matches(t.matches,oracle(text,pattern,requested));
    for (const auto& step:t.steps) {
      EXPECT_EQ(step.byte,static_cast<unsigned char>(text[step.position]));
      ASSERT_EQ(step.states.size(),budget+1);
      for (int d=0;d<=budget;++d) for (std::size_t j=0;j<pattern.size();++j) {
        bool active=false;
        if (j<=step.position) {
          int errors=0;
          const auto start=step.position-j;
          for (std::size_t q=0;q<=j;++q) errors+=text[start+q]!=pattern[q];
          active=errors<=d;
        }
        EXPECT_EQ(bool(step.states[d]&(std::uint64_t{1}<<j)),active);
      }
    }
  }
}

TEST(BitapTrace,OverlapsEmptyPatternsAndShortTextsHaveExplicitSemantics) {
  same_matches(bitap_matches("AAAA","AA"),{{0,2,0},{1,3,0},{2,4,0}});
  same_matches(bitap_matches("ab",""),{{0,0,0},{1,1,0},{2,2,0}});
  const auto empty=bitap_trace("ab","",INT_MAX);
  EXPECT_TRUE(empty.steps.empty());
  EXPECT_EQ(empty.bytes_scanned,0);
  EXPECT_EQ(empty.word_updates,0);
  EXPECT_EQ(empty.effective_budget,0);
  EXPECT_TRUE(bitap_matches("AB","ABCD",INT_MAX).empty());
  EXPECT_EQ(bitap_fuzzy_search("AB","ABCD",INT_MAX),-1);
}

TEST(BitapTrace,ArbitraryBytesAndEmbeddedNulAreMatchedWithoutCharacterConversion) {
  std::string all;
  for (int byte=0;byte<256;++byte) all.push_back(static_cast<char>(byte));
  for (std::size_t offset:{0,1,63,127,193}) {
    const auto pattern=all.substr(offset,63);
    same_matches(bitap_matches(all,pattern,2),oracle(all,pattern,2));
    EXPECT_EQ(bitap_search(all,pattern),offset);
  }
  same_matches(bitap_matches(std::string("a\0ba\0b",6),std::string("a\0b",3)),{{0,3,0},{3,6,0}});
}

TEST(BitapTrace,WordLimitsPreserveLegacySentinelsAndModernValidation) {
  const std::string text(80,'x'),p62(62,'x'),p63(63,'x'),p64(64,'x');
  EXPECT_EQ(bitap_search(text,p63),0);
  EXPECT_EQ(bitap_search(text,p64),-1);
  EXPECT_EQ(bitap_fuzzy_search(text,p62,0),61);
  EXPECT_EQ(bitap_fuzzy_search(text,p63,0),-1);
  EXPECT_EQ(bitap_matches(text,p63).size(),18);
  EXPECT_THROW(bitap_matches(text,p64),std::invalid_argument);
  EXPECT_THROW(bitap_trace(text,p64),std::invalid_argument);
}

TEST(BitapTrace,AnUnboundedRequestedBudgetIsClampedToThePatternLength) {
  const auto t=bitap_trace("abcdefgh","WXYZ",INT_MAX);
  EXPECT_EQ(t.effective_budget,4);
  EXPECT_EQ(t.word_updates,8*5);
  ASSERT_EQ(t.matches.size(),5);
  for (const auto& match:t.matches) EXPECT_EQ(match.substitutions,4);
  EXPECT_EQ(bitap_fuzzy_search("abcdefgh","WXYZ",INT_MAX),3);
}

TEST(BitapTrace,SubstitutionsDoNotImplementInsertionDeletionMatching) {
  EXPECT_TRUE(bitap_matches("ABXCD","ABCD",1).empty());
  EXPECT_TRUE(bitap_matches("ABD","ABCD",1).empty());
}
