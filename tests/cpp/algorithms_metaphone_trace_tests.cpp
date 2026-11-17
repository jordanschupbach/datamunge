#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <datamunge/algorithms/metaphone.hpp>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
void verify(const std::string& input) {
  auto        t      = metaphone_trace(input);
  std::size_t cursor = 0, output = 0;
  std::string emitted;
  for (auto& s : t.steps) {
    ASSERT_EQ(s.begin, cursor);
    ASSERT_GT(s.end, s.begin);
    ASSERT_LE(s.end, t.analysis.normalized.size());
    EXPECT_EQ(s.current, static_cast<unsigned char>(t.analysis.normalized[s.begin]));
    EXPECT_EQ(s.has_next, s.begin + 1 < t.analysis.normalized.size());
    EXPECT_EQ(s.has_next2, s.begin + 2 < t.analysis.normalized.size());
    EXPECT_EQ(
        s.next, s.has_next ? static_cast<unsigned char>(t.analysis.normalized[s.begin + 1]) : 0);
    EXPECT_EQ(
        s.next2, s.has_next2 ? static_cast<unsigned char>(t.analysis.normalized[s.begin + 2]) : 0);
    EXPECT_EQ(s.output_before, output);
    emitted += s.emitted;
    output += s.emitted.size();
    EXPECT_EQ(s.output_after, output);
    EXPECT_FALSE(s.rule.empty());
    cursor = s.end;
  }
  EXPECT_EQ(cursor, t.analysis.normalized.size());
  EXPECT_EQ(t.analysis.consumed, cursor);
  EXPECT_EQ(t.analysis.iterations, t.steps.size());
  EXPECT_EQ(t.analysis.emitted, output);
  for (char& c : emitted)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  EXPECT_EQ(t.analysis.key, emitted);
  EXPECT_EQ(metaphone(input), emitted);
  EXPECT_EQ(
      t.captured_entries,
      input.size() + t.analysis.normalized.size() + t.steps.size() + 2 * output);
  EXPECT_LE(output, 2 * input.size());
}
} // namespace
TEST(MetaphoneTrace, OriginalWordsAndIndependentHandCheckedContextRules) {
  const std::vector<std::string> expected{
      "NT", "RT", "0MPSN", "SFR", "S", "ET", "KM", "NMN", "PSXLJ", "WTNST", "NM", "FLKM"};
  auto words = metaphone_demo_words();
  ASSERT_EQ(words.size(), expected.size());
  for (std::size_t i = 0; i < words.size(); ++i) {
    EXPECT_EQ(metaphone(words[i]), expected[i]);
    verify(words[i]);
  }
  for (auto pair : std::vector<std::pair<std::string, std::string>>{
           {"ph", "F"},       {"ch", "X"},     {"th", "0"},           {"dge", "J"},
           {"cia", "X"},      {"sio", "X"},    {"tia", "X"},          {"tch", "X"},
           {"x", "S"},        {"ax", "AKS"},   {"mb", "M"},           {"amb", "AM"},
           {"amber", "AMBR"}, {"whale", "WL"}, {"cc", "KK"},          {"bb", "B"},
           {"h", "H"},        {"ah", "A"},     {"Catherine", "K0RN"}, {"Kathryn", "K0RN"}}) {
    EXPECT_EQ(metaphone(pair.first), pair.second) << pair.first;
    verify(pair.first);
  }
}
TEST(MetaphoneTrace, EveryShortRuleAlphabetWordAllAsciiPairsAndArbitraryBytesActualSpans) {
  const std::string alphabet = "achknpx";
  for (std::size_t n = 0; n <= 5; ++n) {
    std::size_t count = 1;
    for (std::size_t k = 0; k < n; ++k)
      count *= alphabet.size();
    for (std::size_t code = 0; code < count; ++code) {
      std::string input(n, ' ');
      auto        bits = code;
      for (char& c : input) {
        c = alphabet[bits % alphabet.size()];
        bits /= alphabet.size();
      }
      verify(input);
    }
  }
  for (unsigned a = 0; a < 128; ++a)
    for (unsigned b = 0; b < 128; ++b)
      verify(std::string{static_cast<char>(a), static_cast<char>(b)});
  std::mt19937 rng(3);
  for (int trial = 0; trial < 1000; ++trial) {
    std::string bytes(rng() % 40, '\0');
    for (char& c : bytes)
      c = static_cast<char>(rng());
    verify(bytes);
  }
}
TEST(MetaphoneTrace, ActualOnsetsDuplicateConsumptionEmptyQuotasAndOwnership) {
  auto onset = metaphone_trace("Knight");
  EXPECT_EQ(onset.analysis.onset_removed, 1);
  EXPECT_EQ(onset.analysis.normalized, "night");
  EXPECT_EQ(onset.analysis.key, "NT");
  auto ph = metaphone_trace("ph");
  ASSERT_EQ(ph.steps.size(), 1);
  EXPECT_EQ(ph.steps[0].begin, 0);
  EXPECT_EQ(ph.steps[0].end, 2);
  EXPECT_EQ(ph.steps[0].emitted, "f");
  EXPECT_EQ(ph.steps[0].rule, "letter p context");
  auto duplicate = metaphone_trace("bb");
  EXPECT_EQ(duplicate.steps[0].rule, "duplicate");
  EXPECT_TRUE(duplicate.steps[0].emitted.empty());
  EXPECT_EQ(duplicate.steps[1].emitted, "b");
  EXPECT_NO_THROW(metaphone_trace("ph", 7));
  EXPECT_THROW(metaphone_trace("ph", 6), std::length_error);
  EXPECT_NO_THROW(metaphone_trace("ph", 0));
  EXPECT_EQ(metaphone_trace("", 1).captured_entries, 0);
  auto name = std::string("a\0x", 3);
  auto t    = metaphone_trace(name);
  name[0]   = 'z';
  EXPECT_EQ(t.input[0], 'a');
  EXPECT_EQ(t.analysis.normalized[1], 0);
}
