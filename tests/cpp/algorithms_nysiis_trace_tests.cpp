#include <gtest/gtest.h>

#include <cctype>
#include <datamunge/algorithms/nysiis.hpp>
#include <datamunge/algorithms/phonetic.hpp>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NysiisTrace, HandRuleCasesPreserveActualNativeVariant) {
  for (auto& [input, key] : std::vector<std::pair<std::string, std::string>>{
           {"", ""},
           {"MAC", "MC"},
           {"MacDonald", "MCDANALD"},
           {"Catherine", "CATARAN"},
           {"AASCH", "AS"},
           {"AEV", "AAF"},
           {"LEW", "LAE"},
           {"Fischer", "FASSAR"},
           {"Fisher", "FASAR"},
           {"Knuth", "NAT"},
           {"PH", "F"},
           {"SCH", "S"},
           {"LEE", "LY"},
           {"MACIE", "MCY"},
           {"AAY", "Y"},
           {"AS", "A"},
           {"A", "A"},
           {"A-A", "A-"}}) {
    auto t = nysiis_trace(input);
    EXPECT_EQ(t.analysis.key, key) << input;
    EXPECT_EQ(nysiis(input), key);
    EXPECT_EQ(nysiis_analyze(input).key, key);
  }
}
TEST(NysiisTrace, CompleteActualByteSnapshotsTokensOwnershipAndQuotas) {
  std::vector<std::string> inputs   = nysiis_demo_names();
  auto                     examples = nysiis_demo_examples();
  inputs.insert(inputs.end(), examples.begin(), examples.end());
  for (int byte = 0; byte < 256; ++byte) {
    inputs.emplace_back(1, char(byte));
    inputs.push_back(std::string("A") + char(byte) + "EV");
  }
  for (auto& name : inputs) {
    auto t = nysiis_trace(name);
    EXPECT_EQ(t.analysis.key, nysiis(name));
    EXPECT_EQ(t.input, name);
    std::string uppercase = name;
    for (auto& c : uppercase)
      c = char(std::toupper(static_cast<unsigned char>(c)));
    EXPECT_EQ(t.uppercase, uppercase);
    std::size_t units = t.input.size() + t.uppercase.size() + t.rewritten.size() +
                        t.analysis.key.size(),
                tokens = 0, suppressed = 0;
    std::string previous;
    std::size_t next = 1;
    for (auto& s : t.steps) {
      units +=
          1 + s.stage.size() + s.rule.size() + s.before.size() + s.after.size() + s.token.size();
      if (s.stage == "prefix")
        EXPECT_EQ(s.before, uppercase);
      if (s.stage == "suffix")
        EXPECT_EQ(t.rewritten, s.after);
      if (s.stage == "first") {
        EXPECT_TRUE(s.before.empty());
        EXPECT_EQ(s.after, s.token);
        previous = s.after;
      }
      if (s.stage == "interior") {
        EXPECT_EQ(s.before, previous);
        EXPECT_EQ(s.begin, next);
        next = s.end;
        EXPECT_EQ(s.appended, s.token.back() != s.before.back());
        EXPECT_EQ(s.after, s.appended ? s.before + s.token : s.before);
        ++tokens;
        suppressed += !s.appended;
        previous = s.after;
      }
      if (s.stage == "cleanup") {
        EXPECT_EQ(s.before, previous);
        previous = s.after;
      }
    }
    EXPECT_EQ(units, t.captured_entries);
    EXPECT_EQ(tokens, t.analysis.interior_tokens);
    EXPECT_EQ(suppressed, t.analysis.suppressed_tokens);
    if (!name.empty()) {
      EXPECT_EQ(next, t.rewritten.size());
      EXPECT_EQ(previous, t.analysis.key);
      EXPECT_NO_THROW(nysiis_trace(name, t.captured_entries));
      EXPECT_THROW(nysiis_trace(name, t.captured_entries - 1), std::length_error);
    } else
      EXPECT_TRUE(t.steps.empty());
  }
  std::string input = "MacDonald";
  auto        own   = nysiis_trace(input);
  input[0]          = 'Z';
  EXPECT_EQ(own.input, "MacDonald");
}
TEST(NysiisTrace, OriginalCohortIndependentEqualityPairsAndNativeExamples) {
  auto names = nysiis_demo_names();
  auto c     = nysiis_compare_names(names);
  EXPECT_EQ(c.names.size(), 32);
  EXPECT_EQ(c.distinct_soundex, 23);
  EXPECT_EQ(c.distinct_nysiis, 27);
  std::uint64_t sx = 0, ny = 0;
  for (std::size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(c.soundex_keys[i], soundex(names[i]));
    EXPECT_EQ(c.nysiis_keys[i], nysiis(names[i]));
    for (std::size_t j = 0; j < i; ++j) {
      sx += c.soundex_keys[i] == c.soundex_keys[j];
      ny += c.nysiis_keys[i] == c.nysiis_keys[j];
    }
  }
  EXPECT_EQ(c.equal_soundex_pairs, sx);
  EXPECT_EQ(c.equal_nysiis_pairs, ny);
  EXPECT_THROW(nysiis_compare_names(names, 31), std::length_error);
  EXPECT_EQ(nysiis_compare_names({"Miller", "Mueller"}).equal_nysiis_pairs, 1);
  auto empty = nysiis_compare_names({});
  EXPECT_EQ(empty.distinct_nysiis, 0);
  names[0] = "changed";
  EXPECT_EQ(c.names[0], "Anderson");
}
