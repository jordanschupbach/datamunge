#include <gtest/gtest.h>

#include <datamunge/algorithms/luhn.hpp>
#include <random>
using namespace datamunge::algorithms;
namespace {
int reference(const std::string& s, bool payload) {
  int sum = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    int  digit = s[i] - '0';
    bool twice = ((s.size() - 1 - i) % 2) == (payload ? 0u : 1u);
    if (twice)
      digit = digit < 5 ? 2 * digit : 2 * digit - 9;
    sum += digit;
  }
  return sum % 10;
}
} // namespace
TEST(LuhnTrace, ExhaustiveDecimalWordsAndActualRightToLeftSteps) {
  for (int n = 0, power = 1; n <= 4; ++n, power *= 10)
    for (int value = 0; value < power; ++value) {
      std::string s(n, '0');
      int         v = value;
      for (int i = n - 1; i >= 0; --i) {
        s[i] = '0' + v % 10;
        v /= 10;
      }
      for (auto mode : {LuhnMode::Full, LuhnMode::Payload}) {
        auto t = luhn_trace(s, mode);
        EXPECT_EQ(t.analysis.remainder, reference(s, mode == LuhnMode::Payload));
        EXPECT_EQ(t.steps.size(), n);
        EXPECT_EQ(t.captured_entries, 2 * n);
        int residue = 0;
        for (std::size_t k = 0; k < t.steps.size(); ++k) {
          auto& step = t.steps[k];
          EXPECT_EQ(step.index, n - 1 - k);
          EXPECT_EQ(step.digit, s[step.index] - '0');
          EXPECT_EQ(step.raw, step.digit * (step.doubled ? 2 : 1));
          EXPECT_EQ(step.contribution, step.raw > 9 ? step.raw - 9 : step.raw);
          EXPECT_EQ(step.before, residue);
          residue = (residue + step.contribution) % 10;
          EXPECT_EQ(step.after, residue);
        }
        EXPECT_EQ(residue, t.analysis.remainder);
      }
      EXPECT_EQ(luhn_check_digit(s), (10 - reference(s, true)) % 10);
      EXPECT_TRUE(luhn_validate(s + char('0' + luhn_check_digit(s))));
      EXPECT_EQ(luhn_validate(s), !s.empty() && reference(s, false) == 0);
    }
}
TEST(LuhnTrace, OriginalErrorStudyAndBlindSpotIsOnlyDistinctAdjacentPair) {
  auto original = luhn_error_study("1230904560907");
  EXPECT_TRUE(luhn_validate(original.full));
  EXPECT_EQ(original.substitutions, 9 * original.full.size());
  EXPECT_EQ(original.substitutions_detected, original.substitutions);
  EXPECT_LT(original.swaps_detected, original.swaps);
  for (int payload = 0; payload < 100; ++payload) {
    auto study = luhn_error_study(std::to_string(payload));
    for (auto& e : study.cases) {
      EXPECT_EQ(e.remainder, reference(e.changed, false));
      EXPECT_EQ(e.detected, !luhn_validate(e.changed));
      if (e.kind == LuhnErrorKind::Substitution)
        EXPECT_TRUE(e.detected);
      else {
        const auto a = study.full[e.index], b = study.full[e.index + 1];
        const bool blind = (a == '0' && b == '9') || (a == '9' && b == '0');
        EXPECT_EQ(e.detected, !blind);
      }
    }
  }
  EXPECT_EQ(luhn_check_digit("7992739871"), 3);
  EXPECT_TRUE(luhn_validate("79927398713"));
  EXPECT_FALSE(luhn_validate("79927398710"));
}
TEST(LuhnTrace, GuardsEmptyZerosQuotaAndOwnership) {
  EXPECT_FALSE(luhn_validate(""));
  EXPECT_TRUE(luhn_validate("0"));
  EXPECT_TRUE(luhn_validate("0000"));
  EXPECT_EQ(luhn_check_digit(""), 0);
  EXPECT_EQ(luhn_trace("").captured_entries, 0);
  for (auto s :
       {std::string("12 3"), std::string("-1"), std::string("12\0", 3), std::string("\xff", 1)}) {
    EXPECT_THROW(luhn_check_digit(s), std::invalid_argument);
    EXPECT_THROW(luhn_validate(s), std::invalid_argument);
    EXPECT_THROW(luhn_trace(s), std::invalid_argument);
    EXPECT_THROW(luhn_error_study(s), std::invalid_argument);
  }
  EXPECT_THROW(luhn_analyze("", static_cast<LuhnMode>(99)), std::invalid_argument);
  EXPECT_THROW(luhn_trace("123", LuhnMode::Full, 5), std::length_error);
  EXPECT_EQ(luhn_trace("123", LuhnMode::Full, 6).captured_entries, 6);
  std::string s = "123";
  auto        t = luhn_trace(s, LuhnMode::Full, 0);
  s[0]          = '9';
  EXPECT_EQ(t.input, "123");
  auto errors = luhn_error_study("123");
  EXPECT_THROW(luhn_error_study("123", errors.captured_entries - 1), std::length_error);
  EXPECT_EQ(
      luhn_error_study("123", errors.captured_entries).captured_entries, errors.captured_entries);
  EXPECT_GE(luhn_analyze(std::string(10000, '9')).remainder, 0);
  EXPECT_LT(luhn_analyze(std::string(10000, '9')).remainder, 10);
}
