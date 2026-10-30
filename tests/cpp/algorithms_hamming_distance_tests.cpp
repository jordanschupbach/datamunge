#include <gtest/gtest.h>

#include <array>
#include <bitset>
#include <cmath>
#include <datamunge/algorithms/hamming_distance.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(HammingGeometry, ByteAndBitComparisonsHaveDifferentAlphabets) {
  for (unsigned a = 0; a < 256; ++a)
    for (unsigned b = 0; b < 256; ++b) {
      const std::string aa(1, static_cast<char>(a)), bb(1, static_cast<char>(b));
      EXPECT_EQ(hamming_distance(aa, bb), a != b);
      EXPECT_EQ(
          hamming_bit_distance(a, b),
          hamming_distance(std::bitset<8>(a).to_string(), std::bitset<8>(b).to_string()));
    }
  EXPECT_EQ(hamming_bit_distance(0, std::numeric_limits<std::uint64_t>::max()), 64);
  EXPECT_EQ(hamming_bit_distance(0, std::uint64_t(1) << 63), 1);
  EXPECT_THROW(hamming_distance(std::string("\xc3\xa9", 2), "e"), std::invalid_argument);
}
TEST(HammingGeometry, ComparisonExposesOriginalBytePositionsAndPrefixCounts) {
  const auto t = hamming_comparison("karolin", "kathrin");
  EXPECT_EQ(t.nrows(), 7);
  EXPECT_EQ(t.double_at("cumulative_distance", 6), 3);
  for (std::size_t i = 0; i < 7; ++i)
    EXPECT_EQ(t.double_at("mismatch", i), i >= 2 && i <= 4);
  const auto bytes = hamming_comparison(std::string("\0\xff", 2), std::string("\0\x7f", 2));
  EXPECT_EQ(bytes.double_at("a_byte", 1), 255);
  EXPECT_EQ(bytes.double_at("b_byte", 1), 127);
  EXPECT_EQ(hamming_comparison("", "").nrows(), 0);
  EXPECT_THROW(hamming_comparison("a", ""), std::invalid_argument);
}
TEST(HammingGeometry, CubeMetricAndDistancesAgreeForEveryTriple) {
  std::vector<std::string> cube;
  for (unsigned i = 0; i < 8; ++i)
    cube.push_back(std::bitset<3>(i).to_string());
  const auto d = hamming_distance_matrix(cube);
  for (std::size_t i = 0; i < 8; ++i)
    for (std::size_t j = 0; j < 8; ++j) {
      EXPECT_EQ(d(i, j), hamming_bit_distance(i, j));
      EXPECT_EQ(d(i, j), d(j, i));
      EXPECT_EQ(d(i, j) == 0, i == j);
      for (std::size_t k = 0; k < 8; ++k)
        EXPECT_LE(d(i, k), d(i, j) + d(j, k));
    }
  EXPECT_EQ(hamming_distance_matrix({}).rows(), 0);
  EXPECT_EQ(hamming_distance_matrix({"", ""})(0, 1), 0);
  EXPECT_THROW(hamming_distance_matrix({"a", "bb"}), std::invalid_argument);
}
TEST(HammingGeometry, RealignmentCanStrictlyImproveTheEditDistance) {
  EXPECT_EQ(hamming_distance("abcdef", "bcdefa"), 6);
  EXPECT_EQ(levenshtein_distance("abcdef", "bcdefa"), 2);
}
TEST(HammingGeometry, RepetitionCodeCorrectsEveryOneBitErrorButCanMiscorrectTwo) {
  const std::vector<std::string> code    = {"000", "111"};
  const auto                     summary = hamming_code_summary(code);
  EXPECT_EQ(summary.double_at("minimum_distance", 0), 3);
  EXPECT_EQ(summary.double_at("detectable_errors", 0), 2);
  EXPECT_EQ(summary.double_at("correctable_errors", 0), 1);
  for (unsigned i = 0; i < 8; ++i) {
    const auto q        = std::bitset<3>(i).to_string();
    const auto t        = hamming_nearest(code, q);
    const auto distance = hamming_distance("000", q);
    EXPECT_EQ(t.double_at("nearest", distance <= 1 ? 0 : 1), 1);
  }
  EXPECT_EQ(hamming_nearest(code, "110").double_at("nearest", 1), 1);
}
TEST(HammingGeometry, ParityCodeReturnsAllNearestTiesAndRejectsInvalidCodebooks) {
  const std::vector<std::string> code = {"000", "011", "101", "110"};
  const auto                     t    = hamming_nearest(code, "001");
  EXPECT_EQ(t.double_at("nearest", 0), 1);
  EXPECT_EQ(t.double_at("nearest", 1), 1);
  EXPECT_EQ(t.double_at("nearest", 2), 1);
  EXPECT_EQ(t.double_at("nearest", 3), 0);
  EXPECT_EQ(hamming_code_summary(code).double_at("correctable_errors", 0), 0);
  EXPECT_THROW(hamming_code_summary({"000"}), std::invalid_argument);
  EXPECT_THROW(hamming_code_summary({"000", "000"}), std::invalid_argument);
  EXPECT_THROW(hamming_code_summary({"0", "11"}), std::invalid_argument);
  EXPECT_THROW(hamming_nearest({}, ""), std::invalid_argument);
  EXPECT_THROW(hamming_nearest({"0"}, "11"), std::invalid_argument);
  EXPECT_EQ(hamming_nearest({""}, "").double_at("nearest", 0), 1);
}
TEST(HammingGeometry, BinaryAndFourSymbolShellsMatchCompleteEnumeration) {
  for (std::size_t q : {2u, 4u}) {
    const std::size_t        n = q == 2 ? 8 : 4, total = 256;
    const auto               t = hamming_shells(n, q);
    std::vector<std::size_t> counts(n + 1, 0);
    for (std::size_t word = 0; word < total; ++word) {
      std::size_t rest = word, mismatches = 0;
      for (std::size_t j = 0; j < n; ++j) {
        mismatches += rest % q != 0;
        rest /= q;
      }
      ++counts[mismatches];
    }
    std::size_t cumulative  = 0;
    double      probability = 0;
    for (std::size_t k = 0; k <= n; ++k) {
      cumulative += counts[k];
      probability += t.double_at("probability", k);
      EXPECT_EQ(t.double_at("shell_size", k), counts[k]);
      EXPECT_EQ(t.double_at("ball_volume", k), cumulative);
      EXPECT_NEAR(t.double_at("cumulative_probability", k), double(cumulative) / total, 1e-15);
    }
    EXPECT_NEAR(probability, 1, 1e-15);
  }
  EXPECT_EQ(hamming_shells(0).double_at("shell_size", 0), 1);
  EXPECT_THROW(hamming_shells(3, 1), std::invalid_argument);
  EXPECT_THROW(hamming_shells(1024), std::overflow_error);
  EXPECT_THROW(hamming_shells(std::numeric_limits<std::size_t>::max()), std::overflow_error);
}
TEST(HammingGeometry, PlotUsesOriginalColumnsAndRejectsUnreadableCases) {
  const auto p = plot_hamming_comparison("1011101", "1001001");
  EXPECT_FALSE(p.to_svg().empty());
  EXPECT_THROW(plot_hamming_comparison("", ""), std::invalid_argument);
  EXPECT_THROW(plot_hamming_comparison("a", "bb"), std::invalid_argument);
  EXPECT_THROW(
      plot_hamming_comparison(std::string(65, 'a'), std::string(65, 'a')), std::invalid_argument);
}
