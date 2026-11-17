#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/algorithms/burrows_wheeler.hpp>
#include <datamunge/algorithms/move_to_front.hpp>
#include <numeric>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(MTFTrace, ExhaustiveBytesIndependentAlphabetReplayAndActualWork) {
  const std::array<unsigned char, 5> symbols{0, 65, 66, 128, 255};
  for (int length = 0, count = 1; length <= 6; ++length, count *= 5)
    for (int code = 0; code < count; ++code) {
      int         q = code;
      std::string input;
      for (int i = 0; i < length; ++i) {
        input.push_back(char(symbols[q % 5]));
        q /= 5;
      }
      auto             t = mtf_encode_trace(input);
      std::vector<int> alphabet(256);
      std::iota(alphabet.begin(), alphabet.end(), 0);
      std::vector<int> expected;
      std::size_t      shifts = 0;
      for (std::size_t i = 0; i < input.size(); ++i) {
        int   byte = static_cast<unsigned char>(input[i]);
        auto  pos  = std::find(alphabet.begin(), alphabet.end(), byte);
        int   rank = int(pos - alphabet.begin());
        auto& s    = t.steps[i];
        EXPECT_EQ(s.byte, byte);
        EXPECT_EQ(s.rank, rank);
        EXPECT_EQ(s.front_before, alphabet.front());
        EXPECT_EQ(s.comparisons, rank + 1);
        EXPECT_EQ(s.shifts, rank);
        expected.push_back(rank);
        shifts += rank;
        alphabet.erase(pos);
        alphabet.insert(alphabet.begin(), byte);
      }
      EXPECT_EQ(t.result.indices, expected);
      EXPECT_EQ(mtf_encode(input), expected);
      EXPECT_EQ(t.result.shifts, shifts);
      EXPECT_EQ(t.result.comparisons, shifts + input.size());
      EXPECT_EQ(t.captured_entries, 3 * input.size());
      auto inverse = mtf_decode_trace(expected);
      EXPECT_EQ(inverse.result.bytes, input);
      EXPECT_EQ(mtf_decode(expected), input);
      EXPECT_EQ(inverse.result.shifts, shifts);
      EXPECT_EQ(inverse.result.comparisons, 0);
      EXPECT_EQ(inverse.captured_entries, 3 * input.size());
      auto summary = mtf_index_summary(expected);
      auto sorted  = expected;
      std::sort(sorted.begin(), sorted.end());
      EXPECT_EQ(summary.count, expected.size());
      if (!sorted.empty())
        EXPECT_EQ(
            summary.median, (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2.);
      else
        EXPECT_EQ(summary.median, 0);
      EXPECT_EQ(summary.zeros + summary.one_to_three + summary.four_or_more, summary.count);
    }
}
TEST(MTFTrace, CompleteRankValidationQuotasOwnershipAndOriginalBWT) {
  for (auto bad : std::vector<std::vector<int>>{{-1}, {256}, {65, 0, -1}, {65, 256}}) {
    EXPECT_THROW(mtf_decode(bad), std::invalid_argument);
    EXPECT_THROW(mtf_decode_trace(bad, 1), std::invalid_argument);
    EXPECT_THROW(mtf_index_summary(bad), std::invalid_argument);
  }
  EXPECT_THROW(mtf_encode_trace("ABBA", 11), std::length_error);
  EXPECT_NO_THROW(mtf_encode_trace("ABBA", 12));
  EXPECT_NO_THROW(mtf_encode_trace("ABBA", 0));
  EXPECT_THROW(mtf_decode_trace({65, 66, 0, 1}, 11), std::length_error);
  auto text  = mtf_demo_text();
  auto bw    = bwt_transform(text);
  auto ranks = mtf_encode(bw.transformed);
  EXPECT_EQ(mtf_decode(ranks), bw.transformed);
  EXPECT_EQ(bwt_inverse({mtf_decode(ranks), bw.index}), text);
  auto input = std::string("\0\xff\0\xff", 4);
  auto t     = mtf_encode_trace(input);
  input.clear();
  EXPECT_EQ(t.input_bytes.size(), 4);
  EXPECT_EQ(mtf_decode(t.result.indices), t.input_bytes);
  auto s = mtf_index_summary({0, 0, 1, 1});
  EXPECT_EQ(s.median, .5);
  EXPECT_EQ(s.entropy_bits, 1);
}
