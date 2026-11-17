#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/hamming_code_analysis.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(Hamming74Trace, CompleteIndependentNearestCodewordAndParityCertificates) {
  auto a = hamming74_audit();
  EXPECT_EQ(a.minimum_distance, 3);
  EXPECT_EQ(a.codeword_weights, (std::array<std::size_t, 8>{1, 0, 0, 7, 7, 0, 0, 1}));
  for (unsigned word = 0; word < 128; ++word) {
    std::array<int, 7> c{};
    for (int i = 0; i < 7; ++i)
      c[i] = (word >> i) & 1;
    auto t = hamming74_decode_trace(c);
    auto d = hamming74_decode(c);
    EXPECT_EQ(t.received, c);
    EXPECT_EQ(d.data, t.decoded.data);
    EXPECT_EQ(d.error_position, t.syndrome);
    EXPECT_EQ(t.syndrome, hamming74_syndrome(c));
    int syndrome = 0;
    for (int r = 0; r < 3; ++r) {
      int parity = 0;
      for (int j = 0; j < 7; ++j)
        if ((j + 1) & (1 << r))
          parity += c[j];
      syndrome += (parity % 2) * (1 << r);
      int sum = 0;
      for (int j = 0; j < 4; ++j) {
        const auto& q = t.checks[r];
        EXPECT_EQ(q.bits[j], c[q.positions[j] - 1]);
        sum += q.bits[j];
        EXPECT_EQ(q.prefix[j], sum % 2);
      }
    }
    EXPECT_EQ(t.syndrome, syndrome);
    EXPECT_EQ(t.work.parity_bit_visits, 12);
    EXPECT_EQ(t.work.parity_xors, 9);
    EXPECT_EQ(t.work.data_reads, 4);
    EXPECT_EQ(t.work.flips, syndrome != 0);
    int best = 8, winners = 0, winner = -1;
    for (int id = 0; id < 16; ++id) {
      int dist = 0;
      for (int j = 0; j < 7; ++j)
        dist += c[j] != a.codewords[id][j];
      if (dist < best) {
        best    = dist;
        winners = 1;
        winner  = id;
      } else if (dist == best)
        ++winners;
    }
    EXPECT_EQ(winners, 1);
    EXPECT_LE(best, 1);
    EXPECT_EQ(t.corrected_word, a.codewords[winner]);
    EXPECT_EQ(
        t.decoded.data,
        (std::array<int, 4>{(winner >> 3) & 1, (winner >> 2) & 1, (winner >> 1) & 1, winner & 1}));
    auto copy = t;
    copy.checks[0].bits[0] ^= 1;
    copy.received[0] ^= 1;
    EXPECT_EQ(t.received, c);
    EXPECT_NE(copy.checks[0].bits, t.checks[0].bits);
  }
  for (int w = 0; w <= 7; ++w) {
    EXPECT_EQ(a.recovered[w], w <= 1 ? a.patterns[w] : 0);
    if (w == 1 || w == 2)
      EXPECT_EQ(a.nonzero_syndromes[w], a.patterns[w]);
  }
}
TEST(Hamming74Trace, MisCorrectionNonbinaryDomainsAndModel) {
  auto c = hamming74_encode({1, 0, 1, 1});
  c[0] ^= 1;
  c[1] ^= 1;
  auto t = hamming74_decode_trace(c);
  EXPECT_EQ(t.syndrome, 3);
  EXPECT_TRUE(t.decoded.corrected);
  EXPECT_NE(t.decoded.data, (std::array<int, 4>{1, 0, 1, 1}));
  for (int value : {-100, -1, 2, 8, std::numeric_limits<int>::max()})
    for (int i = 0; i < 7; ++i) {
      std::array<int, 7> bad{};
      bad[i] = value;
      EXPECT_THROW(hamming74_decode(bad), std::invalid_argument);
      EXPECT_THROW(hamming74_decode_trace(bad), std::invalid_argument);
      EXPECT_THROW(hamming74_syndrome(bad), std::invalid_argument);
    }
  for (int value : {-1, 2, std::numeric_limits<int>::min()})
    for (int i = 0; i < 4; ++i) {
      std::array<int, 4> bad{};
      bad[i] = value;
      EXPECT_THROW(hamming74_encode(bad), std::invalid_argument);
    }
  EXPECT_DOUBLE_EQ(hamming74_bsc_recovery(0), 1);
  EXPECT_DOUBLE_EQ(hamming74_bsc_recovery(1), 0);
  EXPECT_NEAR(hamming74_bsc_recovery(.1), .8503056, 1e-14);
  EXPECT_THROW(hamming74_bsc_recovery(-.1), std::invalid_argument);
  EXPECT_THROW(
      hamming74_bsc_recovery(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}
