#include <datamunge/algorithms/berlekamp_massey.hpp>
#include <datamunge/algorithms/lfsr.hpp>
#include <datamunge/datasets/csv.hpp>
#include <datamunge/datasets/datasets.hpp>

#include <gtest/gtest.h>

#include <random>
#include <vector>

namespace alg = datamunge::algorithms;

namespace {

std::vector<int> primitive(int degree, std::vector<int> middle_taps) {
  std::vector<int> c(static_cast<std::size_t>(degree) + 1, 0);
  c[0] = 1;
  c[static_cast<std::size_t>(degree)] = 1;
  for (const int t : middle_taps) c[static_cast<std::size_t>(t)] = 1;
  return c;
}

}  // namespace

TEST(Lfsr, SequenceMatchesSmallExampleAndPeriodIsMaximal) {
  EXPECT_EQ(alg::lfsr_sequence({1, 1, 0, 1}, {0, 0, 1}, 14),
            (std::vector<int>{0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1}));
  EXPECT_EQ(alg::lfsr_period({1, 1, 0, 1}, {0, 0, 1}), 7);
  const auto c17 = primitive(17, {3});
  std::vector<int> seed(17, 0);
  seed[4] = 1;
  EXPECT_EQ(alg::lfsr_period(c17, seed), (1LL << 17) - 1);
  // x^4 + x^2 + 1 = (x^2 + x + 1)^2 is not primitive: shorter period.
  EXPECT_LT(alg::lfsr_period({1, 0, 1, 0, 1}, {0, 0, 0, 1}), 15);
  EXPECT_THROW((void)alg::lfsr_sequence({0, 1}, {1}, 3), std::invalid_argument);
  EXPECT_THROW((void)alg::lfsr_period({1, 1, 0, 1}, {0, 0, 0}), std::invalid_argument);
}

TEST(Lfsr, BerlekampMasseyRecoversTheRegisterFromTwoLBits) {
  const auto c = primitive(31, {3});
  std::mt19937 rng(3);
  std::vector<int> seed(31);
  for (auto& b : seed) b = static_cast<int>(rng() & 1);
  seed[0] = 1;
  const auto s = alg::lfsr_sequence(c, seed, 62);
  const auto fit = alg::berlekamp_massey(s);
  EXPECT_EQ(fit.length, 31);
  EXPECT_EQ(fit.polynomial, c);
}

TEST(Lfsr, ProfileMatchesPrefixByPrefixBerlekampMassey) {
  std::mt19937 rng(11);
  std::vector<int> s(150);
  for (auto& b : s) b = static_cast<int>(rng() & 1);
  const auto profile = alg::linear_complexity_profile(s);
  ASSERT_EQ(profile.size(), s.size());
  for (std::size_t k = 1; k <= s.size(); ++k)
    EXPECT_EQ(profile[k - 1], alg::berlekamp_massey(std::vector<int>(s.begin(), s.begin() + static_cast<long>(k))).length);
}

TEST(Lfsr, ExpectedComplexityMatchesExhaustiveEnumeration) {
  for (std::size_t n = 1; n <= 12; ++n) {
    double total = 0.0;
    for (unsigned code = 0; code < (1u << n); ++code) {
      std::vector<int> s(n);
      for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<int>((code >> i) & 1);
      total += alg::berlekamp_massey(s).length;
    }
    EXPECT_NEAR(total / static_cast<double>(1u << n), alg::expected_linear_complexity(n), 1e-12) << n;
  }
}

TEST(Lfsr, GeffeComplexityIsL1L2PlusL2L3PlusL3) {
  const auto c3 = primitive(3, {1}), c4 = primitive(4, {1}), c5 = primitive(5, {2});
  const std::size_t n = 400;
  const auto x1 = alg::lfsr_sequence(c3, {1, 0, 0}, n);
  const auto x2 = alg::lfsr_sequence(c4, {0, 1, 0, 0}, n);
  const auto x3 = alg::lfsr_sequence(c5, {1, 1, 0, 0, 0}, n);
  const auto z = alg::geffe_sequence(x1, x2, x3);
  EXPECT_EQ(alg::berlekamp_massey(z).length, 3 * 4 + 4 * 5 + 5);
}

TEST(Lfsr, CorrelationScanMatchesDirectComparison) {
  const auto c = primitive(7, {1});
  std::mt19937 rng(5);
  std::vector<int> observed(60);
  for (auto& b : observed) b = static_cast<int>(rng() % 3);  // 2 marks an unknown bit
  const auto score = alg::lfsr_correlation_scan(c, observed);
  ASSERT_EQ(score.size(), 128u);
  for (long long code = 0; code < 128; ++code) {
    const auto s = alg::lfsr_sequence(c, alg::lfsr_seed(code, 7), observed.size());
    int agree = 0;
    for (std::size_t i = 0; i < s.size(); ++i) agree += observed[i] == s[i];
    EXPECT_EQ(score[static_cast<std::size_t>(code)], agree) << code;
  }
  EXPECT_EQ(alg::lfsr_correlation_scan({1, 1, 0, 1}, {0, 0, 1, 1, 1, 0, 1})[4], 7);
}

TEST(Lfsr, BitsAndCsvRoundTrip) {
  EXPECT_EQ(alg::bytes_to_bits("A"), (std::vector<int>{0, 1, 0, 0, 0, 0, 0, 1}));
  EXPECT_EQ(alg::bits_to_bytes(alg::bytes_to_bits("Hello\n")), "Hello\n");
  EXPECT_EQ(alg::xor_bits({1, 0, 1}, {1, 1, 0}), (std::vector<int>{0, 1, 1}));

  const auto faithful = datamunge::datasets::faithful();
  const auto text = datamunge::datasets::to_csv(faithful);
  EXPECT_EQ(text.substr(0, 18), "eruptions,waiting\n");
  const auto back = datamunge::datasets::parse_csv(text);
  ASSERT_EQ(back.nrows(), faithful.nrows());
  for (std::size_t r = 0; r < faithful.nrows(); ++r) {
    EXPECT_EQ(std::get<double>(back.value("eruptions", r)), std::get<double>(faithful.value("eruptions", r)));
    EXPECT_EQ(std::get<double>(back.value("waiting", r)), std::get<double>(faithful.value("waiting", r)));
  }
  datamunge::dstruct::DataFrame quoted;
  quoted.add_column("name", datamunge::dstruct::DataFrame::string_column_type({std::string("a,b"), std::string("say \"hi\""), std::nullopt}));
  const auto q = datamunge::datasets::to_csv(quoted);
  EXPECT_EQ(q, "name\n\"a,b\"\n\"say \"\"hi\"\"\"\nNA\n");
}
