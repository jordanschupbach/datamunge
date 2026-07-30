#include <gtest/gtest.h>

#include <datamunge/algorithms/levenshtein_coding.hpp>
#include <datamunge/algorithms/package_merge.hpp>
#include <datamunge/algorithms/range_coding.hpp>
#include <datamunge/algorithms/shannon_fano_elias.hpp>
#include <datamunge/algorithms/truncated_binary.hpp>
#include <datamunge/algorithms/unary_coding.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

// ---- Unary ------------------------------------------------------------------

TEST(Unary, RoundTrip) {
    std::vector<std::uint64_t> v = {0, 1, 2, 5, 3, 0, 7};
    EXPECT_EQ(unary_decode(unary_encode(v), (int)v.size()), v);
    std::vector<bool> b;
    unary_encode_value(3, b);
    EXPECT_EQ(b, (std::vector<bool>{true, true, true, false}));  // 3 -> 1110
}

// ---- Truncated binary -------------------------------------------------------

TEST(TruncatedBinary, KnownCodewordsAndRoundTrip) {
    auto code = [](std::uint64_t v, std::uint64_t n) {
        std::vector<bool> b; truncated_binary_encode_value(v, n, b);
        std::string s; for (bool x : b) s += x ? '1' : '0'; return s;
    };
    // n = 5: first 3 values in 2 bits, last 2 in 3 bits.
    EXPECT_EQ(code(0, 5), "00");
    EXPECT_EQ(code(2, 5), "10");
    EXPECT_EQ(code(3, 5), "110");
    EXPECT_EQ(code(4, 5), "111");

    std::vector<std::uint64_t> v = {0, 1, 2, 3, 4, 2, 0};
    auto                       bits = truncated_binary_encode(v, 5);
    EXPECT_EQ(truncated_binary_decode(bits, 5, (int)v.size()), v);
}

// ---- Levenshtein ------------------------------------------------------------

TEST(Levenshtein, RoundTripRange) {
    for (std::uint64_t n = 0; n <= 2000; ++n) {
        std::vector<bool> b;
        levenshtein_encode_value(n, b);
        auto d = levenshtein_decode(b, 1);
        ASSERT_EQ(d.size(), 1u);
        EXPECT_EQ(d[0], n);
    }
}

TEST(Levenshtein, SmallCodewords) {
    auto code = [](std::uint64_t n) {
        std::vector<bool> b; levenshtein_encode_value(n, b);
        std::string s; for (bool x : b) s += x ? '1' : '0'; return s;
    };
    EXPECT_EQ(code(0), "0");
    EXPECT_EQ(code(1), "10");
}

// ---- Shannon-Fano-Elias -----------------------------------------------------

TEST(ShannonFanoElias, RoundTripAndPrefixFree) {
    std::vector<std::uint32_t> freq = {4, 3, 2, 1};
    std::vector<int>           msg  = {0, 1, 0, 2, 0, 3, 1, 0};
    auto                       bits = shannon_fano_elias_encode(msg, freq);
    EXPECT_EQ(shannon_fano_elias_decode(bits, freq, (int)msg.size()), msg);

    // Prefix-free: no codeword is a prefix of another.
    auto cb = shannon_fano_elias_codebook(freq);
    for (std::size_t i = 0; i < cb.size(); ++i)
        for (std::size_t j = 0; j < cb.size(); ++j)
            if (i != j && cb[i].size() <= cb[j].size()) {
                bool prefix = std::equal(cb[i].begin(), cb[i].end(), cb[j].begin());
                EXPECT_FALSE(prefix) << "codeword " << i << " prefixes " << j;
            }
}

// ---- Range coding -----------------------------------------------------------

TEST(RangeCoding, RoundTrip) {
    std::vector<std::uint32_t> freq = {50, 20, 20, 10};
    std::vector<int>           msg  = {0, 1, 0, 2, 0, 0, 3, 1, 0, 2, 0, 0, 1, 0, 3, 0};
    auto                       bytes = range_encode(msg, freq);
    EXPECT_EQ(range_decode(bytes, freq, (int)msg.size()), msg);
}

TEST(RangeCoding, CompressesSkewedSource) {
    std::vector<std::uint32_t> freq = {90, 6, 3, 1};
    std::vector<int>           msg;
    for (int i = 0; i < 400; ++i) msg.push_back(i % 25 == 0 ? 1 : 0);
    auto bytes = range_encode(msg, freq);
    EXPECT_EQ(range_decode(bytes, freq, (int)msg.size()), msg);
    EXPECT_LT(bytes.size() * 8.0 / msg.size(), 1.0);  // < 1 bit/symbol on this skewed source
}

// ---- Package-merge ----------------------------------------------------------

TEST(PackageMerge, KraftEqualityAndLengthLimit) {
    std::vector<std::uint64_t> w = {20, 17, 6, 3, 2, 2, 2, 1, 1};
    for (int L : {4, 5, 6}) {
        auto   len   = package_merge(w, L);
        double kraft = 0;
        int    maxl  = 0;
        for (int l : len) { kraft += std::pow(2.0, -l); maxl = std::max(maxl, l); }
        EXPECT_NEAR(kraft, 1.0, 1e-9) << "L=" << L;   // a complete prefix code
        EXPECT_LE(maxl, L) << "L=" << L;              // respects the length limit
    }
}

TEST(PackageMerge, TighterLimitCostsMore) {
    // Loosening the limit can only lower (or equal) the optimal weighted length.
    std::vector<std::uint64_t> w = {20, 17, 6, 3, 2, 2, 2, 1, 1};
    auto cost = [&](int L) {
        auto len = package_merge(w, L);
        long c = 0;
        for (std::size_t i = 0; i < w.size(); ++i) c += (long)w[i] * len[i];
        return c;
    };
    EXPECT_GE(cost(4), cost(5));
    EXPECT_GE(cost(5), cost(6));
}
