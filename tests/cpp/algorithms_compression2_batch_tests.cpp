#include <gtest/gtest.h>

#include <datamunge/algorithms/burrows_wheeler.hpp>
#include <datamunge/algorithms/elias_gamma.hpp>
#include <datamunge/algorithms/lz77.hpp>
#include <datamunge/algorithms/lzw.hpp>
#include <datamunge/algorithms/move_to_front.hpp>
#include <datamunge/algorithms/shannon_fano.hpp>

#include <cstdint>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

namespace {
const std::string kText =
    "the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.";
}

TEST(LZ77, RoundTrips) {
    for (const std::string& s : {kText, std::string("aaaaaaaa"), std::string("abcabcabcabc"),
                                 std::string("x"), std::string("")}) {
        auto toks = lz77_encode(s, 64, 32);
        EXPECT_EQ(lz77_decode(toks), s);
    }
    // Repetitive text compresses to fewer tokens than characters.
    EXPECT_LT(lz77_encode(kText).size(), kText.size());
}

TEST(LZW, RoundTripsAndCompresses) {
    for (const std::string& s : {kText, std::string("TOBEORNOTTOBEORTOBEORNOT"), std::string("aaaaaa"),
                                 std::string("z")}) {
        auto codes = lzw_encode(s);
        EXPECT_EQ(lzw_decode(codes), s);
    }
    EXPECT_LT(lzw_encode(kText).size(), kText.size());
}

TEST(BurrowsWheeler, RoundTripsAndClustersRuns) {
    for (const std::string& s : {std::string("banana"), kText, std::string("mississippi"),
                                 std::string("abracadabra")}) {
        auto bw = bwt_transform(s);
        EXPECT_EQ(bw.transformed.size(), s.size());
        EXPECT_EQ(bwt_inverse(bw), s);
    }
    // "banana" -> the classic BWT (grouping suffix contexts).
    auto bw = bwt_transform("banana");
    EXPECT_EQ(bw.transformed, "nnbaaa");
}

TEST(MoveToFront, RoundTripsAndShrinksClusteredData) {
    for (const std::string& s : {kText, std::string("aaaabbbbcccc"), std::string("q")}) {
        auto idx = mtf_encode(s);
        EXPECT_EQ(mtf_decode(idx), s);
    }
    // On BWT output (clustered), MTF produces many small indices (lots of zeros).
    auto        bw  = bwt_transform("mississippi");
    auto        idx = mtf_encode(bw.transformed);
    std::size_t zeros = 0;
    for (int v : idx) if (v == 0) ++zeros;
    EXPECT_GT(zeros, 0u);
}

TEST(ShannonFano, PrefixCodeBelowByteBaseline) {
    auto   code = shannon_fano_build(kText);
    double avg  = shannon_fano_average_bits(code, kText);
    EXPECT_GT(avg, 0.0);
    EXPECT_LT(avg, 8.0);  // beats a fixed 8-bit encoding
    // Prefix-free: no codeword is a prefix of another.
    std::vector<std::string> cw;
    for (const auto& [c, s] : code.codes) cw.push_back(s);
    for (std::size_t a = 0; a < cw.size(); ++a)
        for (std::size_t b = 0; b < cw.size(); ++b)
            if (a != b) EXPECT_FALSE(cw[a].size() <= cw[b].size() && cw[b].compare(0, cw[a].size(), cw[a]) == 0);
}

TEST(EliasGamma, RoundTripsAndKnownCodes) {
    // Known Elias gamma codewords.
    EXPECT_EQ(elias_gamma_encode_one(1), "1");
    EXPECT_EQ(elias_gamma_encode_one(2), "010");
    EXPECT_EQ(elias_gamma_encode_one(3), "011");
    EXPECT_EQ(elias_gamma_encode_one(4), "00100");
    EXPECT_EQ(elias_gamma_encode_one(9), "0001001");
    // Round-trip a sequence.
    std::vector<std::uint64_t> vals = {1, 2, 3, 4, 5, 10, 100, 1000, 7, 42};
    EXPECT_EQ(elias_gamma_decode(elias_gamma_encode(vals)), vals);
}
