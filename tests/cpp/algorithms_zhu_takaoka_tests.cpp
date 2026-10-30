#include <datamunge/algorithms/string_search_extra.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace alg = datamunge::algorithms;
TEST(ZhuTakaoka, CompiledTraceCountsTheActualSearch) {
    const alg::ZhuTakaoka matcher("aa");
    const auto work = matcher.scan("aaaa");
    EXPECT_EQ(work.positions, (std::vector<std::size_t>{0,1,2}));
    EXPECT_EQ(work.comparisons, 6);
    EXPECT_EQ(work.alignments, 3);
    const auto trace = matcher.trace("aaaa");
    EXPECT_EQ(trace.nrows(), work.alignments);
    double sum = 0;
    for (std::size_t i = 0; i < trace.nrows(); ++i) {
        sum += trace.double_at("comparisons", i);
        EXPECT_EQ(trace.double_at("advance", i), 1);
        EXPECT_EQ(trace.double_at("match", i), 1);
        EXPECT_EQ(trace.double_at("digram_shift", i), 0);
    }
    EXPECT_EQ(sum, work.comparisons);
    EXPECT_EQ(matcher.search("a"), (std::vector<std::size_t>{}));
    EXPECT_TRUE(matcher.trace("a").empty());
}
TEST(ZhuTakaoka, DigramShiftsIncludePrefixBoundaryAndExcludeFinalPair) {
    const auto table = alg::ZhuTakaoka("Miss.").digram_table("Mis.");
    ASSERT_EQ(table.nrows(), 16);
    EXPECT_EQ(table.double_at("shift", 1), 3); // Mi
    EXPECT_EQ(table.double_at("shift", 10), 1); // ss
    EXPECT_EQ(table.double_at("shift", 12), 4); // .M, prefix overlap
    EXPECT_EQ(table.double_at("shift", 11), 5); // s., terminal pair excluded
    EXPECT_THROW(alg::ZhuTakaoka("x").digram_table("xy"), std::invalid_argument);
    EXPECT_EQ(alg::ZhuTakaoka("ab").digram_table("").nrows(), 0);
}
TEST(ZhuTakaoka, TextbookComparisonCountAndBinaryBytes) {
    const auto example = alg::ZhuTakaoka("GCAGAGAG").scan("GCATCGCAGAGAGTATACAGTACG");
    EXPECT_EQ(example.positions, (std::vector<std::size_t>{5}));
    EXPECT_EQ(example.comparisons, 14);
    const std::string pattern{"\0\xff", 2};
    const std::string text{"a\0\xff\0\xffz", 6};
    const alg::ZhuTakaoka matcher(pattern);
    EXPECT_EQ(matcher.search(text), (std::vector<std::size_t>{1,3}));
    EXPECT_EQ(alg::boyer_moore_horspool_search(text, pattern), matcher.search(text));
    EXPECT_TRUE(alg::ZhuTakaoka("").scan(text).positions.empty());
    EXPECT_EQ(alg::ZhuTakaoka("a").scan("banana").comparisons, 6);
    EXPECT_EQ(alg::ZhuTakaoka("a").search("banana"), (std::vector<std::size_t>{1,3,5}));
}
TEST(ZhuTakaoka, CorpusCountersAndRecordBoundaries) {
    datamunge::dstruct::DataFrame frame;
    frame.add_column("text", std::vector<std::string>{"abab"});
    const auto table = alg::string_search_comparison(frame,"text",{"ab"});
    ASSERT_EQ(table.nrows(), 3);
    EXPECT_EQ(table.double_at("comparisons", 0), 5);
    EXPECT_EQ(table.double_at("comparisons", 1), 4);
    EXPECT_EQ(table.double_at("comparisons", 2), 4);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(table.double_at("matches", i), 2);
        EXPECT_EQ(table.double_at("agrees_with_naive", i), 1);
    }
    datamunge::dstruct::DataFrame split;
    split.add_column("text", std::vector<std::string>{"ab", "a"});
    const auto no_cross = alg::string_search_comparison(split,"text",{"aba", ""});
    for (std::size_t i = 0; i < no_cross.nrows(); ++i) EXPECT_EQ(no_cross.double_at("matches", i), 0);
    EXPECT_THROW(alg::string_search_comparison(frame,"missing",{"a"}), std::out_of_range);
    frame.set_null("text", 0);
    EXPECT_THROW(alg::string_search_comparison(frame,"text",{"a"}), std::invalid_argument);
}
