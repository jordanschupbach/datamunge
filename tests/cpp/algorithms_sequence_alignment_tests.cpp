#include <gtest/gtest.h>

#include <datamunge/algorithms/sequence_alignment.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using datamunge::algorithms::Alignment;
using datamunge::algorithms::AlignmentScoring;
using datamunge::algorithms::hirschberg;
using datamunge::algorithms::needleman_wunsch;
using datamunge::algorithms::smith_waterman;

namespace {

// Recompute an alignment's score directly from its two gapped rows, as an independent check.
int recompute_score(const Alignment& al, AlignmentScoring s) {
    int score = 0;
    for (std::size_t k = 0; k < al.a_aligned.size(); ++k) {
        const char x = al.a_aligned[k];
        const char y = al.b_aligned[k];
        if (x == '-' || y == '-')
            score += s.gap;
        else
            score += (x == y) ? s.match : s.mismatch;
    }
    return score;
}

std::string strip_gaps(const std::string& s) {
    std::string out;
    for (char c : s)
        if (c != '-') out.push_back(c);
    return out;
}

bool is_substring(const std::string& sub, const std::string& full) {
    return full.find(sub) != std::string::npos;
}

std::string random_string(std::mt19937& rng, std::size_t max_len, int alphabet_size) {
    std::uniform_int_distribution<std::size_t> len_dist(0, max_len);
    std::uniform_int_distribution<int>         chr_dist(0, alphabet_size - 1);
    std::string                                s(len_dist(rng), 'A');
    for (char& c : s) c = static_cast<char>('A' + chr_dist(rng));
    return s;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Needleman-Wunsch (global)
// ------------------------------------------------------------------------------------------------

TEST(NeedlemanWunsch, KnownSmallAlignment) {
    const AlignmentScoring s{1, -1, -1};
    const Alignment        al = needleman_wunsch("AAAC", "AGC", s);
    EXPECT_EQ(al.score, 0);
    EXPECT_EQ(al.a_aligned, "AAAC");
    EXPECT_EQ(al.b_aligned, "-AGC");
    EXPECT_EQ(recompute_score(al, s), al.score);
}

TEST(NeedlemanWunsch, GapAndEmptyBoundaries) {
    EXPECT_EQ(needleman_wunsch("", "").score, 0);
    EXPECT_EQ(needleman_wunsch("ABC", "").score, 3 * AlignmentScoring{}.gap); // all gaps
    EXPECT_EQ(needleman_wunsch("", "ABCD").score, 4 * AlignmentScoring{}.gap);
    EXPECT_EQ(needleman_wunsch("ABC", "ABC").score, 3 * AlignmentScoring{}.match);
}

TEST(NeedlemanWunsch, ScoreConsistencyAndRecovery) {
    std::mt19937 rng(101);
    for (int trial = 0; trial < 3000; ++trial) {
        const AlignmentScoring s{2, -1, -2};
        const std::string      a  = random_string(rng, 9, 4);
        const std::string      b  = random_string(rng, 9, 4);
        const Alignment        al = needleman_wunsch(a, b, s);
        EXPECT_EQ(al.a_aligned.size(), al.b_aligned.size());
        EXPECT_EQ(strip_gaps(al.a_aligned), a);          // global: whole strings recovered
        EXPECT_EQ(strip_gaps(al.b_aligned), b);
        EXPECT_EQ(recompute_score(al, s), al.score);     // reported score matches the rows
    }
}

// ------------------------------------------------------------------------------------------------
// Smith-Waterman (local)
// ------------------------------------------------------------------------------------------------

TEST(SmithWaterman, LocalBeatsGlobalOnFlank) {
    // Default scoring {2,-1,-2}. The leading 'A' of "AAGT" is dropped by the local aligner rather
    // than gap-penalized, so the local score (6) exceeds the global score (4).
    const Alignment sw = smith_waterman("AAGT", "AGT");
    EXPECT_EQ(sw.score, 6);
    EXPECT_EQ(sw.a_aligned, "AGT");
    EXPECT_EQ(sw.b_aligned, "AGT");
    EXPECT_EQ(needleman_wunsch("AAGT", "AGT").score, 4);
}

TEST(SmithWaterman, NoPositiveAlignmentIsZero) {
    const Alignment sw = smith_waterman("AAAA", "TTTT"); // nothing matches
    EXPECT_EQ(sw.score, 0);
    EXPECT_TRUE(sw.a_aligned.empty());
    EXPECT_TRUE(sw.b_aligned.empty());
}

TEST(SmithWaterman, ScoreConsistencyAndSubstrings) {
    std::mt19937 rng(202);
    for (int trial = 0; trial < 3000; ++trial) {
        const AlignmentScoring s{2, -1, -2};
        const std::string      a  = random_string(rng, 9, 4);
        const std::string      b  = random_string(rng, 9, 4);
        const Alignment        sw = smith_waterman(a, b, s);
        EXPECT_GE(sw.score, 0);
        EXPECT_EQ(sw.a_aligned.size(), sw.b_aligned.size());
        EXPECT_EQ(recompute_score(sw, s), sw.score);
        // The aligned rows, with gaps removed, are substrings of the originals.
        EXPECT_TRUE(is_substring(strip_gaps(sw.a_aligned), a));
        EXPECT_TRUE(is_substring(strip_gaps(sw.b_aligned), b));
        // A local optimum is never worse than doing nothing, nor better than the all-match bound.
        EXPECT_LE(sw.score, static_cast<int>(std::min(a.size(), b.size())) * s.match);
    }
}

// ------------------------------------------------------------------------------------------------
// Hirschberg (linear-space global)
// ------------------------------------------------------------------------------------------------

TEST(Hirschberg, MatchesNeedlemanWunschScore) {
    std::mt19937 rng(303);
    for (int trial = 0; trial < 3000; ++trial) {
        const AlignmentScoring s = (trial % 2 == 0) ? AlignmentScoring{2, -1, -2}
                                                    : AlignmentScoring{1, -1, -1};
        const std::string a = random_string(rng, 12, 3);
        const std::string b = random_string(rng, 12, 3);

        const Alignment hb = hirschberg(a, b, s);
        const Alignment nw = needleman_wunsch(a, b, s);
        EXPECT_EQ(hb.score, nw.score);                 // same optimal score as full NW
        EXPECT_EQ(strip_gaps(hb.a_aligned), a);        // a valid global alignment of the whole strings
        EXPECT_EQ(strip_gaps(hb.b_aligned), b);
        EXPECT_EQ(recompute_score(hb, s), hb.score);   // and its rows really achieve that score
    }
}

// Dynamic Time Warping, the fourth member of the sequence-alignment family, is provided by the
// geometry module (datamunge::geometry::dynamic_time_warping) and is exercised by
// geometry_dtw_and_frechet_tests.cpp; it is not re-tested here.
