#include <gtest/gtest.h>

#include <datamunge/algorithms/blast.hpp>
#include <datamunge/algorithms/bloom_filter.hpp>
#include <datamunge/algorithms/maximum_parsimony.hpp>
#include <datamunge/algorithms/signed_reversals.hpp>
#include <datamunge/algorithms/upgma.hpp>
#include <datamunge/algorithms/velvet.hpp>

#include <string>
#include <vector>

using namespace datamunge::algorithms;

// ---- Bloom filter -----------------------------------------------------------

TEST(BloomFilter, NoFalseNegatives) {
    BloomFilter bf(2000, BloomFilter::optimal_num_hashes(2000, 200));
    for (int i = 0; i < 200; ++i) bf.add(bloom_hash64("key" + std::to_string(i)));
    for (int i = 0; i < 200; ++i)
        EXPECT_TRUE(bf.maybe_contains(bloom_hash64("key" + std::to_string(i))));  // never a false negative
}

TEST(BloomFilter, FalsePositiveRateNearPrediction) {
    BloomFilter bf(4000, BloomFilter::optimal_num_hashes(4000, 400));
    for (int i = 0; i < 400; ++i) bf.add(bloom_hash64("m" + std::to_string(i)));
    int fp = 0, trials = 4000;
    for (int i = 100000; i < 100000 + trials; ++i)
        if (bf.maybe_contains(bloom_hash64("m" + std::to_string(i)))) ++fp;
    double rate = static_cast<double>(fp) / trials;
    EXPECT_LT(rate, 3.0 * bf.expected_false_positive_rate(400) + 0.01);  // within a few x of predicted
}

// ---- UPGMA ------------------------------------------------------------------

TEST(UPGMA, ClassicFiveTaxaExample) {
    std::vector<std::vector<double>> d = {{0, 17, 21, 31, 23},
                                          {17, 0, 30, 34, 21},
                                          {21, 30, 0, 28, 39},
                                          {31, 34, 28, 0, 43},
                                          {23, 21, 39, 43, 0}};
    auto u = upgma(d);
    ASSERT_EQ(u.merges.size(), 4u);           // n-1 merges for 5 taxa
    // Closest pair is (0,1) at distance 17 -> height 8.5, first merge.
    EXPECT_EQ(u.merges[0].left, 0);
    EXPECT_EQ(u.merges[0].right, 1);
    EXPECT_DOUBLE_EQ(u.merges[0].height, 8.5);
    // Ultrametric: heights are non-decreasing toward the root.
    for (std::size_t i = 1; i < u.merges.size(); ++i)
        EXPECT_GE(u.merges[i].height, u.merges[i - 1].height);
}

// ---- Maximum parsimony (Fitch) ----------------------------------------------

TEST(MaximumParsimony, FitchScore) {
    ParsimonyTree t;
    t.root = 6;
    t.children.assign(7, {-1, -1});
    t.children[4] = {0, 1};
    t.children[5] = {2, 3};
    t.children[6] = {4, 5};

    // Character A C A G (states 0 1 0 2): needs 2 changes on this balanced tree.
    std::vector<int> col(7, -1);
    col[0] = 0; col[1] = 1; col[2] = 0; col[3] = 2;
    EXPECT_EQ(fitch_small_parsimony(t, col), 2);

    // All leaves identical -> 0 changes.
    std::vector<int> same(7, -1);
    same[0] = same[1] = same[2] = same[3] = 1;
    EXPECT_EQ(fitch_small_parsimony(t, same), 0);

    // Column matrix score sums per-character scores.
    EXPECT_EQ(parsimony_score(t, {col, same}), 2);
}

// ---- Sorting by signed reversals --------------------------------------------

TEST(SignedReversals, ReversedIdentityIsOneReversal) {
    // [-3,-2,-1] -> reverse the whole thing -> [1,2,3]. Distance 1.
    auto r = sort_by_reversals({-3, -2, -1});
    EXPECT_EQ(r.distance, 1);
    EXPECT_EQ(r.reversals.size(), 1u);
}

TEST(SignedReversals, IdentityIsZero) {
    EXPECT_EQ(sort_by_reversals({1, 2, 3, 4}).distance, 0);
    EXPECT_EQ(signed_breakpoints({1, 2, 3, 4}), 0);       // sorted -> no breakpoints
    EXPECT_GT(signed_breakpoints({3, 1, 2}), 0);
}

TEST(SignedReversals, DistanceMatchesReconstruction) {
    std::vector<int> p = {2, -4, -3, 1};
    auto             r = sort_by_reversals(p);
    // Apply the returned reversals and check we reach the identity.
    std::vector<int> cur = p;
    for (auto [i, j] : r.reversals) cur = detail::apply_reversal(cur, i, j);
    EXPECT_EQ(cur, (std::vector<int>{1, 2, 3, 4}));
    EXPECT_EQ((int)r.reversals.size(), r.distance);
}

// ---- BLAST ------------------------------------------------------------------

TEST(Blast, FindsSharedRegion) {
    std::string db    = "ACGTACGTGGATCCAATTGGCCAAGGTT";
    std::string query = "TTGGATCCAATT";     // shares "TGGATCCAATT" with db
    auto        hsps  = blast_search(query, db, 4);
    ASSERT_FALSE(hsps.empty());
    // Best HSP is the exact shared run: 11 matched positions * 2 = 22.
    EXPECT_EQ(hsps[0].score, 22);
    EXPECT_EQ(hsps[0].length, 11);
}

TEST(Blast, NoSeedNoHit) {
    auto hsps = blast_search("AAAAAA", "TTTTTT", 4);
    EXPECT_TRUE(hsps.empty());
}

// ---- Velvet (de Bruijn assembly) --------------------------------------------

TEST(Velvet, ReconstructsSequenceFromKmers) {
    std::string              orig = "ATGGCGTGCAAT";
    int                      k    = 4;
    std::vector<std::string> reads;
    for (int i = 0; i + k <= (int)orig.size(); ++i) reads.push_back(orig.substr(i, k));
    EXPECT_EQ(velvet_assemble(reads, k), orig);
}

TEST(Velvet, AssemblesFromOverlappingReads) {
    // Two overlapping reads of ACGTACGTACGT... style sequence.
    std::string orig = "TTACGGCATTAC";
    int         k    = 3;
    std::vector<std::string> reads = {orig.substr(0, 8), orig.substr(4, 8)};
    std::string              a     = velvet_assemble(reads, k);
    EXPECT_FALSE(a.empty());
}
