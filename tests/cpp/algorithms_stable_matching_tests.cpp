#include <gtest/gtest.h>

#include <datamunge/algorithms/stable_matching.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using datamunge::algorithms::gale_shapley;
using datamunge::algorithms::is_stable;
using datamunge::algorithms::kUnmatched;
using datamunge::algorithms::StableMatchingResult;

namespace {

std::vector<std::vector<std::size_t>> random_preferences(std::size_t n, std::mt19937_64& rng) {
    std::vector<std::vector<std::size_t>> p(n, std::vector<std::size_t>(n));
    for (auto& list : p) {
        std::iota(list.begin(), list.end(), 0);
        std::shuffle(list.begin(), list.end(), rng);
    }
    return p;
}

std::size_t rank_of(const std::vector<std::size_t>& pref, std::size_t target) {
    return static_cast<std::size_t>(std::find(pref.begin(), pref.end(), target) - pref.begin());
}

} // namespace

TEST(GaleShapley, EveryoneGetsMutualFirstChoice) {
    // proposer i and reviewer i rank each other first -> matched i<->i, all first choices.
    const std::size_t n = 4;
    std::vector<std::vector<std::size_t>> p(n), r(n);
    for (std::size_t i = 0; i < n; ++i) {
        p[i].push_back(i);
        r[i].push_back(i);
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) { p[i].push_back(j); r[i].push_back(j); }
    }
    const auto m = gale_shapley(p, r);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(m.proposer_match[i], i);
        EXPECT_EQ(m.reviewer_match[i], i);
    }
    EXPECT_TRUE(is_stable(m, p, r));
    EXPECT_EQ(m.proposals, n); // each proposer accepted on its first proposal
}

TEST(GaleShapley, DetectsAnUnstableMatching) {
    const std::size_t n = 2;
    // Both proposers prefer reviewer 0; both reviewers prefer proposer 0.
    std::vector<std::vector<std::size_t>> p = {{0, 1}, {0, 1}};
    std::vector<std::vector<std::size_t>> r = {{0, 1}, {0, 1}};
    // The swapped matching 0<->1, 1<->0 is unstable: proposer 0 and reviewer 0 both prefer each other.
    StableMatchingResult bad;
    bad.proposer_match = {1, 0};
    bad.reviewer_match = {1, 0};
    EXPECT_FALSE(is_stable(bad, p, r));
    // Gale-Shapley itself returns the stable one (0<->0, 1<->1).
    const auto m = gale_shapley(p, r);
    EXPECT_TRUE(is_stable(m, p, r));
    EXPECT_EQ(m.proposer_match[0], 0u);
}

TEST(GaleShapley, AlwaysStableAndCompleteOnRandomInstances) {
    std::mt19937_64 rng(2024);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 2 + rng() % 14;
        const auto p = random_preferences(n, rng);
        const auto r = random_preferences(n, rng);
        const auto m = gale_shapley(p, r);
        ASSERT_TRUE(is_stable(m, p, r)) << "trial " << trial;
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NE(m.proposer_match[i], kUnmatched);
            EXPECT_NE(m.reviewer_match[i], kUnmatched);
        }
    }
}

TEST(GaleShapley, ProposerOptimalityBeatsReviewerProposing) {
    // Every proposer's partner under proposer-proposing is at least as good (rank <=) as under
    // reviewer-proposing -- the proposer-optimality theorem.
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 100; ++trial) {
        const std::size_t n = 3 + rng() % 10;
        const auto p = random_preferences(n, rng);
        const auto r = random_preferences(n, rng);
        const auto proposer_opt = gale_shapley(p, r);
        const auto reviewer_opt = gale_shapley(r, p); // reviewers propose
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t rank_prop = rank_of(p[i], proposer_opt.proposer_match[i]);
            const std::size_t rank_rev = rank_of(p[i], reviewer_opt.reviewer_match[i]);
            EXPECT_LE(rank_prop, rank_rev) << "trial " << trial << " proposer " << i;
        }
    }
}

TEST(GaleShapley, RejectsMalformedInput) {
    EXPECT_THROW(gale_shapley({{0, 1}}, {{0}, {1}}), std::invalid_argument);        // unequal sizes
    EXPECT_THROW(gale_shapley({{0}}, {{0, 1}}), std::invalid_argument);            // list not length n
    EXPECT_THROW(gale_shapley({{0, 0}, {1, 0}}, {{0, 1}, {0, 1}}), std::invalid_argument); // not a permutation
}
