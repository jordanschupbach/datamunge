#include <gtest/gtest.h>

#include <datamunge/algorithms/odds_algorithm.hpp>
#include <datamunge/algorithms/subset_sum.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Odds algorithm ----------------

namespace {

// Probability of stopping on the last success using threshold t (0-based start):
//   P(win | t) = (prod_{i>=t} q_i) * (sum_{i>=t} p_i/q_i).
double win_prob_threshold(const std::vector<double>& p, std::size_t t) {
    double prod = 1.0, sum = 0.0;
    for (std::size_t i = t; i < p.size(); ++i) {
        const double q = 1.0 - p[i];
        prod *= q;
        sum += p[i] / q;
    }
    return prod * sum;
}

} // namespace

TEST(OddsAlgorithm, MaximisesOverAllThresholds) {
    std::mt19937_64                        rng(1);
    std::uniform_real_distribution<double> pr(0.01, 0.95);
    for (int t = 0; t < 2000; ++t) {
        const int           n = 1 + static_cast<int>(rng() % 20);
        std::vector<double> p(n);
        for (auto& x : p) x = pr(rng);

        const auto r = odds_algorithm(p);
        // Brute force the best threshold.
        double best = 0;
        for (std::size_t s = 0; s < p.size(); ++s) best = std::max(best, win_prob_threshold(p, s));
        EXPECT_NEAR(r.win_probability, best, 1e-9) << "n=" << n;
        // The reported threshold must itself achieve the optimum.
        EXPECT_NEAR(win_prob_threshold(p, r.threshold - 1), best, 1e-9) << "n=" << n;
    }
}

TEST(OddsAlgorithm, SecretaryProblemApproachesOneOverE) {
    // p_i = 1/i reproduces the classic secretary problem: win prob -> 1/e.
    const int           n = 1000;
    std::vector<double> p(n);
    for (int i = 0; i < n; ++i) p[i] = 1.0 / (i + 1);
    const auto r = odds_algorithm(p);
    EXPECT_NEAR(r.win_probability, 1.0 / std::exp(1.0), 0.02);
    // Threshold near n/e.
    EXPECT_NEAR(static_cast<double>(r.threshold), n / std::exp(1.0), n * 0.05);
}

// ---------------- Subset sum ----------------

namespace {

// Brute-force existence + witness over all subsets (n small).
bool brute_subset_sum(const std::vector<long long>& nums, long long target) {
    const int n = static_cast<int>(nums.size());
    for (std::uint64_t m = 0; m < (std::uint64_t{1} << n); ++m) {
        long long s = 0;
        for (int j = 0; j < n; ++j)
            if (m & (std::uint64_t{1} << j)) s += nums[j];
        if (s == target) return true;
    }
    return false;
}

} // namespace

TEST(SubsetSum, MatchesBruteForceRandom) {
    std::mt19937_64                        rng(2);
    std::uniform_int_distribution<int>     len(0, 18);
    std::uniform_int_distribution<long long> val(-50, 50);
    for (int t = 0; t < 3000; ++t) {
        const int              n = len(rng);
        std::vector<long long> nums(n);
        for (auto& x : nums) x = val(rng);
        const long long target = val(rng) * 3;

        const auto r     = subset_sum(nums, target);
        const bool exists = brute_subset_sum(nums, target);
        EXPECT_EQ(r.found, exists) << "t=" << t;
        if (r.found) {
            long long s = 0;
            for (std::size_t idx : r.indices) s += nums[idx];
            EXPECT_EQ(s, target) << "witness wrong t=" << t;
        }
    }
}

TEST(SubsetSum, KnownCases) {
    // Target 0 is always achievable by the empty subset.
    EXPECT_TRUE(subset_sum({3, 1, 4, 1, 5}, 0).found);
    EXPECT_TRUE(subset_sum({}, 0).found);
    EXPECT_FALSE(subset_sum({}, 5).found);

    auto r = subset_sum({3, 34, 4, 12, 5, 2}, 9); // 4 + 5
    ASSERT_TRUE(r.found);
    long long s = 0;
    for (auto i : r.indices) s += std::vector<long long>{3, 34, 4, 12, 5, 2}[i];
    EXPECT_EQ(s, 9);

    EXPECT_FALSE(subset_sum({3, 34, 4, 12, 5, 2}, 100).found);
    // Negatives.
    EXPECT_TRUE(subset_sum({-7, -3, 2, 5, 8}, -1).found); // -3 + 2, or -7 -3 +2 +... etc.
}
