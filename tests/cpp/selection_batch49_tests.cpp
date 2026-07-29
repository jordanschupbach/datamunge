#include <gtest/gtest.h>

#include <datamunge/algorithms/selection_operators.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Fitness-proportionate (roulette-wheel) ----------------

TEST(RouletteWheel, FrequencyMatchesFitnessShare) {
    const std::vector<double> fit{1, 2, 3, 4};
    const double              total = 10;
    const int                 N = 400000;
    const auto                pick = roulette_wheel_sample(fit, N, 1);
    std::vector<int>          count(4, 0);
    for (auto i : pick) ++count[i];
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(static_cast<double>(count[i]) / N, fit[i] / total, 0.01) << "i=" << i;
}

TEST(RouletteWheel, NeverPicksZeroFitness) {
    const std::vector<double> fit{0, 5, 0, 3, 0};
    const auto                pick = roulette_wheel_sample(fit, 20000, 7);
    for (auto i : pick) EXPECT_GT(fit[i], 0.0);
}

// ---------------- Stochastic universal sampling ----------------

TEST(StochasticUniversalSampling, CountsWithinOneOfExpected) {
    const std::vector<double> fit{1, 2, 3, 4, 5};
    double                    total = 0; for (double f : fit) total += f;
    const int                 N = 1000;
    const auto                pick = stochastic_universal_sampling(fit, N, 3);
    ASSERT_EQ(static_cast<int>(pick.size()), N);
    std::vector<int> count(fit.size(), 0);
    for (auto i : pick) ++count[i];
    for (std::size_t i = 0; i < fit.size(); ++i) {
        const double expected = N * fit[i] / total;
        EXPECT_LE(std::fabs(count[i] - expected), 1.0 + 1e-9) << "i=" << i; // SUS guarantee
    }
}

// ---------------- Tournament selection ----------------

TEST(TournamentSelection, MonotoneInFitnessAndPressureGrowsWithK) {
    const std::vector<double> fit{1, 2, 3, 4};
    const int                 N = 400000;

    const auto pick3 = tournament_sample(fit, N, 3, 1);
    std::vector<int> c3(4, 0);
    for (auto i : pick3) ++c3[i];
    // frequency strictly increases with fitness
    EXPECT_LT(c3[0], c3[1]);
    EXPECT_LT(c3[1], c3[2]);
    EXPECT_LT(c3[2], c3[3]);

    // larger tournament -> more pressure: the best is chosen even more often
    const auto pick2 = tournament_sample(fit, N, 2, 1);
    std::vector<int> c2(4, 0);
    for (auto i : pick2) ++c2[i];
    EXPECT_GT(c3[3], c2[3]); // k=3 selects the fittest more than k=2
}
