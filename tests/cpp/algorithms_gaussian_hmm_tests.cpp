#include <datamunge/algorithms/gaussian_hmm.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace alg = datamunge::algorithms;

namespace {
// Two well-separated sticky states.
std::pair<std::vector<double>, std::vector<std::size_t>> simulate(std::size_t T, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    std::normal_distribution<double> z(0, 1);
    std::vector<double> x;
    std::vector<std::size_t> q;
    std::size_t s = 0;
    for (std::size_t t = 0; t < T; ++t) {
        if (t > 0 && u(rng) < (s == 0 ? 0.1 : 0.2)) s = 1 - s;
        q.push_back(s);
        x.push_back(s == 0 ? -1.0 + 0.5 * z(rng) : 2.0 + 0.8 * z(rng));
    }
    return {x, q};
}
}  // namespace

TEST(GaussianHmm, RecoversParametersAndIncreasesLikelihood) {
    const auto [x, q] = simulate(4000, 3);
    const auto fit = alg::baum_welch_gaussian(x, alg::gaussian_hmm_initial(x, 2));
    for (std::size_t k = 1; k < fit.log_likelihood_history.size(); ++k)
        EXPECT_GE(fit.log_likelihood_history[k], fit.log_likelihood_history[k - 1] - 1e-8);
    const auto& m = fit.model;
    const std::size_t lo = m.mean[0] < m.mean[1] ? 0 : 1, hi = 1 - lo;
    EXPECT_NEAR(m.mean[lo], -1.0, 0.05);
    EXPECT_NEAR(m.mean[hi], 2.0, 0.08);
    EXPECT_NEAR(m.sd[lo], 0.5, 0.05);
    EXPECT_NEAR(m.sd[hi], 0.8, 0.06);
    EXPECT_NEAR(m.transition[lo][hi], 0.1, 0.03);
    EXPECT_NEAR(m.transition[hi][lo], 0.2, 0.04);
    const auto path = alg::gaussian_hmm_viterbi(m, x);
    std::size_t agree = 0;
    for (std::size_t t = 0; t < x.size(); ++t) agree += (path[t] == hi) == (q[t] == 1);
    EXPECT_GT(agree, 0.97 * x.size());
    double ll = 0;
    const auto g = alg::gaussian_hmm_posteriors(m, x, &ll);
    EXPECT_NEAR(ll, fit.log_likelihood, 1e-9);
    EXPECT_NEAR(g[10][0] + g[10][1], 1.0, 1e-12);
    // The independent mixture fits worse on Markov data.
    const auto mix = alg::baum_welch_gaussian(x, alg::gaussian_hmm_initial(x, 2), {500, 1e-8, 1e-3, true});
    EXPECT_LT(mix.log_likelihood, fit.log_likelihood);
    EXPECT_NEAR(mix.model.transition[0][0], mix.model.transition[1][0], 1e-12);
    EXPECT_THROW(alg::gaussian_hmm_initial({}, 2), std::invalid_argument);
}

TEST(GaussianHmm, SampleHmmFollowsModel) {
    alg::HiddenMarkovModel h;
    h.initial    = {1.0, 0.0};
    h.transition = {{0.9, 0.1}, {0.5, 0.5}};
    h.emission   = {{1.0, 0.0}, {0.0, 1.0}};
    const auto s = alg::sample_hmm(h, 20000, 1);
    EXPECT_EQ(s.states.front(), 0u);
    std::size_t from0 = 0, stay0 = 0;
    for (std::size_t t = 0; t + 1 < s.states.size(); ++t)
        if (s.states[t] == 0) {
            ++from0;
            stay0 += s.states[t + 1] == 0;
        }
    EXPECT_NEAR(static_cast<double>(stay0) / from0, 0.9, 0.01);
    EXPECT_EQ(s.states, s.observations);
}
