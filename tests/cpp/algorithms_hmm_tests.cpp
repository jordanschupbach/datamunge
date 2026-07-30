#include <gtest/gtest.h>

#include <datamunge/algorithms/hmm.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using datamunge::algorithms::baum_welch;
using datamunge::algorithms::forward_backward;
using datamunge::algorithms::HiddenMarkovModel;
using datamunge::algorithms::viterbi;

namespace {

// The canonical textbook example (Rabiner / Wikipedia "Viterbi algorithm"):
// two health states {Healthy=0, Fever=1}, three symptoms {normal=0, cold=1, dizzy=2}.
HiddenMarkovModel health_model() {
    HiddenMarkovModel h;
    h.initial    = {0.6, 0.4};
    h.transition = {{0.7, 0.3}, {0.4, 0.6}};
    h.emission   = {{0.5, 0.4, 0.1}, {0.1, 0.3, 0.6}};
    return h;
}

// Brute-force P(O | lambda) by summing the joint over every one of N^T state paths.
double brute_force_likelihood(const HiddenMarkovModel& h, const std::vector<std::size_t>& obs) {
    const std::size_t n = h.num_states();
    const std::size_t t = obs.size();
    std::vector<std::size_t> path(t, 0);
    double                   total = 0.0;
    const std::size_t        combos = static_cast<std::size_t>(std::pow(static_cast<double>(n), static_cast<double>(t)));
    for (std::size_t code = 0; code < combos; ++code) {
        std::size_t c = code;
        for (std::size_t i = 0; i < t; ++i) {
            path[i] = c % n;
            c /= n;
        }
        double p = h.initial[path[0]] * h.emission[path[0]][obs[0]];
        for (std::size_t i = 1; i < t; ++i) p *= h.transition[path[i - 1]][path[i]] * h.emission[path[i]][obs[i]];
        total += p;
    }
    return total;
}

// Brute-force the single most likely path and its joint log-probability.
std::pair<std::vector<std::size_t>, double> brute_force_viterbi(const HiddenMarkovModel&        h,
                                                                const std::vector<std::size_t>& obs) {
    const std::size_t        n = h.num_states();
    const std::size_t        t = obs.size();
    std::vector<std::size_t> path(t, 0), best_path(t, 0);
    double                   best = -1.0;
    const std::size_t        combos = static_cast<std::size_t>(std::pow(static_cast<double>(n), static_cast<double>(t)));
    for (std::size_t code = 0; code < combos; ++code) {
        std::size_t c = code;
        for (std::size_t i = 0; i < t; ++i) {
            path[i] = c % n;
            c /= n;
        }
        double p = h.initial[path[0]] * h.emission[path[0]][obs[0]];
        for (std::size_t i = 1; i < t; ++i) p *= h.transition[path[i - 1]][path[i]] * h.emission[path[i]][obs[i]];
        if (p > best) {
            best      = p;
            best_path = path;
        }
    }
    return {best_path, std::log(best)};
}

}  // namespace

TEST(HmmViterbi, KnownHealthExample) {
    const HiddenMarkovModel        h   = health_model();
    const std::vector<std::size_t> obs = {0, 1, 2};  // normal, cold, dizzy
    const auto                     r   = viterbi(h, obs);
    // The well-known answer is Healthy, Healthy, Fever.
    ASSERT_EQ(r.states.size(), 3u);
    EXPECT_EQ(r.states[0], 0u);
    EXPECT_EQ(r.states[1], 0u);
    EXPECT_EQ(r.states[2], 1u);
}

TEST(HmmViterbi, MatchesBruteForceOnRandomModels) {
    std::mt19937_64                        rng(12345);
    std::uniform_real_distribution<double> u(0.05, 1.0);
    for (int trial = 0; trial < 40; ++trial) {
        HiddenMarkovModel h;
        const std::size_t n = 3, m = 3;
        h.initial.resize(n);
        h.transition.assign(n, std::vector<double>(n));
        h.emission.assign(n, std::vector<double>(m));
        double s = 0.0;
        for (auto& x : h.initial) x = u(rng), s += x;
        for (auto& x : h.initial) x /= s;
        for (std::size_t i = 0; i < n; ++i) {
            double rs = 0.0;
            for (auto& x : h.transition[i]) x = u(rng), rs += x;
            for (auto& x : h.transition[i]) x /= rs;
            double es = 0.0;
            for (auto& x : h.emission[i]) x = u(rng), es += x;
            for (auto& x : h.emission[i]) x /= es;
        }
        std::vector<std::size_t> obs(5);
        std::uniform_int_distribution<std::size_t> sym(0, m - 1);
        for (auto& o : obs) o = sym(rng);

        const auto got  = viterbi(h, obs);
        const auto want = brute_force_viterbi(h, obs);
        EXPECT_NEAR(got.log_probability, want.second, 1e-9);
        // The recovered path must achieve the same joint probability (there may be ties).
        EXPECT_EQ(got.states, want.first);
    }
}

TEST(HmmForwardBackward, LikelihoodMatchesBruteForce) {
    const HiddenMarkovModel        h   = health_model();
    const std::vector<std::size_t> obs = {0, 1, 2, 2, 1, 0};
    const auto                     fb  = forward_backward(h, obs);
    EXPECT_NEAR(std::exp(fb.log_likelihood), brute_force_likelihood(h, obs), 1e-9);
}

TEST(HmmForwardBackward, PosteriorsAreProperDistributions) {
    const HiddenMarkovModel        h   = health_model();
    const std::vector<std::size_t> obs = {0, 1, 2, 0, 2};
    const auto                     fb  = forward_backward(h, obs);
    for (const auto& row : fb.gamma) {
        double s = 0.0;
        for (double g : row) {
            EXPECT_GE(g, -1e-12);
            EXPECT_LE(g, 1.0 + 1e-12);
            s += g;
        }
        EXPECT_NEAR(s, 1.0, 1e-9);
    }
}

TEST(HmmBaumWelch, LogLikelihoodIsMonotoneNonDecreasing) {
    // Generate sequences from a known model, then fit from a perturbed start.
    const HiddenMarkovModel truth = health_model();
    std::mt19937_64         rng(7);
    auto                    sample_seq = [&](std::size_t len) {
        std::vector<std::size_t>               obs(len);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::size_t                            state = (u(rng) < truth.initial[0]) ? 0 : 1;
        for (std::size_t t = 0; t < len; ++t) {
            double        r = u(rng), c = 0.0;
            std::size_t   sym = 0;
            for (std::size_t k = 0; k < truth.num_symbols(); ++k) {
                c += truth.emission[state][k];
                if (r <= c) {
                    sym = k;
                    break;
                }
            }
            obs[t]      = sym;
            double  rt  = u(rng), ct = 0.0;
            std::size_t nxt = 0;
            for (std::size_t j = 0; j < truth.num_states(); ++j) {
                ct += truth.transition[state][j];
                if (rt <= ct) {
                    nxt = j;
                    break;
                }
            }
            state = nxt;
        }
        return obs;
    };
    std::vector<std::vector<std::size_t>> seqs;
    for (int i = 0; i < 20; ++i) seqs.push_back(sample_seq(15));

    HiddenMarkovModel init;
    init.initial    = {0.5, 0.5};
    init.transition = {{0.6, 0.4}, {0.5, 0.5}};
    init.emission   = {{0.4, 0.4, 0.2}, {0.2, 0.3, 0.5}};

    const auto out = baum_welch(seqs, init, 50, 1e-9);
    ASSERT_GE(out.log_likelihood_history.size(), 2u);
    for (std::size_t i = 1; i < out.log_likelihood_history.size(); ++i)
        EXPECT_GE(out.log_likelihood_history[i], out.log_likelihood_history[i - 1] - 1e-6);

    // Rows of the learned parameters remain valid probability distributions.
    for (const auto& row : out.model.transition) {
        double s = 0.0;
        for (double v : row) s += v;
        EXPECT_NEAR(s, 1.0, 1e-6);
    }
    for (const auto& row : out.model.emission) {
        double s = 0.0;
        for (double v : row) s += v;
        EXPECT_NEAR(s, 1.0, 1e-6);
    }
}

TEST(Hmm, EmptyObservationSequence) {
    const HiddenMarkovModel h = health_model();
    const auto              r = viterbi(h, {});
    EXPECT_TRUE(r.states.empty());
    const auto fb = forward_backward(h, {});
    EXPECT_EQ(fb.log_likelihood, 0.0);
}
