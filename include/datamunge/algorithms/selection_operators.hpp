#pragma once

// The classic selection operators of evolutionary computation: given the fitness
// of each individual in a population, choose which ones "reproduce". Selection is
// what turns random variation into directed search -- it sets the *selection
// pressure*, the bias toward fitter individuals. Three standard schemes:
//
//   * Fitness-proportionate (roulette-wheel) selection -- probability of choosing
//     an individual is its fitness divided by the total, as if spinning a roulette
//     wheel whose slot sizes are the fitnesses.
//   * Stochastic universal sampling (SUS) -- one spin with N equally-spaced
//     pointers picks all N parents at once, giving the same expected counts as
//     roulette but far less variance (no individual is over- or under-picked by
//     more than one).
//   * Tournament selection -- repeatedly draw k random individuals and keep the
//     fittest; the tournament size k tunes the selection pressure directly.
//
// All require non-negative fitnesses.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
struct SelRng {
    std::uint64_t s;
    explicit SelRng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double        uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    std::size_t   below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};

// Index of the first cumulative-sum bucket strictly exceeding `target`.
inline std::size_t cumulative_pick(const std::vector<double>& cum, double target) {
    std::size_t lo = 0, hi = cum.size() - 1;
    while (lo < hi) {
        const std::size_t mid = (lo + hi) / 2;
        if (cum[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
} // namespace detail

// Draw `n` indices by fitness-proportionate (roulette-wheel) selection.
inline std::vector<std::size_t> roulette_wheel_sample(const std::vector<double>& fitness, int n,
                                                      std::uint64_t seed = 1) {
    std::vector<double> cum(fitness.size());
    double              total = 0;
    for (std::size_t i = 0; i < fitness.size(); ++i) { total += fitness[i]; cum[i] = total; }
    detail::SelRng           rng(seed);
    std::vector<std::size_t> out;
    out.reserve(n);
    for (int k = 0; k < n; ++k) out.push_back(detail::cumulative_pick(cum, rng.uniform() * total));
    return out;
}

// Draw `n` indices by stochastic universal sampling: one random offset, `n`
// equally-spaced pointers. Same expected counts as roulette, lower variance.
inline std::vector<std::size_t> stochastic_universal_sampling(const std::vector<double>& fitness, int n,
                                                              std::uint64_t seed = 1) {
    std::vector<double> cum(fitness.size());
    double              total = 0;
    for (std::size_t i = 0; i < fitness.size(); ++i) { total += fitness[i]; cum[i] = total; }
    detail::SelRng           rng(seed);
    const double             step  = total / n;
    double                   ptr   = rng.uniform() * step;
    std::vector<std::size_t> out;
    out.reserve(n);
    std::size_t idx = 0;
    for (int k = 0; k < n; ++k) {
        while (idx < cum.size() - 1 && cum[idx] <= ptr) ++idx;
        out.push_back(idx);
        ptr += step;
    }
    return out;
}

// Draw `n` indices by tournament selection: each pick is the fittest of `k`
// random contestants.
inline std::vector<std::size_t> tournament_sample(const std::vector<double>& fitness, int n, int k,
                                                  std::uint64_t seed = 1) {
    detail::SelRng           rng(seed);
    std::vector<std::size_t> out;
    out.reserve(n);
    const std::size_t m = fitness.size();
    for (int s = 0; s < n; ++s) {
        std::size_t best = rng.below(m);
        for (int c = 1; c < k; ++c) {
            const std::size_t cand = rng.below(m);
            if (fitness[cand] > fitness[best]) best = cand;
        }
        out.push_back(best);
    }
    return out;
}

} // namespace datamunge::algorithms
