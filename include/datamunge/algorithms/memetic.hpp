#pragma once

// Memetic algorithm: a genetic algorithm hybridized with local search -- the
// "genes" evolve by selection, crossover, and mutation as usual, but every new
// individual is first refined to a local optimum before competing. The name
// (Dawkins's "meme") captures the idea that each individual also *learns* within
// its lifetime, not just inherits. The population-level exploration of the GA
// combines with the exploitation of local search, so memetic algorithms typically
// reach better solutions in fewer generations than a plain GA on hard
// combinatorial problems. Here it is applied to MAX-CUT.

#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

struct MemeticMaxCutResult {
    double           value{0};
    std::vector<int> side; // 0/1 per vertex
};

namespace detail {
struct MemRng {
    std::uint64_t s;
    explicit MemRng(std::uint64_t seed) : s(seed ? seed : 0x2545F4914F6CDD1DULL) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double        uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    std::size_t   below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};
} // namespace detail

// Maximize the cut of an n-vertex weighted graph (symmetric adjacency list of
// (neighbor, weight)) by a memetic algorithm.
inline MemeticMaxCutResult memetic_max_cut(int n,
                                           const std::vector<std::vector<std::pair<int, double>>>& adj,
                                           int pop_size = 30, int generations = 40,
                                           std::uint64_t seed = 1) {
    detail::MemRng rng(seed);

    auto cut_value = [&](const std::vector<int>& s) {
        double v = 0;
        for (int u = 0; u < n; ++u)
            for (const auto& [w, wt] : adj[u])
                if (u < w && s[u] != s[w]) v += wt;
        return v;
    };
    // Local search: flip any vertex that increases the cut, to a local optimum.
    auto local_search = [&](std::vector<int>& s) {
        bool improved = true;
        while (improved) {
            improved = false;
            for (int u = 0; u < n; ++u) {
                double delta = 0;
                for (const auto& [w, wt] : adj[u]) delta += (s[w] != s[u]) ? -wt : wt;
                if (delta > 1e-12) { s[u] ^= 1; improved = true; }
            }
        }
    };

    // Initialize a locally-optimized population.
    std::vector<std::vector<int>> pop(pop_size, std::vector<int>(n));
    std::vector<double>           fit(pop_size);
    for (int p = 0; p < pop_size; ++p) {
        for (int u = 0; u < n; ++u) pop[p][u] = static_cast<int>(rng.next() & 1);
        local_search(pop[p]);
        fit[p] = cut_value(pop[p]);
    }
    auto best_index = [&] { int b = 0; for (int p = 1; p < pop_size; ++p) if (fit[p] > fit[b]) b = p; return b; };

    for (int gen = 0; gen < generations; ++gen) {
        std::vector<std::vector<int>> next;
        std::vector<double>           nfit;
        next.reserve(pop_size);
        nfit.reserve(pop_size);
        // Elitism: carry the best forward.
        const int elite = best_index();
        next.push_back(pop[elite]);
        nfit.push_back(fit[elite]);

        while (static_cast<int>(next.size()) < pop_size) {
            // Tournament selection (size 3) of two parents.
            auto tourney = [&] { std::size_t b = rng.below(pop_size); for (int t = 0; t < 2; ++t) { std::size_t c = rng.below(pop_size); if (fit[c] > fit[b]) b = c; } return b; };
            const auto& A = pop[tourney()];
            const auto& B = pop[tourney()];
            // Uniform crossover.
            std::vector<int> child(n);
            for (int u = 0; u < n; ++u) child[u] = (rng.next() & 1) ? A[u] : B[u];
            // Mutation: flip each bit with probability 1/n.
            for (int u = 0; u < n; ++u) if (rng.uniform() < 1.0 / n) child[u] ^= 1;
            // Local refinement (the "memetic" step).
            local_search(child);
            next.push_back(child);
            nfit.push_back(cut_value(child));
        }
        pop.swap(next);
        fit.swap(nfit);
    }

    const int b = best_index();
    return {fit[b], pop[b]};
}

} // namespace datamunge::algorithms
