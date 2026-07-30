#pragma once

/// \file quine_mccluskey.hpp
/// \brief The Quine-McCluskey algorithm for Boolean function minimization.
///
/// Quine-McCluskey is the tabular, mechanizable cousin of the Karnaugh map: it minimizes a
/// Boolean function given as a set of *minterms* (and optional *don't-cares*). It first finds
/// all *prime implicants* by repeatedly merging pairs of terms that differ in a single bit
/// (replacing that bit with a dash), then selects a minimum subset of prime implicants that
/// covers every minterm -- the *essential* prime implicants plus, for the remainder, a minimum
/// cover found by Petrick's method. The result is a minimum sum-of-products expression. This
/// module returns the prime implicants, the essential ones, and a minimal cover.

#include <datamunge/algorithms/petrick.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// A product term (implicant): fixed bits plus dashed (don't-care) positions.
struct Implicant {
    std::uint32_t bits;  ///< Value on the non-dashed positions.
    std::uint32_t mask;  ///< 1 = this position is a dash (don't care).

    bool operator==(const Implicant& o) const { return bits == o.bits && mask == o.mask; }
    /// Does this implicant cover minterm \c m?
    bool covers(std::uint32_t m) const { return ((m ^ bits) & ~mask) == 0; }
};

/// Result of Quine-McCluskey minimization.
struct QuineMcCluskeyResult {
    std::vector<Implicant> prime_implicants;  ///< All prime implicants.
    std::vector<Implicant> essential;         ///< Essential prime implicants.
    std::vector<Implicant> minimal_cover;     ///< A minimum sum-of-products cover.
};

namespace detail {
inline int popcount32(std::uint32_t x) {
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}
}  // namespace detail

/// \brief Render an implicant as a variable string over \c n variables (e.g. "1-0-").
inline std::string implicant_to_string(const Implicant& imp, int n) {
    std::string s;
    for (int i = n - 1; i >= 0; --i) {
        if (imp.mask & (1u << i)) s += '-';
        else                     s += ((imp.bits >> i) & 1) ? '1' : '0';
    }
    return s;
}

/// \brief Minimize a Boolean function via Quine-McCluskey (+ Petrick for the covering step).
///
/// \param n          number of Boolean variables.
/// \param minterms   inputs where the function is 1.
/// \param dontcares  inputs where the output is unspecified (usable in implicants, but need
///                   not be covered).
inline QuineMcCluskeyResult quine_mccluskey(int n, std::vector<std::uint32_t> minterms,
                                            std::vector<std::uint32_t> dontcares = {}) {
    QuineMcCluskeyResult result;

    // --- Prime-implicant generation by iterated single-bit merging. ---
    std::vector<Implicant> current;
    for (std::uint32_t m : minterms)  current.push_back({m, 0});
    for (std::uint32_t m : dontcares) current.push_back({m, 0});
    // Dedup.
    std::sort(current.begin(), current.end(),
              [](const Implicant& a, const Implicant& b) { return a.bits < b.bits; });
    current.erase(std::unique(current.begin(), current.end()), current.end());

    std::vector<Implicant> primes;
    while (!current.empty()) {
        std::vector<char>      used(current.size(), 0);
        std::vector<Implicant> next;
        for (std::size_t i = 0; i < current.size(); ++i)
            for (std::size_t j = i + 1; j < current.size(); ++j) {
                if (current[i].mask != current[j].mask) continue;      // same dash pattern
                std::uint32_t diff = current[i].bits ^ current[j].bits;
                if (diff && (diff & (diff - 1)) == 0) {                // differ in exactly one bit
                    used[i] = used[j] = 1;
                    next.push_back({current[i].bits & ~diff, current[i].mask | diff});
                }
            }
        for (std::size_t i = 0; i < current.size(); ++i)
            if (!used[i]) primes.push_back(current[i]);                // unmerged -> prime
        std::sort(next.begin(), next.end(), [](const Implicant& a, const Implicant& b) {
            return a.bits != b.bits ? a.bits < b.bits : a.mask < b.mask;
        });
        next.erase(std::unique(next.begin(), next.end()), next.end());
        current = std::move(next);
    }
    // Dedup primes.
    std::sort(primes.begin(), primes.end(), [](const Implicant& a, const Implicant& b) {
        return a.bits != b.bits ? a.bits < b.bits : a.mask < b.mask;
    });
    primes.erase(std::unique(primes.begin(), primes.end()), primes.end());
    result.prime_implicants = primes;

    // --- Coverage chart over the minterms (don't-cares need not be covered). ---
    std::vector<std::vector<int>> covers_of_minterm(minterms.size());
    for (std::size_t mi = 0; mi < minterms.size(); ++mi)
        for (std::size_t pi = 0; pi < primes.size(); ++pi)
            if (primes[pi].covers(minterms[mi])) covers_of_minterm[mi].push_back(static_cast<int>(pi));

    // --- Essential prime implicants: the sole cover of some minterm. ---
    std::vector<char> is_essential(primes.size(), 0);
    for (const auto& cov : covers_of_minterm)
        if (cov.size() == 1) is_essential[cov[0]] = 1;
    for (std::size_t pi = 0; pi < primes.size(); ++pi)
        if (is_essential[pi]) result.essential.push_back(primes[pi]);

    // --- Minimal cover: essentials, plus Petrick over the still-uncovered minterms. ---
    std::vector<char> covered(minterms.size(), 0);
    for (std::size_t mi = 0; mi < minterms.size(); ++mi)
        for (int pi : covers_of_minterm[mi])
            if (is_essential[pi]) { covered[mi] = 1; break; }

    std::vector<std::vector<int>> remaining;
    for (std::size_t mi = 0; mi < minterms.size(); ++mi)
        if (!covered[mi]) remaining.push_back(covers_of_minterm[mi]);

    std::vector<int> chosen;
    for (std::size_t pi = 0; pi < primes.size(); ++pi)
        if (is_essential[pi]) chosen.push_back(static_cast<int>(pi));

    if (!remaining.empty()) {
        PetrickResult pr = petrick(remaining);
        if (!pr.minimal_covers.empty()) {
            // Prefer the extra cover with the fewest total literals.
            const std::vector<int>* best = &pr.minimal_covers.front();
            int best_lits = INT32_MAX;
            for (const auto& cover : pr.minimal_covers) {
                int lits = 0;
                for (int pi : cover) lits += n - detail::popcount32(primes[pi].mask);
                if (lits < best_lits) { best_lits = lits; best = &cover; }
            }
            for (int pi : *best) chosen.push_back(pi);
        }
    }

    std::sort(chosen.begin(), chosen.end());
    chosen.erase(std::unique(chosen.begin(), chosen.end()), chosen.end());
    for (int pi : chosen) result.minimal_cover.push_back(primes[pi]);
    return result;
}

}  // namespace datamunge::algorithms
