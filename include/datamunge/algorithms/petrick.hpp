#pragma once

/// \file petrick.hpp
/// \brief Petrick's method: minimal prime-implicant cover from a coverage chart.
///
/// After Quine-McCluskey lists the prime implicants of a Boolean function, one must still
/// choose the *fewest* of them that together cover every minterm. Petrick's method solves
/// this exactly. For each minterm it writes a *sum* of the prime implicants covering it; the
/// product of these sums (a product-of-sums) is true exactly when a chosen set covers
/// everything. Multiplying it out into a sum-of-products (with absorption) enumerates all
/// *irredundant* covers, and the smallest product term is a minimum cover. This module
/// performs that expansion over prime-implicant indices.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace datamunge::algorithms {

/// Result of Petrick's method.
struct PetrickResult {
    std::vector<std::vector<int>> minimal_covers;  ///< All covers of minimum size (PI-index sets).
};

/// \brief Find the minimum prime-implicant covers by Petrick's method.
///
/// \param covers_of_minterm  covers_of_minterm[i] = prime-implicant indices covering minterm i.
/// \return every cover achieving the minimum number of prime implicants.
inline PetrickResult petrick(const std::vector<std::vector<int>>& covers_of_minterm) {
    // Each product term is a set of PI indices (their AND). Start with the "1" term {}.
    std::vector<std::set<int>> terms = {{}};

    for (const auto& clause : covers_of_minterm) {
        if (clause.empty()) continue;  // an uncoverable minterm contributes nothing
        std::vector<std::set<int>> next;
        for (const auto& term : terms)
            for (int pi : clause) {
                std::set<int> t = term;
                t.insert(pi);   // distribute: (term) AND (pi)
                next.push_back(std::move(t));
            }
        // Absorption: drop any term that is a superset of another (X + XY = X).
        std::sort(next.begin(), next.end(),
                  [](const std::set<int>& a, const std::set<int>& b) { return a.size() < b.size(); });
        std::vector<std::set<int>> reduced;
        for (const auto& t : next) {
            bool absorbed = false;
            for (const auto& r : reduced)
                if (std::includes(t.begin(), t.end(), r.begin(), r.end())) { absorbed = true; break; }
            if (!absorbed) reduced.push_back(t);
        }
        terms = std::move(reduced);
    }

    PetrickResult result;
    std::size_t   best = SIZE_MAX;
    for (const auto& t : terms) best = std::min(best, t.size());
    for (const auto& t : terms)
        if (t.size() == best) result.minimal_covers.emplace_back(t.begin(), t.end());
    return result;
}

}  // namespace datamunge::algorithms
