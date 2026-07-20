#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::algebra {

/// @brief Monomial orderings supported for multivariate polynomial leading-term selection.
///        Lex: pure lexicographic (variable 0 dominates). Grlex: total degree first, then
///        lexicographic to break ties. Grevlex: total degree first, then REVERSE
///        lexicographic (smallest trailing-variable exponent wins) to break ties -- the usual
///        default for Groebner basis computations since it tends to keep intermediate
///        polynomials smaller than Grlex.
enum class MonomialOrder { Lex, Grlex, Grevlex };

namespace detail {

[[nodiscard]] inline int total_degree(const std::vector<int>& exponents) {
    int sum = 0;
    for (int e : exponents) sum += e;
    return sum;
}

/// @brief Returns -1, 0, or 1 as monomial `a` is smaller, equal to, or larger than `b` under
///        `order`. `a` and `b` must have the same length.
[[nodiscard]] inline int compare_monomials(const std::vector<int>& a, const std::vector<int>& b, MonomialOrder order) {
    if (order == MonomialOrder::Lex) {
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
        return 0;
    }

    const int da = total_degree(a);
    const int db = total_degree(b);
    if (da != db) return da < db ? -1 : 1;

    if (order == MonomialOrder::Grlex) {
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
        return 0;
    }

    // Grevlex: compare from the last variable backward; a LARGER trailing exponent means a
    // SMALLER monomial under this tie-break.
    for (std::size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) return a[i] > b[i] ? -1 : 1;
    }
    return 0;
}

} // namespace detail

} // namespace datamunge::algebra
