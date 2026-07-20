#pragma once

#include <datamunge/algebra/polynomial.hpp>

#include <utility>
#include <vector>

namespace datamunge::algebra {

/// @brief Descartes' rule of signs: the number of sign changes in p's coefficient sequence
///        (highest to lowest degree, skipping zero coefficients) -- an upper bound on the
///        number of positive real roots of p (counted with multiplicity); the true count
///        differs from this bound by a nonnegative even number.
[[nodiscard]] inline int descartes_sign_changes(const Polynomial& p) {
    int changes = 0;
    int prev_sign = 0;
    for (int i = p.degree(); i >= 0; --i) {
        const double c = p.coefficient(i);
        if (c == 0.0) continue;
        const int s = c > 0 ? 1 : -1;
        if (prev_sign != 0 && s != prev_sign) ++changes;
        prev_sign = s;
    }
    return changes;
}

/// @brief Upper bound on the number of NEGATIVE real roots of p, via Descartes' rule applied
///        to p(-x) (same even-gap caveat as descartes_sign_changes()).
[[nodiscard]] inline int descartes_negative_root_bound(const Polynomial& p) {
    std::vector<double> flipped = p.coefficients();
    for (std::size_t i = 1; i < flipped.size(); i += 2) flipped[i] = -flipped[i];
    return descartes_sign_changes(Polynomial(std::move(flipped)));
}

} // namespace datamunge::algebra
