#pragma once

#include <datamunge/algebra/poly_gcd.hpp>
#include <datamunge/algebra/polynomial.hpp>

#include <vector>

namespace datamunge::algebra {

/// @brief One factor of a square-free decomposition: `factor` (itself square-free, i.e. no
///        repeated roots) appears with multiplicity `multiplicity` in the original polynomial.
struct SquareFreeFactor {
    Polynomial factor;
    int multiplicity = 0;
};

/// @brief Yun's algorithm: factors p (char-0 coefficients, doubles here) into square-free
///        pieces, each paired with its multiplicity, such that p equals the product of
///        `factor_i ^ multiplicity_i` up to an overall constant. Each returned `factor` is
///        itself square-free (no repeated roots) and the `multiplicity`-i factors are pairwise
///        coprime.
[[nodiscard]] inline std::vector<SquareFreeFactor> square_free_factorization(const Polynomial& p) {
    std::vector<SquareFreeFactor> factors;
    if (p.degree() <= 0) return factors;

    Polynomial a = p;
    Polynomial b = p.derivative();
    Polynomial c = poly_gcd(a, b);
    Polynomial w = a.divmod(c).first;
    Polynomial y = b.divmod(c).first;

    int i = 1;
    while (w.degree() > 0) {
        Polynomial z = y.subtract(w.derivative());
        Polynomial g = poly_gcd(w, z);
        if (g.degree() > 0) factors.push_back({monic(g), i});
        w = w.divmod(g).first;
        y = z.divmod(g).first;
        ++i;
    }
    return factors;
}

} // namespace datamunge::algebra
