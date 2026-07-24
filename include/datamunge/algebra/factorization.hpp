#pragma once

#include <datamunge/algebra/polynomial.hpp>
#include <datamunge/algebra/polynomial_roots.hpp>
#include <datamunge/algebra/square_free.hpp>

#include <cmath>
#include <vector>

namespace datamunge::algebra {

/// @brief One irreducible-over-R factor of a real factorization: `factor` is either linear
///        (x - r) for a real root r, or an irreducible real quadratic (x^2 + bx + c, for a
///        complex-conjugate root pair), appearing with `multiplicity` in the original
///        polynomial.
struct RealFactor {
    Polynomial factor;
    int multiplicity = 0;
};

/// @brief Factors p over the reals into irreducible linear/quadratic pieces plus the overall
///        scale needed to reconstruct p exactly:
///        p == leading_coefficient * prod(factors[i].factor ^ factors[i].multiplicity).
struct RealFactorization {
    std::vector<RealFactor> factors;
    double leading_coefficient = 1.0;

    [[nodiscard]] std::size_t num_factors() const { return factors.size(); }
    [[nodiscard]] const RealFactor& factor_at(std::size_t index) const { return factors.at(index); }
};

/// @brief Factors p over the reals: splits it into square-free pieces via
///        square_free_factorization() (each with its own multiplicity), then finds every root
///        of each piece via complex_roots() -- safe here since a square-free piece has, by
///        construction, only distinct roots, exactly the case complex_roots() converges best
///        on -- and pairs up complex-conjugate roots into real quadratic factors, leaving real
///        roots as linear factors. This is the numeric analog of "factor()" in an exact CAS:
///        for a floating-point polynomial with no assumed rational structure, linear and
///        quadratic real factors are as far as factoring over R can go (see rational_roots()
///        for an exact search restricted to genuinely rational roots instead).
/// @param conjugate_tolerance Two roots are treated as a real root (imaginary part ~ 0) or a
///        conjugate pair (matching real parts, opposite imaginary parts) when within this
///        tolerance of the corresponding exact relationship.
[[nodiscard]] inline RealFactorization factor_over_reals(const Polynomial& p, double conjugate_tolerance = 1e-6) {
    RealFactorization result;
    if (p.is_zero() || p.degree() <= 0) {
        result.leading_coefficient = p.is_zero() ? 0.0 : p.coefficient(0);
        return result;
    }
    result.leading_coefficient = p.coefficient(p.degree());

    for (const auto& sf : square_free_factorization(p)) {
        auto roots = complex_roots(sf.factor);
        std::vector<bool> used(roots.size(), false);

        for (std::size_t i = 0; i < roots.size(); ++i) {
            if (used[i]) continue;
            used[i] = true;

            if (std::fabs(roots[i].im) <= conjugate_tolerance) {
                result.factors.push_back({Polynomial({-roots[i].re, 1.0}), sf.multiplicity});
                continue;
            }

            bool paired = false;
            for (std::size_t j = i + 1; j < roots.size(); ++j) {
                if (used[j]) continue;
                if (std::fabs(roots[j].re - roots[i].re) <= conjugate_tolerance &&
                    std::fabs(roots[j].im + roots[i].im) <= conjugate_tolerance) {
                    used[j] = true;
                    const double b = -2.0 * roots[i].re;
                    const double c = roots[i].re * roots[i].re + roots[i].im * roots[i].im;
                    result.factors.push_back({Polynomial({c, b, 1.0}), sf.multiplicity});
                    paired = true;
                    break;
                }
            }
            // A real-coefficient polynomial's non-real roots always come in conjugate pairs;
            // an unpaired one here only happens from numerical noise right at the tolerance
            // boundary, so fall back to treating it as (near-)real rather than dropping it.
            if (!paired) result.factors.push_back({Polynomial({-roots[i].re, 1.0}), sf.multiplicity});
        }
    }

    return result;
}

} // namespace datamunge::algebra
