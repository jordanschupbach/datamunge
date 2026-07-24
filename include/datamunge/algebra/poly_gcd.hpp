#pragma once

#include <datamunge/algebra/polynomial.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace datamunge::algebra {

/// @brief Monic-normalizes p by dividing through by its leading coefficient (no-op on the zero
///        polynomial).
[[nodiscard]] inline Polynomial monic(const Polynomial& p) {
    if (p.is_zero()) return p;
    return p.scale(1.0 / p.coefficient(p.degree()));
}

namespace detail {

/// @brief True when every coefficient of `r` is negligible relative to `scale_reference`'s
///        largest-magnitude coefficient. The classical Euclidean algorithm's termination test
///        (`remainder.is_zero()`) assumes a remainder that is mathematically zero comes back
///        as *exactly* zero -- true over an exact field, but not over doubles: a genuine
///        (e.g. repeated) common factor can leave a remainder like 1.78e-15 instead of 0.0,
///        which Polynomial::is_zero()'s exact-equality check treats as a nonzero polynomial,
///        making the Euclidean algorithm take one more spurious step and converge on a
///        meaningless near-zero-constant "GCD" instead of the true, higher-degree one. Treating
///        such a remainder as zero (this function) fixes that without weakening
///        Polynomial::is_zero() itself, which is used elsewhere for genuine exact-zero checks
///        (e.g. rejecting division by the zero polynomial).
[[nodiscard]] inline bool is_negligible_remainder(const Polynomial& r, const Polynomial& scale_reference,
                                                   double relative_tolerance = 1e-9) {
    if (r.is_zero()) return true;
    double scale = 0.0;
    for (double c : scale_reference.coefficients()) scale = std::max(scale, std::fabs(c));
    if (scale == 0.0) return true;
    double max_r = 0.0;
    for (double c : r.coefficients()) max_r = std::max(max_r, std::fabs(c));
    return max_r <= relative_tolerance * scale;
}

} // namespace detail

/// @brief GCD of two polynomials via the classical Euclidean algorithm (repeated
///        divmod-and-swap), returned monic.
[[nodiscard]] inline Polynomial poly_gcd(Polynomial a, Polynomial b) {
    while (!b.is_zero()) {
        auto [q, r] = a.divmod(b);
        (void)q;
        a = b;
        b = detail::is_negligible_remainder(r, b) ? Polynomial(0.0) : r;
    }
    return monic(a);
}

/// @brief Result of the extended Euclidean algorithm on two polynomials: `gcd`, plus the
///        Bezout coefficients `s`/`t` such that `s.multiply(a).add(t.multiply(b)) == gcd`.
struct PolyExtendedGcdResult {
    Polynomial gcd;
    Polynomial s;
    Polynomial t;
};

/// @brief Extended Euclidean algorithm: computes gcd(a, b) along with Bezout coefficients s, t
///        satisfying `s*a + t*b == gcd`. `gcd` is returned monic (s and t are scaled to match).
[[nodiscard]] inline PolyExtendedGcdResult poly_extended_gcd(const Polynomial& a, const Polynomial& b) {
    Polynomial old_r = a, r = b;
    Polynomial old_s(1.0), s(0.0);
    Polynomial old_t(0.0), t(1.0);

    while (!r.is_zero()) {
        auto [q, rem] = old_r.divmod(r);
        if (detail::is_negligible_remainder(rem, r)) rem = Polynomial(0.0);

        Polynomial new_r = rem;
        old_r = r;
        r = new_r;

        Polynomial new_s = old_s.subtract(q.multiply(s));
        old_s = s;
        s = new_s;

        Polynomial new_t = old_t.subtract(q.multiply(t));
        old_t = t;
        t = new_t;
    }

    if (!old_r.is_zero()) {
        const double lead = old_r.coefficient(old_r.degree());
        old_r = old_r.scale(1.0 / lead);
        old_s = old_s.scale(1.0 / lead);
        old_t = old_t.scale(1.0 / lead);
    }
    return {old_r, old_s, old_t};
}

/// @brief GCD via a pseudo-remainder sequence: like poly_gcd(), but each step scales the
///        dividend by lc(divisor)^(degree gap + 1) before dividing (pseudo-division), the
///        classical technique subresultant-PRS algorithms are built on. This is the basic
///        (unscaled) variant -- it avoids the fraction-heavy divisions plain Euclidean division
///        would need over an exact (e.g. integer) coefficient field, but does not add the
///        further subresultant coefficient-growth control a full implementation would. Over
///        doubles it is numerically a different (and typically larger-magnitude-intermediate)
///        path to the same monic result as poly_gcd().
[[nodiscard]] inline Polynomial poly_gcd_pseudo_remainder_sequence(Polynomial a, Polynomial b) {
    if (b.is_zero()) return monic(a);
    while (!b.is_zero()) {
        if (a.degree() < b.degree()) std::swap(a, b);
        if (b.is_zero()) break;
        const int delta = a.degree() - b.degree();
        const double lc = b.coefficient(b.degree());
        Polynomial scaled = a.scale(std::pow(lc, delta + 1));
        auto [q, rem] = scaled.divmod(b);
        (void)q;
        a = b;
        b = detail::is_negligible_remainder(rem, b) ? Polynomial(0.0) : rem;
    }
    return monic(a);
}

} // namespace datamunge::algebra
