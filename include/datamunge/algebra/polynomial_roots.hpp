#pragma once

#include <datamunge/algebra/poly_gcd.hpp>
#include <datamunge/algebra/polynomial.hpp>
#include <datamunge/algebra/sturm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace datamunge::algebra {

/// @brief A complex number, used to report polynomial roots that are not real. Plain
///        re/im fields (no arithmetic operators exposed) matching Polynomial's convention of
///        a member `operator==` over a free one (SWIG's Python backend maps a free
///        `operator==` to a single module-level `__eq__` that collides across every type that
///        defines one).
struct Complex {
    double re{0.0};
    double im{0.0};

    [[nodiscard]] double modulus() const { return std::sqrt(re * re + im * im); }
    [[nodiscard]] bool operator==(const Complex& other) const { return re == other.re && im == other.im; }
    [[nodiscard]] bool operator!=(const Complex& other) const { return !(*this == other); }
};

namespace detail {

// Minimal complex arithmetic for the Durand-Kerner iteration below -- a separate, non-exposed
// type from Complex so no arithmetic operators need to survive into any language binding.
struct Cplx {
    double re{0.0};
    double im{0.0};
};

[[nodiscard]] inline Cplx c_add(Cplx a, Cplx b) { return {a.re + b.re, a.im + b.im}; }
[[nodiscard]] inline Cplx c_sub(Cplx a, Cplx b) { return {a.re - b.re, a.im - b.im}; }
[[nodiscard]] inline Cplx c_mul(Cplx a, Cplx b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
[[nodiscard]] inline Cplx c_div(Cplx a, Cplx b) {
    const double denom = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / denom, (a.im * b.re - a.re * b.im) / denom};
}
[[nodiscard]] inline double c_abs(Cplx a) { return std::sqrt(a.re * a.re + a.im * a.im); }

[[nodiscard]] inline Cplx c_eval(const Polynomial& p, Cplx x) {
    Cplx result{0.0, 0.0};
    for (auto it = p.coefficients().rbegin(); it != p.coefficients().rend(); ++it) {
        result = c_add(c_mul(result, x), Cplx{*it, 0.0});
    }
    return result;
}

} // namespace detail

/// @brief All n = p.degree() complex roots of p (real roots included, with a negligible
///        imaginary part), via the Durand-Kerner (Weierstrass) method: simultaneous
///        fixed-point iteration z_k <- z_k - p(z_k) / prod_{j != k}(z_k - z_j), starting from n
///        points evenly spaced (with a fixed angular offset to avoid real-axis symmetry, which
///        would otherwise stall convergence to any non-real root of a real-coefficient
///        polynomial) around a circle whose radius is a Cauchy bound on root magnitude.
///        Converges quadratically once guesses are close, but -- unlike Sturm's theorem-based
///        real_roots(), which isolates each real root to a provably correct interval before
///        refining it -- this is a general iterative method with no convergence guarantee.
///        For best results p should be square-free (see square_free_factorization()); repeated
///        roots make two of the simultaneous iterates chase the same point, which both slows
///        convergence and risks a near-zero denominator late in the iteration.
/// @throws std::invalid_argument if p has degree 0 (a nonzero constant has no roots; the zero
///         polynomial is degenerate).
[[nodiscard]] inline std::vector<Complex> complex_roots(const Polynomial& p, std::size_t max_iterations = 200,
                                                         double tolerance = 1e-12) {
    if (p.degree() <= 0) throw std::invalid_argument("complex_roots: p must have degree >= 1");

    const Polynomial mp = monic(p);
    const int n = mp.degree();
    const double bound = std::max(1.0, detail::cauchy_root_bound(mp));

    std::vector<detail::Cplx> z(static_cast<std::size_t>(n));
    constexpr double kPi = 3.14159265358979323846;
    for (int k = 0; k < n; ++k) {
        const double angle = 2.0 * kPi * static_cast<double>(k) / static_cast<double>(n) + 0.5;
        z[static_cast<std::size_t>(k)] = {bound * std::cos(angle), bound * std::sin(angle)};
    }

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        double max_delta = 0.0;
        for (int k = 0; k < n; ++k) {
            const std::size_t sk = static_cast<std::size_t>(k);
            detail::Cplx denom{1.0, 0.0};
            for (int j = 0; j < n; ++j) {
                if (j == k) continue;
                denom = detail::c_mul(denom, detail::c_sub(z[sk], z[static_cast<std::size_t>(j)]));
            }
            const detail::Cplx delta = detail::c_div(detail::c_eval(mp, z[sk]), denom);
            z[sk] = detail::c_sub(z[sk], delta);
            max_delta = std::max(max_delta, detail::c_abs(delta));
        }
        if (max_delta < tolerance) break;
    }

    std::vector<Complex> roots;
    roots.reserve(static_cast<std::size_t>(n));
    for (const auto& zk : z) roots.push_back(Complex{zk.re, zk.im});
    return roots;
}

/// @brief All rational roots of p, via the Rational Root Theorem: every rational root a/b in
///        lowest terms has a dividing the constant term and b dividing the leading coefficient,
///        so trying every such ratio (after first dividing out any factors of x, i.e. any
///        roots at exactly 0) finds every rational root exactly -- a finite, exact search, in
///        contrast to real_roots()'s numeric bisection or complex_roots()'s general iteration.
/// @param tolerance Used both to verify p's coefficients are within `tolerance` of integers
///        (required for the theorem to apply) and to test each candidate ratio's residual.
/// @throws std::invalid_argument if p's coefficients are not within `tolerance` of integers.
[[nodiscard]] inline std::vector<double> rational_roots(const Polynomial& p, double tolerance = 1e-9) {
    if (p.degree() <= 0) return {};

    std::vector<std::int64_t> coeffs;
    coeffs.reserve(p.coefficients().size());
    for (double c : p.coefficients()) {
        const double rounded = std::round(c);
        if (std::fabs(c - rounded) > tolerance)
            throw std::invalid_argument("rational_roots: p must have (near-)integer coefficients");
        coeffs.push_back(static_cast<std::int64_t>(rounded));
    }

    std::vector<double> roots;

    // Factor out x: a run of zero low-order coefficients means 0 is a root.
    std::size_t shift = 0;
    while (shift + 1 < coeffs.size() && coeffs[shift] == 0) ++shift;
    if (shift > 0) roots.push_back(0.0);
    coeffs.erase(coeffs.begin(), coeffs.begin() + static_cast<std::ptrdiff_t>(shift));
    if (coeffs.size() <= 1) return roots; // p was a constant times a power of x

    const std::int64_t constant_term = std::llabs(coeffs.front());
    const std::int64_t leading_term = std::llabs(coeffs.back());

    auto divisors_of = [](std::int64_t value) {
        std::vector<std::int64_t> divs;
        for (std::int64_t d = 1; d * d <= value; ++d) {
            if (value % d == 0) {
                divs.push_back(d);
                if (d != value / d) divs.push_back(value / d);
            }
        }
        return divs;
    };

    std::vector<double> reduced_coeffs(coeffs.begin(), coeffs.end());
    const Polynomial reduced(std::move(reduced_coeffs));

    for (std::int64_t a : divisors_of(constant_term)) {
        for (std::int64_t b : divisors_of(leading_term)) {
            for (double sign : {1.0, -1.0}) {
                const double candidate = sign * static_cast<double>(a) / static_cast<double>(b);
                if (std::fabs(reduced.evaluate(candidate)) > tolerance) continue;
                const bool duplicate =
                    std::any_of(roots.begin(), roots.end(), [&](double r) { return std::fabs(r - candidate) < 1e-9; });
                if (!duplicate) roots.push_back(candidate);
            }
        }
    }

    std::sort(roots.begin(), roots.end());
    return roots;
}

} // namespace datamunge::algebra
