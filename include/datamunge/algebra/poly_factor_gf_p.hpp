#pragma once

/// @file poly_factor_gf_p.hpp
/// @brief Factoring polynomials over the prime field GF(p): Berlekamp's algorithm for square-free
///        polynomials, square-free factorization, complete factorization with multiplicities,
///        and counts of irreducible polynomials.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algebra {

namespace detail {

/// Polynomial over GF(p): ascending-degree coefficients, each reduced mod p.
using GfPoly = std::vector<long long>;

/// Least non-negative residue of a modulo p.
[[nodiscard]] inline long long gf_mod(long long a, long long p) {
    const long long r = a % p;
    return r < 0 ? r + p : r;
}

/// Drops trailing zero coefficients, leaving at least the constant term.
[[nodiscard]] inline GfPoly gf_trim(GfPoly a) {
    while (a.size() > 1 && a.back() == 0) a.pop_back();
    if (a.empty()) a.push_back(0);
    return a;
}

/// Degree of a trimmed polynomial (0 for constants, including zero).
[[nodiscard]] inline int gf_degree(const GfPoly& a) { return static_cast<int>(a.size()) - 1; }

/// True when a trimmed polynomial is the zero polynomial.
[[nodiscard]] inline bool gf_is_zero(const GfPoly& a) { return a.size() == 1 && a[0] == 0; }

/// Inverse of a modulo p by the extended Euclidean algorithm; throws if none exists.
[[nodiscard]] inline long long gf_mod_inverse(long long a, long long p) {
    long long r0 = p, r1 = gf_mod(a, p), s0 = 0, s1 = 1;
    while (r1 != 0) {
        const long long q = r0 / r1;
        const long long r2 = r0 - q * r1;
        r0 = r1;
        r1 = r2;
        const long long s2 = s0 - q * s1;
        s0 = s1;
        s1 = s2;
    }
    if (r0 != 1) throw std::invalid_argument("no inverse mod p (p is not prime, or the value is a multiple of p)");
    return gf_mod(s0, p);
}

/// Quotient and remainder of a divided by divisor over GF(p); throws on a zero divisor.
[[nodiscard]] inline std::pair<GfPoly, GfPoly> gf_divmod(GfPoly a, const GfPoly& divisor, long long p) {
    a = gf_trim(std::move(a));
    const GfPoly b = gf_trim(divisor);
    if (gf_is_zero(b)) throw std::invalid_argument("division by the zero polynomial");
    const int da = gf_degree(a);
    const int db = gf_degree(b);
    if (da < db) return {GfPoly{0}, a};

    GfPoly q(static_cast<std::size_t>(da - db + 1), 0);
    const long long lead_inv = gf_mod_inverse(b[static_cast<std::size_t>(db)], p);
    for (int k = da - db; k >= 0; --k) {
        const std::size_t sk = static_cast<std::size_t>(k);
        const long long coef = gf_mod(a[sk + static_cast<std::size_t>(db)] * lead_inv, p);
        q[sk] = coef;
        for (int j = 0; j <= db; ++j) {
            const std::size_t idx = sk + static_cast<std::size_t>(j);
            a[idx] = gf_mod(a[idx] - coef * b[static_cast<std::size_t>(j)], p);
        }
    }
    return {gf_trim(q), gf_trim(a)};
}

/// Monic greatest common divisor of a and b over GF(p).
[[nodiscard]] inline GfPoly gf_gcd(GfPoly a, GfPoly b, long long p) {
    a = gf_trim(std::move(a));
    b = gf_trim(std::move(b));
    while (!gf_is_zero(b)) {
        auto [q, r] = gf_divmod(a, b, p);
        (void)q;
        a = b;
        b = r;
    }
    if (!gf_is_zero(a)) {
        const long long inv = gf_mod_inverse(a.back(), p);
        for (auto& c : a) c = gf_mod(c * inv, p);
    }
    return a;
}

/// @brief Multiplies `poly` by x mod `modulus` (i.e. shift-then-reduce), used to build up
///        x^(i*p) mod f one multiplication-by-x at a time in berlekamp_factor().
[[nodiscard]] inline GfPoly gf_mul_x_mod(const GfPoly& poly, const GfPoly& modulus, long long p) {
    GfPoly shifted(poly.size() + 1, 0);
    for (std::size_t k = 0; k < poly.size(); ++k) shifted[k + 1] = poly[k];
    return gf_divmod(shifted, modulus, p).second;
}

/// @brief Product of two polynomials over GF(p).
[[nodiscard]] inline GfPoly gf_mul(const GfPoly& a, const GfPoly& b, long long p) {
    if (gf_is_zero(a) || gf_is_zero(b)) return {0};
    GfPoly r(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != 0)
            for (std::size_t j = 0; j < b.size(); ++j) r[i + j] = gf_mod(r[i + j] + a[i] * b[j], p);
    return gf_trim(r);
}

/// @brief Formal derivative over GF(p); it vanishes for p-th powers \f$g(x^p)\f$.
[[nodiscard]] inline GfPoly gf_derivative(const GfPoly& a, long long p) {
    if (a.size() <= 1) return {0};
    GfPoly d(a.size() - 1);
    for (std::size_t i = 1; i < a.size(); ++i) d[i - 1] = gf_mod(a[i] * static_cast<long long>(i % static_cast<std::size_t>(p)), p);
    return gf_trim(d);
}

/// @brief Square-free decomposition of a monic polynomial over GF(p) (Yun's algorithm, with the
///        p-th-root step needed in characteristic p): pairs (g, m) with f = prod g^m, every g
///        monic and square-free and the g pairwise coprime.
[[nodiscard]] inline std::vector<std::pair<GfPoly, int>> gf_square_free(const GfPoly& f, long long p) {
    std::vector<std::pair<GfPoly, int>> out;
    if (gf_degree(f) <= 0) return out;
    const auto pth_root_parts = [&](const GfPoly& c) {
        // c = h(x^p), and a^(1/p) = a in GF(p), so c^(1/p) = h(x).
        GfPoly h;
        for (std::size_t i = 0; i < c.size(); i += static_cast<std::size_t>(p)) h.push_back(c[i]);
        for (auto& [g, m] : gf_square_free(gf_trim(h), p)) out.push_back({g, m * static_cast<int>(p)});
    };
    const GfPoly fp = gf_derivative(f, p);
    if (gf_is_zero(fp)) {
        pth_root_parts(f);
        return out;
    }
    GfPoly c = gf_gcd(f, fp, p);
    GfPoly w = gf_divmod(f, c, p).first;
    for (int i = 1; gf_degree(w) > 0; ++i) {
        const GfPoly y = gf_gcd(w, c, p);
        const GfPoly z = gf_divmod(w, y, p).first;
        if (gf_degree(z) > 0) out.push_back({z, i});
        w = y;
        c = gf_divmod(c, y, p).first;
    }
    if (gf_degree(c) > 0) pth_root_parts(c);
    return out;
}

/// @brief Basis of the Berlekamp subalgebra of a monic polynomial @p f of degree n >= 1: the
///        polynomials v of degree < n with \f$v^p\equiv v \pmod f\f$, as the null space of
///        \f$Q-I\f$, where row i of Q holds the coefficients of \f$x^{ip}\bmod f\f$. For square-free
///        f its dimension is the number of irreducible factors. The first vector is the constant 1.
[[nodiscard]] inline std::vector<GfPoly> berlekamp_null_space(const GfPoly& f, long long p) {
    const int n = gf_degree(f);
    // Q[i] = coefficients of x^(i*p) mod f, as a length-n row (padded/truncated to degree < n).
    std::vector<std::vector<long long>> Q(static_cast<std::size_t>(n), std::vector<long long>(static_cast<std::size_t>(n), 0));
    GfPoly current{1};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= gf_degree(current) && j < n; ++j)
            Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = current[static_cast<std::size_t>(j)];
        GfPoly next = current;
        for (long long step = 0; step < p; ++step) next = gf_mul_x_mod(next, f, p);
        current = next;
    }

    // v^p = sum_i v_i x^(ip) = sum_i v_i Q[i] (mod f), so v^p = v means v^T (Q - I) = 0: the null
    // space of M = (Q - I)^T. (Using Q - I itself gives a space of the right dimension but the
    // wrong vectors, which can leave a factor unsplit.)
    std::vector<std::vector<long long>> M(static_cast<std::size_t>(n), std::vector<long long>(static_cast<std::size_t>(n)));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] =
                gf_mod(Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] - (i == j ? 1 : 0), p);

    // Gaussian elimination to row-reduced echelon form, tracking pivot columns.
    std::vector<int> pivot_col(static_cast<std::size_t>(n), -1);
    int rank = 0;
    for (int col = 0; col < n && rank < n; ++col) {
        int pivot_row = -1;
        for (int row = rank; row < n; ++row) {
            if (M[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] != 0) {
                pivot_row = row;
                break;
            }
        }
        if (pivot_row < 0) continue;
        std::swap(M[static_cast<std::size_t>(pivot_row)], M[static_cast<std::size_t>(rank)]);

        const long long inv = gf_mod_inverse(M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(col)], p);
        for (int k = 0; k < n; ++k)
            M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(k)] = gf_mod(M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(k)] * inv, p);

        for (int row = 0; row < n; ++row) {
            if (row == rank) continue;
            const long long factor = M[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            if (factor == 0) continue;
            for (int k = 0; k < n; ++k) {
                const std::size_t rk = static_cast<std::size_t>(row), kk = static_cast<std::size_t>(k), rankk = static_cast<std::size_t>(rank);
                M[rk][kk] = gf_mod(M[rk][kk] - factor * M[rankk][kk], p);
            }
        }
        pivot_col[static_cast<std::size_t>(rank)] = col;
        ++rank;
    }

    std::vector<bool> is_pivot_col(static_cast<std::size_t>(n), false);
    for (int r = 0; r < rank; ++r) is_pivot_col[static_cast<std::size_t>(pivot_col[static_cast<std::size_t>(r)])] = true;

    std::vector<GfPoly> basis;
    for (int free_col = 0; free_col < n; ++free_col) {
        if (is_pivot_col[static_cast<std::size_t>(free_col)]) continue;
        GfPoly v(static_cast<std::size_t>(n), 0);
        v[static_cast<std::size_t>(free_col)] = 1;
        for (int r = 0; r < rank; ++r)
            v[static_cast<std::size_t>(pivot_col[static_cast<std::size_t>(r)])] =
                gf_mod(-M[static_cast<std::size_t>(r)][static_cast<std::size_t>(free_col)], p);
        basis.push_back(gf_trim(v));
    }

    return basis;
}

} // namespace detail

/// @brief Factors `poly` (integer coefficients in ascending-degree order) over GF(p), for a
///        small prime p, via Berlekamp's algorithm: builds the Berlekamp matrix Q (row i is
///        x^(i*p) mod poly), finds a basis for the null space of Q - I over GF(p) by Gaussian
///        elimination, then splits `poly` using gcd(current_factor, basis_vector - c) for each
///        residue c in 0..p-1. Deterministic and exact for any prime p; less efficient than a
///        randomized Cantor-Zassenhaus equal-degree split for large p, but simpler and complete
///        on its own (matches this module's "simple, correct" scoping used elsewhere).
///
/// @param poly Must be monic and SQUARE-FREE mod p (repeated factors should first be removed,
///        e.g. by dividing out gcd(poly, poly') mod p) -- Berlekamp's null-space construction
///        assumes distinct irreducible factors.
/// @return The monic irreducible factors of poly mod p (ascending-degree coefficient vectors,
///         each entry in [0, p)), one per distinct irreducible factor.
/// @throws std::invalid_argument if p is not prime (surfaces as a failed modular inverse) or
///         poly is the zero/a constant polynomial.
[[nodiscard]] inline std::vector<std::vector<long long>> berlekamp_factor(std::vector<long long> poly, long long p) {
    using detail::GfPoly;

    GfPoly f = detail::gf_trim(std::move(poly));
    for (auto& c : f) c = detail::gf_mod(c, p);
    f = detail::gf_trim(f);
    const int n = detail::gf_degree(f);
    if (n <= 0) throw std::invalid_argument("poly must have degree >= 1");

    // Normalize to monic.
    if (f.back() != 1) {
        const long long inv = detail::gf_mod_inverse(f.back(), p);
        for (auto& c : f) c = detail::gf_mod(c * inv, p);
    }

    const std::vector<GfPoly> basis = detail::berlekamp_null_space(f, p);
    const int nullity = static_cast<int>(basis.size());  // number of distinct irreducible factors

    // Split f by successive basis vectors: for each nontrivial null-space vector v, refine
    // every current factor via gcd(factor, v - c) for c in 0..p-1.
    std::vector<GfPoly> factors{f};
    for (const auto& v : basis) {
        if (static_cast<int>(factors.size()) >= nullity) break;
        std::vector<GfPoly> refined;
        for (const auto& factor : factors) {
            if (detail::gf_degree(factor) <= 1) {
                refined.push_back(factor);
                continue;
            }
            std::vector<GfPoly> pieces;
            GfPoly remaining = factor;
            for (long long c = 0; c < p && detail::gf_degree(remaining) > 0; ++c) {
                GfPoly shifted = detail::gf_trim(v);
                shifted[0] = detail::gf_mod(shifted[0] - c, p);
                GfPoly g = detail::gf_gcd(remaining, shifted, p);
                if (detail::gf_degree(g) > 0) {
                    pieces.push_back(g);
                    remaining = detail::gf_divmod(remaining, g, p).first;
                }
            }
            if (detail::gf_degree(remaining) > 0) pieces.push_back(remaining);
            if (pieces.empty()) pieces.push_back(factor);
            for (auto& piece : pieces) refined.push_back(std::move(piece));
        }
        factors = std::move(refined);
    }

    return factors;
}

/// @brief Binding-friendly wrapper around berlekamp_factor() using doubles (exact up to 2^53)
///        for both the input polynomial and the prime, matching this module's
///        std::size_t/fixed-width-integer-avoidance convention for SWIG-facing signatures
///        (`std::vector<long long>` has no proven binding template in this codebase, while
///        `std::vector<double>` already does). See berlekamp_factor() for the algorithm and
///        preconditions.
[[nodiscard]] inline std::vector<std::vector<double>> berlekamp_factor_mod(const std::vector<double>& poly, double prime) {
    std::vector<long long> int_poly;
    int_poly.reserve(poly.size());
    for (double c : poly) int_poly.push_back(static_cast<long long>(std::llround(c)));

    const auto factors = berlekamp_factor(int_poly, static_cast<long long>(std::llround(prime)));

    std::vector<std::vector<double>> result;
    result.reserve(factors.size());
    for (const auto& factor : factors) {
        std::vector<double> factor_d;
        factor_d.reserve(factor.size());
        for (long long c : factor) factor_d.push_back(static_cast<double>(c));
        result.push_back(std::move(factor_d));
    }
    return result;
}

/// Complete factorization over GF(p) from factor_gf_p().
struct GfFactorization {
    long long                           leading = 0;     ///< Leading coefficient, in [1, p).
    std::vector<std::vector<long long>> factors;         ///< Distinct monic irreducible factors, by degree then coefficients.
    std::vector<int>                    multiplicities;  ///< Multiplicity of each factor.
};

/// @brief Number of irreducible factors of a monic square-free polynomial over GF(p), by
///        Berlekamp's theorem: the dimension of the null space of \f$(Q-I)^{\mathsf T}\f$, that is
///        \f$n-\operatorname{rank}(Q-I)\f$. A square-free polynomial is irreducible exactly when it is 1.
/// @param poly ascending coefficients (made monic and reduced mod p first).
/// @param p a prime.
/// @return the number of distinct irreducible factors (for square-free input).
/// @throws std::invalid_argument if poly is constant.
[[nodiscard]] inline int berlekamp_factor_count(std::vector<long long> poly, long long p) {
    using detail::GfPoly;
    GfPoly f = detail::gf_trim(std::move(poly));
    for (auto& c : f) c = detail::gf_mod(c, p);
    f = detail::gf_trim(f);
    if (detail::gf_degree(f) <= 0) throw std::invalid_argument("berlekamp_factor_count: degree must be >= 1");
    const long long inv = detail::gf_mod_inverse(f.back(), p);
    for (auto& c : f) c = detail::gf_mod(c * inv, p);
    return static_cast<int>(detail::berlekamp_null_space(f, p).size());
}

/// @brief Complete factorization of a polynomial over GF(p): leading coefficient times monic
///        irreducible factors with multiplicities. The square-free decomposition separates
///        repeated factors (taking p-th roots where the derivative vanishes), and Berlekamp's
///        algorithm splits each square-free part.
/// @param poly ascending coefficients; reduced mod p first.
/// @param p a prime.
/// @return the leading coefficient, the factors and their multiplicities.
/// @throws std::invalid_argument for the zero polynomial.
[[nodiscard]] inline GfFactorization factor_gf_p(std::vector<long long> poly, long long p) {
    using detail::GfPoly;
    GfPoly f = detail::gf_trim(std::move(poly));
    for (auto& c : f) c = detail::gf_mod(c, p);
    f = detail::gf_trim(f);
    if (detail::gf_is_zero(f)) throw std::invalid_argument("factor_gf_p: zero polynomial");
    GfFactorization out;
    out.leading = f.back();
    const long long inv = detail::gf_mod_inverse(f.back(), p);
    for (auto& c : f) c = detail::gf_mod(c * inv, p);
    std::vector<std::pair<GfPoly, int>> found;
    for (const auto& [part, m] : detail::gf_square_free(f, p))
        for (auto& g : berlekamp_factor(part, p)) found.push_back({std::move(g), m});
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size()) return a.first.size() < b.first.size();
        return a.first < b.first;
    });
    for (auto& [g, m] : found) {
        out.factors.push_back(std::move(g));
        out.multiplicities.push_back(m);
    }
    return out;
}

/// @brief Number of monic irreducible polynomials of degree @p d over GF(p), by Gauss's formula
///        \f$I_d=\frac1d\sum_{e\mid d}\mu(d/e)\,p^{e}\f$, as a double (exact while below 2^53).
/// @param p a prime.
/// @param d degree, at least 1.
/// @return \f$I_d\f$.
[[nodiscard]] inline double count_monic_irreducibles(long long p, int d) {
    if (d < 1) throw std::invalid_argument("count_monic_irreducibles: degree must be >= 1");
    const auto mobius = [](int m) {
        int sign = 1;
        for (int q = 2; q * q <= m; ++q)
            if (m % q == 0) {
                m /= q;
                if (m % q == 0) return 0;
                sign = -sign;
            }
        return m > 1 ? -sign : sign;
    };
    double total = 0.0;
    for (int e = 1; e <= d; ++e)
        if (d % e == 0) total += mobius(d / e) * std::pow(static_cast<double>(p), e);
    return total / d;
}

/// @brief Expected number of distinct monic irreducible factors of a uniformly random monic
///        polynomial of degree @p n over GF(p): each irreducible g of degree d <= n divides it with
///        probability \f$p^{-d}\f$, so the expectation is \f$\sum_{d=1}^{n}I_d\,p^{-d}\approx H_n\f$.
/// @param p a prime.
/// @param n degree, at least 1.
/// @return the expected count.
[[nodiscard]] inline double expected_distinct_irreducible_factors(long long p, int n) {
    double total = 0.0;
    for (int d = 1; d <= n; ++d) total += count_monic_irreducibles(p, d) / std::pow(static_cast<double>(p), d);
    return total;
}

} // namespace datamunge::algebra
