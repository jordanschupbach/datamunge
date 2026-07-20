#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algebra {

namespace detail {

using GfPoly = std::vector<long long>; // ascending-degree coefficients, reduced mod p

[[nodiscard]] inline long long gf_mod(long long a, long long p) {
    const long long r = a % p;
    return r < 0 ? r + p : r;
}

[[nodiscard]] inline GfPoly gf_trim(GfPoly a) {
    while (a.size() > 1 && a.back() == 0) a.pop_back();
    if (a.empty()) a.push_back(0);
    return a;
}

[[nodiscard]] inline int gf_degree(const GfPoly& a) { return static_cast<int>(a.size()) - 1; }

[[nodiscard]] inline bool gf_is_zero(const GfPoly& a) { return a.size() == 1 && a[0] == 0; }

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

    // Q[i] = coefficients of x^(i*p) mod f, as a length-n row (padded/truncated to degree < n).
    std::vector<std::vector<long long>> Q(static_cast<std::size_t>(n), std::vector<long long>(static_cast<std::size_t>(n), 0));
    GfPoly current{1};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= detail::gf_degree(current) && j < n; ++j)
            Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = current[static_cast<std::size_t>(j)];
        GfPoly next = current;
        for (long long step = 0; step < p; ++step) next = detail::gf_mul_x_mod(next, f, p);
        current = next;
    }

    // M = Q - I, over GF(p).
    std::vector<std::vector<long long>> M(static_cast<std::size_t>(n), std::vector<long long>(static_cast<std::size_t>(n)));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                detail::gf_mod(Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] - (i == j ? 1 : 0), p);

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

        const long long inv = detail::gf_mod_inverse(M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(col)], p);
        for (int k = 0; k < n; ++k)
            M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(k)] = detail::gf_mod(M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(k)] * inv, p);

        for (int row = 0; row < n; ++row) {
            if (row == rank) continue;
            const long long factor = M[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            if (factor == 0) continue;
            for (int k = 0; k < n; ++k) {
                const std::size_t rk = static_cast<std::size_t>(row), kk = static_cast<std::size_t>(k), rankk = static_cast<std::size_t>(rank);
                M[rk][kk] = detail::gf_mod(M[rk][kk] - factor * M[rankk][kk], p);
            }
        }
        pivot_col[static_cast<std::size_t>(rank)] = col;
        ++rank;
    }
    const int nullity = n - rank; // == number of distinct irreducible factors of a square-free poly

    std::vector<bool> is_pivot_col(static_cast<std::size_t>(n), false);
    for (int r = 0; r < rank; ++r) is_pivot_col[static_cast<std::size_t>(pivot_col[static_cast<std::size_t>(r)])] = true;

    std::vector<GfPoly> basis;
    for (int free_col = 0; free_col < n; ++free_col) {
        if (is_pivot_col[static_cast<std::size_t>(free_col)]) continue;
        GfPoly v(static_cast<std::size_t>(n), 0);
        v[static_cast<std::size_t>(free_col)] = 1;
        for (int r = 0; r < rank; ++r)
            v[static_cast<std::size_t>(pivot_col[static_cast<std::size_t>(r)])] =
                detail::gf_mod(-M[static_cast<std::size_t>(r)][static_cast<std::size_t>(free_col)], p);
        basis.push_back(detail::gf_trim(v));
    }

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

} // namespace datamunge::algebra
