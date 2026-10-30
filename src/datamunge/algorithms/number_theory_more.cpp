/// @file number_theory_more.cpp
/// @brief Pollard rho and Pohlig-Hellman logarithms, Lucas sequences and probable-prime tests.

#include <datamunge/algorithms/number_theory_more.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <random>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Modular arithmetic helpers for the discrete-log routines.
namespace {

/// @brief (a * b) mod m via 128-bit arithmetic.
std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    return static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) % m);
}
/// @brief a^e mod m by square-and-multiply.
std::uint64_t powmod(std::uint64_t a, std::uint64_t e, std::uint64_t m) {
    std::uint64_t r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = mulmod(r, a, m);
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}
/// @brief Greatest common divisor.
std::uint64_t gcd_u(std::uint64_t a, std::uint64_t b) {
    while (b) { const std::uint64_t t = a % b; a = b; b = t; }
    return a;
}
/// @brief Inverse of a mod m (m need not be prime); 0 if not invertible.
std::uint64_t modinv(std::uint64_t a, std::uint64_t m) {
    long long g = m, x = 0, x1 = 1, r = a % m;
    while (r) {
        const long long q = g / r;
        g -= q * r; std::swap(g, r);
        x -= q * x1; std::swap(x, x1);
    }
    if (g != 1) return 0;
    return static_cast<std::uint64_t>(((x % static_cast<long long>(m)) + static_cast<long long>(m)) % static_cast<long long>(m));
}

/// @brief Small discrete log in a cyclic group of order q (q small): base^x = target, via a table.
long long small_log(std::uint64_t base, std::uint64_t target, std::uint64_t p, std::uint64_t q) {
    std::uint64_t cur = 1 % p;
    for (std::uint64_t j = 0; j < q; ++j) {
        if (cur == target % p) return static_cast<long long>(j);
        cur = mulmod(cur, base, p);
    }
    return -1;
}

/// @brief Prime factorisation of n by trial division, as (prime, exponent) pairs.
std::vector<std::pair<std::uint64_t, int>> factor_order(std::uint64_t n) {
    std::vector<std::pair<std::uint64_t, int>> f;
    for (std::uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) { int e = 0; while (n % d == 0) { n /= d; ++e; } f.emplace_back(d, e); }
    if (n > 1) f.emplace_back(n, 1);
    return f;
}

} // namespace

long long pollard_rho_log(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t order) {
    if (h % p == 1 % p) return 0;
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL ^ (g * 1000003ULL + h));

    auto step = [&](std::uint64_t& x, std::uint64_t& a, std::uint64_t& b) {
        switch (x % 3) {
        case 0: x = mulmod(x, h, p); b = (b + 1) % order; break;
        case 1: x = mulmod(x, x, p); a = (2 * a) % order; b = (2 * b) % order; break;
        default: x = mulmod(x, g, p); a = (a + 1) % order; break;
        }
    };

    for (int attempt = 0; attempt < 40; ++attempt) {
        // Random starting point x0 = g^a0 h^b0.
        std::uint64_t a0 = rng() % order, b0 = rng() % order;
        std::uint64_t x = mulmod(powmod(g, a0, p), powmod(h, b0, p), p), a = a0, b = b0;
        std::uint64_t X = x, A = a, B = b;
        for (std::uint64_t i = 0; i < 8ULL * order + 64; ++i) {
            step(x, a, b);
            step(X, A, B);
            step(X, A, B);
            if (x == X) {
                const long long r = ((static_cast<long long>(B) - static_cast<long long>(b)) % static_cast<long long>(order) + static_cast<long long>(order)) % static_cast<long long>(order);
                const long long num = ((static_cast<long long>(a) - static_cast<long long>(A)) % static_cast<long long>(order) + static_cast<long long>(order)) % static_cast<long long>(order);
                if (r == 0) break; // degenerate collision; restart
                const std::uint64_t d = gcd_u(static_cast<std::uint64_t>(r), order);
                if (num % static_cast<long long>(d) != 0) break;
                const std::uint64_t m   = order / d;
                const std::uint64_t rr  = (static_cast<std::uint64_t>(r) / d) % m;
                const std::uint64_t nn  = (static_cast<std::uint64_t>(num) / d) % m;
                const std::uint64_t inv = modinv(rr, m);
                if (inv == 0) break;
                const std::uint64_t x0 = mulmod(nn, inv, m);
                for (std::uint64_t k = 0; k < d; ++k) {
                    const std::uint64_t cand = (x0 + k * m) % order;
                    if (powmod(g, cand, p) == h % p) return static_cast<long long>(cand);
                }
                break;
            }
        }
    }
    return -1;
}

long long pohlig_hellman_log(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t order) {
    const auto    factors = factor_order(order);
    std::uint64_t crt_x = 0, crt_mod = 1;

    for (const auto& [q, e] : factors) {
        std::uint64_t qe = 1;
        for (int i = 0; i < e; ++i) qe *= q;

        // gamma has order q; solve digit by digit for x mod qe.
        const std::uint64_t gamma = powmod(g, order / q, p);
        const std::uint64_t ginv  = powmod(g, order - 1, p); // g^{-1} = g^{ord-1} (works for our uses via order)
        std::uint64_t       xi    = 0;
        std::uint64_t       qk    = 1;
        for (int k = 0; k < e; ++k) {
            const std::uint64_t gpow = powmod(ginv, xi, p);          // g^{-xi}
            const std::uint64_t hk   = powmod(mulmod(h, gpow, p), order / (qk * q), p);
            const long long     dk   = small_log(gamma, hk, p, q);
            if (dk < 0) return -1;
            xi += static_cast<std::uint64_t>(dk) * qk;
            qk *= q;
        }
        // CRT-merge (xi mod qe) into (crt_x mod crt_mod); qe is coprime to crt_mod.
        const std::uint64_t inv = modinv(crt_mod % qe, qe);
        const std::uint64_t t   = mulmod(((xi % qe + qe) - crt_x % qe) % qe, inv, qe);
        crt_x   = crt_x + crt_mod * t;
        crt_mod = crt_mod * qe;
    }
    return (powmod(g, crt_x % order, p) == h % p) ? static_cast<long long>(crt_x % order) : -1;
}

/// @brief Primality helpers.
namespace {

/// @brief Strong probable-prime test of odd n to base a.
bool miller_rabin_base(std::uint64_t n, std::uint64_t a) {
    if (a % n == 0) return true;
    std::uint64_t d = n - 1;
    int           s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    std::uint64_t x = powmod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int r = 1; r < s; ++r) { x = mulmod(x, x, n); if (x == n - 1) return true; }
    return false;
}

/// @brief (a + b) mod n for a, b < n without overflow.
std::uint64_t addmod(std::uint64_t a, std::uint64_t b, std::uint64_t n) { return a >= n - b ? a - (n - b) : a + b; }

/// @brief (a - b) mod n for a, b < n.
std::uint64_t submod(std::uint64_t a, std::uint64_t b, std::uint64_t n) { return a >= b ? a - b : a + (n - b); }

/// @brief A signed value reduced into [0, n).
std::uint64_t reduce_signed(long long v, std::uint64_t n) {
    if (v >= 0) return static_cast<std::uint64_t>(v) % n;
    const auto m = static_cast<std::uint64_t>(-(v + 1)) % n;  // (|v| - 1) mod n, safe at LLONG_MIN
    return n - 1 - m;                                           // -|v| = -(m + 1) mod n
}

/// @brief True when @p n is a perfect square.
bool is_square(std::uint64_t n) {
    // r stays below 2^32, so r * r cannot overflow.
    std::uint64_t r = std::min<std::uint64_t>(static_cast<std::uint64_t>(std::sqrt(static_cast<double>(n))), 0xFFFFFFFFULL);
    while (r > 0 && r * r > n) --r;
    while (r < 0xFFFFFFFFULL && (r + 1) * (r + 1) <= n) ++r;
    return r * r == n;
}

} // namespace

int jacobi_symbol(long long a, std::uint64_t n) {
    if (n == 0 || n % 2 == 0) throw std::invalid_argument("jacobi_symbol: n must be odd and positive");
    std::uint64_t x = reduce_signed(a, n), m = n;
    int result = 1;
    while (x != 0) {
        while ((x & 1) == 0) {
            x >>= 1;
            if (m % 8 == 3 || m % 8 == 5) result = -result;
        }
        std::swap(x, m);
        if (x % 4 == 3 && m % 4 == 3) result = -result;
        x %= m;
    }
    return m == 1 ? result : 0;
}

SelfridgeParameters selfridge_parameters(std::uint64_t n) {
    if (n < 3 || n % 2 == 0) throw std::invalid_argument("selfridge_parameters: n must be odd and at least 3");
    SelfridgeParameters out;
    if (is_square(n)) {
        out.perfect_square = true;
        return out;
    }
    long long D = 5;
    while (jacobi_symbol(D, n) != -1) D = D > 0 ? -(D + 2) : -(D - 2);  // 5, -7, 9, -11, 13, ...
    out.D = D;
    out.Q = (1 - D) / 4;
    return out;
}

LucasTerms lucas_sequence_mod(std::uint64_t k, long long P, long long Q, std::uint64_t n) {
    if (n % 2 == 0) throw std::invalid_argument("lucas_sequence_mod: n must be odd");
    LucasTerms out;
    if (n == 1) return out;
    const std::uint64_t Pm = reduce_signed(P, n), Qm = reduce_signed(Q, n);
    const std::uint64_t Dm = submod(mulmod(Pm, Pm, n), mulmod(4 % n, Qm, n), n);
    const std::uint64_t half = (n + 1) / 2;  // inverse of 2
    if (k == 0) return LucasTerms{0, 2 % n, 1 % n};
    std::uint64_t U = 1 % n, V = Pm, Qk = Qm;  // index 1
    int hb = 63;
    while (!((k >> hb) & 1)) --hb;
    for (int i = hb - 1; i >= 0; --i) {
        U  = mulmod(U, V, n);
        V  = submod(mulmod(V, V, n), addmod(Qk, Qk, n), n);
        Qk = mulmod(Qk, Qk, n);
        if ((k >> i) & 1) {
            const std::uint64_t nU = mulmod(addmod(mulmod(Pm, U, n), V, n), half, n);
            const std::uint64_t nV = mulmod(addmod(mulmod(Dm, U, n), mulmod(Pm, V, n), n), half, n);
            U  = nU;
            V  = nV;
            Qk = mulmod(Qk, Qm, n);
        }
    }
    return LucasTerms{U, V, Qk};
}

bool lucas_probable_prime(std::uint64_t n) {
    if (n == 2) return true;
    if (n < 2 || n % 2 == 0) return false;
    const auto s = selfridge_parameters(n);
    if (s.perfect_square) return false;
    if (n == 3 || n == 5) return true;
    return lucas_sequence_mod(n + 1, s.P, s.Q, n).U == 0;
}

bool strong_lucas_probable_prime(std::uint64_t n) {
    if (n == 2) return true;
    if (n < 2 || n % 2 == 0) return false;
    const auto sp = selfridge_parameters(n);
    if (sp.perfect_square) return false;
    if (n == 3 || n == 5) return true;
    std::uint64_t d = n + 1;  // n < 2^64 - 1 here, so no overflow for odd n
    int s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        ++s;
    }
    auto t = lucas_sequence_mod(d, sp.P, sp.Q, n);
    if (t.U == 0 || t.V == 0) return true;
    for (int r = 1; r < s; ++r) {
        t.V  = submod(mulmod(t.V, t.V, n), addmod(t.Qk, t.Qk, n), n);
        t.Qk = mulmod(t.Qk, t.Qk, n);
        if (t.V == 0) return true;
    }
    return false;
}

bool baillie_psw(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t sp : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
        if (n == sp) return true;
        if (n % sp == 0) return false;
    }
    if (!miller_rabin_base(n, 2)) return false;
    return strong_lucas_probable_prime(n);
}

} // namespace datamunge::algorithms
