#include <datamunge/algorithms/number_theory_more.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {

std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    return static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) % m);
}
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
std::uint64_t gcd_u(std::uint64_t a, std::uint64_t b) {
    while (b) { const std::uint64_t t = a % b; a = b; b = t; }
    return a;
}
// modular inverse of a mod m (m need not be prime); returns 0 if not invertible.
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

// Small discrete log in a cyclic group of order q (q small): base^x = target, via a table.
long long small_log(std::uint64_t base, std::uint64_t target, std::uint64_t p, std::uint64_t q) {
    std::uint64_t cur = 1 % p;
    for (std::uint64_t j = 0; j < q; ++j) {
        if (cur == target % p) return static_cast<long long>(j);
        cur = mulmod(cur, base, p);
    }
    return -1;
}

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

namespace {

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

int jacobi(long long a, long long n) { // n odd > 0
    a %= n; if (a < 0) a += n;
    int result = 1;
    while (a != 0) {
        while ((a & 1) == 0) { a >>= 1; const long long r = n % 8; if (r == 3 || r == 5) result = -result; }
        std::swap(a, n);
        if ((a % 4 == 3) && (n % 4 == 3)) result = -result;
        a %= n;
    }
    return n == 1 ? result : 0;
}

bool is_square(std::uint64_t n) {
    std::uint64_t r = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(n)));
    for (std::uint64_t x = (r > 2 ? r - 2 : 0); x <= r + 2; ++x) if (x * x == n) return true;
    return false;
}

// Strong Lucas probable prime test (Selfridge parameters).
bool strong_lucas(std::uint64_t n) {
    if (is_square(n)) return false;
    long long D = 5;
    while (jacobi(D, static_cast<long long>(n)) != -1) {
        D = D > 0 ? -(D + 2) : -(D - 2); // 5, -7, 9, -11, 13, ...
        if (D > 100000 || D < -100000) return false;
    }
    const std::uint64_t Dm   = static_cast<std::uint64_t>(((D % static_cast<long long>(n)) + static_cast<long long>(n)) % static_cast<long long>(n));
    const long long     Qs   = (1 - D) / 4;
    const std::uint64_t Qm   = static_cast<std::uint64_t>(((Qs % static_cast<long long>(n)) + static_cast<long long>(n)) % static_cast<long long>(n));
    const std::uint64_t P    = 1;
    const std::uint64_t inv2 = (n + 1) / 2;

    std::uint64_t d = n + 1;
    int           s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }

    // Compute U_d, V_d via binary of d (from the top bit down).
    std::uint64_t U = 1, V = P % n, Qk = Qm;
    int           hb = 63;
    while (!((d >> hb) & 1)) --hb;
    for (int i = hb - 1; i >= 0; --i) {
        // double
        U  = mulmod(U, V, n);
        V  = (mulmod(V, V, n) + n - mulmod(2 % n, Qk, n)) % n;
        Qk = mulmod(Qk, Qk, n);
        if ((d >> i) & 1) {
            // add one: (U,V) <- ((P U + V)/2, (D U + P V)/2), Qk <- Qk*Q
            const std::uint64_t nU = mulmod((mulmod(P, U, n) + V) % n, inv2, n);
            const std::uint64_t nV = mulmod((mulmod(Dm, U, n) + mulmod(P, V, n)) % n, inv2, n);
            U  = nU;
            V  = nV;
            Qk = mulmod(Qk, Qm, n);
        }
    }
    if (U == 0 || V == 0) return true;
    for (int r = 1; r < s; ++r) {
        V  = (mulmod(V, V, n) + n - mulmod(2 % n, Qk, n)) % n;
        Qk = mulmod(Qk, Qk, n);
        if (V == 0) return true;
    }
    return false;
}

} // namespace

bool baillie_psw(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t sp : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
        if (n == sp) return true;
        if (n % sp == 0) return false;
    }
    if (!miller_rabin_base(n, 2)) return false;
    return strong_lucas(n);
}

} // namespace datamunge::algorithms
