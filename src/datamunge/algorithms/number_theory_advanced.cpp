#include <datamunge/algorithms/number_theory_advanced.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace {

using u64  = std::uint64_t;
using u128 = unsigned __int128;
using i128 = __int128;

u64 mulmod(u64 a, u64 b, u64 m) { return static_cast<u64>(static_cast<u128>(a) * b % m); }

u64 powmod(u64 a, u64 e, u64 m) {
    u64 r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = mulmod(r, a, m);
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}

u64 gcd_u64(u64 a, u64 b) {
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

// Extended gcd on signed 128-bit; returns g and sets x,y with a*x + b*y = g.
i128 extgcd(i128 a, i128 b, i128& x, i128& y) {
    if (b == 0) { x = 1; y = 0; return a; }
    i128 x1, y1;
    i128 g = extgcd(b, a % b, x1, y1);
    x      = y1;
    y      = x1 - (a / b) * y1;
    return g;
}

// Inverse of a modulo m (m > 0), assuming gcd(a, m) == 1. Result in [0, m).
i128 invmod(i128 a, i128 m) {
    i128 x, y;
    extgcd(((a % m) + m) % m, m, x, y);
    return ((x % m) + m) % m;
}

u64 isqrt_u64(u64 n) {
    if (n == 0) return 0;
    u64 x = static_cast<u64>(std::sqrt(static_cast<double>(n)));
    while (x > 0 && x > n / x) --x;
    while ((x + 1) <= n / (x + 1) && (x + 1) * (x + 1) <= n) ++x;
    return x;
}

i128 isqrt_i128(i128 n) {
    if (n <= 0) return 0;
    i128 x = static_cast<i128>(std::sqrt(static_cast<long double>(n)));
    while (x > 0 && x * x > n) --x;
    while ((x + 1) * (x + 1) <= n) ++x;
    return x;
}

// ---- elliptic-curve arithmetic mod n for Lenstra ECM ----

struct Point {
    i128 x{0}, y{0};
    bool inf{true};
};

// Add P and Q on y^2 = x^3 + a x + b (mod n). If a slope denominator shares a
// nontrivial factor with n, set `factor` and return.
Point ec_add(const Point& P, const Point& Q, i128 a, i128 n, u64& factor) {
    if (P.inf) return Q;
    if (Q.inf) return P;
    const i128 xp = P.x, yp = P.y, xq = Q.x, yq = Q.y;
    i128       num, den;
    if (xp == xq && (yp + yq) % n == 0) return Point{0, 0, true}; // P + (-P) = O
    if (xp == xq && yp == yq) {
        num = (3 * xp % n * xp + a) % n;
        den = (2 * yp) % n;
    } else {
        num = ((yq - yp) % n + n) % n;
        den = ((xq - xp) % n + n) % n;
    }
    const i128 g = [&] { i128 x, y; return extgcd(((den % n) + n) % n, n, x, y); }();
    if (g != 1 && g != n) { factor = static_cast<u64>(g); return Point{0, 0, true}; }
    if (g == n) return Point{0, 0, true}; // denominator ~ 0 mod n: treat as O
    const i128 lam = ((num % n + n) % n) * invmod(den, n) % n;
    i128       xr  = ((lam * lam - xp - xq) % n + n) % n;
    i128       yr  = ((lam * (xp - xr) - yp) % n + n) % n;
    return Point{xr, yr, false};
}

Point ec_mul(Point P, u64 k, i128 a, i128 n, u64& factor) {
    Point R{0, 0, true};
    while (k) {
        if (k & 1) {
            R = ec_add(R, P, a, n, factor);
            if (factor) return R;
        }
        P = ec_add(P, P, a, n, factor);
        if (factor) return R;
        k >>= 1;
    }
    return R;
}

struct XorShift {
    u64 s;
    explicit XorShift(u64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

// ---- AKS helpers ----

bool is_perfect_power(u64 n) {
    for (int b = 2; (1ull << b) <= n; ++b) {
        // integer b-th root
        u64 lo = 1, hi = 1ull << (63 / b + 1);
        while (hi - lo > 1) {
            u64 mid = lo + (hi - lo) / 2;
            // mid^b vs n, guard overflow
            u128 p = 1;
            bool over = false;
            for (int i = 0; i < b; ++i) { p *= mid; if (p > n) { over = true; break; } }
            if (!over && static_cast<u64>(p) == n) return true;
            if (over || static_cast<u64>(p) > n) hi = mid; else lo = mid;
        }
    }
    return false;
}

u64 mult_order(u64 n, u64 r) {
    if (gcd_u64(n % r, r) != 1) return 0;
    u64 k = 1, cur = n % r;
    while (cur != 1) { cur = mulmod(cur, n % r, r); ++k; if (k > r) return 0; }
    return k;
}

u64 euler_phi(u64 r) {
    u64 res = r, m = r;
    for (u64 p = 2; p * p <= m; ++p)
        if (m % p == 0) { while (m % p == 0) m /= p; res -= res / p; }
    if (m > 1) res -= res / m;
    return res;
}

// Multiply two polynomials mod (x^r - 1) with coefficients mod n.
std::vector<u64> polymul_mod(const std::vector<u64>& A, const std::vector<u64>& B, u64 r, u64 n) {
    std::vector<u64> C(r, 0);
    for (u64 i = 0; i < r; ++i) {
        if (!A[i]) continue;
        for (u64 j = 0; j < r; ++j) {
            if (!B[j]) continue;
            const u64 k = (i + j) % r;
            C[k]        = (C[k] + mulmod(A[i], B[j], n)) % n;
        }
    }
    return C;
}

// (x + a)^n mod (x^r - 1, n).
std::vector<u64> poly_pow_xa(u64 a, u64 n, u64 r) {
    std::vector<u64> result(r, 0);
    result[0] = 1 % n; // 1
    std::vector<u64> base(r, 0);
    base[0]              = a % n;
    base[1 % r]          = (base[1 % r] + 1) % n; // x  (if r==1, wraps to index 0)
    u64 e                = n;
    while (e) {
        if (e & 1) result = polymul_mod(result, base, r, n);
        base = polymul_mod(base, base, r, n);
        e >>= 1;
    }
    return result;
}

} // namespace

PellSolution chakravala(u64 N) {
    const u64 s = isqrt_u64(N);
    if (s * s == N) return {0, 0, false}; // perfect square: no nontrivial solution

    i128 a = static_cast<i128>(s);
    i128 b = 1;
    i128 k = a * a - static_cast<i128>(N); // a^2 - N b^2 = k

    int guard = 0;
    while (k != 1 && ++guard < 100000) {
        i128 absk = k < 0 ? -k : k;
        // m ≡ -a * b^{-1} (mod absk), chosen nearest to sqrt(N), m >= 1.
        i128 m;
        if (absk == 1) {
            m = static_cast<i128>(s); // free choice: nearest to sqrt(N)
            if (m < 1) m = 1;
        } else {
            i128 binv = invmod(((b % absk) + absk) % absk, absk);
            i128 r0   = ((-a % absk + absk) % absk) * binv % absk; // residue class
            // pick m = r0 + t*absk closest to s, with m >= 1
            i128 t    = (static_cast<i128>(s) - r0) / absk;
            i128 best = 0;
            i128 bestval = -1;
            for (i128 dt = -1; dt <= 1; ++dt) {
                i128 cand = r0 + (t + dt) * absk;
                if (cand < 1) continue;
                i128 diff = cand * cand - static_cast<i128>(N);
                if (diff < 0) diff = -diff;
                if (bestval < 0 || diff < bestval) { bestval = diff; best = cand; }
            }
            if (bestval < 0) best = r0 >= 1 ? r0 : r0 + absk; // fallback
            m = best;
        }
        const i128 a_new = (a * m + static_cast<i128>(N) * b) / absk;
        const i128 b_new = (a + b * m) / absk;
        const i128 k_new = (m * m - static_cast<i128>(N)) / k;
        a = a_new < 0 ? -a_new : a_new;
        b = b_new < 0 ? -b_new : b_new;
        k = k_new;
    }
    return {a, b, k == 1};
}

u64 lenstra_ecm(u64 n, u64 seed) {
    if (n <= 1) return n;
    if (n % 2 == 0) return 2;
    for (u64 p : {3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull, 41ull, 43ull, 47ull})
        if (n % p == 0) return p;

    XorShift        rng(seed);
    const u64       B         = 300;   // smoothness bound (stage 1)
    const int       maxTrials = 3000;
    // Sieve primes up to B once.
    std::vector<u64> primes;
    std::vector<bool> comp(B + 1, false);
    for (u64 p = 2; p <= B; ++p)
        if (!comp[p]) { primes.push_back(p); for (u64 q = p * p; q <= B; q += p) comp[q] = true; }

    for (int t = 0; t < maxTrials; ++t) {
        const i128 x0 = static_cast<i128>(rng.next() % n);
        const i128 y0 = static_cast<i128>(rng.next() % n);
        const i128 a  = static_cast<i128>(rng.next() % n);
        // b chosen so (x0, y0) lies on the curve (reduce each term mod n first).
        const i128 y2 = y0 * y0 % n;
        const i128 x3 = x0 * x0 % n * x0 % n;
        const i128 ax = a * x0 % n;
        const i128 b  = ((y2 - x3 - ax) % n + 3 * n) % n;
        // Skip singular curves: 4a^3 + 27b^2 == 0 (mod n).
        const i128 a3   = a * a % n * a % n;
        const i128 b2   = b * b % n;
        const i128 disc = ((4 * a3 % n + 27 * b2 % n) % n + n) % n;
        u64        g    = gcd_u64(static_cast<u64>(((disc % n) + n) % n), n);
        if (g == n) continue;
        if (g > 1) return g;

        Point P{x0, y0, false};
        u64   factor = 0;
        for (u64 q : primes) {
            u64 qe = q;
            while (qe <= B / q) qe *= q; // largest power of q not exceeding B
            P = ec_mul(P, qe, a, static_cast<i128>(n), factor);
            if (factor) break;
        }
        if (factor > 1 && factor < n) return factor;
    }
    return n; // failure
}

bool aks_is_prime(u64 n) {
    if (n < 2) return false;
    if (n < 4) return true; // 2, 3
    if (n % 2 == 0) return false;
    // Step 1: perfect power => composite.
    if (is_perfect_power(n)) return false;

    const double log2n = std::log2(static_cast<double>(n));
    const u64    limit = static_cast<u64>(log2n * log2n) + 1;

    // Step 2: find the smallest r with ord_r(n) > log2(n)^2.
    u64 r = 2;
    for (;; ++r) {
        if (gcd_u64(n % r, r) != 1) continue; // ensure invertibility for order
        const u64 ord = mult_order(n, r);
        if (ord > limit) break;
        if (r >= n) break;
    }

    // Step 3: small gcd checks.
    for (u64 a = 2; a <= r && a < n; ++a) {
        const u64 g = gcd_u64(a, n);
        if (g > 1 && g < n) return false;
    }

    // Step 4: if n <= r, n is prime.
    if (n <= r) return true;

    // Step 5: polynomial congruence (x + a)^n == x^n + a in (Z/n)[x]/(x^r - 1).
    const u64 phi_r = euler_phi(r);
    const u64 amax  = static_cast<u64>(std::sqrt(static_cast<double>(phi_r)) * log2n);
    for (u64 a = 1; a <= amax; ++a) {
        std::vector<u64> lhs = poly_pow_xa(a, n, r);
        // rhs = x^(n mod r) + a
        std::vector<u64> rhs(r, 0);
        rhs[n % r] = (rhs[n % r] + 1) % n;
        rhs[0]     = (rhs[0] + a) % n;
        if (lhs != rhs) return false;
    }
    return true;
}

} // namespace datamunge::algorithms
