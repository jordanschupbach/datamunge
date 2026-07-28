#include <datamunge/algorithms/index_calculus.hpp>

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
    while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; }
    return r;
}
u64 gcd_u64(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }

i128 extgcd(i128 a, i128 b, i128& x, i128& y) {
    if (b == 0) { x = 1; y = 0; return a; }
    i128 x1, y1;
    i128 g = extgcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}
// Inverse of a mod m (requires gcd(a, m) == 1).
u64 invmod(u64 a, u64 m) {
    i128 x, y;
    extgcd(static_cast<i128>(a % m), static_cast<i128>(m), x, y);
    return static_cast<u64>(((x % static_cast<i128>(m)) + static_cast<i128>(m)) % static_cast<i128>(m));
}

struct XorShift {
    u64 s;
    explicit XorShift(u64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

// Try to factor v over the factor base; returns true and fills exps if smooth.
bool smooth(u64 v, const std::vector<u64>& fb, std::vector<int>& exps) {
    exps.assign(fb.size(), 0);
    for (std::size_t i = 0; i < fb.size(); ++i)
        while (v % fb[i] == 0) { v /= fb[i]; ++exps[i]; }
    return v == 1;
}

// Solve the linear system A L = b (mod mod) by RREF, choosing pivots invertible
// modulo `mod`. Returns false if some variable stays unpivoted.
bool solve_mod(std::vector<std::vector<u64>> A, std::vector<u64> b, u64 mod, int nvars,
               std::vector<u64>& L) {
    const int rows = static_cast<int>(A.size());
    std::vector<int> where(nvars, -1);
    int              prow = 0;
    for (int col = 0; col < nvars && prow < rows; ++col) {
        int sel = -1;
        for (int r = prow; r < rows; ++r)
            if (gcd_u64(A[r][col] % mod, mod) == 1) { sel = r; break; }
        if (sel == -1) continue;
        std::swap(A[prow], A[sel]);
        std::swap(b[prow], b[sel]);
        where[col]     = prow;
        const u64 inv  = invmod(A[prow][col], mod);
        for (int c = 0; c < nvars; ++c) A[prow][c] = mulmod(A[prow][c], inv, mod);
        b[prow] = mulmod(b[prow], inv, mod);
        for (int r = 0; r < rows; ++r) {
            if (r == prow || A[r][col] == 0) continue;
            const u64 f = A[r][col];
            for (int c = 0; c < nvars; ++c)
                A[r][c] = (A[r][c] + mod - mulmod(f, A[prow][c], mod)) % mod;
            b[r] = (b[r] + mod - mulmod(f, b[prow], mod)) % mod;
        }
        ++prow;
    }
    L.assign(nvars, 0);
    for (int col = 0; col < nvars; ++col) {
        if (where[col] == -1) return false; // undetermined variable
        L[col] = b[where[col]];
    }
    return true;
}

} // namespace

DiscreteLogResult index_calculus(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t seed) {
    const u64 order = p - 1; // assume g is a generator
    if (h % p == 1) return {0, true};
    if (g % p == h % p) return {1, true};

    // Factor base of small primes <= B.
    const double ln = std::log(static_cast<double>(p));
    int          B  = static_cast<int>(3.0 * std::exp(0.5 * std::sqrt(ln * std::log(ln))));
    if (B < 30) B = 30;
    if (B > 4000) B = 4000;
    std::vector<u64>  fb;
    std::vector<bool> comp(B + 1, false);
    for (int q = 2; q <= B; ++q) {
        if (comp[q]) continue;
        for (int m = q * q; m <= B; m += q) comp[m] = true;
        fb.push_back(static_cast<u64>(q));
    }
    const int fbs = static_cast<int>(fb.size());

    XorShift rng(seed);

    // Collect smooth relations g^k ≡ prod fb^e  =>  sum e_i L_i ≡ k (mod order).
    std::vector<std::vector<u64>> A;
    std::vector<u64>              bvec;
    std::vector<int>              exps;
    const int                     want = fbs + 20;
    long                          attempts = 0;
    while (static_cast<int>(A.size()) < want && attempts++ < 4000000L) {
        const u64 k = 1 + rng.next() % (order - 1);
        const u64 v = powmod(g, k, p);
        if (smooth(v, fb, exps)) {
            std::vector<u64> row(fbs);
            for (int i = 0; i < fbs; ++i) row[i] = static_cast<u64>(exps[i]) % order;
            A.push_back(std::move(row));
            bvec.push_back(k % order);
        }
    }
    if (static_cast<int>(A.size()) < fbs) return {0, false};

    std::vector<u64> L;
    if (!solve_mod(A, bvec, order, fbs, L)) return {0, false};

    // Express log h via one more smooth relation h * g^s.
    for (long t = 0; t < 4000000L; ++t) {
        const u64 s = rng.next() % order;
        const u64 v = mulmod(h % p, powmod(g, s, p), p);
        if (smooth(v, fb, exps)) {
            i128 x = 0;
            for (int i = 0; i < fbs; ++i)
                x = (x + static_cast<i128>(exps[i]) * static_cast<i128>(L[i])) % static_cast<i128>(order);
            x = (x - static_cast<i128>(s) % static_cast<i128>(order) + static_cast<i128>(order)) % static_cast<i128>(order);
            const u64 xr = static_cast<u64>(x);
            if (powmod(g, xr, p) == h % p) return {xr, true};
        }
    }
    return {0, false};
}

} // namespace datamunge::algorithms
