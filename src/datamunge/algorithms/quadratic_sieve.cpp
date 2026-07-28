#include <datamunge/algorithms/quadratic_sieve.hpp>

#include <datamunge/algorithms/number_theory.hpp> // tonelli_shanks

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
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
u64 isqrt_u64(u64 n) {
    if (n == 0) return 0;
    u64 x = static_cast<u64>(std::sqrt(static_cast<double>(n)));
    while (x > 0 && x > n / x) --x;
    while ((x + 1) <= n / (x + 1) && (x + 1) * (x + 1) <= n) ++x;
    return x;
}
int legendre(u64 a, u64 p) {
    const u64 r = powmod(a % p, (p - 1) / 2, p);
    return r == 0 ? 0 : (r == 1 ? 1 : -1);
}

} // namespace

std::uint64_t quadratic_sieve(std::uint64_t n) {
    if (n % 2 == 0) return 2;
    const u64 root = isqrt_u64(n);
    if (root * root == n) return root; // perfect square

    // Factor base bound (generous, so the factor base has enough primes to yield
    // several independent congruences).
    const double ln  = std::log(static_cast<double>(n));
    double       Lb  = std::exp(0.5 * std::sqrt(ln * std::log(ln)));
    int          B   = static_cast<int>(Lb * 2.0);
    if (B < 60) B = 60;
    if (B > 6000) B = 6000;

    // Factor base: 2 and odd primes p <= B with (n | p) = 1. Any p dividing n is
    // returned directly as a factor.
    std::vector<u64> fb;
    fb.push_back(2);
    std::vector<bool> comp(B + 1, false);
    for (int p = 3; p <= B; p += 2) {
        if (comp[p]) continue;
        for (int q = p * p; q <= B; q += p) comp[q] = true;
        if (n % static_cast<u64>(p) == 0) return static_cast<u64>(p);
        if (legendre(n, static_cast<u64>(p)) == 1) fb.push_back(static_cast<u64>(p));
    }
    const int fbs = static_cast<int>(fb.size());

    const u64 s = root + 1; // ceil(sqrt(n))
    // Sieve roots for each odd prime.
    std::vector<std::array<u64, 2>> roots(fbs);
    std::vector<bool>               has_two_roots(fbs, false);
    for (int k = 1; k < fbs; ++k) { // skip fb[0] = 2
        const auto ms = tonelli_shanks(n % fb[k], fb[k]);
        roots[k]      = {ms.root, (fb[k] - ms.root) % fb[k]};
        has_two_roots[k] = ms.exists;
    }

    // Collected relations: x value, full exponent vector over fb, parity bitset.
    struct Rel {
        u64                 x;
        std::vector<int>    exps; // exponents over fb
    };
    std::vector<Rel>              rels;
    const int                     words = (fbs + 63) / 64;
    std::vector<std::vector<u64>> parity;      // parity bitset per relation
    std::vector<std::vector<u64>> pivotVec(fbs); // GF(2) pivots by column
    std::vector<std::vector<u64>> pivotComp(fbs);
    std::vector<bool>             pivotUsed(fbs, false);

    u64  M       = static_cast<u64>(fbs) * 100 + 3000;
    u64  offset  = 0;
    int  needed  = fbs + 25;
    long guard   = 0;

    while (static_cast<int>(rels.size()) < needed && guard++ < 120) {
        // Sieve a fresh window [s+offset, s+offset+M).
        std::vector<i128> q(M);
        for (u64 i = 0; i < M; ++i) {
            const u64 x = s + offset + i;
            q[i]        = static_cast<i128>(x) * x - static_cast<i128>(n);
        }
        // Divide out the factor-base primes at their roots.
        for (int k = 1; k < fbs; ++k) {
            if (!has_two_roots[k]) continue;
            const u64 p = fb[k];
            for (int rr = 0; rr < 2; ++rr) {
                const u64 r0 = roots[k][rr];
                // first x >= s+offset with x ≡ r0 (mod p)
                u64 start = s + offset;
                u64 rem   = start % p;
                u64 add   = (r0 + p - rem) % p;
                for (u64 i = add; i < M; i += p) {
                    while (q[i] % static_cast<i128>(p) == 0) q[i] /= static_cast<i128>(p);
                }
                if (roots[k][0] == roots[k][1]) break;
            }
        }
        // Divide out 2 (x odd => Q even).
        for (u64 i = 0; i < M; ++i)
            while (q[i] % 2 == 0 && q[i] != 0) q[i] /= 2;

        // Smooth positions have q[i] reduced to 1; recompute exact exponents.
        for (u64 i = 0; i < M && static_cast<int>(rels.size()) < needed; ++i) {
            if (q[i] != 1) continue;
            const u64  x  = s + offset + i;
            i128       qq = static_cast<i128>(x) * x - static_cast<i128>(n);
            std::vector<int> exps(fbs, 0);
            for (int k = 0; k < fbs; ++k)
                while (qq % static_cast<i128>(fb[k]) == 0) { qq /= static_cast<i128>(fb[k]); ++exps[k]; }
            if (qq != 1) continue; // not fully smooth (should not happen)

            // Parity bitset and GF(2) reduction with composition tracking.
            std::vector<u64> pv(words, 0), comp_(0);
            const int        ridx = static_cast<int>(rels.size());
            const int        cwords = (needed + 64) / 64 + 1;
            comp_.assign(cwords, 0);
            comp_[ridx / 64] |= (u64{1} << (ridx % 64));
            for (int k = 0; k < fbs; ++k)
                if (exps[k] & 1) pv[k / 64] ^= (u64{1} << (k % 64));

            // Reduce against existing pivots.
            for (int col = 0; col < fbs; ++col) {
                if (!(pv[col / 64] & (u64{1} << (col % 64)))) continue;
                if (pivotUsed[col]) {
                    for (int w = 0; w < words; ++w) pv[w] ^= pivotVec[col][w];
                    for (int w = 0; w < static_cast<int>(comp_.size()) && w < static_cast<int>(pivotComp[col].size()); ++w)
                        comp_[w] ^= pivotComp[col][w];
                } else {
                    // Register as a new pivot.
                    pivotVec[col]  = pv;
                    pivotComp[col] = comp_;
                    pivotUsed[col] = true;
                    goto added;
                }
            }
            // pv reduced to zero => linear dependency.
            {
                // Assemble the congruence of squares from comp_.
                u128 X = 1;
                std::vector<int> esum(fbs, 0);
                std::vector<int> members;
                for (int r = 0; r <= ridx; ++r)
                    if (comp_[r / 64] & (u64{1} << (r % 64))) members.push_back(r);
                for (int r : members) {
                    if (r == ridx) {
                        X = static_cast<u128>(X) * (x % n) % n;
                        for (int k = 0; k < fbs; ++k) esum[k] += exps[k];
                    } else {
                        X = static_cast<u128>(X) * (rels[r].x % n) % n;
                        for (int k = 0; k < fbs; ++k) esum[k] += rels[r].exps[k];
                    }
                }
                u128 Y = 1;
                for (int k = 0; k < fbs; ++k)
                    Y = static_cast<u128>(Y) * powmod(fb[k], static_cast<u64>(esum[k] / 2), n) % n;
                const u64 xx = static_cast<u64>(X % n), yy = static_cast<u64>(Y % n);
                const u64 d  = gcd_u64(xx >= yy ? xx - yy : yy - xx, n);
                if (d != 1 && d != n) return d;
            }
            added:;
            rels.push_back({x, std::move(exps)});
        }
        offset += M;
    }
    return n; // failed to factor within the budget
}

} // namespace datamunge::algorithms
