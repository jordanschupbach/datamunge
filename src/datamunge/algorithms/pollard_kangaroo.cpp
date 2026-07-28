#include <datamunge/algorithms/pollard_kangaroo.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

namespace {

using u64  = std::uint64_t;
using u128 = unsigned __int128;

u64 mulmod(u64 a, u64 b, u64 m) { return static_cast<u64>(static_cast<u128>(a) * b % m); }

u64 powmod(u64 a, u64 e, u64 m) {
    u64 r = 1 % m;
    a %= m;
    while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; }
    return r;
}

u64 isqrt(u64 n) {
    if (n == 0) return 0;
    u64 x = static_cast<u64>(std::sqrt(static_cast<double>(n)));
    while (x > 0 && x > n / x) --x;
    while ((x + 1) <= n / (x + 1) && (x + 1) * (x + 1) <= n) ++x;
    return x;
}

struct XorShift {
    u64 s;
    explicit XorShift(u64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

} // namespace

KangarooResult pollard_kangaroo(u64 g, u64 h, u64 P, u64 a, u64 b, u64 seed) {
    if (b < a) return {0, false, 0};
    // Trivial endpoints first.
    if (powmod(g, a, P) == h) return {a, true, 0};
    if (powmod(g, b, P) == h) return {b, true, 0};

    const u64 w = b - a;
    const u64 m = std::max<u64>(2, isqrt(w));
    const int L = 32; // number of distinct jump sizes

    u64 total_jumps = 0;
    for (int attempt = 0; attempt < 8; ++attempt) {
        XorShift rng(seed + 0x9E3779B97F4A7C15ULL * static_cast<u64>(attempt + 1));
        // Jump sizes with mean ~ m, and their precomputed g^s.
        std::vector<u64> s(L), gs(L);
        for (int j = 0; j < L; ++j) {
            s[j]  = 1 + rng.next() % (2 * m);
            gs[j] = powmod(g, s[j], P);
        }

        const u64 N = 8 * m + 64; // tame jumps

        // Tame kangaroo: start at g^b, lay a trail of (point -> exponent).
        std::unordered_map<u64, u64> trail;
        trail.reserve(static_cast<std::size_t>(N) * 2);
        u64 y = powmod(g, b, P), dist = 0;
        trail[y] = b;
        for (u64 i = 0; i < N; ++i) {
            const int j = static_cast<int>(y % static_cast<u64>(L));
            y           = mulmod(y, gs[j], P);
            dist += s[j];
            trail[y] = b + dist;
            ++total_jumps;
        }

        // Wild kangaroo: start at h = g^x, walk until it lands on the trail.
        const u64 wbound = 20 * m + 256;
        u64       yw = h, dw = 0;
        for (u64 i = 0; i < wbound; ++i) {
            auto it = trail.find(yw);
            if (it != trail.end()) {
                const u64 texp = it->second; // = x + dw  (mod ord); here exact
                if (texp >= dw) {
                    const u64 x = texp - dw;
                    if (x >= a && x <= b && powmod(g, x, P) == h)
                        return {x, true, total_jumps};
                }
            }
            const int j = static_cast<int>(yw % static_cast<u64>(L));
            yw          = mulmod(yw, gs[j], P);
            dw += s[j];
            ++total_jumps;
        }
    }
    return {0, false, total_jumps};
}

} // namespace datamunge::algorithms
