#include <datamunge/algorithms/berlekamp.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace {

using i64  = std::int64_t;
using u64  = std::uint64_t;
using u128 = unsigned __int128;

i64 mod(i64 a, i64 p) { return ((a % p) + p) % p; }
i64 mulmod(i64 a, i64 b, i64 p) { return static_cast<i64>(static_cast<u128>(mod(a, p)) * mod(b, p) % p); }

i64 powmod(i64 a, u64 e, i64 p) {
    i64 r = 1 % p;
    a     = mod(a, p);
    while (e) { if (e & 1) r = mulmod(r, a, p); a = mulmod(a, a, p); e >>= 1; }
    return r;
}

i64 invmod(i64 a, i64 p) { return powmod(a, static_cast<u64>(p - 2), p); }

void trim(BPoly& f) { while (!f.empty() && f.back() == 0) f.pop_back(); }
int  deg(const BPoly& f) { return static_cast<int>(f.size()) - 1; }

BPoly reduce(BPoly f, i64 p) {
    for (auto& c : f) c = mod(c, p);
    trim(f);
    return f;
}

BPoly mul(const BPoly& a, const BPoly& b, i64 p) {
    if (a.empty() || b.empty()) return {};
    BPoly r(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i])
            for (std::size_t j = 0; j < b.size(); ++j) r[i + j] = mod(r[i + j] + mulmod(a[i], b[j], p), p);
    trim(r);
    return r;
}

BPoly rem(BPoly a, const BPoly& b, i64 p) {
    a            = reduce(a, p);
    const i64 iv = invmod(b.back(), p);
    while (deg(a) >= deg(b) && !a.empty()) {
        const int sh = deg(a) - deg(b);
        const i64 c  = mulmod(a.back(), iv, p);
        for (std::size_t i = 0; i < b.size(); ++i) a[i + sh] = mod(a[i + sh] - mulmod(c, b[i], p), p);
        trim(a);
    }
    return a;
}

BPoly monic(BPoly f, i64 p) {
    f = reduce(f, p);
    if (f.empty()) return f;
    const i64 iv = invmod(f.back(), p);
    for (auto& c : f) c = mulmod(c, iv, p);
    return f;
}

BPoly gcd(BPoly a, BPoly b, i64 p) {
    a = reduce(a, p);
    b = reduce(b, p);
    while (!b.empty()) { BPoly r = rem(a, b, p); a = b; b = r; }
    return monic(a, p);
}

// base^e mod m.
BPoly powmod_poly(BPoly base, u64 e, const BPoly& m, i64 p) {
    BPoly r{1};
    base = rem(base, m, p);
    while (e) {
        if (e & 1) r = rem(mul(r, base, p), m, p);
        base = rem(mul(base, base, p), m, p);
        e >>= 1;
    }
    return reduce(r, p);
}

struct XorShift {
    u64 s;
    explicit XorShift(u64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

// Quotient f / g over GF(p).
BPoly quotient(const BPoly& f, const BPoly& g, i64 p) {
    BPoly a = reduce(f, p);
    BPoly q(a.size() >= g.size() ? a.size() - g.size() + 1 : 0, 0);
    const i64 iv = invmod(g.back(), p);
    while (deg(a) >= deg(g) && !a.empty()) {
        const int sh = deg(a) - deg(g);
        const i64 c  = mulmod(a.back(), iv, p);
        q[sh]        = c;
        for (std::size_t i = 0; i < g.size(); ++i) a[i + sh] = mod(a[i + sh] - mulmod(c, g[i], p), p);
        trim(a);
    }
    trim(q);
    return q;
}

} // namespace

std::vector<std::int64_t> berlekamp_roots(const BPoly& f_in, std::int64_t p, std::uint64_t seed) {
    BPoly f = monic(f_in, p);
    std::vector<i64> roots;
    if (deg(f) <= 0) return roots;

    // g = gcd(f, x^p - x) = product of the distinct linear factors (all roots).
    BPoly xp = powmod_poly(BPoly{0, 1}, static_cast<u64>(p), f, p); // x^p mod f
    if (xp.size() < 2) xp.resize(2, 0);
    xp[1]   = mod(xp[1] - 1, p); // x^p - x
    trim(xp);
    BPoly g = gcd(f, xp, p);
    if (deg(g) <= 0) return roots; // no roots in GF(p)

    XorShift rng(seed);
    // Split g into individual roots.
    std::vector<BPoly> stack{g};
    while (!stack.empty()) {
        BPoly cur = stack.back();
        stack.pop_back();
        const int d = deg(cur);
        if (d <= 0) continue;
        if (d == 1) { roots.push_back(mod(-cur[0], p)); continue; }
        // one split step
        const i64 delta = static_cast<i64>(rng.next() % static_cast<u64>(p));
        BPoly     xpd{delta, 1};
        BPoly     h = powmod_poly(xpd, static_cast<u64>((p - 1) / 2), cur, p);
        if (!h.empty()) h[0] = mod(h[0] - 1, p);
        trim(h);
        BPoly gg = gcd(cur, h, p);
        if (deg(gg) > 0 && deg(gg) < d) {
            stack.push_back(gg);
            stack.push_back(quotient(cur, gg, p));
        } else {
            stack.push_back(cur); // retry with a new delta
        }
    }
    std::sort(roots.begin(), roots.end());
    return roots;
}

} // namespace datamunge::algorithms
