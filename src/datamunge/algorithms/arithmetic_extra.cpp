#include <datamunge/algorithms/arithmetic_extra.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
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

} // namespace

double goldschmidt_division(double numerator, double denominator) {
    const double s = denominator < 0.0 ? -1.0 : 1.0;
    const double d = std::fabs(denominator);

    int          e = 0;
    const double m = std::frexp(d, &e); // d = m * 2^e, m in [0.5, 1)
    double       N = std::ldexp(numerator, -e);
    double       D = m;
    // Iterate factor f = 2 - D, scaling both N and D; D -> 1 quadratically.
    for (int i = 0; i < 60; ++i) {
        const double f = 2.0 - D;
        N *= f;
        D *= f;
        if (std::fabs(D - 1.0) < 1e-16) break;
    }
    return s * N;
}

std::uint64_t montgomery_multiply(std::uint64_t a, std::uint64_t b, std::uint64_t n) {
    // n must be odd. R = 2^64. Compute n' = -n^{-1} mod R by Newton's iteration mod 2^k.
    std::uint64_t ninv = n; // n^{-1} mod 2^? , converges (n odd)
    for (int i = 0; i < 5; ++i) ninv *= 2 - n * ninv; // doubles correct bits each step (2,4,8,...,64)
    const std::uint64_t nprime = static_cast<std::uint64_t>(0) - ninv; // -n^{-1} mod 2^64

    auto redc = [&](__uint128_t T) -> std::uint64_t {
        const std::uint64_t m = static_cast<std::uint64_t>(T) * nprime; // (T mod R) * n' mod R
        const __uint128_t   t = (T + static_cast<__uint128_t>(m) * n) >> 64;
        return static_cast<std::uint64_t>(t) >= n ? static_cast<std::uint64_t>(t) - n
                                                  : static_cast<std::uint64_t>(t);
    };

    // R^2 mod n, then two Montgomery multiplications give a*b mod n.
    const std::uint64_t Rmod = static_cast<std::uint64_t>((static_cast<__uint128_t>(1) << 64) % n);
    const std::uint64_t R2   = mulmod(Rmod, Rmod, n);

    const std::uint64_t aR  = redc(static_cast<__uint128_t>(a % n) * R2); // a*R mod n
    return redc(static_cast<__uint128_t>(aR) * (b % n));                  // aR*b*R^{-1} = a*b mod n
}

long long cipolla_sqrt(std::uint64_t n, std::uint64_t p) {
    n %= p;
    if (p == 2) return static_cast<long long>(n);
    if (n == 0) return 0;
    if (powmod(n, (p - 1) / 2, p) != 1) return -1; // not a quadratic residue

    // Find a with a^2 - n a non-residue.
    std::uint64_t a = 0, w = 0;
    for (a = 1; a < p; ++a) {
        w = (mulmod(a, a, p) + p - n) % p;
        if (powmod(w, (p - 1) / 2, p) == p - 1) break;
    }

    // Exponentiate (a + sqrt(w)) to (p+1)/2 in F_{p^2}: elements (x + y*sqrt(w)).
    auto mul = [&](std::uint64_t x1, std::uint64_t y1, std::uint64_t x2, std::uint64_t y2,
                   std::uint64_t& rx, std::uint64_t& ry) {
        rx = (mulmod(x1, x2, p) + mulmod(mulmod(y1, y2, p), w, p)) % p;
        ry = (mulmod(x1, y2, p) + mulmod(x2, y1, p)) % p;
    };

    std::uint64_t rx = 1, ry = 0;         // result = 1
    std::uint64_t bx = a, by = 1;         // base = a + sqrt(w)
    std::uint64_t exp = (p + 1) / 2;
    while (exp) {
        if (exp & 1) { std::uint64_t nx, ny; mul(rx, ry, bx, by, nx, ny); rx = nx; ry = ny; }
        std::uint64_t nx, ny;
        mul(bx, by, bx, by, nx, ny);
        bx = nx; by = ny;
        exp >>= 1;
    }
    // ry should be 0; rx is the square root.
    return static_cast<long long>(rx);
}

std::vector<std::uint64_t> addition_chain(std::uint64_t e) {
    // Binary-method chain (always valid), used as-is for large e.
    auto binary_chain = [](std::uint64_t e) {
        std::vector<std::uint64_t> chain{1};
        if (e == 1) return chain;
        int hb = 63;
        while (!((e >> hb) & 1)) --hb;
        std::uint64_t cur = 1;
        for (int b = hb - 1; b >= 0; --b) {
            cur *= 2;
            chain.push_back(cur);
            if ((e >> b) & 1) { cur += 1; chain.push_back(cur); }
        }
        return chain;
    };

    if (e <= 1) return {1};
    if (e > 4096) return binary_chain(e); // optimal search only for small e

    // Iterative-deepening DFS for a shortest chain.
    std::vector<std::uint64_t>                 chain{1};
    std::vector<std::uint64_t>                 result;
    std::function<bool(int)>                   dfs = [&](int max_len) {
        const std::uint64_t last = chain.back();
        if (last == e) { result = chain; return true; }
        if (static_cast<int>(chain.size()) >= max_len) return false;
        // Optimistic bound: doubling each remaining step must be able to reach e.
        std::uint64_t reach = last;
        for (int k = static_cast<int>(chain.size()); k < max_len; ++k) reach <<= 1;
        if (reach < e) return false;
        // Try sums of pairs, larger first, keeping the chain strictly increasing.
        for (int i = static_cast<int>(chain.size()) - 1; i >= 0; --i)
            for (int j = i; j >= 0; --j) {
                const std::uint64_t s = chain[i] + chain[j];
                if (s <= last || s > e) continue;
                chain.push_back(s);
                if (dfs(max_len)) return true;
                chain.pop_back();
            }
        return false;
    };

    const auto fallback = binary_chain(e);
    for (int len = 1; len <= static_cast<int>(fallback.size()); ++len) {
        chain.assign(1, 1);
        if (dfs(len)) return result;
    }
    return fallback;
}

} // namespace datamunge::algorithms
