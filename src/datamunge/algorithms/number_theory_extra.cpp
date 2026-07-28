#include <datamunge/algorithms/number_theory_extra.hpp>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace datamunge::algorithms {

namespace {

std::uint64_t gcd_u(std::uint64_t a, std::uint64_t b) {
    while (b) {
        const std::uint64_t t = a % b;
        a                     = b;
        b                     = t;
    }
    return a;
}

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

// floor(sqrt(x)) for a 128-bit x.
std::uint64_t isqrt128(__uint128_t x) {
    if (x == 0) return 0;
    std::uint64_t r = static_cast<std::uint64_t>(std::sqrtl(static_cast<long double>(x)));
    while (static_cast<__uint128_t>(r) * r > x) --r;
    while (static_cast<__uint128_t>(r + 1) * (r + 1) <= x) ++r;
    return r;
}

std::vector<std::uint64_t> first_primes(int count) {
    std::vector<std::uint64_t> primes;
    for (std::uint64_t c = 2; static_cast<int>(primes.size()) < count; ++c) {
        bool prime = true;
        for (std::uint64_t p : primes) {
            if (p * p > c) break;
            if (c % p == 0) { prime = false; break; }
        }
        if (prime) primes.push_back(c);
    }
    return primes;
}

} // namespace

std::vector<std::uint64_t> trial_division_factorize(std::uint64_t n) {
    std::vector<std::uint64_t> factors;
    while (n % 2 == 0) { factors.push_back(2); n /= 2; }
    for (std::uint64_t d = 3; d * d <= n; d += 2)
        while (n % d == 0) { factors.push_back(d); n /= d; }
    if (n > 1) factors.push_back(n);
    return factors;
}

std::uint64_t fermat_factor(std::uint64_t n) {
    if (n % 2 == 0) return 2;
    std::uint64_t a = isqrt128(n);
    if (a * a < n) ++a;
    for (int iter = 0; iter < 5000000; ++iter, ++a) {
        const __uint128_t b2 = static_cast<__uint128_t>(a) * a - n;
        const std::uint64_t b = isqrt128(b2);
        if (static_cast<__uint128_t>(b) * b == b2) {
            const std::uint64_t f = a - b;
            if (f > 1 && f < n) return f; // a-b==1 means n is prime
            return n;
        }
    }
    return n;
}

std::uint64_t pollard_p_minus_1(std::uint64_t n, std::uint64_t bound) {
    if (n % 2 == 0) return 2;
    std::uint64_t a = 2;
    for (std::uint64_t j = 2; j <= bound; ++j) {
        a = powmod(a, j, n);
        // Test the gcd at every step: we want the *first* j where one prime's order divides the
        // accumulated exponent but the other's does not (a later j may make both smooth -> g == n).
        const std::uint64_t g = gcd_u(a == 0 ? n : a - 1, n);
        if (g > 1 && g < n) return g;
        if (g == n) return 0; // over-smooth: all factors collapsed at once
    }
    return 0;
}

std::uint64_t dixon_factor(std::uint64_t n, int base_size) {
    const std::vector<std::uint64_t> base = first_primes(base_size);
    const std::size_t                m    = base.size();

    // Collect smooth relations: x with x^2 mod n fully factoring over `base`.
    std::vector<std::uint64_t>        xs;      // the x values
    std::vector<std::vector<int>>     expo;    // full exponent vectors (for building Y)
    std::vector<std::uint64_t>        parity;  // exponent parity as a bitmask over base primes

    const std::uint64_t start = isqrt128(n) + 1;
    const std::size_t   want  = m + 1; // one more than the base -> a GF(2) dependency must exist
    for (std::uint64_t x = start; x < start + 2000000 && xs.size() < want; ++x) {
        std::uint64_t q = mulmod(x % n, x % n, n);
        if (q == 0) continue;
        std::uint64_t    r = q;
        std::vector<int> e(m, 0);
        for (std::size_t i = 0; i < m; ++i)
            while (r % base[i] == 0) { r /= base[i]; ++e[i]; }
        if (r != 1) continue; // not smooth over the factor base

        std::uint64_t mask = 0;
        for (std::size_t i = 0; i < m; ++i)
            if (e[i] & 1) mask |= (1ULL << i);
        xs.push_back(x);
        expo.push_back(e);
        parity.push_back(mask);
    }
    if (xs.size() < 2) return 0;

    // GF(2) elimination; `combo[r]` tracks which original relations XOR into row r.
    const std::size_t          rows = xs.size();
    std::vector<std::uint64_t> combo(rows, 0);
    for (std::size_t r = 0; r < rows; ++r) combo[r] = (1ULL << r);

    std::vector<std::uint64_t> mat = parity;
    std::size_t                row = 0;
    for (std::size_t col = 0; col < m && row < rows; ++col) {
        std::size_t piv = row;
        while (piv < rows && !((mat[piv] >> col) & 1)) ++piv;
        if (piv == rows) continue;
        std::swap(mat[piv], mat[row]);
        std::swap(combo[piv], combo[row]);
        for (std::size_t r = 0; r < rows; ++r)
            if (r != row && ((mat[r] >> col) & 1)) {
                mat[r] ^= mat[row];
                combo[r] ^= combo[row];
            }
        ++row;
    }

    // Any all-zero parity row with a nonempty combination is a square congruence; try each.
    for (std::size_t r = 0; r < rows; ++r) {
        if (mat[r] != 0 || combo[r] == 0) continue;
        std::vector<int> total(m, 0);
        std::uint64_t    X = 1;
        for (std::size_t k = 0; k < rows; ++k)
            if ((combo[r] >> k) & 1) {
                X = mulmod(X, xs[k] % n, n);
                for (std::size_t i = 0; i < m; ++i) total[i] += expo[k][i];
            }
        std::uint64_t Y = 1;
        for (std::size_t i = 0; i < m; ++i) Y = mulmod(Y, powmod(base[i], total[i] / 2, n), n);

        const std::uint64_t d = gcd_u(X > Y ? X - Y : Y - X, n);
        if (d > 1 && d < n) return d;
    }
    return 0;
}

} // namespace datamunge::algorithms
