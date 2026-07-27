#include <datamunge/algorithms/number_theory.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Overflow-safe modular multiply and exponentiation via 128-bit intermediates.
std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    return static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) % m);
}
std::uint64_t powmod(std::uint64_t base, std::uint64_t exp, std::uint64_t m) {
    std::uint64_t r = 1 % m;
    base %= m;
    while (exp > 0) {
        if (exp & 1ULL) r = mulmod(r, base, m);
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    return r;
}

// ---- big-integer helpers on little-endian base-10 digit vectors, for Karatsuba ----

std::vector<int> to_digits(const std::string& s) {
    std::vector<int> d;
    d.reserve(s.size());
    for (auto it = s.rbegin(); it != s.rend(); ++it) d.push_back(*it - '0');
    while (d.size() > 1 && d.back() == 0) d.pop_back();
    if (d.empty()) d.push_back(0);
    return d;
}

std::string from_digits(std::vector<int> d) {
    while (d.size() > 1 && d.back() == 0) d.pop_back();
    std::string s;
    s.reserve(d.size());
    for (auto it = d.rbegin(); it != d.rend(); ++it) s.push_back(static_cast<char>('0' + *it));
    return s;
}

std::vector<int> add_digits(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out;
    int              carry = 0;
    for (std::size_t i = 0; i < a.size() || i < b.size() || carry; ++i) {
        int s = carry;
        if (i < a.size()) s += a[i];
        if (i < b.size()) s += b[i];
        out.push_back(s % 10);
        carry = s / 10;
    }
    if (out.empty()) out.push_back(0);
    return out;
}

// a - b, assuming a >= b (both normalized).
std::vector<int> sub_digits(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out;
    int              borrow = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        int s = a[i] - borrow - (i < b.size() ? b[i] : 0);
        if (s < 0) {
            s += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push_back(s);
    }
    while (out.size() > 1 && out.back() == 0) out.pop_back();
    return out;
}

std::vector<int> shift_digits(const std::vector<int>& a, std::size_t k) { // multiply by 10^k
    if (a.size() == 1 && a[0] == 0) return a;
    std::vector<int> out(k, 0);
    out.insert(out.end(), a.begin(), a.end());
    return out;
}

std::vector<int> schoolbook(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<long long> acc(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            acc[i + j] += static_cast<long long>(a[i]) * b[j];
    std::vector<int> out(acc.size(), 0);
    long long        carry = 0;
    for (std::size_t i = 0; i < acc.size(); ++i) {
        long long cur = acc[i] + carry;
        out[i]        = static_cast<int>(cur % 10);
        carry         = cur / 10;
    }
    while (carry) {
        out.push_back(static_cast<int>(carry % 10));
        carry /= 10;
    }
    while (out.size() > 1 && out.back() == 0) out.pop_back();
    return out;
}

std::vector<int> karatsuba(const std::vector<int>& x, const std::vector<int>& y) {
    if (x.size() <= 32 || y.size() <= 32) return schoolbook(x, y);
    const std::size_t m = std::min(x.size(), y.size()) / 2;

    const std::vector<int> x0(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(m));
    const std::vector<int> x1(x.begin() + static_cast<std::ptrdiff_t>(m), x.end());
    const std::vector<int> y0(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(m));
    const std::vector<int> y1(y.begin() + static_cast<std::ptrdiff_t>(m), y.end());

    const std::vector<int> z0 = karatsuba(x0, y0);
    const std::vector<int> z2 = karatsuba(x1, y1);
    const std::vector<int> z1 =
        sub_digits(sub_digits(karatsuba(add_digits(x0, x1), add_digits(y0, y1)), z2), z0);

    // z2 * 10^(2m) + z1 * 10^m + z0
    std::vector<int> out = add_digits(shift_digits(z2, 2 * m), shift_digits(z1, m));
    out                  = add_digits(out, z0);
    return out;
}

} // namespace

std::uint64_t binary_gcd(std::uint64_t a, std::uint64_t b) {
    if (a == 0) return b;
    if (b == 0) return a;
    const int shift = std::countr_zero(a | b); // common factors of two
    a >>= std::countr_zero(a);
    do {
        b >>= std::countr_zero(b);
        if (a > b) std::swap(a, b);
        b -= a; // both odd here, so the difference is even
    } while (b != 0);
    return a << shift;
}

ModSqrt tonelli_shanks(std::uint64_t n, std::uint64_t p) {
    n %= p;
    if (n == 0) return ModSqrt{true, 0};
    if (p == 2) return ModSqrt{true, n & 1ULL};
    if (powmod(n, (p - 1) / 2, p) != 1) return ModSqrt{false, 0}; // Euler: n is a non-residue

    auto smaller = [&](std::uint64_t r) { return std::min(r, p - r); };
    if (p % 4 == 3) return ModSqrt{true, smaller(powmod(n, (p + 1) / 4, p))}; // direct shortcut

    // Write p - 1 = Q * 2^S with Q odd.
    std::uint64_t Q = p - 1;
    std::uint64_t S = 0;
    while ((Q & 1ULL) == 0) {
        Q >>= 1;
        ++S;
    }
    // A quadratic non-residue z.
    std::uint64_t z = 2;
    while (powmod(z, (p - 1) / 2, p) != p - 1) ++z;

    std::uint64_t M = S;
    std::uint64_t c = powmod(z, Q, p);
    std::uint64_t t = powmod(n, Q, p);
    std::uint64_t R = powmod(n, (Q + 1) / 2, p);
    while (t != 1) {
        std::uint64_t i  = 0;
        std::uint64_t tt = t;
        while (tt != 1) {
            tt = mulmod(tt, tt, p);
            ++i;
            if (i == M) return ModSqrt{false, 0}; // should not happen for prime p
        }
        std::uint64_t b = c;
        for (std::uint64_t j = 0; j + 1 < M - i; ++j) b = mulmod(b, b, p);
        M = i;
        c = mulmod(b, b, p);
        t = mulmod(t, c, p);
        R = mulmod(R, b, p);
    }
    return ModSqrt{true, smaller(R)};
}

std::string karatsuba_multiply(const std::string& a, const std::string& b) {
    return from_digits(karatsuba(to_digits(a), to_digits(b)));
}

DiscreteLog baby_step_giant_step(std::uint64_t base, std::uint64_t target, std::uint64_t p) {
    base %= p;
    target %= p;
    const std::uint64_t n = p - 1; // order of the multiplicative group of a prime field
    const std::uint64_t m = static_cast<std::uint64_t>(std::ceil(std::sqrt(static_cast<double>(n))));

    // Baby steps: base^j for j in [0, m), keeping the smallest j for each value.
    std::unordered_map<std::uint64_t, std::uint64_t> table;
    table.reserve(m * 2);
    std::uint64_t e = 1 % p;
    for (std::uint64_t j = 0; j < m; ++j) {
        table.emplace(e, j); // emplace keeps the first (smallest) j
        e = mulmod(e, base, p);
    }
    // Giant steps of size m: multiply target by base^{-m} = base^{n-m} each iteration.
    const std::uint64_t factor = powmod(base, n - (m % n), p);
    std::uint64_t       gamma  = target;
    for (std::uint64_t i = 0; i <= m; ++i) {
        auto it = table.find(gamma);
        if (it != table.end()) {
            const std::uint64_t x = i * m + it->second;
            if (powmod(base, x, p) == target) return DiscreteLog{true, x};
        }
        gamma = mulmod(gamma, factor, p);
    }
    return DiscreteLog{false, 0};
}

} // namespace datamunge::algorithms
