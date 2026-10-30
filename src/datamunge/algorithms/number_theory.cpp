/// @file number_theory.cpp
/// @brief Binary GCD, Tonelli-Shanks, Karatsuba and baby-step giant-step.

#include <datamunge/algorithms/number_theory.hpp>
#include <datamunge/algorithms/pollard_rho.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

/// @brief Modular arithmetic and decimal-digit helpers.
namespace {

/// @brief (a * b) mod m via a 128-bit intermediate.
std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    return static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) % m);
}
/// @brief base^exp mod m by square-and-multiply.
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

/// @brief Little-endian decimal digits of @p s, without leading zeros.
std::vector<int> to_digits(const std::string& s) {
    std::vector<int> d;
    d.reserve(s.size());
    for (auto it = s.rbegin(); it != s.rend(); ++it) d.push_back(*it - '0');
    while (d.size() > 1 && d.back() == 0) d.pop_back();
    if (d.empty()) d.push_back(0);
    return d;
}

/// @brief The decimal string of little-endian digits @p d.
std::string from_digits(std::vector<int> d) {
    while (d.size() > 1 && d.back() == 0) d.pop_back();
    std::string s;
    s.reserve(d.size());
    for (auto it = d.rbegin(); it != d.rend(); ++it) s.push_back(static_cast<char>('0' + *it));
    return s;
}

/// @brief Digit-vector sum a + b.
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

/// @brief Digit-vector difference a - b, assuming a >= b (both normalized).
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

/// @brief Digit vector a times 10^k.
std::vector<int> shift_digits(const std::vector<int>& a, std::size_t k) { // multiply by 10^k
    if (a.size() == 1 && a[0] == 0) return a;
    std::vector<int> out(k, 0);
    out.insert(out.end(), a.begin(), a.end());
    return out;
}

/// @brief Schoolbook product of two digit vectors.
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

/// @brief Karatsuba product of two digit vectors.
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

BsgsTrace baby_step_giant_step_trace(std::uint64_t base, std::uint64_t target, std::uint64_t p,
                                     std::uint64_t order, std::uint64_t table_size, bool record) {
    if (p < 2) throw std::invalid_argument("baby_step_giant_step_trace: p must be at least 2");
    base %= p;
    target %= p;
    if (base == 0) throw std::invalid_argument("baby_step_giant_step_trace: base must be a unit mod p");
    BsgsTrace out;
    const std::uint64_t n = order == 0 ? p - 1 : order;
    std::uint64_t       m = table_size;
    if (m == 0) {
        m = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(n)));
        while (m * m < n) ++m;
    }
    m           = std::max<std::uint64_t>(1, std::min(m, n));
    out.order   = n;
    out.table_size = m;

    std::unordered_map<std::uint64_t, std::uint64_t> table;
    table.reserve(m * 2);
    std::uint64_t e = 1 % p;
    for (std::uint64_t j = 0; j < m; ++j) {
        table.emplace(e, j);
        if (record) out.baby.push_back(e);
        e = mulmod(e, base, p);
        ++out.baby_steps;
    }
    // e = base^m now; the giant stride is its inverse, base^{(p-1) - m mod (p-1)}.
    const std::uint64_t factor = powmod(base, (p - 1) - (m % (p - 1)), p);
    const std::uint64_t giants = (n + m - 1) / m;  // i = 0 .. giants-1 covers x < n
    std::uint64_t       gamma  = target;
    for (std::uint64_t i = 0; i < giants; ++i) {
        ++out.giant_steps;
        if (record) out.giant.push_back(gamma);
        auto it = table.find(gamma);
        if (it != table.end()) {
            out.exists      = true;
            out.giant_index = i;
            out.baby_index  = it->second;
            out.exponent    = i * m + it->second;
            return out;
        }
        gamma = mulmod(gamma, factor, p);
    }
    return out;
}

std::uint64_t element_order(std::uint64_t g, std::uint64_t p) {
    if (p < 2) throw std::invalid_argument("element_order: p must be at least 2");
    g %= p;
    if (g == 0) throw std::invalid_argument("element_order: g must be a unit mod p");
    if (p == 2) return 1;
    std::uint64_t order = p - 1;
    auto          factors = factorize(p - 1);
    factors.erase(std::unique(factors.begin(), factors.end()), factors.end());
    for (std::uint64_t q : factors)
        while (order % q == 0 && powmod(g, order / q, p) == 1) order /= q;
    return order;
}

std::uint64_t primitive_root(std::uint64_t p) {
    if (p < 2) throw std::invalid_argument("primitive_root: p must be at least 2");
    if (p == 2) return 1;
    for (std::uint64_t g = 2; g < p; ++g)
        if (element_order(g, p) == p - 1) return g;
    return 0;
}

} // namespace datamunge::algorithms
