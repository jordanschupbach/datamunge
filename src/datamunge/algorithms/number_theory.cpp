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

} // namespace





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
