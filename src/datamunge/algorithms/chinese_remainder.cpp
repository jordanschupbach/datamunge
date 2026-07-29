#include <datamunge/algorithms/chinese_remainder.hpp>

#include <stdexcept>

namespace datamunge::algorithms {

namespace {

// Extended Euclidean algorithm: returns g = gcd(a, b) and sets x, y with a*x + b*y = g.
// Inlined here so this translation unit is self-contained. Iterative to avoid recursion depth.
std::int64_t ext_gcd(std::int64_t a, std::int64_t b, std::int64_t& x, std::int64_t& y) {
    std::int64_t old_r = a, r = b;
    std::int64_t old_s = 1, s = 0;
    std::int64_t old_t = 0, t = 1;
    while (r != 0) {
        const std::int64_t q = old_r / r;
        std::int64_t tmp = old_r - q * r; old_r = r; r = tmp;
        tmp = old_s - q * s; old_s = s; s = tmp;
        tmp = old_t - q * t; old_t = t; t = tmp;
    }
    x = old_s;
    y = old_t;
    return old_r; // == a*x + b*y
}

// Normalize a residue into [0, m) for a positive modulus m.
std::int64_t normalize(std::int64_t r, std::int64_t m) {
    r %= m;
    if (r < 0) r += m;
    return r;
}

} // namespace

CrtSolution chinese_remainder(const std::vector<std::int64_t>& remainders,
                              const std::vector<std::int64_t>& moduli) {
    if (remainders.size() != moduli.size())
        throw std::invalid_argument("chinese_remainder: remainders and moduli must have equal length");
    for (const std::int64_t m : moduli)
        if (m <= 0)
            throw std::invalid_argument("chinese_remainder: every modulus must be strictly positive");

    // The empty system is vacuously satisfied by every integer: x == 0 (mod 1).
    CrtSolution acc{true, 0, 1};

    for (std::size_t i = 0; i < moduli.size(); ++i) {
        const std::int64_t m1 = acc.modulus;
        const std::int64_t r1 = acc.remainder;              // already in [0, m1)
        const std::int64_t m2 = moduli[i];
        const std::int64_t r2 = normalize(remainders[i], m2);

        // Merge  x == r1 (mod m1)  and  x == r2 (mod m2).
        std::int64_t p = 0, q = 0;
        const std::int64_t g = ext_gcd(m1, m2, p, q);       // m1*p + m2*q = g
        const std::int64_t diff = r2 - r1;
        if (diff % g != 0) {                                // gcd(m1,m2) must divide (r1 - r2)
            acc.solvable = false;
            acc.remainder = 0;
            acc.modulus = 0;
            return acc;
        }

        // Solution: x = r1 + m1 * t, where t == (diff/g) * (m1/g)^{-1} (mod m2/g). Because
        // m1*p == g (mod m2), p is that inverse scaled by g, so t == (diff/g) * p (mod m2/g).
        const std::int64_t md = m2 / g;                     // m2 / gcd
        std::int64_t t = static_cast<std::int64_t>(
            (static_cast<__int128>(diff / g) % md) * (p % md) % md); // 128-bit to avoid overflow
        if (t < 0) t += md;

        const __int128 lcm = static_cast<__int128>(m1) * md;         // = m1 * m2 / g = lcm(m1,m2)
        __int128 x = (static_cast<__int128>(r1) + static_cast<__int128>(m1) * t) % lcm;
        if (x < 0) x += lcm;

        acc.remainder = static_cast<std::int64_t>(x);
        acc.modulus = static_cast<std::int64_t>(lcm);
    }

    return acc;
}

} // namespace datamunge::algorithms
