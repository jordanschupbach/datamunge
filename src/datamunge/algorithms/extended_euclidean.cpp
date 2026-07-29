#include <datamunge/algorithms/extended_euclidean.hpp>

#include <stdexcept>

namespace datamunge::algorithms {

ExtendedGcd extended_gcd(std::int64_t a, std::int64_t b) {
    // Maintain the two invariants  old_s*a + old_t*b == old_r  and  s*a + t*b == r  throughout,
    // shrinking (old_r, r) by repeated remainder until r reaches zero. The Euclidean step
    //     (old_r, r) <- (r, old_r - q*r)   with q = old_r / r
    // is mirrored on the coefficient pairs, so when r hits 0 the surviving (old_r, old_s, old_t)
    // is exactly (gcd, x, y) with a*x + b*y == gcd.
    std::int64_t old_r = a, r = b;
    std::int64_t old_s = 1, s = 0;
    std::int64_t old_t = 0, t = 1;

    while (r != 0) {
        const std::int64_t q = old_r / r;
        std::int64_t tmp;
        tmp = old_r - q * r; old_r = r; r = tmp; // remainder step
        tmp = old_s - q * s; old_s = s; s = tmp; // coefficient of a
        tmp = old_t - q * t; old_t = t; t = tmp; // coefficient of b
    }

    // Truncating integer division can leave a negative gcd when an input is negative. Negating all
    // three terms preserves a*x + b*y == gcd while forcing the gcd non-negative (gcd(0, 0) = 0).
    if (old_r < 0) {
        old_r = -old_r;
        old_s = -old_s;
        old_t = -old_t;
    }
    return ExtendedGcd{old_r, old_s, old_t};
}

std::int64_t modular_inverse(std::int64_t a, std::int64_t m) {
    if (m <= 0)
        throw std::invalid_argument("modular_inverse: modulus m must be positive");

    const ExtendedGcd g = extended_gcd(a, m);
    if (g.gcd != 1)
        throw std::invalid_argument("modular_inverse: a and m are not coprime, so no inverse exists");

    // g.x satisfies a*g.x + m*g.y == 1, i.e. a*g.x == 1 (mod m). Fold g.x into the canonical
    // residue [0, m).
    std::int64_t inv = g.x % m;
    if (inv < 0) inv += m;
    return inv;
}

} // namespace datamunge::algorithms
