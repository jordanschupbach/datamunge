#include <datamunge/algorithms/pollard_rho.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {

namespace {

// Modular multiply via a 128-bit intermediate: correct for any modulus up to 2^64-1, where a plain
// 64-bit product would overflow.
std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    return static_cast<std::uint64_t>(static_cast<unsigned __int128>(a) * b % m);
}

// base^exp mod m by square-and-multiply, each product reduced through mulmod.
std::uint64_t powmod(std::uint64_t base, std::uint64_t exp, std::uint64_t m) {
    std::uint64_t result = 1 % m;
    base %= m;
    while (exp) {
        if (exp & 1) result = mulmod(result, base, m);
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    return result;
}

// Deterministic Miller-Rabin. The twelve bases {2,3,...,37} are a proven witness set for every
// n < 3.3 * 10^24, hence for all 64-bit n: the test is exact, never merely probabilistic.
bool is_prime(std::uint64_t n) {
    static constexpr std::uint64_t small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (std::uint64_t p : small_primes) {
        if (n % p == 0) return n == p; // p divides n: prime iff n is p itself
    }
    if (n < 2) return false;

    // Write n - 1 = d * 2^s with d odd.
    std::uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }

    for (std::uint64_t a : small_primes) {
        std::uint64_t x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue; // a is not a witness
        bool composite = true;
        for (int r = 1; r < s; ++r) {
            x = mulmod(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false; // a witnesses that n is composite
    }
    return true;
}

// Recursively split m into primes, appending them (unsorted) to out.
void factor_recursive(std::uint64_t m, std::vector<std::uint64_t>& out) {
    if (m == 1) return;
    if (is_prime(m)) { out.push_back(m); return; }
    const std::uint64_t d = pollard_rho_factor(m);
    factor_recursive(d, out);
    factor_recursive(m / d, out);
}

} // namespace

std::uint64_t pollard_rho_factor(std::uint64_t n) {
    if (n < 2) return n;       // 0 or 1: nothing to split
    if (n % 2 == 0) return 2;  // even: 2 is a factor
    if (is_prime(n)) return n; // prime: no nontrivial factor (documented behavior)

    // Try deterministic constants c = 1, 2, 3, ... in g(x) = (x^2 + c) mod n until one exposes a
    // nontrivial gcd. Brent's variant keeps a saved value x, advances y in rounds whose length
    // doubles, and multiplies the differences |x - y| into a running product q so a single
    // gcd(q, n) certifies a whole block of steps at once.
    for (std::uint64_t c = 1; c < n; ++c) {
        const auto g = [n, c](std::uint64_t v) { return (mulmod(v, v, n) + c) % n; };

        std::uint64_t x = 2, y = 2, ys = 2, q = 1, d = 1, r = 1;
        constexpr std::uint64_t batch = 128;

        do {
            x = y;
            for (std::uint64_t i = 0; i < r; ++i) y = g(y);
            std::uint64_t k = 0;
            while (k < r && d == 1) {
                ys = y;
                const std::uint64_t steps = std::min(batch, r - k);
                for (std::uint64_t i = 0; i < steps; ++i) {
                    y = g(y);
                    q = mulmod(q, x < y ? y - x : x - y, n);
                }
                d = std::gcd(q, n);
                k += steps;
            }
            r *= 2;
        } while (d == 1);

        if (d == n) {
            // The batch folded several collisions together, collapsing the gcd all the way to n.
            // Replay the last block one step at a time to isolate a single-factor gcd.
            do {
                ys = g(ys);
                d = std::gcd(x < ys ? ys - x : x - ys, n);
            } while (d == 1);
        }

        if (d != 1 && d != n) return d; // nontrivial factor found
        // otherwise this c produced only trivial gcds; move on to the next constant
    }
    return n; // unreachable for composite n
}

std::vector<std::uint64_t> factorize(std::uint64_t n) {
    if (n == 0) throw std::invalid_argument("factorize: 0 has no prime factorization");

    std::vector<std::uint64_t> factors;
    // Peel powers of two directly (keeps the rho search on odd numbers), then recurse.
    while (n % 2 == 0) { factors.push_back(2); n /= 2; }
    factor_recursive(n, factors);
    std::sort(factors.begin(), factors.end());
    return factors;
}

} // namespace datamunge::algorithms
