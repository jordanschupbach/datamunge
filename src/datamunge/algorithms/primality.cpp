#include <datamunge/algorithms/primality.hpp>

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace {

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

// Distinct prime factors of x by trial division.
std::vector<std::uint64_t> distinct_prime_factors(std::uint64_t x) {
    std::vector<std::uint64_t> out;
    for (std::uint64_t d = 2; d * d <= x; ++d)
        if (x % d == 0) {
            out.push_back(d);
            while (x % d == 0) x /= d;
        }
    if (x > 1) out.push_back(x);
    return out;
}

// SplitMix64 step, for reproducible base selection without pulling in the random module.
std::uint64_t splitmix(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z              = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z              = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

} // namespace

bool fermat_probable_prime(std::uint64_t n, int rounds, std::uint64_t seed) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    std::uint64_t state = seed;
    for (int r = 0; r < rounds; ++r) {
        const std::uint64_t a = 2 + splitmix(state) % (n - 3); // base in [2, n-2]
        // Fermat's congruence: a^(n-1) must be 1 mod a prime. Any a that fails (including one
        // sharing a factor with n, for which the power is never 1) proves n composite.
        if (powmod(a, n - 1, n) != 1) return false;
    }
    return true; // probable prime for every base tried
}

bool lucas_primality_test(std::uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    const std::vector<std::uint64_t> factors = distinct_prime_factors(n - 1);
    for (std::uint64_t a = 2; a < n; ++a) {
        if (powmod(a, n - 1, n) != 1) continue; // a must satisfy Fermat's congruence first
        bool witness = true;
        for (std::uint64_t q : factors)
            if (powmod(a, (n - 1) / q, n) == 1) { // order of a is a proper divisor of n-1
                witness = false;
                break;
            }
        if (witness) return true; // a has order exactly n-1 -> n is prime
    }
    return false; // no primitive root exists -> n is composite
}

std::vector<std::uint64_t> sundaram_primes(std::uint64_t limit) {
    std::vector<std::uint64_t> primes;
    if (limit < 2) return primes;
    primes.push_back(2);
    if (limit == 2) return primes;

    const std::uint64_t     k = (limit - 1) / 2; // we test odd numbers 2i+1 for 1 <= i <= k
    std::vector<bool>       marked(k + 1, false);
    for (std::uint64_t i = 1; i <= k; ++i)
        for (std::uint64_t j = i; i + j + 2 * i * j <= k; ++j) marked[i + j + 2 * i * j] = true;
    for (std::uint64_t i = 1; i <= k; ++i)
        if (!marked[i]) primes.push_back(2 * i + 1);
    return primes;
}

std::vector<std::uint64_t> atkin_primes(std::uint64_t limit) {
    std::vector<std::uint64_t> primes;
    if (limit < 2) return primes;

    std::vector<bool> sieve(limit + 1, false);
    for (std::uint64_t x = 1; x * x <= limit; ++x)
        for (std::uint64_t y = 1; y * y <= limit; ++y) {
            std::uint64_t n = 4 * x * x + y * y;
            if (n <= limit && (n % 12 == 1 || n % 12 == 5)) sieve[n] = !sieve[n];
            n = 3 * x * x + y * y;
            if (n <= limit && n % 12 == 7) sieve[n] = !sieve[n];
            if (x > y) {
                n = 3 * x * x - y * y;
                if (n <= limit && n % 12 == 11) sieve[n] = !sieve[n];
            }
        }
    // Remove multiples of squares of primes.
    for (std::uint64_t r = 5; r * r <= limit; ++r)
        if (sieve[r])
            for (std::uint64_t m = r * r; m <= limit; m += r * r) sieve[m] = false;

    if (limit >= 2) primes.push_back(2);
    if (limit >= 3) primes.push_back(3);
    for (std::uint64_t n = 5; n <= limit; ++n)
        if (sieve[n]) primes.push_back(n);
    return primes;
}

} // namespace datamunge::algorithms
