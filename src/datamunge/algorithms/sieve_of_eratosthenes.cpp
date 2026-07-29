#include <datamunge/algorithms/sieve_of_eratosthenes.hpp>

namespace datamunge::algorithms {

std::vector<bool> prime_sieve(std::uint64_t n) {
    if (n < 2) return {}; // no primes below 2

    std::vector<bool> is_prime(n + 1, true);
    is_prime[0] = false;
    is_prime[1] = false;

    // Cross out multiples of each prime p, starting at p*p (smaller multiples of p already carry a
    // smaller prime factor and were struck earlier). Once p*p > n every composite <= n has been
    // marked, since such a composite has a prime factor <= sqrt(n); hence the outer loop stops
    // there. p*p is computed in 64-bit and guarded so it cannot overflow the loop condition.
    for (std::uint64_t p = 2; p * p <= n; ++p) {
        if (!is_prime[p]) continue;
        for (std::uint64_t m = p * p; m <= n; m += p) is_prime[m] = false;
    }
    return is_prime;
}

std::vector<std::uint64_t> primes_up_to(std::uint64_t n) {
    const std::vector<bool> is_prime = prime_sieve(n);
    std::vector<std::uint64_t> primes;
    for (std::uint64_t i = 2; i < is_prime.size(); ++i)
        if (is_prime[i]) primes.push_back(i);
    return primes;
}

} // namespace datamunge::algorithms
