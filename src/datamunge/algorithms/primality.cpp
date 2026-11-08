/// @file primality.cpp
/// @brief Implementations of the primality tests and sieves declared in primality.hpp.

#include <cstdint>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <datamunge/algorithms/primality.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief Modular arithmetic and helpers shared by the tests.
namespace {

/// @brief (a * b) mod m without overflow, via 128-bit arithmetic.
std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
  return static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) % m);
}
/// @brief base^exp mod m by square-and-multiply.
std::uint64_t powmod(std::uint64_t base, std::uint64_t exp, std::uint64_t m) {
  std::uint64_t r = 1 % m;
  base %= m;
  while (exp > 0) {
    if (exp & 1ULL)
      r = mulmod(r, base, m);
    base = mulmod(base, base, m);
    exp >>= 1;
  }
  return r;
}

/// @brief Distinct prime factors of x by trial division.
std::vector<std::uint64_t> distinct_prime_factors(std::uint64_t x) {
  std::vector<std::uint64_t> out;
  for (std::uint64_t d = 2; d * d <= x; ++d)
    if (x % d == 0) {
      out.push_back(d);
      while (x % d == 0)
        x /= d;
    }
  if (x > 1)
    out.push_back(x);
  return out;
}

/// @brief SplitMix64 step, for reproducible base selection without pulling in the random module.
std::uint64_t splitmix(std::uint64_t& state) {
  std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/// @brief Share ordinary/work/trace decisions and actual seeded round execution.
/// @param n Unsigned candidate, with zero-work small/even filters.
/// @param rounds Positive requested maximum, checked before filters.
/// @param seed Initial private SplitMix64 state.
/// @param trace Optional history being constructed, null for constant-storage calls.
/// @param max_entries Whole-record cap; zero disables optional quota.
/// @return Actual decision and last-round state; errors expose no partial trace.
FermatPrimalityOutcome fermat_primality_kernel(
    std::uint64_t         n,
    int                   rounds,
    std::uint64_t         seed,
    FermatPrimalityTrace* trace,
    std::size_t           max_entries) {
  if (rounds <= 0)
    throw std::invalid_argument("Fermat primality rounds must be positive");
  FermatPrimalityOutcome out;
  out.n                = n;
  out.seed             = seed;
  out.requested_rounds = rounds;
  if (n < 2)
    return out;
  if (n == 2 || n == 3) {
    out.probable_prime = true;
    out.stop           = FermatPrimalityStop::SmallPrime;
    return out;
  }
  if (n % 2 == 0) {
    out.stop = FermatPrimalityStop::Even;
    return out;
  }
  std::uint64_t state = seed;
  for (int r = 0; r < rounds; ++r) {
    if (trace && ((max_entries && trace->rows.size() >= max_entries) ||
                  trace->rows.size() == trace->rows.max_size()))
      throw std::length_error("Fermat primality capture record limit exceeded");
    const auto base = 2 + splitmix(state) % (n - 3);
    const auto row  = fermat_base_test(n, base);
    if (trace) {
      trace->rows.push_back(row);
      ++trace->captured_entries;
    }
    ++out.completed_rounds;
    out.last_base    = row.base;
    out.last_gcd     = row.common_divisor;
    out.last_residue = row.residue;
    if (!row.passes) {
      out.stop = FermatPrimalityStop::Witness;
      return out;
    }
  }
  out.probable_prime = true;
  out.stop           = FermatPrimalityStop::RoundLimit;
  return out;
}

} // namespace

bool fermat_probable_prime(std::uint64_t n, int rounds, std::uint64_t seed) {
  return fermat_probable_prime_work(n, rounds, seed).probable_prime;
}
FermatBaseResult fermat_base_test(std::uint64_t n, std::uint64_t base) {
  if (n < 2 || !base || base >= n)
    throw std::invalid_argument("Fermat base requires n>=2 and 1<=base<n");
  FermatBaseResult out;
  out.n              = n;
  out.base           = base;
  out.common_divisor = std::gcd(base, n);
  out.residue        = mod_pow(base, n - 1, n);
  out.passes         = out.residue == 1;
  return out;
}
FermatPrimalityOutcome fermat_probable_prime_work(std::uint64_t n, int rounds, std::uint64_t seed) {
  return fermat_primality_kernel(n, rounds, seed, nullptr, 0);
}
FermatPrimalityTrace fermat_probable_prime_trace(
    std::uint64_t n,
    int           rounds,
    std::uint64_t seed,
    std::size_t   max_entries) {
  FermatPrimalityTrace trace;
  trace.outcome = fermat_primality_kernel(n, rounds, seed, &trace, max_entries);
  return trace;
}
FermatBaseCensus fermat_base_census(
    std::uint64_t n,
    std::uint64_t first,
    std::uint64_t last,
    std::uint64_t max_bases) {
  if (n < 2 || !first || first > last || last > n)
    throw std::invalid_argument("Fermat census requires n>=2 and 1<=first<=last<=n");
  const auto count = last - first;
  if (max_bases && count > max_bases)
    throw std::length_error("Fermat census work limit exceeded");
  FermatBaseCensus out;
  out.n     = n;
  out.first = first;
  out.last  = last;
  for (auto base = first; base < last; ++base) {
    const auto r = fermat_base_test(n, base);
    ++out.total;
    if (r.common_divisor == 1)
      ++out.coprime;
    if (r.passes)
      ++out.passes;
  }
  return out;
}
std::string_view fermat_primality_stop_name(FermatPrimalityStop stop) noexcept {
  switch (stop) {
    case FermatPrimalityStop::BelowTwo:
      return "below two";
    case FermatPrimalityStop::SmallPrime:
      return "small prime";
    case FermatPrimalityStop::Even:
      return "even";
    case FermatPrimalityStop::Witness:
      return "witness";
    case FermatPrimalityStop::RoundLimit:
      return "rounds passed";
  }
  return "unknown";
}

bool lucas_primality_test(std::uint64_t n) {
  if (n < 2)
    return false;
  if (n == 2)
    return true;
  if (n % 2 == 0)
    return false;

  const std::vector<std::uint64_t> factors = distinct_prime_factors(n - 1);
  for (std::uint64_t a = 2; a < n; ++a) {
    if (powmod(a, n - 1, n) != 1)
      continue; // a must satisfy Fermat's congruence first
    bool witness = true;
    for (std::uint64_t q : factors)
      if (powmod(a, (n - 1) / q, n) == 1) { // order of a is a proper divisor of n-1
        witness = false;
        break;
      }
    if (witness)
      return true; // a has order exactly n-1 -> n is prime
  }
  return false; // no primitive root exists -> n is composite
}

std::vector<std::uint64_t> sundaram_primes(std::uint64_t limit) {
  std::vector<std::uint64_t> primes;
  if (limit < 2)
    return primes;
  primes.push_back(2);
  if (limit == 2)
    return primes;

  const std::uint64_t k = (limit - 1) / 2; // we test odd numbers 2i+1 for 1 <= i <= k
  std::vector<bool>   marked(k + 1, false);
  for (std::uint64_t i = 1; i <= k; ++i)
    for (std::uint64_t j = i; i + j + 2 * i * j <= k; ++j)
      marked[i + j + 2 * i * j] = true;
  for (std::uint64_t i = 1; i <= k; ++i)
    if (!marked[i])
      primes.push_back(2 * i + 1);
  return primes;
}

} // namespace datamunge::algorithms
