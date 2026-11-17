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



} // namespace datamunge::algorithms
