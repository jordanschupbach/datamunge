/// @file sieve_of_eratosthenes.cpp
/// @brief One overflow-safe actual marking producer.
#include <datamunge/algorithms/sieve_of_eratosthenes.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Produce actual marks and optional history.
/// @param n Inclusive bound.
/// @param o Policies.
/// @param capture Retain crossing stages.
/// @return Actual owned output.
PrimeSieveRun run(std::uint64_t n, const PrimeSieveOptions& o, bool capture) {
  PrimeSieveRun r;
  if (o.max_n && n > o.max_n)
    throw std::length_error("sieve: bound");
  if (n < 2)
    return r;
  if (n >= std::numeric_limits<std::size_t>::max() || n + 1 > r.mask.max_size())
    throw std::length_error("sieve: mask capacity");
  r.mask.assign(static_cast<std::size_t>(n + 1), true);
  r.mask[0] = r.mask[1] = false;
  std::size_t records   = 0;
  for (std::uint64_t p = 2; p <= n / p; ++p) {
    if (!r.mask[p])
      continue;
    PrimeSieveStage stage;
    stage.prime = p;
    if (capture) {
      if (records == std::numeric_limits<std::size_t>::max() ||
          (o.max_records && records >= o.max_records))
        throw std::length_error("sieve: records");
      ++records;
    }
    for (std::uint64_t m = p * p;;) {
      if (r.strikes == std::numeric_limits<std::size_t>::max() ||
          (o.max_strikes && r.strikes >= o.max_strikes))
        throw std::length_error("sieve: strikes");
      ++r.strikes;
      if (r.mask[m])
        ++stage.newly_composite;
      r.mask[m] = false;
      if (capture) {
        if (records == std::numeric_limits<std::size_t>::max() ||
            (o.max_records && records >= o.max_records))
          throw std::length_error("sieve: records");
        ++records;
        stage.strikes.push_back(m);
      }
      if (n - m < p)
        break;
      m += p;
    }
    if (capture)
      r.stages.push_back(std::move(stage));
  }
  for (std::uint64_t i = 2; i <= n; ++i)
    if (r.mask[i])
      r.primes.push_back(i);
  return r;
}
} // namespace
PrimeSieveRun prime_sieve_run(std::uint64_t n, const PrimeSieveOptions& o) {
  return run(n, o, false);
}
PrimeSieveRun prime_sieve_trace(std::uint64_t n, const PrimeSieveOptions& o) {
  return run(n, o, true);
}
std::vector<bool> prime_sieve(std::uint64_t n) {
  return prime_sieve_run(n).mask;
}
std::vector<std::uint64_t> primes_up_to(std::uint64_t n) {
  return prime_sieve_run(n).primes;
}
} // namespace datamunge::algorithms
