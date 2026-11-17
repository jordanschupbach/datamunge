#pragma once
/// @file sieve_of_eratosthenes.hpp
/// @brief Bounded inclusive prime sieve with actual strike histories.
#include <cstddef>
#include <cstdint>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief PrimeSieveOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::PrimeSieveOptions value{};
/// std::print("{}\n",value.max_n); // 100000000
/// @endcode
struct PrimeSieveOptions {
  std::uint64_t max_n{100000000}; ///< Inclusive allocation bound; zero disables.
  std::size_t   max_strikes{
      1000000000}; ///< Executed composite assignments, including repeat marks; zero disables.
  std::size_t max_records{
      1000000}; ///< Captured actual prime stages and individual strikes; zero disables.
};
/// @brief PrimeSieveStage stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::PrimeSieveStage value{};
/// std::print("{}\n",value.prime); // 0
/// @endcode
struct PrimeSieveStage {
  std::uint64_t              prime{}; ///< Actual crossing prime.
  std::vector<std::uint64_t> strikes; ///< Actual visited multiples p*p,p*p+p,..., inclusive.
  std::size_t                newly_composite{}; ///< Previously true bits actually cleared.
};
/// @brief PrimeSieveRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::PrimeSieveRun value{};
/// std::print("{}\n",value.strikes); // 0
/// @endcode
struct PrimeSieveRun {
  std::vector<bool>
      mask; ///< Empty below two for legacy compatibility; otherwise n+1 primality bits.
  std::vector<std::uint64_t> primes;   ///< Actual ascending surviving integers.
  std::size_t strikes{};               ///< Actual mark attempts, including previously cleared bits.
  std::vector<PrimeSieveStage> stages; ///< Optional actual crossing stages.
};
/// @brief Produce an inclusive Sieve of Eratosthenes.
/// Starts each crossing prime at p*p and stops at p<=n/p, avoiding overflow. Every composite has a least prime factor no larger than its square root, so all and only composites are removed. O(n log log n) marking time, O(n) bit mask plus output storage; optional strike history adds actual work storage. Complete n+1 geometry checked before allocation. No shared state; outputs own storage. Zero disables policies only.
/// @param n Inclusive nonnegative upper bound.
/// @param options Allocation, actual strike and capture policies.
/// @return Owned actual mask, primes, work and optional crossing stages.
/// @throws std::length_error Policy or arithmetic/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::prime_sieve_run(10);
/// std::print("{} {}\n",r.primes,r.strikes); // [2, 3, 5, 7] 5
/// @endcode
/// @par References
/// - Hardy and Wright, An Introduction to the Theory of Numbers, 6th edition, 2008, chapter22.
/// @see primes_up_to()
PrimeSieveRun prime_sieve_run(std::uint64_t n, const PrimeSieveOptions& options = {});
/// @brief Produce an inclusive Sieve of Eratosthenes with actual crossing stages.
/// Starts each crossing prime at p*p and stops at p<=n/p, avoiding overflow. Every composite has a least prime factor no larger than its square root, so all and only composites are removed. O(n log log n) marking time, O(n) bit mask plus output storage; optional strike history adds actual work storage. Complete n+1 geometry checked before allocation. No shared state; outputs own storage. Zero disables policies only.
/// @param n Inclusive nonnegative upper bound.
/// @param options Allocation, actual strike and capture policies.
/// @return Owned actual mask, primes, work and optional crossing stages.
/// @throws std::length_error Policy or arithmetic/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::prime_sieve_trace(10);
/// std::print("{} {}\n",r.primes,r.strikes); // [2, 3, 5, 7] 5
/// @endcode
/// @par References
/// - Hardy and Wright, An Introduction to the Theory of Numbers, 6th edition, 2008, chapter22.
/// @see primes_up_to()
PrimeSieveRun prime_sieve_trace(std::uint64_t n, const PrimeSieveOptions& options = {});
/// @brief Return the preserved inclusive sieve mask.
/// Empty when n<2; otherwise n+1 bits, false at zero/one. Same producer and default policies.
/// @param n Inclusive upper bound.
/// @return Owned primality bits.
/// @throws std::length_error Default bound/work policy or capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prime_sieve(10).size()); // 11
/// @endcode
/// @see prime_sieve_run()
std::vector<bool> prime_sieve(std::uint64_t n);
/// @brief Return all primes at most n in increasing order.
/// Uses the same checked inclusive sieve with default policies.
/// @param n Inclusive upper bound.
/// @return Owned prime list, empty below two.
/// @throws std::length_error Default bound/work policy or capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::primes_up_to(10)); // [2, 3, 5, 7]
/// @endcode
/// @see prime_sieve_run()
std::vector<std::uint64_t> primes_up_to(std::uint64_t n);
} // namespace datamunge::algorithms
