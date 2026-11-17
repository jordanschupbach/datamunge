#pragma once
/// @file trial_division.hpp
/// @brief Checked trial division operations and producer diagnostics.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual trial-division result and remainder-test counters.
/// @par Example
/// @code
/// auto r=dm::algorithms::trial_division_work(12); // {2,2,3}
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
struct TrialDivisionWork {
  std::vector<std::uint64_t> factors;  ///< Prime factors with multiplicity.
  std::size_t                tests{0}; ///< Actual remainder tests, including unsuccessful tests.
  std::size_t                odd_candidates{0}; ///< Distinct odd candidates visited.
};
/// @brief Factor a positive integer by repeated division.
/// Stops using d<=n/d, avoiding d*d overflow. No primality shortcut; hard64-bit primes can hit the work cap. Inputs unchanged; O(sqrt(n)) time, O(log n) storage; calls independent.
/// @param n Positive uint64 integer;1 has no prime factors.
/// @param max_tests Positive remainder-test ceiling, default10000000.
/// @return Prime factors and actual work.
/// @throws std::invalid_argument n is zero.
/// @throws std::length_error Remainder-test ceiling exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::trial_division_work(360);
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
TrialDivisionWork trial_division_work(std::uint64_t n, std::size_t max_tests = 10000000);
/// @brief Return the checked trial-division producer factors.
/// @param n Positive integer,1 returns empty.
/// @return Prime factors.
/// @throws std::invalid_argument Zero input.
/// @throws std::length_error Default work ceiling exceeded.
/// @par Example
/// @code
/// auto f=dm::algorithms::trial_division_factorize(12);
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
std::vector<std::uint64_t> trial_division_factorize(std::uint64_t n);
} // namespace datamunge::algorithms
