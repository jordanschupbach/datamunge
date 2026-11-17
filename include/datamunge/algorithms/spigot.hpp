#pragma once
/// @file spigot.hpp
/// @brief Actual factorial-base e spigot with a rational tail certificate.
#include <array>
#include <cstddef>
#include <string>
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief SpigotOptions owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SpigotOptions value{};
/// std::print("{}\n",value.max_digits); // 4096
/// @endcode
struct SpigotOptions {
  std::size_t max_digits{4096}; ///< Requested fractional digits; zero disables policy.
  std::size_t max_column_updates{
      20000000}; ///< Actual mixed-radix multiply/divide column visits; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Captured digit records plus copied column values; zero disables.
};
/// @brief SpigotDigit owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SpigotDigit value{};
/// std::print("{}\n",value.digit); // 0
/// @endcode
struct SpigotDigit {
  std::size_t index{}; ///< Zero-based actual emitted fractional digit.
  int         digit{}; ///< Actual carry leaving column two.
  std::vector<int>
      remainders; ///< Actual factorial-base state after this emitted digit; slots0/1 unused.
};
/// @brief SpigotRun owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SpigotRun value{};
/// std::print("{}\n",value.certified); // false
/// @endcode
struct SpigotRun {
  std::string
      digits; ///< Actual certified truncated e expansion, always beginning2. including zero-digit output.
  std::size_t columns{}; ///< Actual highest mixed-radix column, requested digits+20.
  std::size_t
      column_updates{}; ///< Actual inner column visits, excludes separate big-integer certificate work.
  bool certified{};     ///< Exact lower/upper rational floors agree with emitted digits.
  std::array<std::size_t, 10>
      frequencies{}; ///< Actual counts of all decimal digits including the leading2, excluding the point.
  std::vector<SpigotDigit> history; ///< Optional actual outer emission states.
};
/// @brief Emit and certify truncated decimal digits of e.
/// Initialize columns2..m to1, m=n_digits+20. Each decimal emission multiplies the state by10 and carries backward through bases m..2. All emitted carries must be decimal digits. A separate exact big-integer certificate forms S_m=sum(k=0..m)1/k!, bounds e strictly between S_m and S_m+1/(m*m!), and requires both scaled rational floors and the actual output prefix to agree. This certifies requested truncation rather than assuming guard columns suffice. Mixed-radix work O(d*(d+20)), storage O(d), optional complete states O(d^2); certificate has additional arbitrary-precision arithmetic/storage. Complete work/capture geometry checked before loops. Intrinsic digits cap100000 bounds small-column arithmetic; policies may be stricter. Owned output, no shared state.
/// @param n_digits Nonnegative requested fractional digit count.
/// @param options Complete digit/work/captured-value policies; zero disables policy.
/// @return Actual certified decimal stream, frequencies, selected work and optional state.
/// @throws std::invalid_argument Negative digit count.
/// @throws std::length_error Intrinsic/policy/container/work/history bound exceeded.
/// @throws std::runtime_error Rational certificate did not isolate the requested decimal prefix.
/// @par Example
/// @code
/// auto r=dm::algorithms::spigot_e_run(5);
/// std::print("{} {}\n",r.digits,r.certified); // 2.71828 true
/// @endcode
/// @par References
/// - A. H. J. Sale, The calculation of e to many significant digits, The Computer Journal11(2),1968, DOI10.1093/comjnl/11.2.229. This fixed-precision mixed-radix variant adds a rational certificate.
/// @see spigot_e()
SpigotRun spigot_e_run(int n_digits, const SpigotOptions& options = {});
/// @brief Emit and certify truncated decimal digits of e with actual factorial-base states.
/// Initialize columns2..m to1, m=n_digits+20. Each decimal emission multiplies the state by10 and carries backward through bases m..2. All emitted carries must be decimal digits. A separate exact big-integer certificate forms S_m=sum(k=0..m)1/k!, bounds e strictly between S_m and S_m+1/(m*m!), and requires both scaled rational floors and the actual output prefix to agree. This certifies requested truncation rather than assuming guard columns suffice. Mixed-radix work O(d*(d+20)), storage O(d), optional complete states O(d^2); certificate has additional arbitrary-precision arithmetic/storage. Complete work/capture geometry checked before loops. Intrinsic digits cap100000 bounds small-column arithmetic; policies may be stricter. Owned output, no shared state.
/// @param n_digits Nonnegative requested fractional digit count.
/// @param options Complete digit/work/captured-value policies; zero disables policy.
/// @return Actual certified decimal stream, frequencies, selected work and optional state.
/// @throws std::invalid_argument Negative digit count.
/// @throws std::length_error Intrinsic/policy/container/work/history bound exceeded.
/// @throws std::runtime_error Rational certificate did not isolate the requested decimal prefix.
/// @par Example
/// @code
/// auto r=dm::algorithms::spigot_e_trace(5);
/// std::print("{} {}\n",r.digits,r.certified); // 2.71828 true
/// @endcode
/// @par References
/// - A. H. J. Sale, The calculation of e to many significant digits, The Computer Journal11(2),1968, DOI10.1093/comjnl/11.2.229. This fixed-precision mixed-radix variant adds a rational certificate.
/// @see spigot_e()
SpigotRun spigot_e_trace(int n_digits, const SpigotOptions& options = {});
/// @brief Return the preserved decimal e spigot string.
/// Same actual producer/certificate with default policies; always retains the decimal point, even for zero digits.
/// @param n_digits Nonnegative fractional count, at most4096 by default.
/// @return Owned certified truncated decimal string.
/// @throws std::invalid_argument Negative count.
/// @throws std::length_error Default complete geometry/work bound exceeded.
/// @throws std::runtime_error Prefix certificate unresolved.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::spigot_e(5)); // 2.71828
/// @endcode
/// @see spigot_e_run()
std::string spigot_e(int n_digits);
} // namespace datamunge::algorithms
