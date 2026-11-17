#pragma once
/// @file sqrt_digits.hpp
/// @brief Exact arbitrary-precision paired-digit square-root extraction.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief SqrtDigitsOptions owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SqrtDigitsOptions value{};
/// std::print("{}\n",value.max_fractional_digits); // 4096
/// @endcode
struct SqrtDigitsOptions {
  std::size_t max_fractional_digits{4096}; ///< Requested fractional count; zero disables.
  std::size_t max_history_characters{
      1000000}; ///< Actual copied root/remainder decimal characters plus records; zero disables.
};
/// @brief SqrtDigit owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SqrtDigit value{};
/// std::print("{}\n",value.digit); // 0
/// @endcode
struct SqrtDigit {
  std::size_t step{};    ///< Actual zero-based integer/fractional digit step.
  int         pair{};    ///< Actual brought-down pair, zero for fractional extension.
  int         digit{};   ///< Actual largest fitting candidate0..9.
  std::string prefix;    ///< Actual nonnegative integer root prefix after this step, no point.
  std::string remainder; ///< Actual exact integer remainder after subtraction.
};
/// @brief SqrtDigitsRun owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SqrtDigitsRun value{};
/// std::print("{}\n",value.candidate_tests); // 0
/// @endcode
struct SqrtDigitsRun {
  std::string digits; ///< Actual truncated expansion, retaining point for zero fractional digits.
  std::size_t candidate_tests{};  ///< Actual descending digit product comparisons.
  std::vector<SqrtDigit> history; ///< Optional actual root/remainder transitions.
};
/// @brief Extract exact truncated square-root digits.
/// Processes decimal input pairs from left to right, then zero pairs. For current root R and brought-down remainder M, choose greatest d in0..9 with (20R+d)d<=M, subtract and append d. Arbitrary-precision integers prevent historical uint64 overflow for full-width radicands or long prefixes. After k fractional digits the point-free integer output is floor(sqrt(n)*10^k); no rounding is performed. The decimal point remains even at zero requested digits. Input n=0 produces zero with requested zeros. At most10 candidate comparisons per emitted digit, each using growing big integers; bit complexity is not constant per digit. Owned outputs, no shared state. Intrinsic fractional cap100000; optional histories separately bounded by copied decimal characters.
/// @param n Complete unsigned64-bit radicand.
/// @param fractional_digits Nonnegative requested count after the point.
/// @param options Complete fractional digit and optional copied-history policies; zero disables policy.
/// @return Owned exact truncated expansion, actual comparison count and optional states.
/// @throws std::invalid_argument Negative fractional digit count.
/// @throws std::length_error Intrinsic, digit, captured-character or container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sqrt_digits_run(2,5);
/// std::print("{}\n",r.digits); // 1.41421
/// @endcode
/// @par References
/// - Classical digit-by-digit square-root algorithm, derived by (10R+d)^2=100R^2+(20R+d)d.
/// @see digit_by_digit_sqrt()
SqrtDigitsRun
sqrt_digits_run(std::uint64_t n, int fractional_digits, const SqrtDigitsOptions& options = {});
/// @brief Extract exact truncated square-root digits with actual prefix/remainder states.
/// Processes decimal input pairs from left to right, then zero pairs. For current root R and brought-down remainder M, choose greatest d in0..9 with (20R+d)d<=M, subtract and append d. Arbitrary-precision integers prevent historical uint64 overflow for full-width radicands or long prefixes. After k fractional digits the point-free integer output is floor(sqrt(n)*10^k); no rounding is performed. The decimal point remains even at zero requested digits. Input n=0 produces zero with requested zeros. At most10 candidate comparisons per emitted digit, each using growing big integers; bit complexity is not constant per digit. Owned outputs, no shared state. Intrinsic fractional cap100000; optional histories separately bounded by copied decimal characters.
/// @param n Complete unsigned64-bit radicand.
/// @param fractional_digits Nonnegative requested count after the point.
/// @param options Complete fractional digit and optional copied-history policies; zero disables policy.
/// @return Owned exact truncated expansion, actual comparison count and optional states.
/// @throws std::invalid_argument Negative fractional digit count.
/// @throws std::length_error Intrinsic, digit, captured-character or container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sqrt_digits_trace(2,5);
/// std::print("{}\n",r.digits); // 1.41421
/// @endcode
/// @par References
/// - Classical digit-by-digit square-root algorithm, derived by (10R+d)^2=100R^2+(20R+d)d.
/// @see digit_by_digit_sqrt()
SqrtDigitsRun
sqrt_digits_trace(std::uint64_t n, int fractional_digits, const SqrtDigitsOptions& options = {});
/// @brief Return the preserved paired-digit square-root string.
/// Same exact arbitrary-precision producer and default bounds; fractional digits are truncated and point is retained.
/// @param n Complete unsigned64-bit radicand.
/// @param frac_digits Nonnegative fractional count, at most4096 by default.
/// @return Owned exact truncated decimal.
/// @throws std::invalid_argument Negative count.
/// @throws std::length_error Default geometry/history capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::digit_by_digit_sqrt(144,4)); // 12.0000
/// @endcode
/// @see sqrt_digits_run()
std::string digit_by_digit_sqrt(std::uint64_t n, int frac_digits);
} // namespace datamunge::algorithms
