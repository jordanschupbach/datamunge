#pragma once
/// @file rounding_functions.hpp
/// @brief Explicit tie modes, checked observed rounding error and actual per-value histories.
#include <cstddef>
#include <string_view>
#include <vector>
/// @brief Canonical scalar rounding and descriptive summaries.
namespace datamunge::algorithms {
/// @brief Nearest-integer tie rules and directed integer rounding.
/// @par Example
/// @code
/// const auto kind=dm::algorithms::RoundMode::HalfToEven;
/// std::print("{}\n",static_cast<int>(kind)); // 0
/// @endcode
enum class RoundMode {
  HalfToEven,       ///< Exact represented halves select an even integer.
  HalfAwayFromZero, ///< Exact halves select the integer with larger magnitude.
  HalfUp,           ///< Exact halves select the integer toward positive infinity.
  HalfDown,         ///< Exact halves select the integer toward negative infinity.
  TowardZero,       ///< Truncate fractional magnitude.
  Ceil,             ///< Smallest represented integer not below the input.
  Floor             ///< Largest represented integer not above the input.
};
/// @brief Round one scalar using the HalfToEven rule.
/// Returns a floating integer without integer narrowing. NaN and infinities propagate unchanged; signed zero propagates unchanged. Finite input is classified by floor and its widened fractional distance, avoiding x+0.5 rounding before the tie decision. O(1), no mutation or shared state. Ties mean an exact represented fractional half, not a decimal textual approximation.
/// @param x Input scalar, including nonfinite values.
/// @return Rounded scalar in the original units.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::round_half_to_even(2.5)); // 2
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see round_with(), rounding_trace()
double round_half_to_even(double x);
/// @brief Round one scalar using the HalfAwayFromZero rule.
/// Returns a floating integer without integer narrowing. NaN and infinities propagate unchanged; signed zero propagates unchanged. Finite input is classified by floor and its widened fractional distance, avoiding x+0.5 rounding before the tie decision. O(1), no mutation or shared state. Ties mean an exact represented fractional half, not a decimal textual approximation.
/// @param x Input scalar, including nonfinite values.
/// @return Rounded scalar in the original units.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::round_half_away_from_zero(2.5)); // 3
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see round_with(), rounding_trace()
double round_half_away_from_zero(double x);
/// @brief Round one scalar using the HalfUp rule.
/// Returns a floating integer without integer narrowing. NaN and infinities propagate unchanged; signed zero propagates unchanged. Finite input is classified by floor and its widened fractional distance, avoiding x+0.5 rounding before the tie decision. O(1), no mutation or shared state. Ties mean an exact represented fractional half, not a decimal textual approximation.
/// @param x Input scalar, including nonfinite values.
/// @return Rounded scalar in the original units.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::round_half_up(2.5)); // 3
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see round_with(), rounding_trace()
double round_half_up(double x);
/// @brief Round one scalar using the HalfDown rule.
/// Returns a floating integer without integer narrowing. NaN and infinities propagate unchanged; signed zero propagates unchanged. Finite input is classified by floor and its widened fractional distance, avoiding x+0.5 rounding before the tie decision. O(1), no mutation or shared state. Ties mean an exact represented fractional half, not a decimal textual approximation.
/// @param x Input scalar, including nonfinite values.
/// @return Rounded scalar in the original units.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::round_half_down(2.5)); // 2
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see round_with(), rounding_trace()
double round_half_down(double x);
/// @brief Round one scalar using the TowardZero rule.
/// Returns a floating integer without integer narrowing. NaN and infinities propagate unchanged; signed zero propagates unchanged. Finite input is classified by floor and its widened fractional distance, avoiding x+0.5 rounding before the tie decision. O(1), no mutation or shared state. Ties mean an exact represented fractional half, not a decimal textual approximation.
/// @param x Input scalar, including nonfinite values.
/// @return Rounded scalar in the original units.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::round_toward_zero(2.5)); // 2
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see round_with(), rounding_trace()
double round_toward_zero(double x);
/// @brief Apply one explicitly selected integer rounding rule.
/// Supports exactly the seven RoundMode enumerators. Finite nearest modes classify the widened original input-minus-floor fraction; directed modes use native floor/ceil/trunc. Nonfinite values and signed zero propagate unchanged. O(1), reentrant, input unchanged.
/// @param x Input scalar.
/// @param mode Supported tie/direction rule.
/// @return Rounded floating scalar.
/// @throws std::invalid_argument Unsupported enum value.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::round_with(-.5,dm::algorithms::RoundMode::HalfUp)); // 0
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see round_mode_name()
double round_with(double x, RoundMode mode);
/// @brief Name one supported rounding rule.
/// Returns static borrowed text valid for the process lifetime; no allocation or mutable shared state.
/// @param mode Supported rule.
/// @return Stable static rule label.
/// @throws std::invalid_argument Unsupported enum value.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::round_mode_name(dm::algorithms::RoundMode::HalfToEven)); // half-to-even
/// @endcode
/// @see round_with()
std::string_view round_mode_name(RoundMode mode);
/// @brief One actual finite rounding decision and the running signed-error mean.
/// @par Example
/// @code
/// const dm::algorithms::RoundingStep value{};
/// std::print("{}\n",value.tie); // false
/// @endcode
struct RoundingStep {
  double      input{};      ///< Actual finite input.
  double      lower{};      ///< Actual floor of the input.
  long double fraction{};   ///< Widened input-minus-floor fraction, protecting near-half decisions.
  bool        tie{};        ///< Whether the represented fraction equals exactly one half.
  double      rounded{};    ///< Actual selected integer-valued output.
  double      error{};      ///< Actual rounded minus input.
  double      mean_error{}; ///< Actual widened running signed-error mean.
};
/// @brief Owned actual scalar decisions and descriptive observed rounding error.
/// @par Example
/// @code
/// const dm::algorithms::RoundingRun value{};
/// std::print("{}\n",value.rounded.size()); // 0
/// @endcode
struct RoundingRun {
  RoundMode           mode{RoundMode::HalfToEven}; ///< Actual selected rule.
  std::vector<double> rounded;                     ///< Actual outputs in caller order.
  double      mean_error{}; ///< Widened mean rounded-minus-input error; zero for empty input.
  std::size_t ties{};       ///< Actual represented half-integers.
  std::vector<RoundingStep> steps; ///< Actual optional per-value decisions.
};
/// @brief Compute actual rounding decisions and observed mean error.
/// Validates the supported mode and every input as finite before producing outputs. Empty input has zero mean/ties and empty outputs. The observed mean error describes this supplied sample, not a population bias theorem. O(N) time and output, with one optional retained record per value. Complete max_values admission precedes allocations; zero disables the policy. Inputs unchanged and histories owned.
/// @param values Complete finite observations in original order.
/// @param mode Supported rounding rule.
/// @param max_values Complete input/output/history count policy, default1000000; zero disables.
/// @return Owned actual rounded values, sample signed-error mean, ties and optional actual records.
/// @throws std::invalid_argument Unsupported rule or nonfinite observation.
/// @throws std::length_error Input/output count policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::rounding_run({.5,1.5},dm::algorithms::RoundMode::HalfToEven);
/// std::print("{} {}\n",r.ties,r.mean_error); // 2 0
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see rounding_bias()
RoundingRun
rounding_run(const std::vector<double>& values, RoundMode mode, std::size_t max_values = 1000000);
/// @brief Compute actual rounding decisions and capture their running errors.
/// Validates the supported mode and every input as finite before producing outputs. Empty input has zero mean/ties and empty outputs. The observed mean error describes this supplied sample, not a population bias theorem. O(N) time and output, with one optional retained record per value. Complete max_values admission precedes allocations; zero disables the policy. Inputs unchanged and histories owned.
/// @param values Complete finite observations in original order.
/// @param mode Supported rounding rule.
/// @param max_values Complete input/output/history count policy, default1000000; zero disables.
/// @return Owned actual rounded values, sample signed-error mean, ties and optional actual records.
/// @throws std::invalid_argument Unsupported rule or nonfinite observation.
/// @throws std::length_error Input/output count policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::rounding_trace({.5,1.5},dm::algorithms::RoundMode::HalfToEven);
/// std::print("{} {}\n",r.ties,r.mean_error); // 2 0
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see rounding_bias()
RoundingRun
rounding_trace(const std::vector<double>& values, RoundMode mode, std::size_t max_values = 1000000);
/// @brief Return the observed mean signed rounding error through the preserved signature.
/// Uses the same checked producer and default million-value bound. Finite samples only; empty returns zero. Uses widened accumulation, without claiming unbiasedness for arbitrary samples.
/// @param values Complete finite sample.
/// @param mode Supported rounding rule.
/// @return Mean rounded-minus-input scalar in the sample units.
/// @throws std::invalid_argument Unsupported rule/nonfinite observation.
/// @throws std::length_error Default input bound exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rounding_bias({.5,1.5},dm::algorithms::RoundMode::HalfUp)); // 0.5
/// @endcode
/// @par References
/// - IEEE754-2019, doi:10.1109/IEEESTD.2019.8766229; half-up/down are the explicitly named tie variants documented here.
/// @see rounding_run()
double rounding_bias(const std::vector<double>& values, RoundMode mode);
/// @brief Generate a canonical controlled report input.
/// Construct the original60 positive half-integers0.5,1.5,...,59.5 exactly. These deliberately tie-heavy observations illustrate sample-specific signed error, not a universal population bias.
/// @return Owned exact synthetic values.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rounding_example_ties().size()); // 60
/// @endcode
/// @see rounding_trace()
std::vector<double> rounding_example_ties();
} // namespace datamunge::algorithms
