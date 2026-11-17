/// @file sqrt_digits.cpp
/// @brief One exact paired-digit producer with actual integer states.
#include <datamunge/algorithms/sqrt_digits.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

#include "detail/decimal_uint.hpp"
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Extract with arbitrary-precision root and remainder.
/// @param n Radicand.
/// @param digits Requested fractional digits.
/// @param o Policies.
/// @param capture Retain actual transitions.
/// @return Actual exact output.
SqrtDigitsRun run(std::uint64_t n, int digits, const SqrtDigitsOptions& o, bool capture) {
  if (digits < 0)
    throw std::invalid_argument("sqrt digits: negative count");
  const auto d = static_cast<std::size_t>(digits);
  if (d > 100000 || (o.max_fractional_digits && d > o.max_fractional_digits))
    throw std::length_error("sqrt digits: count");
  auto s = std::to_string(n);
  if (s.size() % 2)
    s = "0" + s;
  const auto pairs = s.size() / 2, total = pairs + d;
  using cpp_int      = detail::DecimalUInt;
  cpp_int       root = 0, remainder = 0;
  SqrtDigitsRun r;
  std::size_t   retained = 0;
  for (std::size_t step = 0; step < total; ++step) {
    const int pair = step < pairs ? (s[2 * step] - '0') * 10 + s[2 * step + 1] - '0' : 0;
    remainder      = remainder * 100 + pair;
    int     digit  = 9;
    cpp_int product;
    for (; digit >= 0; --digit) {
      product = (20 * root + digit) * digit;
      ++r.candidate_tests;
      if (product <= remainder)
        break;
    }
    remainder -= product;
    root = root * 10 + digit;
    r.digits.push_back(static_cast<char>('0' + digit));
    if (step + 1 == pairs)
      r.digits += '.';
    if (capture) {
      auto prefix = root.convert_to<std::string>(), rem = remainder.convert_to<std::string>();
      for (const auto extra : {std::size_t{1}, prefix.size(), rem.size()}) {
        if (extra > std::numeric_limits<std::size_t>::max() - retained)
          throw std::length_error("sqrt digits: history capacity");
        retained += extra;
        if (o.max_history_characters && retained > o.max_history_characters)
          throw std::length_error("sqrt digits: history");
      }
      r.history.push_back({step, pair, digit, std::move(prefix), std::move(rem)});
    }
  }
  return r;
}
} // namespace
SqrtDigitsRun sqrt_digits_run(std::uint64_t n, int d, const SqrtDigitsOptions& o) {
  return run(n, d, o, false);
}
SqrtDigitsRun sqrt_digits_trace(std::uint64_t n, int d, const SqrtDigitsOptions& o) {
  return run(n, d, o, true);
}
std::string digit_by_digit_sqrt(std::uint64_t n, int d) {
  return sqrt_digits_run(n, d).digits;
}
} // namespace datamunge::algorithms
