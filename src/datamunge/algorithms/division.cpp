/// @file division.cpp
/// @brief Exact64-bit register division and guarded floating-estimate/exact-correction producer histories.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/division.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private exact wide representations and shared producing loops.
namespace {
/// @brief Compiler unsigned128-bit exact arithmetic for complete64-by64 products.
using U128 = unsigned __int128;
/// @brief Compiler signed128-bit register containing non-restoring A, bounded within65 magnitude bits.
using I128 = __int128;
/// @brief Own exact unsigned128 magnitude words.
/// @param x Actual unsigned product/magnitude.
/// @return Lossless high/low pair, nonnegative sign.
DivisionWideValue wide(U128 x) {
  return {std::uint64_t(x >> 64), std::uint64_t(x), false};
}
/// @brief Own exact signed register value as magnitude/sign words.
/// @param x Actual bounded non-restoring register.
/// @return Lossless magnitude/sign, zero nonnegative.
DivisionWideValue signed_wide(I128 x) {
  auto v     = wide(x < 0 ? U128(-x) : U128(x));
  v.negative = x < 0;
  return v;
}
/// @brief Admit known complete logical history units before producer allocation/arithmetic.
/// @param count Complete required units.
/// @param cap Optional policy, zero disables.
/// @throws std::length_error If complete capture policy is exceeded.
void admit(std::size_t count, std::size_t cap) {
  if (cap && count > cap)
    throw std::length_error("integer division: capture quota");
}
/// @brief Execute the original compare-before-subtract fixed64-bit restoring producer.
/// @param n Original dividend.
/// @param d Original divisor,0 sentinel permitted.
/// @param out Optional actual history sink.
/// @param cap Complete history policy, zero disables.
/// @return Exact result and actual comparisons/subtractions, with zero validity explicit.
/// @throws std::length_error If complete history cap is exceeded.
DivisionAnalysis
restoring(std::uint64_t n, std::uint64_t d, RestoringDivisionTrace* out, std::size_t cap) {
  if (out) {
    auto count = 5 + (d ? 64 : 0);
    admit(count, cap);
    out->dividend         = n;
    out->divisor          = d;
    out->captured_entries = count;
  }
  DivisionAnalysis a;
  if (!d)
    return a;
  a.valid_divisor = true;
  std::uint64_t q = 0, r = 0;
  for (int bit = 63; bit >= 0; --bit) {
    RestoringDivisionStep s;
    s.bit              = bit;
    s.quotient_before  = q;
    s.remainder_before = r;
    r                  = (r << 1) | ((n >> bit) & 1);
    s.shifted          = r;
    ++a.comparison_tests;
    s.subtracted = r >= d;
    if (s.subtracted) {
      r -= d;
      q |= std::uint64_t(1) << bit;
      ++a.subtractions;
    } else
      ++a.hypothetical_restores;
    s.quotient_after  = q;
    s.remainder_after = r;
    ++a.bit_steps;
    if (out)
      out->steps.push_back(s);
  }
  a.result = {q, r, 64};
  return a;
}
/// @brief Execute exact signed non-restoring register steps without shifting negative signed integers.
/// @param n Original dividend.
/// @param d Original divisor,0 sentinel permitted.
/// @param out Optional actual history sink.
/// @param cap Complete history policy, zero disables.
/// @return Exact result, actual arithmetic/sign tests and optional final addition.
/// @throws std::length_error If complete history policy is exceeded.
DivisionAnalysis
nonrestoring(std::uint64_t n, std::uint64_t d, NonRestoringDivisionTrace* out, std::size_t cap) {
  if (out) {
    auto count = 6 + (d ? 64 : 0);
    admit(count, cap);
    out->dividend         = n;
    out->divisor          = d;
    out->captured_entries = count;
  }
  DivisionAnalysis a;
  if (!d)
    return a;
  a.valid_divisor         = true;
  I128          remainder = 0;
  const I128    divisor   = d;
  std::uint64_t q         = n;
  for (int i = 0; i < 64; ++i) {
    NonRestoringDivisionStep s;
    s.bit             = 63 - i;
    s.quotient_before = q;
    s.incoming        = (q >> 63) & 1;
    s.before          = signed_wide(remainder);
    remainder         = remainder * 2 + I128(s.incoming);
    s.shifted         = signed_wide(remainder);
    q <<= 1;
    s.subtracted = remainder >= 0;
    ++a.comparison_tests;
    if (s.subtracted) {
      remainder -= divisor;
      ++a.subtractions;
    } else {
      remainder += divisor;
      ++a.additions;
    }
    s.after      = signed_wide(remainder);
    s.output_bit = remainder >= 0;
    ++a.comparison_tests;
    if (s.output_bit)
      q |= 1;
    s.quotient_after = q;
    ++a.bit_steps;
    if (out)
      out->steps.push_back(s);
  }
  if (out)
    out->final_before = signed_wide(remainder);
  ++a.comparison_tests;
  const bool correct = remainder < 0;
  if (correct) {
    remainder += divisor;
    ++a.additions;
  }
  if (out) {
    out->final_corrected = correct;
    out->final_after     = signed_wide(remainder);
  }
  a.result = {q, std::uint64_t(remainder), 64};
  return a;
}
/// @brief Execute original five double reciprocal updates with safe full-width conversion/corrections.
/// @param n Original dividend.
/// @param d Original divisor,0 sentinel permitted.
/// @param out Optional actual-history sink.
/// @param cap Actual online history policy, zero disables.
/// @return Exact quotient/remainder and actual reciprocal/correction work.
/// @throws std::length_error If actual history cap/count geometry is exceeded.
/// @throws std::overflow_error If a floating estimate is invalid.
DivisionAnalysis
newton(std::uint64_t n, std::uint64_t d, IntegerNewtonDivisionTrace* out, std::size_t cap) {
  if (out) {
    admit(10, cap);
    out->dividend         = n;
    out->divisor          = d;
    out->captured_entries = 10;
  }
  /// @brief Admit one actual future update/comparison history record before its append.
  auto record = [&]() {
    if (!out)
      return;
    if (out->captured_entries == std::numeric_limits<std::size_t>::max())
      throw std::length_error("integer Newton: count overflow");
    admit(out->captured_entries + 1, cap);
    ++out->captured_entries;
  };
  DivisionAnalysis a;
  if (!d)
    return a;
  a.valid_divisor = true;
  if (n < d) {
    a.result = {0, n, 0};
    return a;
  }
  double denominator = double(d), r = 1. / denominator;
  if (out) {
    out->rounded_denominator = denominator;
    out->seed                = r;
  }
  for (int i = 0; i < 5; ++i) {
    record();
    auto s = newton_reciprocal_step(denominator, r);
    r      = s.after;
    ++a.reciprocal_updates;
    if (out)
      out->steps.push_back(s);
  }
  double estimate = double(n) * r;
  if (!std::isfinite(estimate) || estimate < 0)
    throw std::overflow_error("integer Newton: invalid quotient estimate");
  const bool    clamped = estimate >= 0x1p64;
  std::uint64_t q = clamped ? std::numeric_limits<std::uint64_t>::max() : std::uint64_t(estimate);
  if (out) {
    out->reciprocal        = r;
    out->quotient_estimate = estimate;
    out->initial_quotient  = q;
    out->estimate_clamped  = clamped;
  }
  for (;;) {
    record();
    IntegerNewtonCheck check;
    check.downward        = true;
    check.quotient_before = q;
    U128 product          = U128(q) * d;
    check.tested_product  = wide(product);
    check.changed         = product > n;
    ++a.comparison_tests;
    if (check.changed) {
      --q;
      ++a.decrements;
    }
    check.quotient_after = q;
    if (out)
      out->checks.push_back(check);
    if (!check.changed)
      break;
  }
  for (;;) {
    record();
    IntegerNewtonCheck check;
    check.downward        = false;
    check.quotient_before = q;
    U128 product          = (U128(q) + 1) * d;
    check.tested_product  = wide(product);
    check.changed         = product <= n;
    ++a.comparison_tests;
    if (check.changed) {
      ++q;
      ++a.increments;
    }
    check.quotient_after = q;
    if (out)
      out->checks.push_back(check);
    if (!check.changed)
      break;
  }
  a.result = {q, n - q * d, 5};
  return a;
}
} // namespace
double division_wide_approximation(DivisionWideValue value) {
  U128   x      = (U128(value.high) << 64) | value.low;
  double result = double(x);
  return value.negative ? -result : result;
}
std::string division_wide_decimal(DivisionWideValue value) {
  U128 x = (U128(value.high) << 64) | value.low;
  if (!x)
    return "0";
  std::string result;
  while (x) {
    result.push_back(char('0' + x % 10));
    x /= 10;
  }
  if (value.negative)
    result.push_back('-');
  std::reverse(result.begin(), result.end());
  return result;
}
DivResultEx restoring_divide(std::uint64_t n, std::uint64_t d) {
  return restoring(n, d, nullptr, 0).result;
}
DivResultEx non_restoring_divide(std::uint64_t n, std::uint64_t d) {
  return nonrestoring(n, d, nullptr, 0).result;
}
DivResultEx newton_raphson_divide(std::uint64_t n, std::uint64_t d) {
  return newton(n, d, nullptr, 0).result;
}
DivisionAnalysis restoring_division_analyze(std::uint64_t n, std::uint64_t d) {
  return restoring(n, d, nullptr, 0);
}
DivisionAnalysis non_restoring_division_analyze(std::uint64_t n, std::uint64_t d) {
  return nonrestoring(n, d, nullptr, 0);
}
DivisionAnalysis integer_newton_division_analyze(std::uint64_t n, std::uint64_t d) {
  return newton(n, d, nullptr, 0);
}
RestoringDivisionTrace restoring_division_trace(std::uint64_t n, std::uint64_t d, std::size_t cap) {
  RestoringDivisionTrace t;
  t.analysis = restoring(n, d, &t, cap);
  return t;
}
NonRestoringDivisionTrace
non_restoring_division_trace(std::uint64_t n, std::uint64_t d, std::size_t cap) {
  NonRestoringDivisionTrace t;
  t.analysis = nonrestoring(n, d, &t, cap);
  return t;
}
IntegerNewtonDivisionTrace
integer_newton_division_trace(std::uint64_t n, std::uint64_t d, std::size_t cap) {
  IntegerNewtonDivisionTrace t;
  t.analysis = newton(n, d, &t, cap);
  return t;
}
std::vector<std::pair<std::uint64_t, std::uint64_t>> integer_newton_demo_pairs() {
  return {
      {100, 7}, {1000000, 999}, {123456789, 12345}, {std::numeric_limits<std::uint64_t>::max(), 7}};
}
std::vector<std::pair<std::uint64_t, std::uint64_t>> nonrestoring_demo_pairs() {
  return {{217, 13}, {100, 7}, {1000000, 999}, {std::numeric_limits<std::uint64_t>::max(), 3}};
}
std::vector<std::uint64_t> division_work_dividends() {
  return {217, 4096, 65535, 1000000, 16777215, 4294967295};
}
} // namespace datamunge::algorithms
