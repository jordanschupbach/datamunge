/// @file srt_division.cpp
/// @brief One actual normalized redundant recurrence and weighted accumulator.
#include <cmath>
#include <datamunge/algorithms/srt_division.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Translation-unit-private checked producing kernels; no shared mutable state.
namespace {
/// @brief Append an actual digit to a floating weighted prefix.
/// @param digit Actual redundant digit.
/// @param weight Mutable current binary weight, initially1.
/// @param value Mutable actual sum, initially0.
void append(int digit, double& weight, double& value) {
  weight *= .5;
  value += digit * weight;
}
/// @brief Execute checked actual remainder transitions once.
/// @param N Numerator.
/// @param D Divisor.
/// @param bits Requested complete length.
/// @param o Policies.
/// @param capture Retain actual states.
/// @return Actual owned stream.
SRTRun run(double N, double D, int bits, const SRTOptions& o, bool capture) {
  if (!std::isfinite(N) || !std::isfinite(D) || D <= 0 || N < 0 || N >= D || bits < 0)
    throw std::invalid_argument("SRT: proper fraction/count");
  const auto count = static_cast<std::size_t>(bits);
  SRTRun     r;
  if ((o.max_bits && count > o.max_bits) || count > r.digits.max_size() ||
      (capture && ((o.max_records && count > o.max_records) || count > r.steps.max_size())))
    throw std::length_error("SRT: complete digit/history geometry");
  int exponent             = 0;
  r.normalized_divisor     = std::frexp(D, &exponent);
  r.normalization_exponent = -exponent;
  r.normalized_numerator   = std::ldexp(N, r.normalization_exponent);
  r.remainder              = r.normalized_numerator;
  r.nominal_tail_bound     = std::ldexp(1., -bits);
  r.digits.reserve(count);
  double weight = 1;
  for (std::size_t j = 0; j < count; ++j) {
    const double before = r.remainder, shifted = 2 * before;
    const int    digit = shifted >= .5 ? 1 : shifted <= -.5 ? -1 : 0;
    r.remainder        = shifted - digit * r.normalized_divisor;
    append(digit, weight, r.value);
    r.digits.push_back(digit);
    ++r.counts[digit + 1];
    if (capture)
      r.steps.push_back({j, before, shifted, digit, r.remainder, weight, r.value});
  }
  return r;
}
} // namespace
SRTRun srt_division_run(double N, double D, int bits, const SRTOptions& o) {
  return run(N, D, bits, o, false);
}
SRTRun srt_division_trace(double N, double D, int bits, const SRTOptions& o) {
  return run(N, D, bits, o, true);
}
std::vector<int> srt_quotient_digits(double N, double D, int bits) {
  return srt_division_run(N, D, bits).digits;
}
double srt_digits_to_value(const std::vector<int>& digits) {
  if (digits.size() > 4096)
    throw std::length_error("SRT: digit policy");
  double weight = 1, value = 0;
  for (int digit : digits) {
    if (digit < -1 || digit > 1)
      throw std::invalid_argument("SRT: redundant digit");
    append(digit, weight, value);
  }
  return value;
}
double srt_divide(double N, double D, int bits) {
  return srt_division_run(N, D, bits).value;
}
} // namespace datamunge::algorithms
