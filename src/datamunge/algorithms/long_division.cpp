/// @file long_division.cpp
/// @brief Checked fixed64-bit quotient/remainder producer and actual step capture.
#include <datamunge/algorithms/long_division.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private shared producing kernel.
namespace {
/// @brief Execute the shared64-step shift/compare/subtract producer.
/// @param dividend Full-range input word.
/// @param divisor Nonzero unsigned divisor, checked before any step.
/// @param steps Optional vector of actual states reserved by the caller.
/// @return Exact quotient/remainder and64 iterations.
DivResult
divide(std::uint64_t dividend, std::uint64_t divisor, std::vector<LongDivisionStep>* steps) {
  if (!divisor)
    throw std::invalid_argument("long division: zero divisor");
  std::uint64_t q = 0, r = 0;
  for (int bit = 63; bit >= 0; --bit) {
    const auto before = r;
    // Before this shift the processed prefix contains at most 63 input bits.
    // Its remainder is no larger than that prefix, so this shift cannot overflow.
    const auto trial    = (r << 1) | ((dividend >> bit) & std::uint64_t{1});
    const bool subtract = trial >= divisor;
    r                   = subtract ? trial - divisor : trial;
    if (subtract)
      q |= std::uint64_t{1} << bit;
    if (steps)
      steps->push_back({bit, before, trial, subtract, q, r});
  }
  return {q, r, 64};
}
} // namespace
DivResult long_division(std::uint64_t dividend, std::uint64_t divisor) {
  return divide(dividend, divisor, nullptr);
}
LongDivisionTrace
long_division_trace(std::uint64_t dividend, std::uint64_t divisor, std::size_t max_entries) {
  if (!divisor)
    throw std::invalid_argument("long division: zero divisor");
  if (max_entries && max_entries < 66)
    throw std::length_error("long division: trace quota");
  LongDivisionTrace t{dividend, divisor, {}, {}};
  t.steps.reserve(64);
  t.result = divide(dividend, divisor, &t.steps);
  return t;
}
} // namespace datamunge::algorithms
