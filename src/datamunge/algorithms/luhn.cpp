/// @file luhn.cpp
/// @brief Shared complete-guarded bounded-residue Luhn producer and exhaustive native error controls.
#include <algorithm>
#include <datamunge/algorithms/luhn.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private byte validation, checked geometry and shared producing residue pass.
namespace {
/// @brief Validate every original byte as ASCII decimal without locale dependence.
/// @param digits Complete borrowed input.
void validate(const std::string& digits) {
  for (unsigned char c : digits)
    if (c < '0' || c > '9')
      throw std::invalid_argument("Luhn: only ASCII decimal bytes are accepted");
}
/// @brief Add nonnegative counts without wraparound.
/// @param a First count.
/// @param b Second count.
/// @return Exact representable sum.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Luhn: count overflow");
  return a + b;
}
/// @brief Multiply counts without wraparound.
/// @param a First count.
/// @param b Second count.
/// @return Exact representable product.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("Luhn: count overflow");
  return a * b;
}
/// @brief Validate once and produce actual parity transforms and a bounded modulo10 residue.
/// @param digits Complete original input.
/// @param mode Full/payload parity convention.
/// @param trace Optional owned actual digit capture.
/// @param quota Optional selected logical capture cap.
/// @return Actual checksum result without unbounded integer-sum accumulation.
LuhnAnalysis
produce(const std::string& digits, LuhnMode mode, LuhnTrace* trace, std::size_t quota) {
  if (mode != LuhnMode::Full && mode != LuhnMode::Payload)
    throw std::invalid_argument("Luhn: invalid mode");
  validate(digits);
  if (trace) {
    const auto entries = multiply(digits.size(), 2);
    if (quota && entries > quota)
      throw std::length_error("Luhn: trace quota");
    trace->captured_entries = entries;
    trace->input            = digits;
    trace->steps.reserve(digits.size());
  }
  LuhnAnalysis out;
  out.mode     = mode;
  out.digits   = digits.size();
  bool doubled = mode == LuhnMode::Payload;
  for (std::size_t i = digits.size(); i-- > 0;) {
    const int  digit        = digits[i] - '0';
    const int  raw          = doubled ? 2 * digit : digit;
    const bool reduced      = raw > 9;
    const int  contribution = reduced ? raw - 9 : raw;
    const int  before       = out.remainder;
    out.remainder           = (before + contribution) % 10;
    out.doubled += doubled;
    out.reductions += reduced;
    if (trace)
      trace->steps.push_back(
          {i, digit, doubled, raw, reduced, contribution, before, out.remainder});
    doubled = !doubled;
  }
  if (mode == LuhnMode::Payload)
    out.check_digit = (10 - out.remainder) % 10;
  else
    out.valid = !digits.empty() && out.remainder == 0;
  return out;
}
} // namespace
int luhn_check_digit(const std::string& payload) {
  return produce(payload, LuhnMode::Payload, nullptr, 0).check_digit;
}
bool luhn_validate(const std::string& full) {
  return produce(full, LuhnMode::Full, nullptr, 0).valid;
}
LuhnAnalysis luhn_analyze(const std::string& digits, LuhnMode mode) {
  return produce(digits, mode, nullptr, 0);
}
LuhnTrace luhn_trace(const std::string& digits, LuhnMode mode, std::size_t max_entries) {
  LuhnTrace t;
  t.analysis = produce(digits, mode, &t, max_entries);
  return t;
}
LuhnErrorStudy luhn_error_study(const std::string& payload, std::size_t max_entries) {
  const int   digit = luhn_check_digit(payload);
  const auto  n     = add(payload.size(), 1);
  std::string full  = payload;
  full.push_back(char('0' + digit));
  std::size_t swaps = 0;
  for (std::size_t i = 0; i + 1 < n; ++i)
    swaps += full[i] != full[i + 1];
  const auto cases   = add(multiply(n, 9), swaps);
  const auto entries = add(add(payload.size(), n), multiply(cases, add(n, 1)));
  if (max_entries && entries > max_entries)
    throw std::length_error("Luhn: complete error-study quota");
  LuhnErrorStudy out;
  out.payload          = payload;
  out.full             = full;
  out.captured_entries = entries;
  out.cases.reserve(cases);
  for (std::size_t i = 0; i < n; ++i)
    for (char c = '0'; c <= '9'; ++c) {
      if (c == full[i])
        continue;
      auto changed      = full;
      changed[i]        = c;
      const auto actual = luhn_analyze(changed);
      ++out.substitutions;
      out.substitutions_detected += !actual.valid;
      out.cases.push_back(
          {LuhnErrorKind::Substitution, i, std::move(changed), !actual.valid, actual.remainder});
    }
  for (std::size_t i = 0; i + 1 < n; ++i) {
    if (full[i] == full[i + 1])
      continue;
    auto changed = full;
    std::swap(changed[i], changed[i + 1]);
    const auto actual = luhn_analyze(changed);
    ++out.swaps;
    out.swaps_detected += !actual.valid;
    out.cases.push_back(
        {LuhnErrorKind::AdjacentSwap, i, std::move(changed), !actual.valid, actual.remainder});
  }
  return out;
}
} // namespace datamunge::algorithms
