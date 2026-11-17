/// @file spigot.cpp
/// @brief One actual factorial-base stream and independent exact rational prefix certificate.
#include <datamunge/algorithms/spigot.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

#include "detail/decimal_uint.hpp"
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Emit the stream and check its exact rational interval.
/// @param requested Fractional count.
/// @param o Policies.
/// @param capture Retain actual outer states.
/// @return Certified actual stream.
SpigotRun run(int requested, const SpigotOptions& o, bool capture) {
  if (requested < 0)
    throw std::invalid_argument("spigot: negative digits");
  const auto d = static_cast<std::size_t>(requested);
  if (d > 100000 || (o.max_digits && d > o.max_digits))
    throw std::length_error("spigot: digits");
  const auto m = d + 20, limit = std::numeric_limits<std::size_t>::max();
  if (d && m - 1 > limit / d)
    throw std::length_error("spigot: work capacity");
  if (o.max_column_updates && d * (m - 1) > o.max_column_updates)
    throw std::length_error("spigot: work policy");
  if (capture &&
      (d && (m + 2 > limit / d || (o.max_history_values && d * (m + 2) > o.max_history_values))))
    throw std::length_error("spigot: history");
  std::vector<int> a(m + 1, 1);
  SpigotRun        r;
  r.digits         = "2.";
  r.columns        = m;
  r.frequencies[2] = 1;
  for (std::size_t j = 0; j < d; ++j) {
    int carry = 0;
    for (std::size_t i = m; i >= 2; --i) {
      const int value = a[i] * 10 + carry;
      a[i]            = value % static_cast<int>(i);
      carry           = value / static_cast<int>(i);
      ++r.column_updates;
    }
    if (carry < 0 || carry > 9)
      throw std::runtime_error("spigot: nondecimal carry");
    r.digits.push_back(static_cast<char>('0' + carry));
    ++r.frequencies[carry];
    if (capture)
      r.history.push_back({j, carry, a});
  }
  using cpp_int     = detail::DecimalUInt;
  cpp_int factorial = 1, numerator = 1, scale = 1;
  for (std::size_t k = 1; k <= m; ++k) {
    factorial *= k;
    numerator = numerator * k + 1;
  }
  for (std::size_t k = 0; k < d; ++k)
    scale *= 10;
  const cpp_int lower = numerator * scale / factorial,
                upper = (numerator * m + 1) * scale / (factorial * m);
  std::string emitted = r.digits;
  emitted.erase(emitted.find('.'), 1);
  if (lower != upper || lower.convert_to<std::string>() != emitted)
    throw std::runtime_error("spigot: unresolved exact prefix certificate");
  r.certified = true;
  return r;
}
} // namespace
SpigotRun spigot_e_run(int n, const SpigotOptions& o) {
  return run(n, o, false);
}
SpigotRun spigot_e_trace(int n, const SpigotOptions& o) {
  return run(n, o, true);
}
std::string spigot_e(int n) {
  return spigot_e_run(n).digits;
}
} // namespace datamunge::algorithms
