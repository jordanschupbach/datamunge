/// @file pollard_p_minus_one.cpp
/// @brief Native factorial powering with checked actual-stage observation and no loop wrap.
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <datamunge/algorithms/pollard_p_minus_one.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Shared fixed-base attempt and optional producing-stage observer.
namespace {
/// @brief Apply the historical factorial schedule with complete scalar validation.
PMinusOneTrace run(std::uint64_t n, std::uint64_t bound, PMinusOneOptions options, bool capture) {
  if (n < 2)
    throw std::invalid_argument("p-minus-one requires n>=2");
  PMinusOneTrace t;
  auto&          r = t.run;
  if (n == 2) {
    r.stop = PMinusOneStop::SmallPrime;
    return t;
  }
  if (n % 2 == 0) {
    r.divisor = 2;
    r.stop    = PMinusOneStop::EvenFactor;
    return t;
  }
  std::uint64_t a = 2;
  for (std::uint64_t j = 2; j <= bound; ++j) {
    if (r.stages == std::numeric_limits<std::size_t>::max() ||
        (options.max_stages && r.stages >= options.max_stages) ||
        (capture && options.max_trace_records && t.stages.size() >= options.max_trace_records))
      throw std::length_error("p-minus-one stage/history policy");
    const auto before = a;
    a                 = mod_pow(a, j, n);
    const auto gcd    = std::gcd(a == 0 ? n : a - 1, n);
    ++r.stages;
    if (capture)
      t.stages.push_back({j, before, a, gcd});
    if (gcd > 1 && gcd < n) {
      r.divisor = gcd;
      r.stop    = PMinusOneStop::ProperFactor;
      return t;
    }
    if (gcd == n) {
      r.stop = PMinusOneStop::Collapsed;
      return t;
    }
    if (j == bound)
      break;
  }
  return t;
}
} // namespace
std::uint64_t pollard_p_minus_1(std::uint64_t n, std::uint64_t bound) {
  return run(n, bound, {}, false).run.divisor;
}
PMinusOneRun pollard_p_minus_one_run(std::uint64_t n, std::uint64_t bound, PMinusOneOptions o) {
  return run(n, bound, o, false).run;
}
PMinusOneTrace pollard_p_minus_one_trace(std::uint64_t n, std::uint64_t bound, PMinusOneOptions o) {
  return run(n, bound, o, true);
}
} // namespace datamunge::algorithms
