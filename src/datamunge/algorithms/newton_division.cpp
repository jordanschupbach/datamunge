/// @file newton_division.cpp
/// @brief Scaled finite floating division and actual shared reciprocal-Newton update producer.
#include <cmath>
#include <datamunge/algorithms/newton_division.hpp>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private finite operand admission and shared scaled quotient producer.
namespace {
/// @brief Compute a checked scaled quotient, optionally retaining actual reciprocal updates.
/// @param n Finite original numerator.
/// @param d Finite original nonzero denominator.
/// @param out Optional actual-history sink.
/// @param cap Actual history cap, zero disables.
/// @return Actual scaling/state metadata and quotient.
/// @throws std::invalid_argument If complete operand domain fails.
/// @throws std::overflow_error If an actual intermediate/final quotient is nonfinite.
/// @throws std::length_error If capture quota is exceeded before an update.
NewtonDivisionRun run(double n, double d, NewtonDivisionTrace* out, std::size_t cap) {
  if (!std::isfinite(n) || !std::isfinite(d) || d == 0)
    throw std::invalid_argument("Newton division: finite operands and nonzero divisor required");
  if (out) {
    if (cap && cap < 3)
      throw std::length_error("Newton division: capture base");
    out->numerator        = n;
    out->denominator      = d;
    out->captured_entries = 3;
  }
  NewtonDivisionRun r;
  const bool        negative = std::signbit(n) != std::signbit(d);
  if (n == 0) {
    r.zero_shortcut = true;
    r.quotient      = std::copysign(0., negative ? -1. : 1.);
    return r;
  }
  r.numerator_mantissa   = std::frexp(std::abs(n), &r.numerator_exponent);
  r.denominator_mantissa = std::frexp(std::abs(d), &r.denominator_exponent);
  r.seed                 = 48. / 17. - (32. / 17.) * r.denominator_mantissa;
  double x               = r.seed;
  for (int i = 0; i < 60; ++i) {
    if (out) {
      if (cap && out->captured_entries == cap)
        throw std::length_error("Newton division: capture quota");
      ++out->captured_entries;
    }
    auto step = newton_reciprocal_step(r.denominator_mantissa, x);
    ++r.updates;
    if (out)
      out->steps.push_back(step);
    r.stationary = std::abs(step.after - x) <= 1e-17 * std::abs(step.after);
    x            = step.after;
    if (r.stationary)
      break;
  }
  r.reciprocal   = x;
  double product = r.numerator_mantissa * x;
  r.quotient =
      std::ldexp(negative ? -product : product, r.numerator_exponent - r.denominator_exponent);
  if (!std::isfinite(r.quotient))
    throw std::overflow_error("Newton division: quotient overflow");
  r.underflowed = r.quotient == 0;
  r.subnormal   = std::fpclassify(r.quotient) == FP_SUBNORMAL;
  return r;
}
} // namespace
double newton_raphson_division(double n, double d) {
  return run(n, d, nullptr, 0).quotient;
}
NewtonDivisionRun newton_division_run(double n, double d) {
  return run(n, d, nullptr, 0);
}
NewtonDivisionTrace newton_division_trace(double n, double d, std::size_t cap) {
  NewtonDivisionTrace t;
  t.run = run(n, d, &t, cap);
  return t;
}
std::vector<std::pair<double, double>> newton_division_demo_pairs() {
  return {{1., 3.}, {-7., 1024.}, {5., 1e-9}, {22., 7.}};
}
NewtonDivisionCheck newton_division_check(std::size_t samples, std::uint64_t seed) {
  if (samples > 1000000)
    throw std::length_error("Newton division study: sample cap");
  std::mt19937_64     rng(seed);
  NewtonDivisionCheck c;
  for (std::size_t i = 0; i < samples; ++i) {
    double mn = .5 + .5 * double(rng() >> 11) * 0x1.0p-53,
           md = .5 + .5 * double(rng() >> 11) * 0x1.0p-53;
    int    en = int(rng() % 1001) - 500, ed = int(rng() % 1001) - 500;
    bool   sn = rng() & 1, sd = rng() & 1;
    double n = std::ldexp(sn ? -mn : mn, en), d = std::ldexp(sd ? -md : md, ed),
           value = newton_raphson_division(n, d), reference = n / d;
    double error = double(std::abs((static_cast<long double>(value) - reference) / reference));
    ++c.checked;
    if (error > c.max_relative_error) {
      c.max_relative_error = error;
      c.worst_index        = i;
    }
  }
  return c;
}
} // namespace datamunge::algorithms
