/// @file newton_reciprocal.cpp
/// @brief Shared checked double reciprocal-Newton update, retaining actual rounded operands.
#include <cmath>
#include <datamunge/algorithms/newton_reciprocal.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
NewtonReciprocalStep newton_reciprocal_step(double d, double x) {
  if (!std::isfinite(d) || d <= 0 || !std::isfinite(x))
    throw std::invalid_argument(
        "reciprocal Newton: positive finite denominator and finite seed required");
  NewtonReciprocalStep s;
  s.before          = x;
  s.product         = d * x;
  s.factor          = 2. - s.product;
  s.after           = x * s.factor;
  s.residual_before = std::fma(-d, x, 1.);
  s.residual_after  = std::fma(-d, s.after, 1.);
  if (!std::isfinite(s.product) || !std::isfinite(s.factor) || !std::isfinite(s.after) ||
      !std::isfinite(s.residual_before) || !std::isfinite(s.residual_after))
    throw std::overflow_error("reciprocal Newton: nonfinite update");
  return s;
}
NewtonReciprocalTrace newton_reciprocal_trace(double d, double x, std::size_t n, std::size_t cap) {
  if (!std::isfinite(d) || d <= 0 || !std::isfinite(x))
    throw std::invalid_argument("reciprocal Newton: complete input domain");
  if (n > std::numeric_limits<std::size_t>::max() - 4 ||
      n > std::vector<NewtonReciprocalStep>{}.max_size() || (cap && n + 4 > cap))
    throw std::length_error("reciprocal Newton: complete history geometry/quota");
  NewtonReciprocalTrace t;
  t.denominator      = d;
  t.seed             = x;
  t.initial_residual = std::fma(-d, x, 1.);
  if (!std::isfinite(t.initial_residual))
    throw std::overflow_error("reciprocal Newton: initial diagnostic overflow");
  t.final            = x;
  t.captured_entries = 4 + n;
  t.steps.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto s  = newton_reciprocal_step(d, t.final);
    t.final = s.after;
    t.steps.push_back(s);
  }
  return t;
}
} // namespace datamunge::algorithms
