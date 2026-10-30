/// @file kernel.cpp
/// @brief summation_name() and summation_error_bound().

#include "datamunge/nn/kernel.hpp"

#include <cmath>
#include <limits>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief Higham's \f$\gamma_k=ku/(1-ku)\f$.
/// @param k  Number of rounding errors.
/// @param u  Unit roundoff.
/// @return \f$\gamma_k\f$, or infinity once \f$ku\ge1\f$.
double gamma(double k, double u) { return k * u < 1.0 ? k * u / (1.0 - k * u) : INFINITY; }

}  // namespace

std::string summation_name(Summation method) {
  switch (method) {
    case Summation::Naive: return "naive";
    case Summation::Pairwise: return "pairwise";
    case Summation::Kahan: return "kahan";
    case Summation::WideAccumulator: return "wide accumulator";
  }
  return "unknown";
}

double summation_error_bound(Summation method, std::int64_t n, DType dtype) {
  if (n < 1) throw std::invalid_argument("summation_error_bound: n must be at least 1");
  const double u = dispatch_floating(dtype, "summation_error_bound",
                                     [](auto tag) { return std::ldexp(1.0, -std::numeric_limits<decltype(tag)>::digits); });
  const double k = static_cast<double>(n - 1);
  switch (method) {
    case Summation::Naive: return gamma(k, u);
    case Summation::Pairwise: return n <= 8 ? gamma(k, u) : gamma(std::ceil(std::log2(static_cast<double>(n))) + 7.0, u);
    case Summation::Kahan: return 2.0 * u + 2.0 * static_cast<double>(n) * u * u;
    case Summation::WideAccumulator: return (dtype == DType::Float64 ? 0.0 : u) + gamma(k, std::ldexp(1.0, -53));
  }
  throw std::invalid_argument("summation_error_bound: unknown method");
}

}  // namespace datamunge::nn
