/// @file thomas.cpp
/// @brief Actual checked tridiagonal sweeps and original sine-forcing input assembly.
#include <cmath>
#include <datamunge/algorithms/thomas.hpp>
#include <numbers>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private finite/pivot checks and actual producer.
namespace {
/// @brief Reject nonfinite actual arithmetic.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Thomas arithmetic");
  return x;
}
/// @brief Reject a zero actual no-pivot divisor before division.
void pivot(double x) {
  if (x == 0)
    throw std::domain_error("Thomas zero pivot");
}
/// @brief Execute owned actual eliminated rows and solution coordinates.
ThomasRun execute(
    const std::vector<double>& sub,
    const std::vector<double>& diag,
    const std::vector<double>& super,
    const std::vector<double>& rhs,
    const ThomasOptions&       o,
    bool                       capture) {
  auto n = diag.size();
  if (sub.size() != n || super.size() != n || rhs.size() != n)
    throw std::invalid_argument("Thomas vector geometry");
  for (const auto* v : {&sub, &diag, &super, &rhs})
    for (double x : *v)
      if (!std::isfinite(x))
        throw std::invalid_argument("Thomas finite inputs");
  if (o.max_rows && n > o.max_rows)
    throw std::length_error("Thomas rows");
  ThomasRun r;
  r.diagonal = diag;
  r.rhs      = rhs;
  r.x.assign(n, 0);
  r.residual.resize(n);
  if (!n)
    return r;
  /// @brief Capture one actual forward or backward state before exceeding the record policy.
  auto record = [&](ThomasStep s) {
    if (!capture)
      return;
    if (o.max_records && r.steps.size() >= o.max_records)
      throw std::length_error("Thomas history");
    r.steps.push_back(s);
  };
  for (std::size_t i = 1; i < n; ++i) {
    pivot(r.diagonal[i - 1]);
    double w = finite(sub[i] / r.diagonal[i - 1]);
    ++r.divisions;
    r.diagonal[i] = finite(r.diagonal[i] - finite(w * super[i - 1]));
    r.rhs[i]      = finite(r.rhs[i] - finite(w * r.rhs[i - 1]));
    ++r.eliminations;
    record({false, i, w, r.diagonal[i], r.rhs[i], 0});
  }
  for (std::size_t i = n; i-- > 0;) {
    pivot(r.diagonal[i]);
    double value = r.rhs[i];
    if (i + 1 < n)
      value = finite(value - finite(super[i] * r.x[i + 1]));
    r.x[i] = finite(value / r.diagonal[i]);
    ++r.divisions;
    ++r.substitutions;
    record({true, i, 0, r.diagonal[i], r.rhs[i], r.x[i]});
  }
  for (std::size_t i = 0; i < n; ++i) {
    double value = finite(diag[i] * r.x[i]);
    if (i)
      value = finite(value + finite(sub[i] * r.x[i - 1]));
    if (i + 1 < n)
      value = finite(value + finite(super[i] * r.x[i + 1]));
    r.residual[i]   = finite(rhs[i] - value);
    r.residual_norm = finite(std::hypot(r.residual_norm, r.residual[i]));
  }
  return r;
}
} // namespace
ThomasRun thomas_run(
    const std::vector<double>& sub,
    const std::vector<double>& diag,
    const std::vector<double>& super,
    const std::vector<double>& rhs,
    const ThomasOptions&       options) {
  return execute(sub, diag, super, rhs, options, false);
}
ThomasRun thomas_trace(
    const std::vector<double>& sub,
    const std::vector<double>& diag,
    const std::vector<double>& super,
    const std::vector<double>& rhs,
    const ThomasOptions&       options) {
  return execute(sub, diag, super, rhs, options, true);
}
std::vector<double> thomas_solve(
    std::vector<double> sub,
    std::vector<double> diag,
    std::vector<double> super,
    std::vector<double> rhs) {
  return thomas_run(sub, diag, super, rhs).x;
}
ThomasGrid thomas_poisson_example(std::size_t n) {
  if (n > 1000000)
    throw std::length_error("Thomas fixture rows");
  ThomasGrid g;
  g.sub.assign(n, -1);
  g.diagonal.assign(n, 2);
  g.super.assign(n, -1);
  g.rhs.resize(n);
  g.coordinates.resize(n);
  g.exact.resize(n);
  double h = 1. / (n + 1);
  for (std::size_t i = 0; i < n; ++i) {
    g.coordinates[i] = (i + 1) * h;
    g.exact[i]       = std::sin(std::numbers::pi * g.coordinates[i]);
    g.sub[i]         = i == 0 ? 0 : -1 / (h * h);
    g.super[i]       = i + 1 == n ? 0 : -1 / (h * h);
    g.diagonal[i]    = 2 / (h * h);
    g.rhs[i]         = std::numbers::pi * std::numbers::pi * g.exact[i];
  }
  return g;
}
} // namespace datamunge::algorithms
