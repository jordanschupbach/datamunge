/// @file hermite.cpp
/// @brief Shared checked Hermite segment arithmetic and single-validation batch entry.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/hermite.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private all-input validation and actual value-producing scalar kernel.
namespace hermite_detail {
/// @brief Validate complete finite node/value/slope arrays before lookup or policy caps.
/// @param xs Nodes, at least2 and strictly increasing.
/// @param ys Equal-sized finite values.
/// @param slopes Equal-sized finite physical derivatives.
/// @throws std::invalid_argument On shapes/nonfinite inputs/nonincreasing nodes.
/// @throws std::overflow_error On nonfinite positive node width.
void validate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& slopes) {
  if (xs.size() < 2 || ys.size() != xs.size() || slopes.size() != xs.size())
    throw std::invalid_argument("Hermite shape mismatch");
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]) || !std::isfinite(slopes[i]))
      throw std::invalid_argument("Hermite finite inputs required");
    if (i) {
      if (!(xs[i] > xs[i - 1]))
        throw std::invalid_argument("Hermite nodes must strictly increase");
      if (!std::isfinite(xs[i] - xs[i - 1]))
        throw std::overflow_error("Hermite width overflow");
    }
  }
}
/// @brief Validate a query before lookup/allocation.
/// @param x Candidate query.
/// @throws std::invalid_argument On nonfinite x.
void query(double x) {
  if (!std::isfinite(x))
    throw std::invalid_argument("Hermite query must be finite");
}
/// @brief Reject nonfinite internal arithmetic without normalizing or clamping it.
/// @param x Computed intermediate.
/// @return Unchanged finite x.
/// @throws std::overflow_error On nonfinite intermediate.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Hermite evaluation overflow");
  return x;
}
/// @brief Evaluate validated inputs with original interval policy and term operation order.
/// @param xs Valid borrowed nodes.
/// @param ys Valid borrowed values.
/// @param m Valid borrowed physical slopes.
/// @param x Valid finite query.
/// @return Owned actual fixed-size arithmetic trace.
/// @throws std::overflow_error On nonfinite normalized coordinate/basis/term/sum.
HermiteEvaluation evaluate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& m,
    double                     x) {
  HermiteEvaluation e;
  e.x            = x;
  e.extrapolated = x < xs.front() || x > xs.back();
  if (x <= xs.front())
    e.interval = 0;
  else if (x >= xs[xs.size() - 2])
    e.interval = xs.size() - 2;
  else
    e.interval = std::size_t(std::upper_bound(xs.begin(), xs.end(), x) - xs.begin()) - 1;
  const auto k = e.interval;
  e.width      = xs[k + 1] - xs[k];
  e.t          = finite((x - xs[k]) / e.width);
  e.t2         = finite(e.t * e.t);
  e.t3         = finite(e.t2 * e.t);
  e.basis      = {
      finite(2 * e.t3 - 3 * e.t2 + 1),
      finite(e.t3 - 2 * e.t2 + e.t),
      finite(-2 * e.t3 + 3 * e.t2),
      finite(e.t3 - e.t2)};
  e.terms = {
      finite(e.basis[0] * ys[k]),
      finite(e.basis[1] * e.width * m[k]),
      finite(e.basis[2] * ys[k + 1]),
      finite(e.basis[3] * e.width * m[k + 1])};
  e.value = finite(e.terms[0] + e.terms[1] + e.terms[2] + e.terms[3]);
  return e;
}
} // namespace hermite_detail
HermiteEvaluation hermite_evaluate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& slopes,
    double                     x) {
  hermite_detail::validate(xs, ys, slopes);
  hermite_detail::query(x);
  return hermite_detail::evaluate(xs, ys, slopes, x);
}
double hermite_interpolate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& slopes,
    double                     x) {
  return hermite_evaluate(xs, ys, slopes, x).value;
}
std::vector<double> hermite_evaluate_many(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& slopes,
    const std::vector<double>& queries,
    std::size_t                max_queries) {
  hermite_detail::validate(xs, ys, slopes);
  for (auto x : queries)
    hermite_detail::query(x);
  std::vector<double> out;
  if ((max_queries && queries.size() > max_queries) || queries.size() > out.max_size())
    throw std::length_error("Hermite complete query limit");
  out.reserve(queries.size());
  for (auto x : queries)
    out.push_back(hermite_detail::evaluate(xs, ys, slopes, x).value);
  return out;
}
} // namespace datamunge::algorithms
