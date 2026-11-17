/// @file levinson.cpp
/// @brief Shared finite checked historical Toeplitz prefix recurrence and independent widened residual observer.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/levinson.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private full input/geometry checks and actual prefix producer.
namespace levinson_detail {
/// @brief Require complete finite input coordinates.
/// @param v Borrowed original values.
/// @throws std::invalid_argument On any nonfinite value.
void finite_input(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      throw std::invalid_argument("Levinson finite input required");
}
/// @brief Check matched full shapes, complete finite values and conservative selected work geometry.
/// @param row Borrowed complete defining row.
/// @param rhs Borrowed complete right-hand side.
/// @throws std::invalid_argument On shape/nonfinite inputs.
/// @throws std::length_error On selected-work/container geometry.
void validate(const std::vector<double>& row, const std::vector<double>& rhs) {
  if (row.size() != rhs.size())
    throw std::invalid_argument("Levinson matched shapes required");
  finite_input(row);
  finite_input(rhs);
  const auto n = row.size(), max = std::numeric_limits<std::size_t>::max();
  if (n && n > max / n / 6)
    throw std::length_error("Levinson complete selected-work geometry");
  if (n > std::vector<double>{}.max_size())
    throw std::length_error("Levinson vector geometry");
}
/// @brief Require a finite actual produced recurrence scalar without changing arithmetic order.
/// @param value Actual double result.
/// @return Unchanged finite double.
/// @throws std::overflow_error On nonfinite producing arithmetic.
double produced(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Levinson nonfinite recurrence arithmetic");
  return value;
}
/// @brief Observe one actual completed prefix with no new solve.
/// @param trace Optional preadmitted capture.
/// @param f Actual forward vector.
/// @param b Actual backward vector.
/// @param x Actual prefix solution.
void state(
    LevinsonTrace*             trace,
    const std::vector<double>& f,
    const std::vector<double>& b,
    const std::vector<double>& x) {
  if (trace)
    trace->states.push_back({x.size(), f, b, x});
}
/// @brief Produce one original-order recurrence, optionally retaining actual states and attempted denominators.
/// @param row Validated original defining row.
/// @param rhs Validated original right-hand side.
/// @param trace Optional completely admitted capture.
/// @return Actual complete or explicit zero-denominator breakdown result/work.
/// @throws std::overflow_error On nonfinite produced arithmetic.
/// @throws std::bad_alloc On runtime/output/capture allocations.
LevinsonAnalysis
run(const std::vector<double>& row, const std::vector<double>& rhs, LevinsonTrace* trace) {
  LevinsonAnalysis out;
  const auto       n = row.size();
  if (!n)
    return out;
  if (row[0] == 0) {
    out.solved          = false;
    out.breakdown_order = 1;
    return out;
  }
  std::vector<double> f{produced(1. / row[0])}, b = f, x{produced(rhs[0] / row[0])};
  out.work.completed_orders = 1;
  state(trace, f, b, x);
  for (std::size_t s = 1; s < n; ++s) {
    double ef = 0, eb = 0, ex = 0;
    for (std::size_t i = 0; i < s; ++i) {
      ef = produced(ef + produced(row[s - i] * f[i]));
      eb = produced(eb + produced(row[i + 1] * b[i]));
      ex = produced(ex + produced(row[s - i] * x[i]));
      out.work.dot_products += 3;
    }
    const double denom = produced(1. - produced(eb * ef));
    if (trace)
      trace->attempts.push_back({s + 1, ef, eb, ex, denom});
    if (denom == 0) {
      out.solved          = false;
      out.breakdown_order = s + 1;
      break;
    }
    std::vector<double> fn(s + 1), bn(s + 1), xn(s + 1);
    for (std::size_t j = 0; j <= s; ++j) {
      const double fp = j < s ? f[j] : 0, bp = j ? b[j - 1] : 0;
      fn[j] = produced(produced(fp - produced(ef * bp)) / denom);
      bn[j] = produced(produced(bp - produced(eb * fp)) / denom);
      out.work.vector_updates += 2;
    }
    for (std::size_t j = 0; j <= s; ++j) {
      const double xp = j < s ? x[j] : 0;
      xn[j]           = produced(xp + produced(produced(rhs[s] - ex) * bn[j]));
      ++out.work.vector_updates;
    }
    f = std::move(fn);
    b = std::move(bn);
    x = std::move(xn);
    ++out.work.completed_orders;
    state(trace, f, b, x);
  }
  out.solution = std::move(x);
  return out;
}
} // namespace levinson_detail
std::vector<double> levinson_solve(const std::vector<double>& row, const std::vector<double>& rhs) {
  auto a = levinson_analyze(row, rhs);
  if (!a.solved)
    throw std::domain_error("Levinson exact-zero leading-block breakdown");
  return std::move(a.solution);
}
LevinsonAnalysis levinson_analyze(const std::vector<double>& row, const std::vector<double>& rhs) {
  levinson_detail::validate(row, rhs);
  return levinson_detail::run(row, rhs, nullptr);
}
LevinsonTrace levinson_trace(
    const std::vector<double>& row,
    const std::vector<double>& rhs,
    std::size_t                max_entries) {
  levinson_detail::validate(row, rhs);
  const auto  n = row.size(), max = std::numeric_limits<std::size_t>::max();
  std::size_t bound = 2 * n;
  for (std::size_t k = 1; k <= n; ++k) {
    const auto extra = 1 + 3 * k + (k > 1);
    if (extra > max - bound)
      throw std::length_error("Levinson capture geometry");
    bound += extra;
  }
  if (max_entries && bound > max_entries)
    throw std::length_error("Levinson conservative complete quota");
  LevinsonTrace out;
  out.first_row = row;
  out.rhs       = rhs;
  out.states.reserve(n);
  out.attempts.reserve(n ? n - 1 : 0);
  out.analysis         = levinson_detail::run(row, rhs, &out);
  out.captured_entries = 2 * n + out.attempts.size();
  for (auto& s : out.states)
    out.captured_entries += 1 + 3 * s.order;
  return out;
}
std::vector<std::vector<double>> symmetric_toeplitz_matrix(const std::vector<double>& row) {
  levinson_detail::finite_input(row);
  const auto n = row.size();
  if (n && (n > std::vector<std::vector<double>>{}.max_size() ||
            n > std::vector<double>{}.max_size() / n))
    throw std::length_error("Toeplitz matrix geometry");
  std::vector<std::vector<double>> out(n, std::vector<double>(n));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      out[i][j] = row[i > j ? i - j : j - i];
  return out;
}
LevinsonResidual levinson_residual(
    const std::vector<double>& row,
    const std::vector<double>& rhs,
    const std::vector<double>& x) {
  if (row.size() != rhs.size() || row.size() != x.size())
    throw std::invalid_argument("Toeplitz residual matched shapes");
  levinson_detail::finite_input(row);
  levinson_detail::finite_input(rhs);
  levinson_detail::finite_input(x);
  long double amax = 0, xmax = 0, bmax = 0, rmax = 0;
  for (std::size_t i = 0; i < row.size(); ++i) {
    long double dot = 0, asum = 0;
    for (std::size_t j = 0; j < row.size(); ++j) {
      const long double a = row[i > j ? i - j : j - i];
      dot += a * x[j];
      asum += std::abs(a);
    }
    amax = std::max(amax, asum);
    xmax = std::max(xmax, std::abs(static_cast<long double>(x[i])));
    bmax = std::max(bmax, std::abs(static_cast<long double>(rhs[i])));
    rmax = std::max(rmax, std::abs(dot - rhs[i]));
    if (!std::isfinite(dot) || !std::isfinite(asum))
      throw std::overflow_error("Toeplitz widened residual range");
  }
  const long double denom = amax * xmax + bmax;
  if (!std::isfinite(denom) || !std::isfinite(rmax))
    throw std::overflow_error("Toeplitz widened residual norm range");
  return {rmax, denom ? rmax / denom : 0};
}
} // namespace datamunge::algorithms
