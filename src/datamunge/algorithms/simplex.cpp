/// @file simplex.cpp
/// @brief One bounded actual primal tableau producer with explicit unresolved status.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/simplex.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Check finite actual tableau arithmetic.
/// @param x Actual computed value.
/// @return Same finite value.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::domain_error("simplex: nonfinite arithmetic");
  return x;
}
/// @brief Execute validated feasible-origin pivots once.
/// @param c Objective.
/// @param A Constraints.
/// @param b Right-hand sides.
/// @param o Policies.
/// @param capture Retain actual post-pivot tableaux.
/// @return Owned actual output.
SimplexRun run(
    const std::vector<double>&              c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SimplexOptions&                   o,
    bool                                    capture) {
  const auto m = A.size(), n = c.size(), limit = std::numeric_limits<std::size_t>::max();
  if (b.size() != m || !std::isfinite(o.tolerance) || o.tolerance <= 0)
    throw std::invalid_argument("simplex: shape/tolerance");
  for (double x : c)
    if (!std::isfinite(x))
      throw std::invalid_argument("simplex: objective");
  for (std::size_t i = 0; i < m; ++i) {
    if (A[i].size() != n || !std::isfinite(b[i]) || b[i] < 0)
      throw std::invalid_argument("simplex: constraint shape/rhs");
    for (double x : A[i])
      if (!std::isfinite(x))
        throw std::invalid_argument("simplex: coefficient");
  }
  if (m == limit || n > limit - m - 1)
    throw std::length_error("simplex: dimensions");
  const auto total = n + m, width = total + 1, height = m + 1;
  if (width > limit / height)
    throw std::length_error("simplex: cells capacity");
  const auto cells = width * height;
  if ((o.max_cells && cells > o.max_cells) || width > std::vector<double>{}.max_size() ||
      height > std::vector<std::vector<double>>{}.max_size())
    throw std::length_error("simplex: tableau policy");
  std::vector<std::vector<double>> T(height, std::vector<double>(width));
  std::vector<std::size_t>         basis(m);
  for (std::size_t i = 0; i < m; ++i) {
    std::copy(A[i].begin(), A[i].end(), T[i].begin());
    T[i][n + i] = 1;
    T[i][total] = b[i];
    basis[i]    = n + i;
  }
  for (std::size_t j = 0; j < n; ++j)
    T[m][j] = -c[j];
  SimplexRun  r;
  std::size_t retained = 0;
  for (;;) {
    std::size_t entering = total;
    for (std::size_t j = 0; j < total; ++j)
      if (T[m][j] < -o.tolerance) {
        entering = j;
        break;
      }
    if (entering == total) {
      r.result.status = SimplexStatus::Optimal;
      break;
    }
    if (r.pivots == o.max_pivots) {
      r.result.status = SimplexStatus::IterationLimit;
      break;
    }
    std::size_t leaving = m;
    double      best    = 0;
    for (std::size_t i = 0; i < m; ++i)
      if (T[i][entering] > o.tolerance) {
        const double ratio = finite(T[i][total] / T[i][entering]);
        if (leaving == m || ratio < best || (ratio == best && basis[i] < basis[leaving])) {
          best    = ratio;
          leaving = i;
        }
      }
    if (leaving == m) {
      r.result = {SimplexStatus::Unbounded, 0, {}};
      return r;
    }
    if (cells > limit - m - 1)
      throw std::length_error("simplex: capture capacity");
    const auto copied = cells + m + 1;
    if (capture) {
      if (copied > limit - retained)
        throw std::length_error("simplex: capture capacity");
      retained += copied;
      if (o.max_history_values && retained > o.max_history_values)
        throw std::length_error("simplex: history");
    }
    const double pivot = T[leaving][entering];
    const auto   old   = basis[leaving];
    auto         count = [&]() {
      if (r.cell_updates == limit)
        throw std::length_error("simplex: cell counter");
      ++r.cell_updates;
    };
    for (double& x : T[leaving]) {
      x = finite(x / pivot);
      count();
    }
    for (std::size_t i = 0; i < height; ++i)
      if (i != leaving) {
        const double factor = T[i][entering];
        if (factor == 0)
          continue;
        for (std::size_t j = 0; j < width; ++j) {
          T[i][j] = finite(T[i][j] - factor * T[leaving][j]);
          count();
        }
      }
    basis[leaving] = entering;
    ++r.pivots;
    if (capture)
      r.history.push_back({entering, old, leaving, pivot, basis, T});
  }
  r.result.x.assign(n, 0);
  for (std::size_t i = 0; i < m; ++i)
    if (basis[i] < n)
      r.result.x[basis[i]] = T[i][total];
  r.result.value = finite(T[m][total]);
  return r;
}
} // namespace
SimplexRun simplex_run(
    const std::vector<double>&              c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SimplexOptions&                   o) {
  return run(c, A, b, o, false);
}
SimplexRun simplex_trace(
    const std::vector<double>&              c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SimplexOptions&                   o) {
  return run(c, A, b, o, true);
}
SimplexResult simplex_maximize(
    const std::vector<double>&              c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b) {
  return simplex_run(c, A, b).result;
}
} // namespace datamunge::algorithms
