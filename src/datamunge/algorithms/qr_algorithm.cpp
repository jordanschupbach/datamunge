/// @file qr_algorithm.cpp
/// @brief Shared checked Householder QR eigenvalue producer.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/qr_algorithm.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical algorithm implementation.
namespace datamunge::algorithms {
/// @brief Private dense QR implementation details.
namespace {
/// @brief Convert checked extended precision arithmetic back to the public scalar.
/// @param value Intermediate result.
/// @return Finite double.
/// @throws std::overflow_error Double result is nonfinite.
double finite(long double value) {
  const double x = static_cast<double>(value);
  if (!std::isfinite(x))
    throw std::overflow_error("qr_eigenvalues: nonfinite arithmetic");
  return x;
}
/// @brief Execute the one shared dense QR solver.
/// @param A Owned input matrix.
/// @param o Validated execution policies.
/// @param capture Retain actual matrices and decisions.
/// @return Converged owned spectrum and optional history.
QrResult solve(QrMatrix A, const QrOptions& o, bool capture) {
  const auto n = A.size();
  if (!std::isfinite(o.tolerance) || o.tolerance <= 0)
    throw std::invalid_argument("qr_eigenvalues: tolerance must be finite and positive");
  if (o.max_order && n > o.max_order)
    throw std::length_error("qr_eigenvalues: order policy");
  for (const auto& row : A) {
    if (row.size() != n)
      throw std::invalid_argument("qr_eigenvalues: nonsquare input");
    for (double x : row)
      if (!std::isfinite(x))
        throw std::invalid_argument("qr_eigenvalues: nonfinite input");
  }
  long double symmetry_scale = 0;
  for (const auto& row : A)
    for (double value : row)
      symmetry_scale = std::max(symmetry_scale, std::fabs(static_cast<long double>(value)));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < i; ++j) {
      const long double a = A[i][j], b = A[j][i];
      if (std::fabs(a - b) > 64 * std::numeric_limits<double>::epsilon() * symmetry_scale)
        throw std::invalid_argument("qr_eigenvalues: nonsymmetric input");
      A[i][j] = A[j][i] = finite(a / 2 + b / 2);
    }
  QrResult    out;
  std::size_t history = 0;
  for (std::size_t m = n; m > 0; --m) {
    std::size_t iteration = 0;
    double      mu        = 0;
    for (;;) {
      long double tail = 0, scale = 0;
      for (std::size_t i = 0; i < m; ++i) {
        scale = std::max(scale, std::fabs(static_cast<long double>(A[i][i])));
        if (i + 1 < m)
          tail = std::hypot(tail, static_cast<long double>(A[m - 1][i]));
      }
      const bool done = tail <= static_cast<long double>(o.tolerance) * scale;
      if (capture && (iteration > 0 || done)) {
        if (n && n > (std::numeric_limits<std::size_t>::max() - 1) / n)
          throw std::length_error("qr_eigenvalues: history size overflow");
        const auto cost = n * n + 1;
        if (cost > std::numeric_limits<std::size_t>::max() - history ||
            (o.max_trace_entries &&
             (history > o.max_trace_entries || cost > o.max_trace_entries - history)))
          throw std::length_error("qr_eigenvalues: history policy");
        history += cost;
        out.steps.push_back({m, iteration, mu, finite(tail), done, A});
      }
      if (done) {
        out.eigenvalues.push_back(A[m - 1][m - 1]);
        ++out.deflations;
        for (std::size_t i = 0; i + 1 < m; ++i)
          A[i][m - 1] = A[m - 1][i] = 0;
        break;
      }
      if (iteration == o.max_iterations)
        throw std::runtime_error("qr_eigenvalues: no convergence within block budget");
      const long double a = A[m - 2][m - 2], b = A[m - 2][m - 1], c = A[m - 1][m - 1],
                        delta = (a - c) / 2;
      const long double denom = std::fabs(delta) + std::hypot(delta, b);
      mu                      = finite(denom > 0 ? c - (delta >= 0 ? 1 : -1) * b * (b / denom) : c);
      QrMatrix R(m, std::vector<double>(m)), Q(m, std::vector<double>(m));
      for (std::size_t i = 0; i < m; ++i) {
        Q[i][i] = 1;
        for (std::size_t j = 0; j < m; ++j)
          R[i][j] = finite(static_cast<long double>(A[i][j]) - (i == j ? mu : 0));
      }
      for (std::size_t k = 0; k < m; ++k) {
        long double norm = 0;
        for (std::size_t i = k; i < m; ++i)
          norm = std::hypot(norm, static_cast<long double>(R[i][k]));
        if (norm == 0)
          continue;
        std::vector<long double> v(m - k);
        for (std::size_t i = k; i < m; ++i)
          v[i - k] = static_cast<long double>(R[i][k]) / norm;
        v[0] += v[0] >= 0 ? 1 : -1;
        long double vn = 0;
        for (auto x : v)
          vn = std::hypot(vn, x);
        for (auto& x : v)
          x /= vn;
        for (std::size_t j = k; j < m; ++j) {
          long double dot = 0;
          for (std::size_t i = k; i < m; ++i)
            dot += v[i - k] * R[i][j];
          for (std::size_t i = k; i < m; ++i)
            R[i][j] = finite(static_cast<long double>(R[i][j]) - 2 * v[i - k] * dot);
        }
        for (std::size_t i = 0; i < m; ++i) {
          long double dot = 0;
          for (std::size_t j = k; j < m; ++j)
            dot += Q[i][j] * v[j - k];
          for (std::size_t j = k; j < m; ++j)
            Q[i][j] = finite(static_cast<long double>(Q[i][j]) - 2 * dot * v[j - k]);
        }
      }
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
          long double dot = 0;
          for (std::size_t k = 0; k < m; ++k)
            dot += static_cast<long double>(R[i][k]) * Q[k][j];
          A[i][j] = finite(dot + (i == j ? mu : 0));
        }
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < i; ++j)
          A[i][j] = A[j][i] =
              finite(static_cast<long double>(A[i][j]) / 2 + static_cast<long double>(A[j][i]) / 2);
      ++iteration;
      ++out.iterations;
    }
  }
  std::sort(out.eigenvalues.begin(), out.eigenvalues.end());
  out.matrix = std::move(A);
  return out;
}
} // namespace
QrResult qr_eigenvalues_run(QrMatrix A, const QrOptions& o) {
  return solve(std::move(A), o, false);
}
QrResult qr_eigenvalues_trace(QrMatrix A, const QrOptions& o) {
  return solve(std::move(A), o, true);
}
std::vector<double> qr_eigenvalues(QrMatrix A, int max_iter, double tol) {
  if (max_iter < 0)
    throw std::invalid_argument("qr_eigenvalues: negative iteration budget");
  QrOptions o;
  o.max_iterations = static_cast<std::size_t>(max_iter);
  o.tolerance      = tol;
  return qr_eigenvalues_run(std::move(A), o).eigenvalues;
}
QrMatrix
qr_tridiagonal(std::size_t n, double diagonal, double off_diagonal, std::size_t max_order) {
  if (!std::isfinite(diagonal) || !std::isfinite(off_diagonal))
    throw std::invalid_argument("qr_tridiagonal: nonfinite coefficient");
  if (max_order && n > max_order)
    throw std::length_error("qr_tridiagonal: order policy");
  QrMatrix a(n, std::vector<double>(n));
  for (std::size_t i = 0; i < n; ++i) {
    a[i][i] = diagonal;
    if (i)
      a[i][i - 1] = a[i - 1][i] = off_diagonal;
  }
  return a;
}
} // namespace datamunge::algorithms
