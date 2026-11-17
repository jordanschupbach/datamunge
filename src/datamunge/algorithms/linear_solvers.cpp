/// @file linear_solvers.cpp
/// @brief Implementations of the dense and matrix-free linear solvers.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <datamunge/algorithms/linear_solvers.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Helpers for linear_solvers.cpp.
namespace {

using Matrix = std::vector<std::vector<double>>; ///< Dense row-major matrix.
using Vector = std::vector<double>;              ///< Dense vector.

/// Dense product A x.
Vector matvec(const Matrix& A, const Vector& x) {
  Vector y(A.size(), 0.0);
  for (std::size_t i = 0; i < A.size(); ++i)
    for (std::size_t j = 0; j < x.size(); ++j)
      y[i] += A[i][j] * x[j];
  return y;
}
/// Dense product A^T x.
Vector matvec_transpose(const Matrix& A, const Vector& x) {
  const std::size_t n = A.size();
  Vector            y(n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      y[j] += A[i][j] * x[i]; // (A^T x)_j = sum_i A[i][j] x[i]
  return y;
}
/// Inner product of a and b.
double dot(const Vector& a, const Vector& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    s += a[i] * b[i];
  return s;
}
/// Euclidean norm of A x - b.
double residual_norm(const Matrix& A, const Vector& x, const Vector& b) {
  const Vector Ax = matvec(A, x);
  double       s  = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i)
    s += (Ax[i] - b[i]) * (Ax[i] - b[i]);
  return std::sqrt(s);
}

} // namespace



IterativeSolution biconjugate_gradient(
    const std::vector<std::vector<double>>& a,
    const std::vector<double>&              b,
    int                                     max_iter,
    double                                  tol) {
  const std::size_t n = a.size();
  IterativeSolution out;
  Vector            x(n, 0.0);
  Vector            r  = b; // residual (x0 = 0)
  Vector            rt = r; // shadow residual
  Vector            p = r, pt = rt;
  double            rho_old = dot(rt, r);

  for (int it = 1; it <= max_iter; ++it) {
    const Vector Ap  = matvec(a, p);
    const Vector Atp = matvec_transpose(a, pt);
    const double pAp = dot(pt, Ap);
    if (std::fabs(pAp) < 1e-30)
      break; // breakdown
    const double alpha = rho_old / pAp;
    for (std::size_t i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      rt[i] -= alpha * Atp[i];
    }
    out.iterations = it;
    out.residual   = std::sqrt(dot(r, r));
    if (out.residual < tol) {
      out.converged = true;
      break;
    }
    const double rho_new = dot(rt, r);
    if (std::fabs(rho_new) < 1e-30)
      break; // breakdown
    const double beta = rho_new / rho_old;
    for (std::size_t i = 0; i < n; ++i) {
      p[i]  = r[i] + beta * p[i];
      pt[i] = rt[i] + beta * pt[i];
    }
    rho_old = rho_new;
  }
  out.x = std::move(x);
  return out;
}

BiCGResult biconjugate_gradient(
    const LinearOperator&      apply,
    const LinearOperator&      apply_transpose,
    const std::vector<double>& b,
    int                        max_iter,
    double                     tol,
    const std::vector<double>& shadow) {
  const std::size_t n = b.size();
  if (!shadow.empty() && shadow.size() != n)
    throw std::invalid_argument(
        "biconjugate_gradient: the shadow residual must have the size of b");
  BiCGResult out;
  out.x.assign(n, 0.0);
  const double bnorm = std::sqrt(dot(b, b));
  if (bnorm == 0.0) {
    out.converged        = true;
    out.residual_history = {0.0};
    return out;
  }
  Vector       r = b, rt = shadow.empty() ? b : shadow, p = r, pt = rt;
  const double rt0 = std::sqrt(dot(rt, rt));
  double       rho = dot(rt, r);
  if (std::fabs(rho) <= 1e-15 * rt0 * bnorm) {
    out.breakdown = true;
    out.residual_history.push_back(1.0);
    return out;
  }
  out.residual_history.push_back(1.0);
  for (int it = 1; it <= max_iter; ++it) {
    const Vector Ap  = apply(p);
    const Vector Atp = apply_transpose(pt);
    out.products += 2;
    if (Ap.size() != n || Atp.size() != n)
      throw std::invalid_argument(
          "biconjugate_gradient: the operators must return vectors of the size of b");
    const double pAp = dot(pt, Ap);
    if (std::fabs(pAp) <= 1e-300 + 1e-15 * std::sqrt(dot(pt, pt) * dot(Ap, Ap))) {
      out.breakdown = true;
      break;
    }
    const double alpha = rho / pAp;
    for (std::size_t i = 0; i < n; ++i) {
      out.x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      rt[i] -= alpha * Atp[i];
    }
    out.iterations   = it;
    const double rel = std::sqrt(dot(r, r)) / bnorm;
    out.residual_history.push_back(rel);
    if (rel <= tol) {
      out.converged = true;
      break;
    }
    const double rho_new = dot(rt, r);
    const double rt_norm = std::sqrt(dot(rt, rt));
    // A shadow that collapses far faster than the residual is a serious breakdown in disguise.
    if (rt_norm <= 1e-10 * rel * rt0 ||
        std::fabs(rho_new) <= 1e-300 + 1e-15 * rt_norm * std::sqrt(dot(r, r))) {
      out.breakdown = true;
      break;
    }
    const double beta = rho_new / rho;
    for (std::size_t i = 0; i < n; ++i) {
      p[i]  = r[i] + beta * p[i];
      pt[i] = rt[i] + beta * pt[i];
    }
    rho = rho_new;
  }
  return out;
}

} // namespace datamunge::algorithms
