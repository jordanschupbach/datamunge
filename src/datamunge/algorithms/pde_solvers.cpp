/// @file
/// @brief Checked uniform Poisson assembly and existing advection/ODE kernels.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <datamunge/algorithms/linear_solvers.hpp>
#include <datamunge/algorithms/pde_solvers.hpp>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

/// @brief Canonical numerical algorithms and their owned results.
namespace datamunge::algorithms {
/// @brief Internal Poisson validation and single-pass assembly.
namespace {
/// @brief Require a representable finite scalar before storing numerical output.
/// @param x Candidate scalar.
/// @return x unchanged when finite.
/// @throws std::overflow_error If x is nonfinite.
double poisson_finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Poisson arithmetic is nonfinite");
  return x;
}

/// @brief Shared checked assembly and canonical solve, with optional owned diagnostics.
/// @param f Interior source callback, invoked exactly n times on success.
/// @param alpha Left boundary value.
/// @param beta Right boundary value.
/// @param n Positive interior count with signed n+2 representable.
/// @param capture Optional diagnostics destination; not exposed on failure.
/// @return Owned nodal solution including both boundaries.
/// @throws std::invalid_argument For invalid controls or nonfinite callback samples.
/// @throws std::length_error For signed-grid or vector capacity overflow.
/// @throws std::overflow_error For nonfinite assembly, solve or stored diagnostics.
/// @throws std::bad_alloc If numeric storage cannot be allocated.
std::vector<double> poisson_kernel(
    const std::function<double(double)>& f,
    double                               alpha,
    double                               beta,
    int                                  n,
    FiniteDifferencePoissonResult*       capture) {
  if (n <= 0 || !f || !std::isfinite(alpha) || !std::isfinite(beta))
    throw std::invalid_argument("Poisson requires positive n, a source and finite boundaries");
  if (n > INT_MAX - 2)
    throw std::length_error("Poisson signed grid count overflow");
  const auto count = static_cast<std::size_t>(n);
  if (count > std::vector<double>{}.max_size() - 2)
    throw std::length_error("Poisson vector capacity");
  const double        h = 1.0 / (n + 1), h2 = h * h;
  const double        diagonal = 2.0 / h2, off = -1.0 / h2;
  const double        left = poisson_finite(alpha / h2), right = poisson_finite(beta / h2);
  std::vector<double> a(count, off), b(count, diagonal), c(count, off), d(count);
  for (std::size_t i = 0; i < count; ++i) {
    d[i] = f((i + 1) * h);
    if (!std::isfinite(d[i]))
      throw std::invalid_argument("Poisson source sample is nonfinite");
  }
  if (capture) {
    capture->h            = h;
    capture->diagonal     = diagonal;
    capture->off_diagonal = off;
    capture->source       = d;
  }
  d[0]         = poisson_finite(d[0] + left);
  d[count - 1] = poisson_finite(d[count - 1] + right);
  if (capture)
    capture->rhs = d;
  const auto          u = thomas_solve(std::move(a), std::move(b), std::move(c), std::move(d));
  std::vector<double> out(count + 2);
  out[0]         = alpha;
  out[count + 1] = beta;
  for (std::size_t i = 0; i < count; ++i)
    out[i + 1] = poisson_finite(u[i]);
  if (capture) {
    capture->x.resize(count + 2);
    capture->x[0]         = 0;
    capture->x[count + 1] = 1;
    for (std::size_t i = 0; i < count; ++i)
      capture->x[i + 1] = (i + 1) * h;
    capture->residual.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      long double value = static_cast<long double>(diagonal) * u[i];
      if (i > 0)
        value += static_cast<long double>(off) * u[i - 1];
      if (i + 1 < count)
        value += static_cast<long double>(off) * u[i + 1];
      value -= capture->rhs[i];
      if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("Poisson residual does not fit double");
      capture->residual[i] = static_cast<double>(value);
      capture->max_abs_residual =
          std::max(capture->max_abs_residual, std::abs(capture->residual[i]));
    }
  }
  return out;
}
} // namespace

std::vector<double> finite_difference_poisson(
    const std::function<double(double)>& f,
    double                               alpha,
    double                               beta,
    int                                  n) {
  return poisson_kernel(f, alpha, beta, n, nullptr);
}

FiniteDifferencePoissonResult finite_difference_poisson_solve(
    const std::function<double(double)>& f,
    double                               alpha,
    double                               beta,
    int                                  n) {
  FiniteDifferencePoissonResult result;
  result.u = poisson_kernel(f, alpha, beta, n, &result);
  return result;
}

std::vector<double>
lax_wendroff_advection(const std::vector<double>& u0, double a, double dx, double dt, int nsteps) {
  const int           m  = static_cast<int>(u0.size());
  const double        nu = a * dt / dx; // Courant number
  std::vector<double> u  = u0;
  for (int step = 0; step < nsteps; ++step) {
    std::vector<double> un(m);
    for (int j = 0; j < m; ++j) {
      const double up = u[(j + 1) % m];
      const double um = u[(j - 1 + m) % m];
      un[j]           = u[j] - 0.5 * nu * (up - um) + 0.5 * nu * nu * (up - 2.0 * u[j] + um);
    }
    u.swap(un);
  }
  return u;
}

std::vector<std::pair<double, double>> trapezoidal_ode(
    const std::function<double(double, double)>& f,
    double                                       y0,
    double                                       t0,
    double                                       t1,
    int                                          steps) {
  const double                           h = (t1 - t0) / steps;
  std::vector<std::pair<double, double>> traj;
  double                                 t = t0, y = y0;
  traj.emplace_back(t, y);
  for (int n = 0; n < steps; ++n) {
    const double tn1 = t0 + (n + 1) * h;
    const double fn  = f(t, y);
    // Solve Y = y + (h/2)(fn + f(tn1, Y)) by Newton with a finite-difference derivative.
    double Y = y + h * fn; // forward-Euler predictor
    for (int it = 0; it < 60; ++it) {
      const double g   = Y - y - 0.5 * h * (fn + f(tn1, Y));
      const double eps = 1e-8 * (std::fabs(Y) + 1e-8);
      const double gp  = 1.0 - 0.5 * h * (f(tn1, Y + eps) - f(tn1, Y)) / eps;
      const double Yn  = Y - g / gp;
      if (std::fabs(Yn - Y) <= 1e-14 * (std::fabs(Yn) + 1e-14)) {
        Y = Yn;
        break;
      }
      Y = Yn;
    }
    y = Y;
    t = tn1;
    traj.emplace_back(t, y);
  }
  return traj;
}

} // namespace datamunge::algorithms
