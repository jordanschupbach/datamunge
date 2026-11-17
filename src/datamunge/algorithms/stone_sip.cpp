/// @file stone_sip.cpp
/// @brief One checked actual SIP factor/update producer and original manufactured grid.
#include <cmath>
#include <datamunge/algorithms/stone_sip.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Translation-unit-private checked producing kernels; no shared mutable state.
namespace {
/// @brief Require representable actual arithmetic.
/// @param value Actual value.
/// @return Same finite value.
double finite(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("SIP: arithmetic range");
  return value;
}
/// @brief Validate complete dimensions and compute capacity-safe cell geometry.
/// @param nx Horizontal order.
/// @param ny Vertical order.
/// @param cap Zero or policy.
/// @return Complete cell count.
std::size_t cells(int nx, int ny, std::size_t cap) {
  if (nx < 0 || ny < 0)
    throw std::invalid_argument("SIP: dimensions");
  const auto x = static_cast<std::size_t>(nx), y = static_cast<std::size_t>(ny);
  if (y && x > std::numeric_limits<std::size_t>::max() / y)
    throw std::length_error("SIP: cells capacity");
  const auto n = x * y;
  if (cap && n > cap)
    throw std::length_error("SIP: cells policy");
  return n;
}
/// @brief Factor and execute actual correction steps once.
/// @param g Complete grid.
/// @param o Controls/policies.
/// @param capture Retain actual intermediate factors/iterates.
/// @return Actual owned result.
StoneRun run(const StoneGrid& g, const StoneOptions& o, bool capture) {
  if (!std::isfinite(o.alpha) || o.alpha < 0 || o.alpha >= 1 || o.max_iterations < 0 ||
      !std::isfinite(o.tolerance) || o.tolerance < 0)
    throw std::invalid_argument("SIP: controls");
  const auto n = cells(g.nx, g.ny, o.max_cells), nx = static_cast<std::size_t>(g.nx),
             ny = static_cast<std::size_t>(g.ny), limit = std::numeric_limits<std::size_t>::max();
  for (const auto* v : {&g.Ap, &g.Aw, &g.Ae, &g.As, &g.An, &g.b}) {
    if (v->size() != n)
      throw std::invalid_argument("SIP: complete vectors");
    for (double value : *v)
      if (!std::isfinite(value))
        throw std::invalid_argument("SIP: finite input");
  }
  for (std::size_t j = 0; j < ny; ++j)
    for (std::size_t i = 0; i < nx; ++i) {
      const auto p = j * nx + i;
      if ((!i && g.Aw[p] != 0) || (i + 1 == nx && g.Ae[p] != 0) || (!j && g.As[p] != 0) ||
          (j + 1 == ny && g.An[p] != 0))
        throw std::invalid_argument("SIP: boundary coefficients");
    }
  StoneRun r;
  r.result.x.assign(n, 0);
  r.Lw.assign(n, 0);
  r.Ls.assign(n, 0);
  r.Lp.assign(n, 0);
  r.Ue.assign(n, 0);
  r.Un.assign(n, 0);
  std::size_t visits = 0, retained = 0;
  auto        count = [&](std::size_t& counter) {
    if (counter == limit || visits == limit || (o.max_grid_visits && visits >= o.max_grid_visits))
      throw std::length_error("SIP: actual grid visits");
    ++counter;
    ++visits;
  };
  auto keep = [&](std::size_t amount) {
    if (amount > limit - retained)
      throw std::length_error("SIP: history capacity");
    retained += amount;
    if (o.max_history_values && retained > o.max_history_values)
      throw std::length_error("SIP: history");
  };
  for (std::size_t j = 0; j < ny; ++j)
    for (std::size_t i = 0; i < nx; ++i) {
      const auto   p   = j * nx + i;
      const double unw = i ? r.Un[p - 1] : 0, uew = i ? r.Ue[p - 1] : 0, ues = j ? r.Ue[p - nx] : 0,
                   uns = j ? r.Un[p - nx] : 0, dw = finite(1 + o.alpha * unw),
                   ds = finite(1 + o.alpha * ues);
      if (dw == 0 || ds == 0)
        throw std::domain_error("SIP: factor denominator");
      r.Lw[p]         = i ? finite(g.Aw[p] / dw) : 0;
      r.Ls[p]         = j ? finite(g.As[p] / ds) : 0;
      const double p1 = finite(o.alpha * r.Lw[p] * unw), p2 = finite(o.alpha * r.Ls[p] * ues);
      r.Lp[p] = finite(g.Ap[p] + p1 + p2 - r.Lw[p] * uew - r.Ls[p] * uns);
      if (r.Lp[p] == 0)
        throw std::domain_error("SIP: zero incomplete pivot");
      r.Ue[p] = finite((g.Ae[p] - p1) / r.Lp[p]);
      r.Un[p] = finite((g.An[p] - p2) / r.Lp[p]);
      count(r.factorized_cells);
      if (capture) {
        keep(5);
        r.factors.push_back({p, dw, ds, p1, p2});
      }
    }
  std::vector<double> residual(n), rho(n), delta(n);
  auto                norm = [&]() {
    double value = 0;
    for (std::size_t j = 0; j < ny; ++j)
      for (std::size_t i = 0; i < nx; ++i) {
        const auto p  = j * nx + i;
        double     ax = g.Ap[p] * r.result.x[p];
        count(r.residual_products);
        if (i) {
          ax += g.Aw[p] * r.result.x[p - 1];
          count(r.residual_products);
        }
        if (i + 1 < nx) {
          ax += g.Ae[p] * r.result.x[p + 1];
          count(r.residual_products);
        }
        if (j) {
          ax += g.As[p] * r.result.x[p - nx];
          count(r.residual_products);
        }
        if (j + 1 < ny) {
          ax += g.An[p] * r.result.x[p + nx];
          count(r.residual_products);
        }
        residual[p] = finite(g.b[p] - ax);
        value       = finite(std::hypot(value, residual[p]));
      }
    return value;
  };
  auto record = [&]() {
    if (capture) {
      if (n == limit)
        throw std::length_error("SIP: iterate capacity");
      keep(n + 1);
      r.history.push_back({r.result.iterations, r.result.residual, r.result.x});
    }
  };
  r.result.residual  = norm();
  r.initial_residual = r.result.residual;
  r.converged        = r.result.residual <= o.tolerance;
  record();
  for (int iteration = 0; iteration < o.max_iterations && !r.converged; ++iteration) {
    for (std::size_t j = 0; j < ny; ++j)
      for (std::size_t i = 0; i < nx; ++i) {
        const auto p = j * nx + i;
        double     v = residual[p];
        if (i)
          v -= r.Lw[p] * rho[p - 1];
        if (j)
          v -= r.Ls[p] * rho[p - nx];
        rho[p] = finite(v / r.Lp[p]);
        count(r.triangular_cells);
      }
    for (std::size_t j = ny; j > 0; --j)
      for (std::size_t i = nx; i > 0; --i) {
        const auto p = (j - 1) * nx + i - 1;
        double     v = rho[p];
        if (i < nx)
          v -= r.Ue[p] * delta[p + 1];
        if (j < ny)
          v -= r.Un[p] * delta[p + nx];
        delta[p] = finite(v);
        count(r.triangular_cells);
      }
    for (std::size_t p = 0; p < n; ++p) {
      r.result.x[p] = finite(r.result.x[p] + delta[p]);
      count(r.correction_assignments);
    }
    r.result.iterations = iteration + 1;
    r.result.residual   = norm();
    r.converged         = r.result.residual <= o.tolerance;
    record();
  }
  return r;
}
} // namespace
StoneRun stone_sip_run(const StoneGrid& g, const StoneOptions& o) {
  return run(g, o, false);
}
StoneRun stone_sip_trace(const StoneGrid& g, const StoneOptions& o) {
  return run(g, o, true);
}
StoneResult stone_sip(
    int                        nx,
    int                        ny,
    const std::vector<double>& ap,
    const std::vector<double>& aw,
    const std::vector<double>& ae,
    const std::vector<double>& as,
    const std::vector<double>& an,
    const std::vector<double>& b,
    double                     alpha,
    int                        cap,
    double                     tol) {
  StoneOptions o;
  o.alpha          = alpha;
  o.max_iterations = cap;
  o.tolerance      = tol;
  return stone_sip_run({nx, ny, ap, aw, ae, as, an, b, {}}, o).result;
}
StoneGrid stone_poisson_example(int nx, int ny, std::uint32_t seed) {
  if (nx < 1 || ny < 1)
    throw std::invalid_argument("SIP: fixture dimensions");
  const auto n = cells(nx, ny, 1000000);
  StoneGrid  g;
  g.nx = nx;
  g.ny = ny;
  g.Ap.assign(n, 4);
  g.Aw.assign(n, 0);
  g.Ae.assign(n, 0);
  g.As.assign(n, 0);
  g.An.assign(n, 0);
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const auto p = static_cast<std::size_t>(j) * nx + i;
      if (i)
        g.Aw[p] = -1;
      if (i + 1 < nx)
        g.Ae[p] = -1;
      if (j)
        g.As[p] = -1;
      if (j + 1 < ny)
        g.An[p] = -1;
    }
  std::mt19937                           rng(seed);
  std::uniform_real_distribution<double> u(-1, 1);
  g.known.resize(n);
  for (double& v : g.known)
    v = u(rng);
  g.b.resize(n);
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const auto p     = static_cast<std::size_t>(j) * nx + i;
      double     value = g.Ap[p] * g.known[p];
      if (i)
        value += g.Aw[p] * g.known[p - 1];
      if (i + 1 < nx)
        value += g.Ae[p] * g.known[p + 1];
      if (j)
        value += g.As[p] * g.known[p - nx];
      if (j + 1 < ny)
        value += g.An[p] * g.known[p + nx];
      g.b[p] = value;
    }
  return g;
}
} // namespace datamunge::algorithms
