/// @file lax_wendroff.cpp
/// @brief Shared checked periodic stencil producer with actual copy/general/widened decisions and complete capture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/lax_wendroff.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Complete coefficient/count validation and actual stencil/diagnostic helpers.
namespace advection_detail {
/// @brief Add a logical producing count without wrap.
/// @param count Mutable actual selected count.
/// @param amount Additional logical records/operations.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& count, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("Lax-Wendroff count overflow");
  count += amount;
}
/// @brief Require finite widened diagnostic arithmetic.
/// @param value Actual widened value.
/// @return Unchanged finite widened value.
/// @throws std::overflow_error On nonfinite widened arithmetic.
long double wide_finite(long double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Lax-Wendroff nonfinite widened arithmetic");
  return value;
}
/// @brief Convert actual widened coefficient/update to a finite double.
/// @param value Actual widened arithmetic result.
/// @return Rounded finite double.
/// @throws std::overflow_error On nonfinite/out-of-double-range value.
double finite_double(long double value) {
  wide_finite(value);
  if (std::abs(value) > std::numeric_limits<double>::max())
    throw std::overflow_error("Lax-Wendroff double range");
  return static_cast<double>(value);
}
/// @brief Validate complete finite grid, parameters, selected-work geometry and optional CFL enforcement.
/// @param u0 Borrowed complete original nonempty finite grid.
/// @param a Actual finite signed speed.
/// @param dx Actual positive finite spacing.
/// @param dt Actual nonnegative finite timestep.
/// @param steps Actual nonnegative complete pass count.
/// @param allow_unstable Explicit CFL opt-out.
/// @return Actual finite computed Courant coefficient.
/// @throws std::invalid_argument On complete input/parameter/enforced CFL violations.
/// @throws std::length_error On complete selected-work count geometry.
/// @throws std::overflow_error On unrepresentable Courant arithmetic.
double validate(
    const std::vector<double>& u0,
    double                     a,
    double                     dx,
    double                     dt,
    int                        steps,
    bool                       allow_unstable) {
  if (u0.empty() || !std::isfinite(a) || !std::isfinite(dx) || dx <= 0 || !std::isfinite(dt) ||
      dt < 0 || steps < 0)
    throw std::invalid_argument("Lax-Wendroff shape/parameters");
  for (double value : u0)
    if (!std::isfinite(value))
      throw std::invalid_argument("Lax-Wendroff finite grid");
  if (u0.size() > std::numeric_limits<std::size_t>::max() / 3 ||
      static_cast<std::size_t>(steps) > std::numeric_limits<std::size_t>::max() / (3 * u0.size()))
    throw std::length_error("Lax-Wendroff complete work geometry");
  const double product = a * dt, ordinary = product / dx;
  const double nu = std::isfinite(product) && std::isfinite(ordinary)
                        ? ordinary
                        : finite_double(static_cast<long double>(a) * dt / dx);
  if (!allow_unstable && std::abs(nu) > 1)
    throw std::invalid_argument("Lax-Wendroff requires abs(Courant)<=1");
  return nu;
}
/// @brief Observe one actual complete state in widened time/mass/L2 arithmetic.
/// @param u Borrowed actual producing complete grid.
/// @param completed Actual completed full step count.
/// @param dx Actual positive spacing.
/// @param dt Actual nonnegative timestep.
/// @param work Mutable actual additional diagnostic count.
/// @return Owned actual state observations.
/// @throws std::length_error On diagnostic count overflow.
/// @throws std::overflow_error On nonfinite widened diagnostic arithmetic.
LaxWendroffState observe(
    const std::vector<double>& u,
    std::size_t                completed,
    double                     dx,
    double                     dt,
    LaxWendroffWork&           work) {
  LaxWendroffState out;
  out.completed_steps = completed;
  out.time            = wide_finite(static_cast<long double>(completed) * dt);
  out.minimum = out.maximum = u.front();
  long double sum = 0, squares = 0;
  for (double value : u) {
    sum         = wide_finite(sum + value);
    squares     = wide_finite(squares + static_cast<long double>(value) * value);
    out.minimum = std::min(out.minimum, value);
    out.maximum = std::max(out.maximum, value);
    add(work.diagnostics_cells, 1);
  }
  out.mass    = wide_finite(sum * dx);
  out.l2_norm = std::sqrt(wide_finite(squares * dx));
  return out;
}
/// @brief Produce actual synchronous periodic stencil updates with optional original/state/cell observation.
/// @param u0 Validated complete finite original values.
/// @param a Validated actual speed.
/// @param dx Validated actual spacing.
/// @param dt Validated actual timestep.
/// @param steps Validated actual complete count.
/// @param allow_unstable Actual explicit CFL policy.
/// @param nu Actual validated finite Courant coefficient.
/// @param trace Optional completely preadmitted producing capture.
/// @return Owned actual final grid/Courant/work.
/// @throws std::length_error On actual operation/container limits.
/// @throws std::overflow_error On nonfinite produced values/diagnostics.
/// @throws std::bad_alloc On runtime/capture allocation failure.
LaxWendroffAnalysis run(
    const std::vector<double>& u0,
    double                     a,
    double                     dx,
    double                     dt,
    int                        steps,
    bool                       allow_unstable,
    double                     nu,
    LaxWendroffTrace*          trace) {
  LaxWendroffAnalysis out;
  out.courant           = nu;
  out.allow_unstable    = allow_unstable;
  std::vector<double> u = u0, next(u.size());
  const auto          n = u.size();
  if (trace) {
    trace->speed    = a;
    trace->spacing  = dx;
    trace->timestep = dt;
    trace->states.push_back(u);
    trace->diagnostics.push_back(observe(u, 0, dx, dt, out.work));
  }
  for (int step = 0; step < steps; ++step) {
    for (std::size_t j = 0; j < n; ++j) {
      const auto   left_index = j ? j - 1 : n - 1, right_index = j + 1 == n ? 0 : j + 1;
      const double left = u[left_index], center = u[j], right = u[right_index];
      add(out.work.stencil_reads, 3);
      LaxWendroffCell cell;
      cell.step        = step;
      cell.index       = j;
      cell.left_index  = left_index;
      cell.right_index = right_index;
      cell.left        = left;
      cell.center      = center;
      cell.right       = right;
      if (nu == 0) {
        cell.kind  = LaxWendroffCellKind::CopyCenter;
        cell.value = center;
        add(out.work.exact_copies, 1);
      } else if (nu == 1 || nu == -1) {
        cell.kind  = nu == 1 ? LaxWendroffCellKind::ShiftRight : LaxWendroffCellKind::ShiftLeft;
        cell.value = nu == 1 ? left : right;
        add(out.work.exact_copies, 1);
      } else {
        const double first_difference = right - left, twice_center = 2. * center,
                     second_difference = right - twice_center + left,
                     first = .5 * nu * first_difference, second = .5 * nu * nu * second_difference,
                     value = center - first + second;
        add(out.work.general_updates, 1);
        if (std::isfinite(first_difference) && std::isfinite(twice_center) &&
            std::isfinite(second_difference) && std::isfinite(first) && std::isfinite(second) &&
            std::isfinite(value)) {
          cell.advective_term = first;
          cell.diffusive_term = second;
          cell.value          = value;
        } else {
          cell.widened        = true;
          cell.advective_term = wide_finite(.5L * nu * (static_cast<long double>(right) - left));
          cell.diffusive_term =
              wide_finite(.5L * nu * nu * (static_cast<long double>(right) - 2.L * center + left));
          cell.value = finite_double(
              static_cast<long double>(center) - cell.advective_term + cell.diffusive_term);
          add(out.work.widened_updates, 1);
        }
      }
      next[j] = cell.value;
      add(out.work.cell_updates, 1);
      if (trace)
        trace->cells.push_back(cell);
    }
    u.swap(next);
    add(out.work.steps, 1);
    if (trace) {
      trace->states.push_back(u);
      trace->diagnostics.push_back(
          observe(u, static_cast<std::size_t>(step) + 1, dx, dt, out.work));
    }
  }
  out.values = std::move(u);
  return out;
}
} // namespace advection_detail
std::complex<double> lax_wendroff_amplification(double courant, double angle) {
  if (!std::isfinite(courant) || !std::isfinite(angle))
    throw std::invalid_argument("Lax-Wendroff finite symbol inputs");
  const long double nu = courant;
  return {
      advection_detail::finite_double(1 + nu * nu * (std::cos(angle) - 1.L)),
      advection_detail::finite_double(-nu * std::sin(angle))};
}
std::vector<double>
lax_wendroff_advection(const std::vector<double>& u0, double a, double dx, double dt, int nsteps) {
  const auto nu = advection_detail::validate(u0, a, dx, dt, nsteps, false);
  return advection_detail::run(u0, a, dx, dt, nsteps, false, nu, nullptr).values;
}
LaxWendroffAnalysis lax_wendroff_analyze(
    const std::vector<double>& u0,
    double                     a,
    double                     dx,
    double                     dt,
    int                        nsteps,
    bool                       allow_unstable) {
  const auto nu = advection_detail::validate(u0, a, dx, dt, nsteps, allow_unstable);
  return advection_detail::run(u0, a, dx, dt, nsteps, allow_unstable, nu, nullptr);
}
LaxWendroffTrace lax_wendroff_trace(
    const std::vector<double>& u0,
    double                     a,
    double                     dx,
    double                     dt,
    int                        nsteps,
    bool                       allow_unstable,
    std::size_t                max_entries) {
  const auto  nu      = advection_detail::validate(u0, a, dx, dt, nsteps, allow_unstable);
  const auto  n       = u0.size();
  std::size_t initial = n;
  advection_detail::add(initial, 1);
  std::size_t per = n;
  advection_detail::add(per, n);
  advection_detail::add(per, 1);
  if (static_cast<std::size_t>(nsteps) > std::numeric_limits<std::size_t>::max() / per)
    throw std::length_error("Lax-Wendroff capture geometry");
  auto capture = initial;
  advection_detail::add(capture, static_cast<std::size_t>(nsteps) * per);
  if (max_entries && capture > max_entries)
    throw std::length_error("Lax-Wendroff trace quota");
  LaxWendroffTrace out;
  out.captured_entries = capture;
  out.states.reserve(static_cast<std::size_t>(nsteps) + 1);
  out.diagnostics.reserve(static_cast<std::size_t>(nsteps) + 1);
  out.cells.reserve(static_cast<std::size_t>(nsteps) * n);
  out.analysis = advection_detail::run(u0, a, dx, dt, nsteps, allow_unstable, nu, &out);
  return out;
}
} // namespace datamunge::algorithms
