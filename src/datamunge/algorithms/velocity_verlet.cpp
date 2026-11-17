/// @file velocity_verlet.cpp
/// @brief Checked producing velocity-Verlet drift/kick with actual acceleration snapshots.
#include <datamunge/algorithms/velocity_verlet.hpp>
namespace datamunge::algorithms {
/// @brief Implement the checked drift/kick producer and capture the same executed operands.
VerletWork velocity_verlet_work(
    const std::function<std::vector<double>(const std::vector<double>&)>& accel,
    std::vector<double>                                                   x,
    std::vector<double>                                                   v,
    double                                                                t0,
    double                                                                t1,
    int                                                                   count,
    std::size_t                                                           max_records) {
  if (!accel || x.empty() || x.size() != v.size() || count <= 0 || count > 1000000 ||
      !std::isfinite(t0) || !std::isfinite(t1))
    throw std::invalid_argument("Verlet inputs");
  if (x.size() > 2000000 / (2 * (std::size_t(count) + 1)) || max_records > 1000000 / (6 * x.size()))
    throw std::length_error("Verlet scalar budget");
  for (double q : x)
    if (!std::isfinite(q))
      throw std::invalid_argument("Verlet position");
  for (double q : v)
    if (!std::isfinite(q))
      throw std::invalid_argument("Verlet velocity");
  VerletWork r;
  r.step_size = (t1 - t0) / count;
  double h    = r.step_size;
  if (!std::isfinite(h) || (h == 0 && t1 != t0))
    throw std::overflow_error("Verlet step");
  r.solution.t.push_back(t0);
  r.solution.x.push_back(x);
  r.solution.v.push_back(v);
  // Validate each actual callback result before any indexed acceleration read.
  auto acceleration = [&](const std::vector<double>& at) {
    ++r.acceleration_calls;
    auto a = accel(at);
    if (a.size() != x.size())
      throw std::invalid_argument("Verlet acceleration shape");
    for (double q : a)
      if (!std::isfinite(q))
        throw std::invalid_argument("Verlet acceleration nonfinite");
    return a;
  };
  std::vector<double> a(x.size());
  if (h != 0)
    a = acceleration(x);
  for (int n = 0; n < count; ++n) {
    VerletStep record;
    if (r.steps.size() < max_records) {
      record.before_x         = x;
      record.before_v         = v;
      record.old_acceleration = a;
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += v[i] * h + .5 * a[i] * h * h;
      if (!std::isfinite(x[i]))
        throw std::overflow_error("Verlet drift");
    }
    auto next = h == 0 ? a : acceleration(x);
    for (std::size_t i = 0; i < v.size(); ++i) {
      v[i] += .5 * (a[i] + next[i]) * h;
      if (!std::isfinite(v[i]))
        throw std::overflow_error("Verlet kick");
    }
    if (r.steps.size() < max_records) {
      record.new_acceleration = next;
      record.after_x          = x;
      record.after_v          = v;
      r.steps.push_back(std::move(record));
    }
    a = std::move(next);
    r.solution.t.push_back(n == count - 1 ? t1 : t0 + (n + 1) * h);
    r.solution.x.push_back(x);
    r.solution.v.push_back(v);
  }
  return r;
}
/// @brief Delegate the legacy trajectory API to the same checked producing kernel.
VerletSolution velocity_verlet(
    const std::function<std::vector<double>(const std::vector<double>&)>& accel,
    std::vector<double>                                                   x,
    std::vector<double>                                                   v,
    double                                                                t0,
    double                                                                t1,
    int                                                                   count) {
  return velocity_verlet_work(accel, std::move(x), std::move(v), t0, t1, count).solution;
}
} // namespace datamunge::algorithms
