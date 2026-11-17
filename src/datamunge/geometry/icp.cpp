/// @file icp.cpp
/// @brief Shared checked planar ICP production and actual-state observations.
#include <cmath>
#include <datamunge/geometry/icp.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Private input/arithmetic checks and the single producing ICP kernel.
namespace {
/// @brief Preserve a finite computed value, rejecting arithmetic overflow/NaN.
/// @param value Computed scalar.
/// @return Unchanged finite value.
/// @throws std::overflow_error On nonfinite value.
double checked(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("ICP nonfinite arithmetic");
  return value;
}
/// @brief Increment an actual selected work count without unsigned wrap.
/// @param count Mutable counter.
/// @throws std::length_error On exhausted size_t.
void count_one(std::size_t& count) {
  if (count == std::numeric_limits<std::size_t>::max())
    throw std::length_error("ICP work count");
  ++count;
}
/// @brief Compute the original double squared distance with finite checks.
/// @param p First finite point.
/// @param q Second finite point.
/// @return Finite rounded squared distance, possibly underflowed zero.
/// @throws std::overflow_error On nonfinite differences/products/sum.
double distance_squared(Point2D p, Point2D q) {
  const double dx = checked(p.x - q.x), dy = checked(p.y - q.y);
  return checked(dx * dx + dy * dy);
}
/// @brief Validate complete inputs and controls before any input copy or working arrays.
/// @param source Complete borrowed source.
/// @param target Complete borrowed target.
/// @param max_iters Requested update budget.
/// @param tol Strict absolute pre-update tolerance.
/// @param options Complete input policy.
/// @throws std::invalid_argument On coordinate/control domain errors.
/// @throws std::length_error On count overflow or complete point cap.
void validate(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    int                         max_iters,
    double                      tol,
    IcpOptions                  options) {
  if (max_iters < 0 || !std::isfinite(tol) || tol < 0)
    throw std::invalid_argument("ICP budget/tolerance");
  const auto max = std::numeric_limits<std::size_t>::max();
  if (source.size() > max - target.size())
    throw std::length_error("ICP input count");
  if (options.max_points && source.size() + target.size() > options.max_points)
    throw std::length_error("ICP complete point cap");
  for (const auto* cloud : {&source, &target})
    for (const auto p : *cloud)
      if (!std::isfinite(p.x) || !std::isfinite(p.y))
        throw std::invalid_argument("ICP finite points required");
}
/// @brief Search actual nearest matches and optionally retain chosen indices/points.
/// Strict-less ascending target order preserves original ties; requires nonempty
/// clouds already validated. Every evaluated distance is counted in the supplied
/// producing or diagnostic counter, including losing candidates.
/// @param current Borrowed actual working source.
/// @param target Borrowed finite nonempty target.
/// @param counter Mutable selected distance count.
/// @param indices Optional output actual target indices, already sized N.
/// @param matched Optional output actual matched target coordinates, already sized N.
/// @return Finite directed nearest-target RMSE.
/// @throws std::length_error On counter overflow.
/// @throws std::overflow_error On squared-distance or sum overflow.
double nearest(
    const std::vector<Point2D>& current,
    const std::vector<Point2D>& target,
    std::size_t&                counter,
    std::vector<std::size_t>*   indices = nullptr,
    std::vector<Point2D>*       matched = nullptr) {
  double sse = 0;
  for (std::size_t i = 0; i < current.size(); ++i) {
    double      best = std::numeric_limits<double>::infinity();
    std::size_t bj   = 0;
    for (std::size_t j = 0; j < target.size(); ++j) {
      count_one(counter);
      const double d = distance_squared(current[i], target[j]);
      if (d < best) {
        best = d;
        bj   = j;
      }
    }
    if (indices)
      (*indices)[i] = bj;
    if (matched)
      (*matched)[i] = target[bj];
    sse = checked(sse + best);
  }
  return checked(std::sqrt(sse / static_cast<double>(current.size())));
}
/// @brief Execute the shared legacy-order finite arithmetic with optional actual capture.
/// Inputs have passed validation; post-error observations never influence updates
/// or the preserved strict pre-update stopping rule. Ordinary calls observe only
/// the final error; capture observes fixed/nearest errors after every update.
/// @param source Complete validated borrowed source.
/// @param target Complete validated borrowed target.
/// @param max_iters Nonnegative update budget.
/// @param tol Finite nonnegative strict tolerance.
/// @param options Online history policy.
/// @param steps Optional initially empty owned capture destination.
/// @return Owned actual final cloud/result/stop/work.
/// @throws std::length_error On complete capture admission/count overflow.
/// @throws std::overflow_error On nonfinite producing/diagnostic arithmetic.
/// @throws std::bad_alloc On allocation failure.
IcpRun execute(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    int                         max_iters,
    double                      tol,
    IcpOptions                  options,
    std::vector<IcpStep>*       steps) {
  IcpRun out;
  out.aligned = source;
  if (source.empty() || target.empty()) {
    out.stop = IcpStop::EmptyCloud;
    return out;
  }
  const auto               n = source.size();
  std::vector<Point2D>     match(n);
  std::vector<std::size_t> indices(n);
  double                   previous = std::numeric_limits<double>::infinity();
  std::size_t              units    = 0;
  for (int iter = 0; iter < max_iters; ++iter) {
    if (steps) {
      const auto max = std::numeric_limits<std::size_t>::max();
      if (n > (max - 1) / 5)
        throw std::length_error("ICP history dimensions");
      const auto next = 1 + 5 * n;
      if (units > max - next ||
          (options.max_history_entries && next > options.max_history_entries - units))
        throw std::length_error("ICP complete history cap");
      units += next;
    }
    IcpStep step;
    if (steps)
      step.before = out.aligned;
    const double rmse =
        nearest(out.aligned, target, out.work.distance_evaluations, &indices, &match);
    Point2D cc{0, 0}, cq{0, 0};
    for (std::size_t i = 0; i < n; ++i) {
      cc.x = checked(cc.x + out.aligned[i].x);
      cc.y = checked(cc.y + out.aligned[i].y);
      cq.x = checked(cq.x + match[i].x);
      cq.y = checked(cq.y + match[i].y);
    }
    const double inv = 1. / static_cast<double>(n);
    cc.x *= inv;
    cc.y *= inv;
    cq.x *= inv;
    cq.y *= inv;
    double a = 0, b = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const double px = checked(out.aligned[i].x - cc.x), py = checked(out.aligned[i].y - cc.y);
      const double qx = checked(match[i].x - cq.x), qy = checked(match[i].y - cq.y);
      a = checked(a + checked(px * qx + py * qy));
      b = checked(b + checked(px * qy - py * qx));
    }
    const double  theta = checked(std::atan2(b, a)), cs = std::cos(theta), sn = std::sin(theta);
    const Point2D dt{
        checked(cq.x - (cs * cc.x - sn * cc.y)), checked(cq.y - (sn * cc.x + cs * cc.y))};
    for (auto& p : out.aligned) {
      const double x = checked(cs * p.x - sn * p.y + dt.x), y = checked(sn * p.x + cs * p.y + dt.y);
      p = {x, y};
      count_one(out.work.point_updates);
    }
    auto& r       = out.result;
    r.angle       = checked(r.angle + theta);
    r.translation = {
        checked(cs * r.translation.x - sn * r.translation.y + dt.x),
        checked(sn * r.translation.x + cs * r.translation.y + dt.y)};
    r.iterations = iter + 1;
    if (steps) {
      double fixed = 0;
      for (std::size_t i = 0; i < n; ++i) {
        count_one(out.work.diagnostic_distance_evaluations);
        fixed = checked(fixed + distance_squared(out.aligned[i], match[i]));
      }
      step.post_fixed_rmse = checked(std::sqrt(fixed / static_cast<double>(n)));
      step.post_nearest_rmse =
          nearest(out.aligned, target, out.work.diagnostic_distance_evaluations);
      step.pre_rmse                = rmse;
      step.after                   = out.aligned;
      step.target_indices          = indices;
      step.incremental_angle       = theta;
      step.incremental_translation = dt;
      r.rmse                       = step.post_nearest_rmse;
      step.pose                    = r;
      steps->push_back(std::move(step));
    }
    if (std::fabs(previous - rmse) < tol) {
      out.stop = IcpStop::PreUpdatePlateau;
      break;
    }
    previous = rmse;
  }
  if (!steps || steps->empty())
    out.result.rmse = nearest(out.aligned, target, out.work.diagnostic_distance_evaluations);
  return out;
}
} // namespace
IcpResult icp(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    int                         max_iters,
    double                      tol) {
  validate(source, target, max_iters, tol, {0, 0});
  return execute(source, target, max_iters, tol, {0, 0}, nullptr).result;
}
IcpRun icp_run(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    int                         max_iters,
    double                      tol,
    IcpOptions                  options) {
  validate(source, target, max_iters, tol, options);
  return execute(source, target, max_iters, tol, options, nullptr);
}
IcpTrace icp_trace(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    int                         max_iters,
    double                      tol,
    IcpOptions                  options) {
  validate(source, target, max_iters, tol, options);
  IcpTrace out;
  out.source = source;
  out.target = target;
  out.run    = execute(source, target, max_iters, tol, options, &out.steps);
  return out;
}
} // namespace datamunge::geometry
