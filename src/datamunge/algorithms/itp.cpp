/// @file itp.cpp
/// @brief One shared finite-value ITP loop with exact-call histories and independent stopping.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/itp.hpp>
#include <numeric>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private checked scalar arithmetic, admission and shared ITP execution.
namespace {
/// @brief Require a finite working-precision scalar.
/// @param x Working scalar.
/// @return Unchanged finite scalar.
/// @throws std::overflow_error On nonfinite arithmetic.
long double finite(long double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("ITP working arithmetic");
  return x;
}
/// @brief Count and execute one actual finite-valued callback.
/// @param f Borrowed callable.
/// @param x Actual double coordinate.
/// @param evaluations Mutable actual callback count.
/// @return Finite callback result.
/// @throws std::length_error On count overflow.
/// @throws std::domain_error On nonfinite callback value; callback exceptions propagate.
double evaluate(const std::function<double(double)>& f, double x, std::size_t& evaluations) {
  if (evaluations == std::numeric_limits<std::size_t>::max())
    throw std::length_error("ITP callback count");
  ++evaluations;
  const double value = f(x);
  if (!std::isfinite(value))
    throw std::domain_error("ITP finite callback required");
  return value;
}
/// @brief Actual wide maximum endpoint distance for a returned/evaluated coordinate.
/// @param a Ordered lower endpoint.
/// @param b Ordered upper endpoint.
/// @param x Coordinate in the interval.
/// @return Working-precision maximum distance.
/// @throws std::overflow_error On an unrepresentable wide distance.
long double radius(double a, double b, double x) {
  return finite(std::max(static_cast<long double>(x) - a, static_cast<long double>(b) - x));
}
/// @brief Compute a capped projection radius without overflowing a large slack exponent.
/// @param epsilon Positive finite coordinate tolerance.
/// @param exponent Remaining nonnegative projection exponent.
/// @param half Current positive wide half-width.
/// @return Radius clipped to [0,half], equivalent to the full feasible band inside the bracket.
/// @throws std::overflow_error On nonfinite working half-width.
long double projection_radius(double epsilon, int exponent, long double half) {
  int e = 0;
  std::frexp(static_cast<long double>(epsilon), &e);
  if (static_cast<long long>(e) + exponent > std::numeric_limits<long double>::max_exponent)
    return half;
  const auto budget = std::ldexp(static_cast<long double>(epsilon), exponent);
  return std::clamp(budget - half, 0.L, half);
}
/// @brief Admit one actual candidate record before its callback.
/// @param out Mutable output with optional owned records.
/// @param options Scalar-record policy.
/// @param capture Whether history is requested.
/// @throws std::length_error On enabled record cap or exhausted vector capacity.
void admit(const ItpTrace& out, const ItpOptions& options, bool capture) {
  if (capture && ((options.max_records && out.steps.size() >= options.max_records) ||
                  out.steps.size() == out.steps.max_size()))
    throw std::length_error("ITP actual candidate record cap");
}
/// @brief Execute the canonical shared interpolation/truncation/projection loop.
/// @param f Borrowed nonempty finite-valued callback.
/// @param a Finite first endpoint.
/// @param b Finite second endpoint, ordered internally.
/// @param o Independent tolerances, bounds and record policy.
/// @param capture Whether to retain actual candidate calls.
/// @return Actual run summary and optional chronological scalar history.
/// @throws std::invalid_argument On controls/callback/endpoints before callbacks.
/// @throws std::domain_error On nonfinite callback returns; callback exceptions propagate.
/// @throws std::overflow_error On nonfinite wide arithmetic.
/// @throws std::length_error On exponent/count/record overflow.
/// @throws std::bad_alloc On history allocation failure.
ItpTrace
execute(const std::function<double(double)>& f, double a, double b, ItpOptions o, bool capture) {
  if (!f || !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(o.x_tolerance) ||
      o.x_tolerance <= 0 || !std::isfinite(o.residual_tolerance) || o.residual_tolerance < 0 ||
      o.max_iterations < 0 || o.extra_iterations < 0)
    throw std::invalid_argument("ITP finite endpoints/valid controls required");
  if (a > b)
    std::swap(a, b);
  const auto initial_width = finite(static_cast<long double>(b) - a);
  ItpTrace   out;
  auto&      r     = out.run;
  r.lower          = a;
  r.upper          = b;
  long double half = initial_width / 2;
  while (half > o.x_tolerance) {
    half /= 2;
    ++r.bisection_bound;
  }
  if (o.extra_iterations > std::numeric_limits<int>::max() - r.bisection_bound)
    throw std::length_error("ITP projection exponent");
  r.projection_bound = r.bisection_bound + o.extra_iterations;
  double fa          = evaluate(f, a, r.evaluations);
  if (fa == 0) {
    r.result = {true, a, 0, 0};
    r.stop   = ItpStop::Endpoint;
    r.lower = r.upper = a;
    r.radius          = 0;
    return out;
  }
  double fb = evaluate(f, b, r.evaluations);
  if (fb == 0) {
    r.result = {true, b, 0, 0};
    r.stop   = ItpStop::Endpoint;
    r.lower = r.upper = b;
    r.radius          = 0;
    return out;
  }
  if (std::signbit(fa) == std::signbit(fb))
    return out;
  int completed = 0;
  for (;;) {
    const double midpoint     = std::midpoint(a, b);
    const auto   mid_radius   = radius(a, b, midpoint);
    ItpStop      final_reason = ItpStop::NoBracket;
    if (mid_radius <= o.x_tolerance)
      final_reason = ItpStop::Interval;
    else if (midpoint == a || midpoint == b)
      final_reason = ItpStop::Stagnation;
    else if (completed >= o.max_iterations)
      final_reason = ItpStop::IterationLimit;
    else if (completed >= r.projection_bound)
      final_reason = ItpStop::ProjectionLimit;
    if (final_reason != ItpStop::NoBracket) {
      admit(out, o, capture);
      const double fm = evaluate(f, midpoint, r.evaluations);
      r.result        = {final_reason == ItpStop::Interval, midpoint, completed, std::abs(fm)};
      r.stop          = final_reason;
      r.lower         = a;
      r.upper         = b;
      r.radius        = mid_radius;
      if (fm == 0) {
        r.result.converged = true;
        r.stop             = ItpStop::ExactValue;
        r.lower = r.upper = midpoint;
        r.radius          = 0;
      } else if (o.residual_tolerance > 0 && std::abs(fm) < o.residual_tolerance) {
        r.result.converged = true;
        r.stop             = ItpStop::Residual;
      }
      if (capture)
        out.steps.push_back(
            {completed, false, a, b, fa, fb, 0, 0, 0, 0, 0, midpoint, fm, mid_radius, false});
      return out;
    }
    const auto width  = finite(static_cast<long double>(b) - a);
    const auto centre = finite(static_cast<long double>(a) + width / 2);
    const auto wa     = std::fabs(static_cast<long double>(fa)),
               wb     = std::fabs(static_cast<long double>(fb));
    // Scale callback magnitudes first, avoiding both a cross-product and wa+wb overflow.
    const auto scale  = std::max(wa, wb);
    const auto weight = (wa / scale) / (wa / scale + wb / scale);
    const auto xf     = finite(static_cast<long double>(a) + width * weight);
    // Algebraically kappa1*width^2, with kappa1=.2/initial_width; avoids squaring.
    const auto delta  = finite(.2L * width * (width / initial_width));
    const int  sigma  = (centre > xf) - (centre < xf);
    const auto xt     = std::abs(centre - xf) > delta ? finite(xf + sigma * delta) : centre;
    const auto rp     = projection_radius(o.x_tolerance, r.projection_bound - completed, width / 2);
    const auto wide_x = finite(std::clamp(xt, centre - rp, centre + rp));
    double     x      = static_cast<double>(wide_x);
    bool       adjusted = false;
    if (!(a < x && x < b)) {
      x        = midpoint;
      adjusted = true;
    }
    adjusted = adjusted || std::fabs(static_cast<long double>(x) - centre) > rp;
    admit(out, o, capture);
    const double fx = evaluate(f, x, r.evaluations);
    ++completed;
    if (capture)
      out.steps.push_back(
          {completed,
           true,
           a,
           b,
           fa,
           fb,
           centre,
           xf,
           delta,
           xt,
           rp,
           x,
           fx,
           radius(a, b, x),
           adjusted});
    r.result = {false, x, completed, std::abs(fx)};
    r.lower  = a;
    r.upper  = b;
    r.radius = radius(a, b, x);
    if (fx == 0) {
      r.result.converged = true;
      r.stop             = ItpStop::ExactValue;
      r.lower = r.upper = x;
      r.radius          = 0;
      return out;
    }
    if (o.residual_tolerance > 0 && std::abs(fx) < o.residual_tolerance) {
      r.result.converged = true;
      r.stop             = ItpStop::Residual;
      return out;
    }
    if (std::signbit(fx) == std::signbit(fa)) {
      a  = x;
      fa = fx;
    } else {
      b  = x;
      fb = fx;
    }
  }
}
} // namespace
std::string_view itp_stop_name(ItpStop stop) {
  switch (stop) {
    case ItpStop::NoBracket:
      return "no bracket";
    case ItpStop::Endpoint:
      return "endpoint";
    case ItpStop::ExactValue:
      return "zero callback";
    case ItpStop::Residual:
      return "residual";
    case ItpStop::Interval:
      return "interval";
    case ItpStop::IterationLimit:
      return "iteration limit";
    case ItpStop::ProjectionLimit:
      return "projection limit";
    case ItpStop::Stagnation:
      return "stagnation";
  }
  return "unknown";
}
RootResult
itp(const std::function<double(double)>& f, double a, double b, double tol, int max_iter) {
  return execute(f, a, b, {tol, tol, max_iter, 1, 0}, false).run.result;
}
ItpRun itp_run(const std::function<double(double)>& f, double a, double b, ItpOptions options) {
  return execute(f, a, b, options, false).run;
}
ItpTrace itp_trace(const std::function<double(double)>& f, double a, double b, ItpOptions options) {
  return execute(f, a, b, options, true);
}
} // namespace datamunge::algorithms
