/// @file root_finding.cpp
/// @brief Scalar root solvers and checked shared plain/Illinois interpolation with actual evidence.
#include <datamunge/algorithms/root_finding.hpp>
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include <cmath>
#include <functional>

namespace datamunge::algorithms {

/// @brief Sign comparison shared by the scalar bracketing methods.
namespace {
/// @brief Compare negative/nonnegative signs without multiplying callback values.
/// @param x First value.
/// @param y Second value.
/// @return True if both compare negative or both compare nonnegative.
bool same_sign(double x, double y) { return (x < 0.0) == (y < 0.0); }
} // namespace

RootResult newton_raphson(const std::function<double(double)>& f, const std::function<double(double)>& df,
                          double x0, double tol, int max_iter) {
    RootResult r;
    double     x = x0;
    for (int i = 1; i <= max_iter; ++i) {
        const double fx = f(x);
        r.iterations    = i;
        r.root          = x;
        r.residual      = std::fabs(fx);
        if (std::fabs(fx) < tol) {
            r.converged = true;
            return r;
        }
        const double d = df(x);
        if (d == 0.0) return r; // flat tangent: cannot proceed
        x = x - fx / d;
    }
    r.root     = x;
    r.residual = std::fabs(f(x));
    if (r.residual < tol) r.converged = true;
    return r;
}

RootResult secant(const std::function<double(double)>& f, double x0, double x1, double tol, int max_iter) {
    RootResult r;
    double     f0 = f(x0);
    double     f1 = f(x1);
    for (int i = 1; i <= max_iter; ++i) {
        r.iterations = i;
        r.root       = x1;
        r.residual   = std::fabs(f1);
        if (std::fabs(f1) < tol) {
            r.converged = true;
            return r;
        }
        const double denom = f1 - f0;
        if (denom == 0.0) return r; // horizontal secant: cannot proceed
        const double x2 = x1 - f1 * (x1 - x0) / denom;
        x0              = x1;
        f0              = f1;
        x1              = x2;
        f1              = f(x2);
    }
    r.root     = x1;
    r.residual = std::fabs(f1);
    if (r.residual < tol) r.converged = true;
    return r;
}

RootResult ridders(const std::function<double(double)>& f, double a, double b, double tol, int max_iter) {
    double     fa = f(a);
    double     fb = f(b);
    RootResult r;
    if (fa == 0.0) return RootResult{true, a, 0, 0.0};
    if (fb == 0.0) return RootResult{true, b, 0, 0.0};
    if (same_sign(fa, fb)) return r;

    for (int i = 1; i <= max_iter; ++i) {
        const double c  = 0.5 * (a + b);
        const double fc = f(c);
        const double s  = std::sqrt(fc * fc - fa * fb);
        if (s == 0.0) return r;
        // Exponential correction toward the root.
        const double d  = c + (c - a) * ((fa >= fb ? 1.0 : -1.0) * fc / s);
        const double fd = f(d);
        r.iterations    = i;
        r.root          = d;
        r.residual      = std::fabs(fd);
        if (std::fabs(fd) < tol) {
            r.converged = true;
            return r;
        }
        // Re-bracket the root among a, c, d.
        if (!same_sign(fc, fd)) {
            a  = c;
            fa = fc;
            b  = d;
            fb = fd;
        } else if (!same_sign(fa, fd)) {
            b  = d;
            fb = fd;
        } else {
            a  = d;
            fa = fd;
        }
        if (std::fabs(b - a) < tol) {
            r.converged = true;
            return r;
        }
    }
    return r;
}

/// @brief Checked finite interpolation, callback capture and shared false-position execution.
namespace false_position_detail {
/// @brief Validate all user controls before callback invocation or optional history admission.
/// @param f Callback required to be nonempty.
/// @param a First endpoint, required finite.
/// @param b Second endpoint, required finite.
/// @param options Supported policy, finite nonnegative tolerances and positive budget.
/// @throws std::invalid_argument For any malformed control.
void validate(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const FalsePositionOptions&          options) {
  if (!f || !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(options.residual_tolerance) ||
      !std::isfinite(options.x_tolerance) || options.residual_tolerance < 0 ||
      options.x_tolerance < 0 || options.max_iterations <= 0 ||
      (options.policy != FalsePositionPolicy::Plain &&
       options.policy != FalsePositionPolicy::Illinois))
    throw std::invalid_argument(
        "false position requires finite endpoints/controls, supported policy, callback and "
        "positive budget");
}
/// @brief Admit the next complete optional records before their arithmetic or callback work.
/// @param trace Optional owned capture destination; null disables history and quota.
/// @param evaluations Number of evaluation records about to be retained.
/// @param steps Number of step records about to be retained.
/// @param limit Whole-record quota; zero disables only quota.
/// @throws std::length_error For checked addition, vector capacity or quota failure.
void admit(
    const FalsePositionTrace* trace,
    std::size_t               evaluations,
    std::size_t               steps,
    std::size_t               limit) {
  if (!trace)
    return;
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (evaluations > maximum - steps)
    throw std::length_error("false-position capture count overflow");
  const auto add = evaluations + steps, have = trace->captured_entries;
  if (add > maximum - have || (limit && (have > limit || add > limit - have)) ||
      evaluations > trace->evaluations.max_size() - trace->evaluations.size() ||
      steps > trace->steps.max_size() - trace->steps.size())
    throw std::length_error("false-position capture exceeds whole-record limit");
}
/// @brief Invoke the actual callback once and optionally retain its finite return value.
/// @param f Borrowed callable; exceptions and side effects belong to caller.
/// @param x Actual coordinate.
/// @param role Initial a=-1, initial b=+1 or trial=0.
/// @param outcome Actual call counter, advanced before invoking f.
/// @param trace Optional pre-admitted evaluation storage.
/// @return Actual finite callback value.
/// @throws std::domain_error If the callback returns a nonfinite value.
/// @throws std::bad_alloc If optional evaluation storage cannot be allocated.
double evaluate(
    const std::function<double(double)>& f,
    double                               x,
    int                                  role,
    FalsePositionOutcome&                outcome,
    FalsePositionTrace*                  trace) {
  ++outcome.evaluations;
  const double value = f(x);
  if (!std::isfinite(value))
    throw std::domain_error("false-position callback returned nonfinite value");
  if (trace) {
    trace->evaluations.push_back({role, x, value});
    ++trace->captured_entries;
  }
  return value;
}
/// @brief Test whether multiplying two finite doubles avoids overflow, without doing that product.
/// @param x Finite factor.
/// @param y Finite factor.
/// @return True if the product magnitude is representable; ordinary underflow/rounding allowed.
bool product_safe(double x, double y) {
  const double ax = std::abs(x), ay = std::abs(y);
  return ax <= 1 || ay <= std::numeric_limits<double>::max() / ax;
}
/// @brief Test whether subtracting two finite doubles avoids overflow, before subtraction.
/// @param x Finite minuend.
/// @param y Finite subtrahend.
/// @return True if the difference magnitude is representable.
bool difference_safe(double x, double y) {
  return std::signbit(x) == std::signbit(y) ||
         std::abs(x) <= std::numeric_limits<double>::max() - std::abs(y);
}
/// @brief Return a finite distance or infinity without raising an intermediate overflow flag.
/// @param x Finite first coordinate.
/// @param y Finite second coordinate.
/// @return Rounded absolute difference, or infinity if that distance exceeds finite double range.
double distance(double x, double y) {
  return difference_safe(x, y) ? std::abs(x - y) : std::numeric_limits<double>::infinity();
}
/// @brief Preserve safe legacy arithmetic, otherwise interpolate with scaled positive weights.
/// Nonzero products must remain normal; otherwise scaled weights avoid accepting
/// a spurious zero numerator caused by product underflow. Ordinary underflow flags
/// can still occur while detecting those products or halving an Illinois weight.
/// @param a Finite first endpoint.
/// @param b Finite second endpoint.
/// @param wa Finite nonzero weight with actual a's sign.
/// @param wb Finite nonzero weight of opposite sign.
/// @return Actual finite point between endpoints and whether the guarded branch was used.
std::pair<double, bool> interpolate(double a, double b, double wa, double wb) {
  const double maximum = std::numeric_limits<double>::max();
  if (product_safe(wb, a) && product_safe(wa, b) && difference_safe(wb, wa)) {
    const double p = wb * a, q = wa * b;
    if ((a == 0 || std::isnormal(p)) && (b == 0 || std::isnormal(q)) && difference_safe(p, q)) {
      const double numerator = p - q, denominator = wb - wa;
      if (denominator != 0 &&
          (std::abs(denominator) >= 1 || std::abs(numerator) <= maximum * std::abs(denominator))) {
        const double c = (wb * a - wa * b) / (wb - wa);
        if (std::isfinite(c) && c >= std::min(a, b) && c <= std::max(a, b))
          return {c, false};
      }
    }
  }
  const double aw = std::abs(wa), bw = std::abs(wb);
  double       fraction;
  if (aw <= bw) {
    const double ratio = aw / bw;
    fraction           = ratio / (1 + ratio);
  } else {
    const double ratio = bw / aw;
    fraction           = 1 / (1 + ratio);
  }
  return {std::lerp(a, b, fraction), true};
}
/// @brief Retain one pre-admitted actual iteration record after its real update/terminal decision.
/// @param trace Optional owned destination.
/// @param step Actual snapshot with chosen transition and branch flags.
/// @throws std::bad_alloc If optional step storage cannot be allocated.
void record(FalsePositionTrace* trace, const FalsePositionStep& step) {
  if (trace) {
    trace->steps.push_back(step);
    ++trace->captured_entries;
  }
}
/// @brief Execute the shared plain/retained-twice Illinois kernel without replay.
/// @param f Borrowed finite-valued callback.
/// @param a First finite endpoint, retaining input orientation.
/// @param b Second finite endpoint.
/// @param options Policy, separate tolerances and positive trial budget.
/// @param trace Optional owned actual evaluation/step destination, initially empty.
/// @param limit Whole optional record quota; zero disables quota only.
/// @return Owned actual result, terminal pre-update state and work counters.
/// @throws std::invalid_argument For malformed controls before callbacks/capture.
/// @throws std::domain_error For nonfinite callback returns; callback exceptions propagate.
/// @throws std::length_error For optional capture admission failure.
/// @throws std::bad_alloc If optional history storage cannot be allocated.
FalsePositionOutcome execute(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const FalsePositionOptions&          options,
    FalsePositionTrace*                  trace,
    std::size_t                          limit) {
  validate(f, a, b, options);
  admit(trace, 2, 0, limit);
  FalsePositionOutcome out;
  double fa = evaluate(f, a, -1, out, trace), fb = evaluate(f, b, 1, out, trace), wa = fa, wb = fb;
  out.a        = a;
  out.b        = b;
  out.f_a      = fa;
  out.f_b      = fb;
  out.weight_a = wa;
  out.weight_b = wb;
  if (fa == 0 || fb == 0) {
    const double root = fa == 0 ? a : b;
    out.result        = {true, root, 0, 0};
    out.stop          = FalsePositionStop::Endpoint;
    out.a = out.b = root;
    out.f_a = out.f_b = out.weight_a = out.weight_b = 0;
    out.radius                                      = 0;
    return out;
  }
  if (same_sign(fa, fb))
    return out;
  int side = 0, iteration = 1;
  for (;;) {
    admit(trace, 1, 1, limit);
    out.a                   = a;
    out.b                   = b;
    out.f_a                 = fa;
    out.f_b                 = fb;
    out.weight_a            = wa;
    out.weight_b            = wb;
    const auto [c, guarded] = interpolate(a, b, wa, wb);
    if (guarded)
      ++out.guarded_interpolations;
    const double fc = evaluate(f, c, 0, out, trace);
    out.result      = {false, c, iteration, std::abs(fc)};
    out.radius      = std::max(distance(c, a), distance(c, b));
    FalsePositionStep step{a, b, fa, fb, wa, wb, c, fc, out.radius, 0, false, guarded};
    std::optional<FalsePositionStop> stop;
    if (fc == 0) {
      stop                 = FalsePositionStop::ExactValue;
      out.result.converged = true;
    } else if (options.residual_tolerance > 0 && std::abs(fc) < options.residual_tolerance) {
      stop                 = FalsePositionStop::Residual;
      out.result.converged = true;
    } else if (options.x_tolerance > 0 && out.radius < options.x_tolerance) {
      stop                 = FalsePositionStop::Interval;
      out.result.converged = true;
    } else if (c == a || c == b)
      stop = FalsePositionStop::Stagnation;
    else if (iteration == options.max_iterations)
      stop = FalsePositionStop::IterationLimit;
    else {
      if (same_sign(fc, fb)) {
        b  = c;
        fb = wb       = fc;
        step.replaced = +1;
        if (options.policy == FalsePositionPolicy::Illinois && side == +1) {
          wa *= .5;
          step.damped = true;
          ++out.weight_halvings;
          if (wa == 0)
            stop = FalsePositionStop::WeightUnderflow;
        }
        side = +1;
      } else {
        a  = c;
        fa = wa       = fc;
        step.replaced = -1;
        if (options.policy == FalsePositionPolicy::Illinois && side == -1) {
          wb *= .5;
          step.damped = true;
          ++out.weight_halvings;
          if (wb == 0)
            stop = FalsePositionStop::WeightUnderflow;
        }
        side = -1;
      }
    }
    record(trace, step);
    if (stop) {
      out.stop = *stop;
      return out;
    }
    ++iteration;
  }
}
} // namespace false_position_detail

/// @brief Name a false-position policy without allocating or throwing.
/// @param policy Any policy value.
/// @return Stable literal, with unknown for unrecognized values.
std::string_view false_position_policy_name(FalsePositionPolicy policy) noexcept {
  switch (policy) {
    case FalsePositionPolicy::Plain:
      return "plain";
    case FalsePositionPolicy::Illinois:
      return "illinois";
  }
  return "unknown";
}
/// @brief Name an actual false-position terminal reason without allocation or exceptions.
/// @param stop Any reason value.
/// @return Stable literal, with unknown for unrecognized values.
std::string_view false_position_stop_name(FalsePositionStop stop) noexcept {
  switch (stop) {
    case FalsePositionStop::NoBracket:
      return "no bracket";
    case FalsePositionStop::Endpoint:
      return "endpoint";
    case FalsePositionStop::ExactValue:
      return "zero callback";
    case FalsePositionStop::Residual:
      return "residual";
    case FalsePositionStop::Interval:
      return "interval";
    case FalsePositionStop::Stagnation:
      return "stagnation";
    case FalsePositionStop::WeightUnderflow:
      return "weight underflow";
    case FalsePositionStop::IterationLimit:
      return "iteration limit";
  }
  return "unknown";
}
/// @brief Preserve the historical compiled interface via checked shared Illinois execution.
/// @param f Borrowed finite-valued callback.
/// @param a First finite endpoint.
/// @param b Second finite endpoint.
/// @param tol Positive finite residual-only tolerance.
/// @param max_iter Positive trial budget.
/// @return Legacy result; public declaration specifies controls, errors and finite arithmetic.
RootResult false_position(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    double                               tol,
    int                                  max_iter) {
  if (!std::isfinite(tol) || tol <= 0)
    throw std::invalid_argument("false-position legacy tolerance must be positive finite");
  return false_position_detail::execute(
             f, a, b, {FalsePositionPolicy::Illinois, tol, 0, max_iter}, nullptr, 0)
      .result;
}
/// @brief Execute the canonical selected policy without retaining optional history.
/// @param f Borrowed finite-valued callback.
/// @param a First finite endpoint.
/// @param b Second finite endpoint.
/// @param options Separate finite tolerances, positive budget and supported policy.
/// @return Owned actual outcome; public declaration gives complete contract and exceptions.
FalsePositionOutcome false_position_work(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const FalsePositionOptions&          options) {
  return false_position_detail::execute(f, a, b, options, nullptr, 0);
}
/// @brief Retain actual callbacks and steps while executing the canonical selected policy once.
/// @param f Borrowed finite-valued callback.
/// @param a First finite endpoint.
/// @param b Second finite endpoint.
/// @param options Supported policy, finite tolerances and positive budget.
/// @param max_entries Whole optional record cap; zero disables quota only.
/// @return Owned actual trace; public declaration gives complete contract and exceptions.
FalsePositionTrace false_position_trace(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const FalsePositionOptions&          options,
    std::size_t                          max_entries) {
  FalsePositionTrace trace;
  trace.outcome = false_position_detail::execute(f, a, b, options, &trace, max_entries);
  return trace;
}

RootResult halley(const std::function<double(double)>& f, const std::function<double(double)>& df,
                  const std::function<double(double)>& d2f, double x0, double tol, int max_iter) {
    RootResult r;
    double     x = x0;
    for (int i = 1; i <= max_iter; ++i) {
        const double fx = f(x);
        r.iterations    = i;
        r.root          = x;
        r.residual      = std::fabs(fx);
        if (std::fabs(fx) < tol) {
            r.converged = true;
            return r;
        }
        const double d     = df(x);
        const double d2    = d2f(x);
        const double denom = 2.0 * d * d - fx * d2;
        if (denom == 0.0) return r;
        x = x - 2.0 * fx * d / denom;
    }
    r.root     = x;
    r.residual = std::fabs(f(x));
    if (r.residual < tol) r.converged = true;
    return r;
}

RootResult muller(const std::function<double(double)>& f, double x0, double x1, double x2, double tol,
                  int max_iter) {
    RootResult r;
    double     f0 = f(x0), f1 = f(x1), f2 = f(x2);
    for (int i = 1; i <= max_iter; ++i) {
        r.iterations = i;
        r.root       = x2;
        r.residual   = std::fabs(f2);
        if (std::fabs(f2) < tol) {
            r.converged = true;
            return r;
        }
        const double h0 = x1 - x0, h1 = x2 - x1;
        const double d0 = (f1 - f0) / h0, d1 = (f2 - f1) / h1;
        const double a  = (d1 - d0) / (h1 + h0); // parabola through the three points
        const double b  = a * h1 + d1;
        const double c  = f2;
        double       disc = b * b - 4.0 * a * c;
        if (disc < 0.0) disc = 0.0; // clamp to target real roots
        const double sq    = std::sqrt(disc);
        const double denom = (std::fabs(b + sq) > std::fabs(b - sq)) ? (b + sq) : (b - sq);
        if (denom == 0.0) return r;
        const double x3 = x2 - 2.0 * c / denom;
        x0 = x1; f0 = f1;
        x1 = x2; f1 = f2;
        x2 = x3; f2 = f(x3);
    }
    r.root     = x2;
    r.residual = std::fabs(f2);
    if (r.residual < tol) r.converged = true;
    return r;
}

RootResult itp(const std::function<double(double)>& f, double a, double b, double tol, int max_iter) {
    double     fa = f(a);
    double     fb = f(b);
    RootResult r;
    if (fa == 0.0) return RootResult{true, a, 0, 0.0};
    if (fb == 0.0) return RootResult{true, b, 0, 0.0};
    if (same_sign(fa, fb)) return r;

    const double eps   = tol;
    const double k1    = 0.2 / (b - a); // scale-dependent truncation parameter
    const double k2    = 2.0;
    const int    n0    = 1;
    const int    n_half = static_cast<int>(std::ceil(std::log2((b - a) / (2.0 * eps))));
    const int    n_max  = n_half + n0;

    for (int j = 0; j < max_iter && (b - a) > 2.0 * eps; ++j) {
        const double xh     = 0.5 * (a + b);                        // bisection midpoint
        const double r_proj = eps * std::pow(2.0, static_cast<double>(n_max - j)) - 0.5 * (b - a);
        const double xf     = (fb * a - fa * b) / (fb - fa);        // interpolation (regula falsi)
        const double sigma  = (xh - xf) >= 0.0 ? 1.0 : -1.0;
        const double delta  = k1 * std::pow(b - a, k2);
        const double xt     = (delta <= std::fabs(xh - xf)) ? xf + sigma * delta : xh; // truncation
        const double xitp   = (std::fabs(xt - xh) <= r_proj) ? xt : xh - sigma * r_proj; // projection
        const double fitp   = f(xitp);
        r.iterations        = j + 1;
        r.root              = xitp;
        r.residual          = std::fabs(fitp);
        if (std::fabs(fitp) < tol) {
            r.converged = true;
            return r;
        }
        if (same_sign(fitp, fa)) { a = xitp; fa = fitp; }
        else { b = xitp; fb = fitp; }
    }
    const double c = 0.5 * (a + b);
    r.root         = c;
    r.residual     = std::fabs(f(c));
    r.converged    = (b - a) <= 2.0 * eps || r.residual < tol;
    return r;
}

} // namespace datamunge::algorithms
