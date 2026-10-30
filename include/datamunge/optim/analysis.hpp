#pragma once
/// @file analysis.hpp
/// @brief Reusable objective fixtures and honest callback traces for optimizer experiments.
#include <cmath>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/optim/function_types.hpp>
#include <datamunge/plot/plot.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
/// @brief Callback-based continuous optimization, declared solver controls and diagnostic fixtures.
namespace datamunge::optim {
/// @brief Owned callback history; events are requests, not accepted iterations.
/// Value calls include rejected line-search/population proposals. Gradient calls
/// may be at look-ahead points. Diagnostic sampling after fitting is separate
/// work and assumes a deterministic objective without callback side effects.
/// @par Example
/// @code
/// dm::optim::OptimizationTrace t;
/// std::print("{}\n",t.calls()); // 0
/// @endcode
/// @see trace_optimize()
class OptimizationTrace {
 public:
  /// @brief Append one callback and its coordinates without evaluating the objective.
  /// @param kind value, gradient, term_value, term_gradient, hessian, constraints, prox or
  /// residuals.
  /// @param x Callback coordinates, copied; dimension must match earlier events.
  /// @param value Known objective or half residual SSE; NaN when unavailable.
  /// @param norm Known gradient norm; NaN when unavailable.
  /// @param terms Nominal number of component gradients charged, zero for non-gradient requests.
  /// @param component Zero-based sampled term index, NaN for full-objective callbacks.
  /// @throws std::invalid_argument Coordinates differ in length from a previous event.
  /// @par Example
  /// @code
  /// dm::optim::OptimizationTrace t;
  /// t.record("term_gradient",{2},NAN,4,1,0);
  /// std::print("{}\n",t.calls()); // 1
  /// @endcode
  void record(const std::string& kind, const std::vector<double>& x,
              double value = NAN, double norm = NAN, std::size_t terms = 0,
              double component = NAN);
  /// @brief Total callback requests captured.
  /// @return Number of stored events; no mutation.
  std::size_t calls() const noexcept { return kinds_.size(); }
  /// @brief Evaluate captured positions after the run, selecting a callback kind.
  /// Does not alter fitting counts/history. Adds one objective call per selected
  /// position, including trials. It is an explicit diagnostic replay, not free work.
  /// @param objective Pure deterministic objective on matching coordinates.
  /// @param kind Callback kind to retain, default value.
  /// @return event/value/gradient_norm/term_gradients/component and x0,x1,... columns.
  /// @throws std::exception Objective callback errors propagate.
  /// @par Example
  /// @code
  /// dm::optim::OptimizationTrace t;
  /// std::print("{}\n",t.calls()); // 0; see trace_optimize for sampling a populated trace.
  /// @endcode
  dstruct::DataFrame sample(ArbitraryFunction& objective,
                            const std::string& kind = "value") const;
  /// @brief Return only stored objective/residual values; no diagnostic callbacks.
  /// @return event/value/gradient_norm/term_gradients and x0,x1,... for value/residual events.
  /// @par Example
  /// @code
  /// dm::optim::OptimizationTrace t;
  /// std::print("{}\n",t.values().nrows()); // 0
  /// @endcode
  dstruct::DataFrame values() const;
  /// @brief Return every recorded callback without invoking the objective.
  /// Component indices identify the sampled term; NaN denotes a full callback.
  /// @return event/kind/value/gradient_norm/term_gradients/component/x0,... columns.
  /// @par Example
  /// @code
  /// dm::optim::OptimizationTrace t;
  /// t.record("term_gradient",{2},NAN,4,1,0);
  /// std::print("{:.0f}\n",t.requests().double_at("component",0)); // 0
  /// @endcode
  dstruct::DataFrame requests() const;
  /// @brief Aggregate callback counts by kind, retaining nominal term-gradient cost.
  /// @return kind/calls table, plus a term_gradient_equivalents row.
  /// @par Example
  /// @code
  /// dm::optim::OptimizationTrace t;
  /// std::print("{}\n",t.counts().nrows()); // 1
  /// @endcode
  dstruct::DataFrame counts() const;

 private:
  std::vector<std::string> kinds_;  ///< Callback names in request order.
  std::vector<std::vector<double>>
      points_;  ///< Captured owned coordinate vectors.
  std::vector<double>
      values_;  ///< Stored objective values, NaN for other request kinds.
  std::vector<double>
      norms_;  ///< Available requested gradient norms, otherwise NaN.
  std::vector<double>
      components_;  ///< Sampled term indices; NaN on full callbacks.
  std::vector<double>
      term_cost_;  ///< Cumulative nominal component-gradient work.
};
/// @brief Result of a recorded optimizer call; no convergence certificate is inferred.
/// exception is empty on normal return. A nonempty exception retains the last
/// mutated coordinates and preceding callback evidence. value is the returned
/// scalar (NaN on failure); checked_value and gradient_norm are post-run checks,
/// excluded from fitting counters. A small smooth-gradient norm is inappropriate
/// for proximal or constrained stationarity; use their mapping/KKT conditions.
/// @par Example
/// @code
/// dm::optim::OptimizationRun r;
/// std::print("{}\n",r.exception.empty()); // true
/// @endcode
struct OptimizationRun {
  std::vector<double> coordinates;  ///< Returned or last mutated coordinates.
  double value{NAN};  ///< Native optimizer return value, NaN if it threw.
  double checked_value{
      NAN};  ///< Independent final objective or half residual SSE.
  double gradient_norm{
      NAN};  ///< Post-run Euclidean gradient norm; NaN for value/residual-only interfaces.
  std::string
      exception;  ///< Failure text, including recorder-detected nonfinite requests.
  OptimizationTrace trace;  ///< Owned callback history and counts.

  /// @brief Present final coordinates, native/checked values and diagnostic status.
  /// @return One-row value/checked_value/gradient_norm/callbacks/exception/x0,x1,... table.
  /// Values, norms and coordinates use compact numeric strings so tiny/huge values
  /// remain visible; callbacks is numeric. Use result fields for numeric calculations.
  /// @par Example
  /// @code
  /// dm::optim::OptimizationRun r;
  /// std::print("{}\n",r.summary().nrows()); // 1
  /// @endcode
  dstruct::DataFrame summary() const;
};
/// @brief Internal checked arithmetic and typed callback forwarding.
namespace detail {
/// @brief Euclidean norm accumulated with hypot to avoid squaring overflow.
/// @param x Numeric components.
/// @return Euclidean norm (nonfinite if any component is nonfinite).
double analysis_norm(const std::vector<double>& x);
/// @brief Half the sum of squared residuals, matching LevenbergMarquardt.
/// @param r Residual vector.
/// @return Half residual SSE.
double analysis_half_sse(const std::vector<double>& r);
/// @brief Decorate exactly one abstract function interface without adding solver calls.
/// Unsupported interface methods are never instantiated. Nonfinite requested
/// values/gradients/residuals are recorded then rejected; this instrumentation
/// makes divergence explicit rather than allowing a native solver to loop on NaN.
/// @tparam Base One function_types.hpp abstract interface.
/// @see trace_optimize()
template <class Base>
class RecordedFunction : public Base {
  Base& source_;  ///< Borrowed objective for the lifetime of this call.
  OptimizationTrace& trace_;  ///< Borrowed trace owned by the run result.
 public:
  /// @brief Borrow an objective and event sink.
  /// @param source Original abstract interface, unchanged apart from its own callback effects.
  /// @param trace Owned-by-caller history sink.
  RecordedFunction(Base& source, OptimizationTrace& trace)
      : source_(source), trace_(trace) {}
  /// @brief Forward and record a full objective request.
  /// @param x Coordinates requested by the optimizer.
  /// @return Original objective value.
  /// @throws std::runtime_error Nonfinite objective.
  double evaluate(const std::vector<double>& x) {
    double v = source_.evaluate(x);
    trace_.record("value", x, v);
    if (!std::isfinite(v))
      throw std::runtime_error("nonfinite objective request");
    return v;
  }
  /// @brief Forward and record a gradient request.
  /// @param x Requested coordinates.
  /// @return Original gradient, unchanged.
  /// @throws std::runtime_error Nonfinite gradient.
  std::vector<double> gradient(const std::vector<double>& x) {
    auto g = source_.gradient(x);
    auto n = analysis_norm(g);
    std::size_t terms = 1;
    if constexpr (std::is_base_of_v<DifferentiableSeparableFunction, Base>)
      terms = source_.num_functions();
    trace_.record("gradient", x, NAN, n, terms);
    if (!std::isfinite(n))
      throw std::runtime_error("nonfinite gradient request");
    return g;
  }
  /// @brief Forward the component count without charging it as a numeric callback.
  /// @return Original number of additive terms.
  std::size_t num_functions() const { return source_.num_functions(); }
  /// @brief Forward one component loss.
  /// @param x Coordinates.
  /// @param i Original component index.
  /// @return Component value.
  double evaluate_term(const std::vector<double>& x, std::size_t i) {
    auto v = source_.evaluate_term(x, i);
    trace_.record("term_value", x, v, NAN, 0, i);
    return v;
  }
  /// @brief Forward and charge one component gradient.
  /// @param x Coordinates.
  /// @param i Component index.
  /// @return Component gradient.
  /// @throws std::runtime_error Nonfinite gradient.
  std::vector<double> gradient_term(const std::vector<double>& x,
                                    std::size_t i) {
    auto g = source_.gradient_term(x, i);
    auto n = analysis_norm(g);
    trace_.record("term_gradient", x, NAN, n, 1, i);
    if (!std::isfinite(n))
      throw std::runtime_error("nonfinite component gradient");
    return g;
  }
  /// @brief Forward the exact Hessian request.
  /// @param x Coordinates.
  /// @return Original Hessian.
  std::vector<std::vector<double>> hessian(const std::vector<double>& x) {
    trace_.record("hessian", x);
    return source_.hessian(x);
  }
  /// @brief Forward equality constraints.
  /// @param x Coordinates.
  /// @return Original equalities.
  std::vector<double> constraints(const std::vector<double>& x) {
    trace_.record("constraints", x);
    return source_.constraints(x);
  }
  /// @brief Forward equality derivatives.
  /// @param x Coordinates.
  /// @return Original equality Jacobian.
  std::vector<std::vector<double>> constraint_jacobian(
      const std::vector<double>& x) {
    trace_.record("constraint_jacobian", x);
    return source_.constraint_jacobian(x);
  }
  /// @brief Forward nonpositive inequalities.
  /// @param x Coordinates.
  /// @return Original inequalities.
  std::vector<double> inequalities(const std::vector<double>& x) {
    trace_.record("inequalities", x);
    return source_.inequalities(x);
  }
  /// @brief Forward inequality derivatives.
  /// @param x Coordinates.
  /// @return Original inequality Jacobian.
  std::vector<std::vector<double>> inequality_jacobian(
      const std::vector<double>& x) {
    trace_.record("inequality_jacobian", x);
    return source_.inequality_jacobian(x);
  }
  /// @brief Forward the proximal map without evaluating the loss.
  /// @param x Smooth-gradient trial point.
  /// @param step Positive proximal scale supplied by the optimizer.
  /// @return Original proximal point.
  std::vector<double> proximal(const std::vector<double>& x, double step) {
    trace_.record("prox", x);
    return source_.proximal(x, step);
  }
  /// @brief Forward and record half residual SSE.
  /// @param x Parameter vector.
  /// @return Original residuals.
  /// @throws std::runtime_error Nonfinite residual sum of squares.
  std::vector<double> residuals(const std::vector<double>& x) {
    auto r = source_.residuals(x);
    auto v = analysis_half_sse(r);
    trace_.record("residuals", x, v);
    if (!std::isfinite(v))
      throw std::runtime_error("nonfinite residual request");
    return r;
  }
  /// @brief Forward the residual Jacobian.
  /// @param x Parameters.
  /// @return Original residual derivatives.
  std::vector<std::vector<double>> jacobian(const std::vector<double>& x) {
    trace_.record("jacobian", x);
    return source_.jacobian(x);
  }
};
}  // namespace detail
/// @brief Run a native optimizer through an explicitly chosen abstract interface.
/// Preserves callback order and adds no fitting evaluations. Captures std::exception
/// text and nonfinite value/gradient/residual requests, retaining the partial trace.
/// Final value/gradient checks are separate post-run callbacks, excluded from counts;
/// use pure objectives for reproducible diagnostic sampling. No native stop reason
/// is invented, and value-call indices must not be labelled accepted iterations.
/// @tparam Base Explicit abstract interface consumed by the selected optimizer.
/// @tparam Optimizer Native optimizer type with optimize(Base&,coordinates,...).
/// @tparam Bounds Optional native bound argument types for population optimizers.
/// @param function Borrowed objective with finite valid dimensions.
/// @param optimizer Configured native optimizer.
/// @param initial Owned starting coordinates; empty is rejected.
/// @param bounds Optional native bound arguments, forwarded unchanged.
/// @return Owned final state and callback trace; check exception before using value.
/// @throws std::invalid_argument Empty initial coordinates.
/// @par Example
/// @code
/// dm::optim::QuadraticObjective f({1},{0});
/// const auto
/// r=dm::optim::trace_optimize<dm::optim::DifferentiableFunction>(f,dm::optim::AdaGrad(),std::vector<double>{1});
/// std::print("{}\n",r.exception.empty()); // true
/// @endcode
/// @see OptimizationTrace, OptimizationRun
template <class Base, class Optimizer, class... Bounds>
OptimizationRun trace_optimize(Base& function, const Optimizer& optimizer,
                               std::vector<double> initial,
                               const Bounds&... bounds) {
  if (initial.empty())
    throw std::invalid_argument("trace_optimize: empty initial point");
  OptimizationRun out;
  out.coordinates = std::move(initial);
  detail::RecordedFunction<Base> recorded(function, out.trace);
  try {
    out.value = optimizer.optimize(recorded, out.coordinates, bounds...);
  } catch (const std::exception& e) {
    out.exception = e.what();
  }
  try {
    if constexpr (std::is_base_of_v<ResidualFunction, Base>)
      out.checked_value =
          detail::analysis_half_sse(function.residuals(out.coordinates));
    else
      out.checked_value = function.evaluate(out.coordinates);
    if constexpr (std::is_base_of_v<DifferentiableFunction, Base>)
      out.gradient_norm =
          detail::analysis_norm(function.gradient(out.coordinates));
  } catch (const std::exception& e) {
    if (out.exception.empty())
      out.exception = std::string("final diagnostic: ") + e.what();
  }
  return out;
}

/// @brief Convex quadratic .5*(x-centre)'H*(x-centre), with optional coupled/rotated axes.
/// H starts diagonal, adds coupling*11', then rotates its first two axes by angle
/// radians. Positive diagonal and nonnegative coupling make H positive definite.
/// Inputs are owned; calls validate finite coordinate shape and never mutate them.
/// @par Example
/// @code
/// dm::optim::QuadraticObjective f({2,8},{0,0});
/// std::print("{:.0f}\n",f.evaluate({1,1})); // 5
/// @endcode
/// @see trace_optimize()
class QuadraticObjective : public HessianFunction {
 public:
  /// @brief Build a finite SPD quadratic from its declared axes and centre.
  /// @param diagonal Positive finite curvature for each axis, nonempty.
  /// @param centre Finite matching optimum coordinates.
  /// @param coupling Nonnegative rank-one coupling, default zero.
  /// @param angle Finite first-plane rotation in radians, default zero.
  /// @throws std::invalid_argument Invalid dimensions or controls.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2},{3});
  /// std::print("{:.0f}\n",f.evaluate({3})); // 0
  /// @endcode
  QuadraticObjective(std::vector<double> diagonal, std::vector<double> centre,
                     double coupling = 0, double angle = 0);
  /// @brief Evaluate the half quadratic form.
  /// @param x Finite point of the configured dimension.
  /// @return Nonnegative loss with minimum zero.
  /// @throws std::invalid_argument Bad coordinates.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2},{0});
  /// std::print("{:.0f}\n",f.evaluate({2})); // 4
  /// @endcode
  double evaluate(const std::vector<double>& x) override;
  /// @brief Return H*(x-centre).
  /// @param x Matching finite coordinates.
  /// @return Owned full gradient.
  /// @throws std::invalid_argument Invalid coordinates.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2},{0});
  /// std::print("{:.0f}\n",f.gradient({2})[0]); // 4
  /// @endcode
  std::vector<double> gradient(const std::vector<double>& x) override;
  /// @brief Return the constant full SPD Hessian.
  /// @param x Matching finite coordinates, validated but not used numerically.
  /// @return Owned row-major nested matrix.
  /// @throws std::invalid_argument Invalid coordinates.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2},{0});
  /// std::print("{:.0f}\n",f.hessian({1})[0][0]); // 2
  /// @endcode
  std::vector<std::vector<double>> hessian(
      const std::vector<double>& x) override;

 private:
  std::vector<double> centre_;               ///< Owned unique minimizer.
  std::vector<std::vector<double>> matrix_;  ///< Owned constant SPD curvature.
};
/// @brief Rosenbrock valley (1-x)^2+100*(y-x^2)^2, Rastrigin or a quadratic-plus-sinusoidal bumpy surface.
/// Rastrigin is 20+sum(x_j^2-10*cos(2*pi*x_j)); their global minima are zero,
/// at (1,1) for Rosenbrock and (0,0) for Rastrigin. Bumpy is
/// (x-3)^2+(y+1)^2+5*sin(x)*cos(y), bounded below by -5; no exact global
/// minimizer is asserted. Analytic derivatives are
/// available even when a derivative-free optimizer is used. No mutable caches.
/// @par Example
/// @code
/// dm::optim::SmoothBenchmark f("rosenbrock");
/// std::print("{:.0f}\n",f.evaluate({1,1})); // 0
/// @endcode
/// @par References
/// - H. H. Rosenbrock (1960), An automatic method for finding the greatest or least value of a
/// function,
///   https://doi.org/10.1093/comjnl/3.3.175
/// @see QuadraticObjective
class SmoothBenchmark : public HessianFunction {
 public:
  /// @brief Select a named two-dimensional benchmark.
  /// @param name rosenbrock (default), rastrigin or bumpy.
  /// @throws std::invalid_argument Unknown benchmark.
  /// @par Example
  /// @code
  /// dm::optim::SmoothBenchmark f("rastrigin");
  /// std::print("{:.0f}\n",f.evaluate({0,0})); // 0
  /// @endcode
  explicit SmoothBenchmark(std::string name = "rosenbrock");
  /// @brief Evaluate the selected objective.
  /// @param x Two finite coordinates.
  /// @return Objective value.
  /// @throws std::invalid_argument Bad coordinates.
  /// @par Example
  /// @code
  /// dm::optim::SmoothBenchmark f;
  /// std::print("{:.0f}\n",f.evaluate({0,0})); // 1
  /// @endcode
  double evaluate(const std::vector<double>& x) override;
  /// @brief Evaluate exact first derivatives.
  /// @param x Two finite coordinates.
  /// @return Owned two-component gradient.
  /// @throws std::invalid_argument Bad coordinates.
  /// @par Example
  /// @code
  /// dm::optim::SmoothBenchmark f;
  /// std::print("{:.0f}\n",f.gradient({0,0})[0]); // -2
  /// @endcode
  std::vector<double> gradient(const std::vector<double>& x) override;
  /// @brief Evaluate exact second derivatives.
  /// @param x Two finite coordinates.
  /// @return Owned two-by-two Hessian, possibly indefinite.
  /// @throws std::invalid_argument Bad coordinates.
  /// @par Example
  /// @code
  /// dm::optim::SmoothBenchmark f;
  /// std::print("{:.0f}\n",f.hessian({0,0})[0][0]); // 2
  /// @endcode
  std::vector<std::vector<double>> hessian(
      const std::vector<double>& x) override;

 private:
  std::string name_;  ///< Validated benchmark name.
};
/// @brief Euclidean projection onto sum(x)=total, with exact KKT control.
/// The optimum is target-(sum(target)-total)/dimension in every coordinate.
/// Raw gradients at a constrained optimum generally do not vanish. Pure owned fixture.
/// @par Example
/// @code
/// dm::optim::EqualityQuadratic f({2,0},1);
/// std::print("{:.2f}\n",f.evaluate({1.5,-.5})); // 0.25
/// @endcode
/// @see trace_optimize()
class EqualityQuadratic : public EqualityConstrainedFunction {
 public:
  /// @brief Own a target and a single sum equality.
  /// @param target Finite nonempty unconstrained target.
  /// @param total Required finite sum, default one.
  /// @throws std::invalid_argument Empty/nonfinite input.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// std::print("{:.1f}\n",f.constraints({1,0})[0]); // 0.0
  /// @endcode
  EqualityQuadratic(std::vector<double> target, double total = 1);
  /// @brief Compute .5*||x-target||^2.
  /// @param x Matching finite point.
  /// @return Scalar .5*||x-target|| squared.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// std::print("{}\n",f.evaluate({1.5,-.5})); // 0.25
  /// @endcode
  double evaluate(const std::vector<double>& x) override;
  /// @brief Compute x-target.
  /// @param x Matching finite point.
  /// @return Owned coordinate-order vector x-target.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// std::print("{}\n",f.gradient({1.5,-.5}).size()); // 2
  /// @endcode
  std::vector<double> gradient(const std::vector<double>& x) override;
  /// @brief Compute the single equality sum(x)-total.
  /// @param x Matching finite point.
  /// @return Owned one-element vector sum(x)-total.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// std::print("{}\n",f.constraints({1.5,-.5}).size()); // 1
  /// @endcode
  std::vector<double> constraints(const std::vector<double>& x) override;
  /// @brief Return the single all-ones constraint derivative row.
  /// @param x Matching finite point.
  /// @return Owned one-row, p-column matrix of ones.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// std::print("{}\n",f.constraint_jacobian({1.5,-.5}).size()); // 1
  /// @endcode
  std::vector<std::vector<double>> constraint_jacobian(
      const std::vector<double>& x) override;

  /// @brief Return the exact equality-constrained optimum.
  /// @return Owned target minus its mean equality excess.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// std::print("{:.1f}\n",f.optimum()[0]); // 1.5
  /// @endcode
  std::vector<double> optimum() const;
  /// @brief Compare a point with the exact primal/dual equality control.
  /// @param x Finite matching coordinates.
  /// @return One row: feasibility, stationarity, distance and exact multiplier.
  /// @throws std::invalid_argument Invalid coordinates.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// std::print("{:.0f}\n",f.diagnostics(f.optimum()).double_at("stationarity",0)); // 0
  /// @endcode
  dstruct::DataFrame diagnostics(const std::vector<double>& x);

 private:
  std::vector<double> target_;  ///< Owned unconstrained target.
  double total_;                ///< Required sum.
};
/// @brief Projection onto x>=0 with a closed-form log-barrier path.
/// The constrained solution is max(target,0) coordinatewise. InteriorPoint needs
/// strictly positive coordinates even though this fixture accepts boundary points.
/// @par Example
/// @code
/// dm::optim::NonnegativeQuadratic f({-1,2});
/// std::print("{}\n",f.inequalities({1,1}).size()); // 2
/// @endcode
/// @see trace_optimize()
class NonnegativeQuadratic : public InequalityConstrainedFunction {
 public:
  /// @brief Own a target for projection onto the nonnegative orthant.
  /// @param target Nonempty finite unconstrained target.
  /// @throws std::invalid_argument Invalid target.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({-1,2});
  /// std::print("{:.1f}\n",f.evaluate({0,2})); // 0.5
  /// @endcode
  explicit NonnegativeQuadratic(std::vector<double> target);
  /// @brief Compute .5*||x-target||^2.
  /// @param x Matching finite coordinates; boundary allowed by fixture.
  /// @return Scalar .5*||x-target|| squared.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({-1,2});
  /// std::print("{}\n",f.evaluate({0,2})); // 0.5
  /// @endcode
  double evaluate(const std::vector<double>& x) override;
  /// @brief Compute x-target.
  /// @param x Matching finite coordinates; boundary allowed by fixture.
  /// @return Owned coordinate-order vector x-target.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({-1,2});
  /// std::print("{}\n",f.gradient({0,2}).size()); // 2
  /// @endcode
  std::vector<double> gradient(const std::vector<double>& x) override;
  /// @brief Return -x, using the solver convention c<=0.
  /// @param x Matching finite coordinates; boundary allowed by fixture.
  /// @return Owned p-vector -x, feasible when every cell is nonpositive.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({-1,2});
  /// std::print("{}\n",f.inequalities({0,2}).size()); // 2
  /// @endcode
  std::vector<double> inequalities(const std::vector<double>& x) override;
  /// @brief Return minus the identity matrix.
  /// @param x Matching finite coordinates; boundary allowed by fixture.
  /// @return Owned p by p negative identity matrix.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({-1,2});
  /// std::print("{}\n",f.inequality_jacobian({0,2}).size()); // 2
  /// @endcode
  std::vector<std::vector<double>> inequality_jacobian(
      const std::vector<double>& x) override;

  /// @brief Return the exact orthant-constrained optimum max(target,0).
  /// @return Owned componentwise projection.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({-1,2});
  /// std::print("{:.0f}\n",f.optimum()[0]); // 0
  /// @endcode
  std::vector<double> optimum() const;
  /// @brief Return the exact minimizer of f(x)-mu*sum(log(x)).
  /// @param mu Finite positive barrier weight.
  /// @return Positive path point (target+sqrt(target^2+4*mu))/2, evaluated stably.
  /// @throws std::invalid_argument Nonpositive/nonfinite mu.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({0});
  /// std::print("{:.0f}\n",f.barrier_optimum(4)[0]); // 2
  /// @endcode
  std::vector<double> barrier_optimum(double mu) const;
  /// @brief Evaluate orthant KKT residuals using candidate multipliers max(gradient,0).
  /// @param x Matching finite point, possibly infeasible.
  /// @return feasibility, stationarity, complementarity and distance to exact optimum.
  /// @throws std::invalid_argument Invalid coordinates.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({-1,2});
  /// std::print("{:.0f}\n",f.diagnostics(f.optimum()).double_at("complementarity",0)); // 0
  /// @endcode
  dstruct::DataFrame diagnostics(const std::vector<double>& x);

 private:
  std::vector<double> target_;  ///< Owned unconstrained target.
};
/// @brief Full-sum linear least squares with inspectable per-row components.
/// evaluate/gradient are sums. SGD and SAGA use mean component gradients;
/// SVRG uses the full gradient divided by n as its mean snapshot. All share the
/// same unpenalised minimizer, but a step size must name its loss convention.
/// @par Example
/// @code
/// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
/// std::print("{}\n",f.num_functions()); // 2
/// @endcode
/// @see trace_optimize()
class LeastSquaresObjective : public DifferentiableSeparableFunction {
 public:
  /// @brief Own a finite observation-by-parameter design and matching response.
  /// @param design Nonempty rectangular matrix, at least one column.
  /// @param response Finite outcomes, one per design row.
  /// @throws std::invalid_argument Empty/ragged/nonfinite or mismatched data.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
  /// std::print("{:.0f}\n",f.evaluate({2})); // 1
  /// @endcode
  LeastSquaresObjective(std::vector<std::vector<double>> design,
                        std::vector<double> response);
  /// @brief Build a fixed numeric Formula design, including its requested intercept.
  /// Missing rows are omitted jointly; fitting does not scale features implicitly.
  /// @param data Labelled fitting frame.
  /// @param formula Numeric response and predictor formula.
  /// @return Owned objective with the resolved expanded design and retained response.
  /// @throws std::invalid_argument Invalid/nonfinite design or response.
  /// @throws std::runtime_error Formula resolution errors.
  /// @par Example
  /// @code
  /// const auto f=dm::optim::LeastSquaresObjective::from_data(
  ///     dm::datasets::iris(),"Petal.Length ~ Petal.Width");
  /// std::print("{}\n",f.num_functions()); // 150
  /// @endcode
  static LeastSquaresObjective from_data(const dstruct::DataFrame& data,
                                         const std::string& formula);
  /// @brief Return the full sum of .5*residual^2.
  /// @param x Finite coefficients matching design columns.
  /// @return Full-sum half SSE, without normalisation by n.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
  /// std::print("{}\n",f.evaluate({2})); // 1
  /// @endcode
  double evaluate(const std::vector<double>& x) override;
  /// @brief Return the full summed gradient X transpose times residual.
  /// @param x Matching finite coefficients.
  /// @return Owned p-vector X transpose times residual, summed over rows.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
  /// std::print("{}\n",f.gradient({2}).size()); // 1
  /// @endcode
  std::vector<double> gradient(const std::vector<double>& x) override;
  /// @brief Return one unscaled .5*residual^2 component.
  /// @param x Matching coefficients.
  /// @param i Observation index in [0,n).
  /// @return Scalar half squared residual for row i, without normalisation.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates; invalid row index.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
  /// std::print("{}\n",f.evaluate_term({2},0)); // 0.5
  /// @endcode
  double evaluate_term(const std::vector<double>& x, std::size_t i) override;
  /// @brief Return one unscaled component gradient.
  /// @param x Matching coefficients.
  /// @param i Observation index in [0,n).
  /// @return Owned p-vector a_i times the row residual.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates; invalid row index.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
  /// std::print("{}\n",f.gradient_term({2},0).size()); // 1
  /// @endcode
  std::vector<double> gradient_term(const std::vector<double>& x,
                                    std::size_t i) override;
  /// @brief Number of row-loss terms.
  /// @return Owned design row count.
  std::size_t num_functions() const override { return response_.size(); }
  /// @brief Compute the gradient Lipschitz constant of the mean squared loss.
  /// Uses the largest eigenvalue of X transpose X/n; no data scaling or penalty.
  /// Dense Gram formation costs O(n*p*p), followed by a symmetric eigensolve.
  /// @return Nonnegative largest eigenvalue; zero for an all-zero design.
  /// @throws std::overflow_error Nonfinite Gram entries or eigensolver result.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1,0},{0,1}},{3,-1});
  /// std::print("{:.1f}\n",f.mean_loss_lipschitz()); // 0.5
  /// @endcode
  double mean_loss_lipschitz() const;
  /// @brief Borrow the frozen observation design.
  /// @return Const matrix view for this object's lifetime.
  const std::vector<std::vector<double>>& design() const noexcept {
    return design_;
  }
  /// @brief Borrow the retained outcomes.
  /// @return Const response view for this object's lifetime.
  const std::vector<double>& response() const noexcept { return response_; }

 private:
  std::vector<std::vector<double>> design_;  ///< Owned observation rows.
  std::vector<double> response_;             ///< Owned outcomes.
};
/// @brief Mean-loss lasso objective with an exact soft-threshold proximal map.
/// No hidden standardisation. Unpenalised coefficient zero supports an explicit
/// intercept column. Stationarity uses the proximal-gradient mapping, not raw
/// smooth-gradient zero. Dense calls own no mutable caches.
/// @par Example
/// @code
/// dm::optim::L1LeastSquaresObjective f({{1}},{3},1);
/// std::print("{:.0f}\n",f.proximal({3},1)[0]); // 2
/// @endcode
/// @see trace_optimize()
class L1LeastSquaresObjective : public ProximalFunction {
 public:
  /// @brief Own mean-squared-loss data and an L1 penalty.
  /// @param design Finite rectangular observation rows.
  /// @param response Matching finite outcomes.
  /// @param lambda Finite nonnegative L1 penalty.
  /// @param penalize_first Whether coefficient zero is penalised; default true, set false for
  /// intercept.
  /// @throws std::invalid_argument Invalid data or penalty.
  /// @par Example
  /// @code
  /// dm::optim::L1LeastSquaresObjective f({{1,0},{0,1}},{3,-1},.5);
  /// std::print("{:.1f}\n",f.proximal({2,-.2},1)[0]); // 1.5
  /// @endcode
  L1LeastSquaresObjective(std::vector<std::vector<double>> design,
                          std::vector<double> response, double lambda,
                          bool penalize_first = true);
  /// @brief Return .5*mean(residual^2)+lambda*L1.
  /// @param x Finite coefficients matching design.
  /// @return Full composite scalar: mean half squared residual plus the declared L1 penalty.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::L1LeastSquaresObjective f({{1,0},{0,1}},{0,0},.5);
  /// std::print("{}\n",f.evaluate({0,0})); // 0
  /// @endcode
  double evaluate(const std::vector<double>& x) override;
  /// @brief Return only the smooth mean-loss gradient.
  /// @param x Matching coefficients.
  /// @return Owned p-vector of the smooth mean-loss gradient; excludes the L1 penalty.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::optim::L1LeastSquaresObjective f({{1,0},{0,1}},{0,0},.5);
  /// std::print("{}\n",f.gradient({0,0}).size()); // 2
  /// @endcode
  std::vector<double> gradient(const std::vector<double>& x) override;
  /// @brief Soft threshold penalised coordinates at step*lambda.
  /// @param x Finite input point.
  /// @param step Finite positive proximal scale.
  /// @return Owned p-vector after coordinate soft thresholding by step*lambda; intercept can be exempt.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates; invalid step.
  /// @par Example
  /// @code
  /// dm::optim::L1LeastSquaresObjective f({{1,0},{0,1}},{0,0},.5);
  /// std::print("{}\n",f.proximal({0,0},1).size()); // 2
  /// @endcode
  std::vector<double> proximal(const std::vector<double>& x,
                               double step) override;

 private:
  LeastSquaresObjective
      least_squares_;    ///< Frozen sum-loss design and outcomes.
  double lambda_;        ///< Mean-loss L1 penalty.
  bool penalize_first_;  ///< Whether coordinate zero participates in L1.
};
/// @brief Analytic exponential, offset-exponential and logistic residuals for Levenberg-Marquardt studies.
/// The optimizer returns half SSE. Parameters are unconstrained; positivity of
/// amplitude/rate is not imposed. Parameter order is (A,k), (A,k,c), or
/// (L,k,mid) for exponential, offset-exponential, or logistic respectively.
/// Logistic evaluation uses a stable inverse link.
/// Callback coordinate dimensions and finiteness are checked.
/// @par Example
/// @code
/// dm::dstruct::DataFrame
/// d;d.add_column("x",std::vector<double>{0,1});d.add_column("y",std::vector<double>{2,1});
/// dm::optim::CurveResidual f(d);
/// std::print("{}\n",f.residuals({2,std::log(2.)}).size()); // 2
/// @endcode
/// @see trace_optimize()
class CurveResidual : public ResidualFunction {
 public:
  /// @brief Own a finite x/y curve-fitting sample.
  /// @param data Nonempty numeric x/y columns; no missing/nonfinite cells.
  /// @param model exponential (A*exp(-k*x)), exponential_offset (A*exp(-k*x)+c), or logistic (L/(1+exp(-k*(x-mid)))).
  /// @throws std::invalid_argument Invalid model/data.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame
  /// d;d.add_column("x",std::vector<double>{0,1});d.add_column("y",std::vector<double>{2,1});
  /// dm::optim::CurveResidual f(d);
  /// std::print("{}\n",f.residuals({2,std::log(2.)}).size()); // 2
  /// @endcode
  CurveResidual(const dstruct::DataFrame& data,
                std::string model = "exponential");
  /// @brief Compute model-minus-observed residuals.
  /// @param x Two exponential or three exponential_offset/logistic finite parameters.
  /// @return Owned n-vector of predictions minus observed outcomes.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame
  /// d;d.add_column("x",std::vector<double>{0,1});d.add_column("y",std::vector<double>{2,1});
  /// dm::optim::CurveResidual f(d);
  /// std::print("{}\n",f.residuals({2,std::log(2.)}).size()); // 2
  /// @endcode
  std::vector<double> residuals(const std::vector<double>& x) override;
  /// @brief Compute the analytic residual derivatives.
  /// @param x Two exponential or three exponential_offset/logistic finite parameters.
  /// @return Owned n by p matrix, observation rows and parameter-order columns.
  /// @throws std::invalid_argument Wrong coordinate count or nonfinite coordinates.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame
  /// d;d.add_column("x",std::vector<double>{0,1});d.add_column("y",std::vector<double>{2,1});
  /// dm::optim::CurveResidual f(d);
  /// std::print("{}\n",f.jacobian({2,std::log(2.)}).size()); // 2
  /// @endcode
  std::vector<std::vector<double>> jacobian(
      const std::vector<double>& x) override;

  /// @brief Evaluate the model at the owned sample inputs without subtracting y.
  /// @param x Matching finite model parameters.
  /// @return Owned fitted curve values in sample order.
  /// @throws std::invalid_argument Invalid parameters.
  /// @par Example
  /// @code
  /// const auto d=dm::optim::optimization_curve_data();dm::optim::CurveResidual f(d);
  /// std::print("{:.1f}\n",f.predict({2.5,.7})[0]); // 2.5
  /// @endcode
  std::vector<double> predict(const std::vector<double>& x);

 private:
  std::vector<double> x_;  ///< Owned sample input coordinates.
  std::vector<double> y_;  ///< Owned observed outcomes.
  std::string model_;      ///< Validated curve family.
};

/// @brief Make declared noisy curve fixtures, including the original offset-exponential/logistic recipes.
/// @param model exponential, exponential_offset or logistic.
/// @param n At least three equally spaced observations, default 30.
/// @param seed Local mt19937_64 seed, default 42; standard normal draws can vary by platform.
/// @return Owned x/y/truth columns. exponential uses x=5*i/(n-1),
/// truth 2.5*exp(-.7*x), noise SD .05. exponential_offset uses x=.25*i,
/// truth 5*exp(-.7*x)+1, noise SD .1. logistic uses x=.2*i,
/// truth 10/(1+exp(-1.5*(x-5))), noise SD .15.
/// @throws std::invalid_argument Invalid model/count.
/// @par Example
/// @code
/// std::print("{}\n",dm::optim::optimization_curve_data().nrows()); // 30
/// @endcode
/// @see CurveResidual
dstruct::DataFrame optimization_curve_data(
    const std::string& model = "exponential", std::size_t n = 30,
    std::uint64_t seed = 42);
/// @brief Plot explicitly sampled callback points from several recorded runs.
/// Replays one pure objective evaluation per selected event after fitting. No
/// accepted-iteration or wall-time claim is made. The y display is log10(max(f-reference,1e-14));
/// flooring is a plotting convention only. Nonfinite sampled values are omitted,
/// while the run exception and final checks remain in summary(). Trials can exceed
/// initial loss. For population studies best_observed shows a running minimum of
/// evaluated trials, not a certified bound or necessarily an accepted trajectory.
/// @param objective Pure deterministic shared objective.
/// @param runs Captured results in legend order, borrowed.
/// @param labels One short label per run.
/// @param kind Callback kind, default value; use gradient for Newton/coordinate/proximal sampling.
/// @param reference Declared minimum or explicit lower-bound reference, default zero.
/// @param best_observed Plot running best observed sample, default false.
/// @param term_axis Use cumulative nominal component-gradient work instead of callback index.
/// @return Owned plot with local legend headroom, ready for save_svg().
/// @throws std::invalid_argument Mismatched labels or no finite points.
/// @par Example
/// @code
/// dm::optim::QuadraticObjective f({1},{0});
/// const auto
/// r=dm::optim::trace_optimize<dm::optim::DifferentiableFunction>(f,dm::optim::AdaGrad(),std::vector<double>{1});
/// dm::optim::optimization_comparison(f,{r},{"AdaGrad"}).save_svg("build/optimization_comparison.svg");
/// @endcode
/// @see OptimizationTrace::sample()
graphics::RPlot optimization_comparison(
    ArbitraryFunction& objective, const std::vector<OptimizationRun>& runs,
    const std::vector<std::string>& labels, const std::string& kind = "value",
    double reference = 0, bool best_observed = false, bool term_axis = false);
/// @brief Euclidean proximal-gradient mapping norm for a declared positive step.
/// This first-order stationarity diagnostic includes nonsmooth penalties through
/// the supplied prox. It performs one gradient and one prox call after fitting.
/// @param f Composite objective whose gradient is only the smooth component.
/// @param x Matching finite point.
/// @param step Finite positive mapping step.
/// @return Norm of (x-prox(x-step*gradient(x),step))/step.
/// @throws std::invalid_argument Invalid step or vector dimensions.
/// @par Example
/// @code
/// dm::optim::L1LeastSquaresObjective f({{1}},{3},1);
/// std::print("{:.0f}\n",dm::optim::proximal_mapping_norm(f,{2},1)); // 0
/// @endcode
/// @see L1LeastSquaresObjective
double proximal_mapping_norm(ProximalFunction& f, const std::vector<double>& x,
                             double step);
/// @brief Compare an analytic residual Jacobian with central finite differences.
/// This is a post-fit diagnostic, not used by LevenbergMarquardt. One pair of
/// residual calls per parameter; truncation/rounding error depends on step.
/// @param f Matching residual/Jacobian interface.
/// @param x Finite parameter point.
/// @param step Positive finite relative perturbation, default 1e-6; scales by max(1,abs(xj)).
/// @return parameter/max_absolute_error table.
/// @throws std::invalid_argument Invalid step or inconsistent derivative dimensions.
/// @par Example
/// @code
/// const auto d=dm::optim::optimization_curve_data();dm::optim::CurveResidual f(d);
/// std::print("{}\n",dm::optim::check_residual_jacobian(f,{2,.5}).nrows()); // 2
/// @endcode
/// @see CurveResidual
dstruct::DataFrame check_residual_jacobian(ResidualFunction& f,
                                           const std::vector<double>& x,
                                           double step = 1e-6);
/// @brief Compute the post-run norm of J transpose times residuals.
/// @param f Analytic matching residual/Jacobian interface.
/// @param x Finite parameter point.
/// @return Euclidean gradient norm of half residual SSE; two separate diagnostic callbacks.
/// @throws std::invalid_argument Inconsistent derivative dimensions.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1});d.add_column("y",std::vector<double>{2,1});
/// dm::optim::CurveResidual f(d);
/// std::print("{:.0f}\n",dm::optim::residual_gradient_norm(f,{2,std::log(2.)})); // 0
/// @endcode
/// @see check_residual_jacobian()
double residual_gradient_norm(ResidualFunction& f,
                              const std::vector<double>& x);
}  // namespace datamunge::optim
