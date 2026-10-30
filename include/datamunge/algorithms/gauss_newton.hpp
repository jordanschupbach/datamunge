#pragma once
/// @file gauss_newton.hpp
/// @brief Full-step nonlinear least squares with scaled Householder QR and explicit iteration
/// evidence.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/qr.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
/// @brief One local residual/Jacobian least-squares problem and its undamped step.
/// All values are owned. J is observations by parameters, r=model-y, and step minimizes
/// ||r+J*step||_2. Columns are scaled to unit norm internally, then the solution is
/// restored to parameter units; no normal equations, regularization or bounds are used.
/// A numerically rank-deficient Jacobian is rejected, not assigned an arbitrary solution.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& p,double){return p[0]*p[0];};
/// const auto a=dm::algorithms::gauss_newton_linearize(f,{0},{2},{1});
/// std::print("{:.6f}\n",a.step[0]); // 0.500000
/// @endcode
/// @see gauss_newton_linearize(), GaussNewtonResult
struct GaussNewtonLinearization {
  std::vector<double> residuals; ///< Owned model-minus-observed residuals, in input row order.
  linalg::DenseMatrix<double> jacobian; ///< Owned observation-by-parameter residual derivatives.
  std::vector<double> step;   ///< Owned full least-squares update in original parameter units.
  double              sse{0}; ///< Sum of squared current residuals, without a factor of one half.
  double              gradient_inf{0}; ///< Infinity norm of computed J^T*r; approximate with finite differences.
  double linearized_sse{0}; ///< Sum of squares of r+J*step, the predicted full-step objective.
};
/// @brief One evaluated parameter state, including its proposed next step.
/// The initial state is iteration zero. step_norm is the norm of the incoming completed
/// update (zero initially), while next_step_norm and linearized_sse describe the proposed
/// outgoing update. Diagnostics are calculated at these params, not the preceding state.
/// @par Example
/// @code
/// const dm::algorithms::GaussNewtonIteration row{};
/// std::print("{}\n",row.iteration); // 0
/// @endcode
/// @see GaussNewtonResult::trace()
struct GaussNewtonIteration {
  int                 iteration{0};    ///< Completed full updates at this state.
  std::vector<double> params;          ///< Owned current parameter vector.
  double              sse{0};          ///< Current sum of squared residuals.
  double              gradient_inf{0}; ///< Current infinity norm of computed J^T*r; approximate with finite differences.
  double step_norm{0};      ///< Euclidean norm of the incoming completed update; zero initially.
  double next_step_norm{0}; ///< Euclidean norm of the proposed undamped update.
  double linearized_sse{0}; ///< Predicted SSE after the proposed undamped update.
};
/// @brief Owned full-step Gauss-Newton fit and its evaluated parameter trajectory.
/// The original params/iterations/sse fields remain first for aggregate compatibility.
/// iterations counts completed updates, including on budget exhaustion. sse and residuals
/// always describe the returned params. step_tolerance_met only certifies that the next
/// proposed step has norm <= tol: it does not prove stationarity, a minimum, global
/// optimality or parameter identification. Check the gradient and model assumptions too.
/// No early rejection or line search hides an increase in the objective.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& p,double x){return p[0]+p[1]*x;};
/// const auto fit=dm::algorithms::gauss_newton(f,{-1,0,1},{0,2,4},{0,0});
/// std::print("{:.6f} {:.6f}\n",fit.params[0],fit.params[1]); // 2.000000 2.000000
/// @endcode
/// @see gauss_newton(), GaussNewtonLinearization
struct GaussNewtonResult {
  std::vector<double> params; ///< Owned final parameters in caller-specified units/order.
  int    iterations{0};       ///< Number of completed full updates, zero when none were taken.
  double sse{0};              ///< SSE recomputed at final params, even on budget exhaustion.
  bool   step_tolerance_met{false}; ///< Whether the final proposed step's norm is <= tol.
  std::vector<double>               residuals; ///< Final model-minus-y residuals in input order.
  std::vector<GaussNewtonIteration> history;   ///< Initial and all completed evaluated states.

  /// @brief Presents the parameter trajectory and current-state diagnostics as a table.
  /// No fitting or model calls occur; work/storage O(history.size()*(params.size()+1)).
  /// @return Owned rows with iteration, sse, gradient_inf, step_norm, next_step_norm,
  /// linearized_sse and p0,p1,... . An empty default result gives an empty table.
  /// @throws std::out_of_range A manually altered history row has too few parameters.
  /// @par Example
  /// @code
  /// const auto f=[](const std::vector<double>& p,double){return p[0]*p[0];};
  /// const auto fit=dm::algorithms::gauss_newton(f,{0},{2},{1},1);
  /// std::print("{}\n",fit.trace().nrows()); // 2
  /// @endcode
  /// @see gauss_newton()
  [[nodiscard]] dstruct::DataFrame trace() const {
    dstruct::DataFrame out;
    if (history.empty())
      return out;
    std::vector<double> iteration, loss, gradient, incoming, next, predicted;
    for (const auto& row : history) {
      iteration.push_back(row.iteration);
      loss.push_back(row.sse);
      gradient.push_back(row.gradient_inf);
      incoming.push_back(row.step_norm);
      next.push_back(row.next_step_norm);
      predicted.push_back(row.linearized_sse);
    }
    out.add_column("iteration", iteration);
    out.add_column("sse", loss);
    out.add_column("gradient_inf", gradient);
    out.add_column("step_norm", incoming);
    out.add_column("next_step_norm", next);
    out.add_column("linearized_sse", predicted);
    for (std::size_t j = 0; j < params.size(); ++j) {
      std::vector<double> values;
      for (const auto& row : history)
        values.push_back(row.params.at(j));
      out.add_column("p" + std::to_string(j), values);
    }
    return out;
  }
};

/// @brief Internal checked arithmetic and Gauss-Newton iteration machinery.
namespace detail {
/// @brief Converts an intermediate result to a finite representable double.
/// @param value Extended-precision intermediate, allowing underflow to zero.
/// @return Finite converted value.
/// @throws std::invalid_argument Nonfinite or unrepresentable arithmetic.
inline double gn_finite(long double value) {
  const auto x = static_cast<double>(value);
  if (!std::isfinite(x))
    throw std::invalid_argument("gauss_newton: nonfinite/unrepresentable model or arithmetic");
  return x;
}
/// @brief Checks complete finite data and a nonempty, at-most-observation-size parameter vector.
/// @param xs Predictor coordinates, no missing values allowed.
/// @param ys Observed responses of equal length.
/// @param params Current finite parameter vector.
/// @throws std::invalid_argument Empty/mismatched/underdetermined input or nonfinite values.
inline void gn_validate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& params) {
  if (xs.empty() || xs.size() != ys.size() || params.empty() || params.size() > xs.size())
    throw std::invalid_argument(
        "gauss_newton: require equal nonempty x/y, and 1..observations parameters");
  for (double x : xs)
    (void)gn_finite(x);
  for (double y : ys)
    (void)gn_finite(y);
  for (double p : params)
    (void)gn_finite(p);
}
/// @brief Computes a finite vector Euclidean norm using scaled hypot accumulation.
/// @param x Borrowed finite vector.
/// @return Finite Euclidean norm.
/// @throws std::invalid_argument Norm not representable as a finite double.
inline double gn_norm(const std::vector<double>& x) {
  double n = 0;
  for (double v : x)
    n = std::hypot(n, v);
  return gn_finite(n);
}
/// @brief Solves a column-scaled local least-squares problem by existing Householder QR.
/// Uses the explicit-Q factorization; no J^T*J is formed. Rank threshold after column
/// scaling is max(m,n)*epsilon*max(abs(diag(R))). This is a numerical test, not a
/// global identifiability assessment. Work O(m*n^2+m^2*n), storage O(m^2+m*n).
/// @param out Owned current residuals and finite Jacobian, filled with step and metrics.
/// @return Completed owned local problem.
/// @throws std::runtime_error Zero column or numerically rank-deficient scaled Jacobian.
/// @throws std::invalid_argument Nonfinite or unrepresentable diagnostics or step.
inline GaussNewtonLinearization gn_finish(GaussNewtonLinearization out) {
  const auto          m = out.jacobian.rows(), n = out.jacobian.cols();
  auto                scaled = out.jacobian;
  std::vector<double> norms(n), rhs(m);
  long double         sse = 0;
  for (std::size_t i = 0; i < m; ++i) {
    rhs[i] = -out.residuals[i];
    sse += static_cast<long double>(rhs[i]) * rhs[i];
  }
  out.sse = gn_finite(sse);
  for (std::size_t j = 0; j < n; ++j) {
    double      norm     = 0;
    long double gradient = 0;
    for (std::size_t i = 0; i < m; ++i) {
      norm = std::hypot(norm, out.jacobian(i, j));
      gradient += static_cast<long double>(out.jacobian(i, j)) * out.residuals[i];
    }
    norms[j]         = gn_finite(norm);
    out.gradient_inf = std::max(out.gradient_inf, std::abs(gn_finite(gradient)));
    if (norm == 0)
      throw std::runtime_error("gauss_newton: zero Jacobian column (rank deficient)");
    for (std::size_t i = 0; i < m; ++i)
      scaled(i, j) /= norm;
  }
  const auto qr = linalg::qr(scaled);
  if (!qr.is_full_rank())
    throw std::runtime_error("gauss_newton: scaled Jacobian is numerically rank deficient");
  out.step = qr.solve(rhs);
  for (std::size_t j = 0; j < n; ++j)
    out.step[j] = gn_finite(static_cast<long double>(out.step[j]) / norms[j]);
  long double predicted = 0;
  for (std::size_t i = 0; i < m; ++i) {
    long double r = out.residuals[i];
    for (std::size_t j = 0; j < n; ++j)
      r += static_cast<long double>(out.jacobian(i, j)) * out.step[j];
    predicted += r * r;
  }
  out.linearized_sse = gn_finite(predicted);
  return out;
}
/// @brief Executes checked full steps, evaluating every returned state and its next step.
/// @tparam Linearize Callable from parameter vector to owned local problem.
/// @param linearize Borrowed local-problem evaluator, may call user callbacks.
/// @param params Owned initial parameters.
/// @param max_iter Nonnegative maximum completed updates.
/// @param tol Finite positive absolute proposed-step norm tolerance.
/// @return Owned result including initial and final-state diagnostics.
/// @throws std::invalid_argument Invalid controls or invalid arithmetic from linearize.
/// @throws std::runtime_error A visited state's Jacobian is rank deficient.
/// @note User callback exceptions propagate, including after an objective-increasing step.
/// No line search, trust region, bounds, weights or damping are applied.
template<class Linearize>
GaussNewtonResult
gn_run(Linearize& linearize, std::vector<double> params, int max_iter, double tol) {
  if (max_iter < 0 || !std::isfinite(tol) || tol <= 0)
    throw std::invalid_argument("gauss_newton: max_iter>=0 and finite tol>0 required");
  GaussNewtonResult out;
  out.params      = std::move(params);
  double incoming = 0;
  for (;;) {
    auto         local = linearize(out.params);
    const double next  = gn_norm(local.step);
    out.sse            = local.sse;
    out.residuals      = local.residuals;
    out.history.push_back(
        {out.iterations,
         out.params,
         local.sse,
         local.gradient_inf,
         incoming,
         next,
         local.linearized_sse});
    out.step_tolerance_met = next <= tol;
    if (out.step_tolerance_met || out.iterations == max_iter)
      return out;
    for (std::size_t j = 0; j < out.params.size(); ++j)
      out.params[j] = gn_finite(static_cast<long double>(out.params[j]) + local.step[j]);
    incoming = next;
    ++out.iterations;
  }
}
} // namespace detail

/// @brief Linearizes model-minus-y residuals with a caller-supplied analytic Jacobian.
/// Computes residuals, full-rank scaled-QR step, current SSE, infinity norm of J^T*r,
/// and predicted linearized SSE. Model and derivative see identical params and x.
/// Inputs remain unchanged; callbacks must be deterministic for fixed inputs.
/// Callback state/thread safety is the caller's responsibility.
/// @tparam Model Callable double(const std::vector<double>&,double).
/// @tparam Jacobian Callable returning a parameter-sized derivative vector for one row.
/// @param model Value callable, passed by value; finite output required.
/// @param jacobian Row-derivative callable, passed by value, in parameter order.
/// @param xs Nonempty finite coordinates, same length as ys and at least params.size().
/// @param ys Nonempty finite observations in matching order; no missing-row omission.
/// @param params Nonempty finite current parameter vector; remains unchanged.
/// @return Owned current local least-squares problem and proposed step.
/// @throws std::invalid_argument Invalid sizes, derivative length, nonfinite values or arithmetic.
/// @throws std::runtime_error Zero/numerically rank-deficient scaled Jacobian.
/// @note User callback exceptions propagate unchanged.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& p,double){return p[0]*p[0];};
/// const auto j=[](const std::vector<double>& p,double){return std::vector<double>{2*p[0]};};
/// const auto a=dm::algorithms::gauss_newton_linearize(f,j,{0},{2},{1});
/// std::print("{:.1f}\n",a.jacobian(0,0)); // 2.0
/// @endcode
/// @par References
/// - B. W. Rust (2002), Fitting Nature's Basic Functions Part III: Exponentials,
///   Sinusoids and Nonlinear Least Squares. Computing in Science & Engineering 4(4).
///   https://www.nist.gov/publications/fitting-natures-basic-functions-part-iii-exponentials-sinusoids-and-nonlinear-least
/// @see gauss_newton(), GaussNewtonLinearization
template<class Model, class Jacobian>
GaussNewtonLinearization gauss_newton_linearize(
    Model                      model,
    Jacobian                   jacobian,
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& params) {
  detail::gn_validate(xs, ys, params);
  GaussNewtonLinearization out;
  out.jacobian = linalg::DenseMatrix<double>(xs.size(), params.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    out.residuals.push_back(
        detail::gn_finite(
            static_cast<long double>(detail::gn_finite(model(params, xs[i]))) - ys[i]));
    const auto row = jacobian(params, xs[i]);
    if (row.size() != params.size())
      throw std::invalid_argument("gauss_newton: wrong Jacobian row length");
    for (std::size_t j = 0; j < params.size(); ++j)
      out.jacobian(i, j) = detail::gn_finite(row[j]);
  }
  return detail::gn_finish(std::move(out));
}
/// @brief Linearizes residuals using scale-aware central finite differences.
/// Step h=cbrt(epsilon)*max(1,abs(params[j])); divides by the actual representable
/// plus-minus parameter span. Central differences require both perturbations to remain
/// in the model's domain. Near a parameter boundary, supply the analytic overload.
/// Costs m*(1+2*n) model calls, plus O(m*n^2+m^2*n) explicit-Q work and O(m^2+m*n)
/// storage. Difference error can limit convergence; no universal precision is promised.
/// @tparam Model Callable double(const std::vector<double>&,double).
/// @param model Finite model callable, passed by value; callback exceptions propagate.
/// @param xs Nonempty finite coordinates, matching ys and at least params.size().
/// @param ys Nonempty finite observed responses; no omitted rows or weights.
/// @param params Nonempty finite current parameter vector; remains unchanged.
/// @return Owned local residuals, Jacobian, full step and diagnostics.
/// @throws std::invalid_argument Invalid sizes, nonfinite model values or arithmetic.
/// @throws std::runtime_error Numerically rank-deficient scaled Jacobian.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& p,double){return p[0]*p[0];};
/// const auto a=dm::algorithms::gauss_newton_linearize(f,{0},{2},{1});
/// std::print("{:.6f}\n",a.step[0]); // 0.500000
/// @endcode
/// @par References
/// - B. W. Rust (2002), Fitting Nature's Basic Functions Part III: Exponentials,
///   Sinusoids and Nonlinear Least Squares. Computing in Science & Engineering 4(4).
///   https://www.nist.gov/publications/fitting-natures-basic-functions-part-iii-exponentials-sinusoids-and-nonlinear-least
/// @see gauss_newton(), GaussNewtonLinearization
template<class Model>
GaussNewtonLinearization gauss_newton_linearize(
    Model                      model,
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& params) {
  /// @brief Differentiates one model row with representable symmetric perturbations.
  /// @param p Current finite parameters.
  /// @param x One observed coordinate.
  /// @return Owned finite parameter derivatives.
  /// @throws std::invalid_argument Invalid perturbed point, output or difference.
  const auto jacobian = [&model](const std::vector<double>& p, double x) {
    std::vector<double> row(p.size());
    for (std::size_t j = 0; j < p.size(); ++j) {
      const double h =
          std::cbrt(std::numeric_limits<double>::epsilon()) * std::max(1., std::abs(p[j]));
      auto plus = p, minus = p;
      plus[j]      = detail::gn_finite(static_cast<long double>(p[j]) + h);
      minus[j]     = detail::gn_finite(static_cast<long double>(p[j]) - h);
      const auto a = detail::gn_finite(model(plus, x)), b = detail::gn_finite(model(minus, x));
      row[j] = detail::gn_finite(
          (static_cast<long double>(a) - b) / (static_cast<long double>(plus[j]) - minus[j]));
    }
    return row;
  };
  return gauss_newton_linearize(model, jacobian, xs, ys, params);
}
/// @brief Fits unweighted nonlinear least squares with analytic derivatives and full QR steps.
/// Solves min_delta ||r+J*delta|| without forming normal equations, updates params by
/// the entire step, and reevaluates residuals/Jacobian. Stops when the proposed next
/// step has Euclidean norm <= tol or max_iter updates are complete. This absolute
/// parameter-unit test is not invariant to reparameterization and is not a minimum
/// certificate. Objective increases are allowed; no globalization, constraints or
/// automatic scaling of the stopping rule is applied. Column scaling only stabilizes
/// the local linear solve. history includes the initial and every completed state.
/// Inputs remain unchanged; callback state/thread safety belongs to the caller.
/// @tparam Model Callable double(const std::vector<double>&,double).
/// @tparam Jacobian Callable returning one parameter-ordered derivative row.
/// @param model Finite model callable, passed by value.
/// @param jacobian Finite analytic derivative callable, passed by value.
/// @param xs Nonempty finite coordinates; observations >= number of parameters.
/// @param ys Finite responses of length xs.size(); no null handling or weighting.
/// @param params0 Owned initial finite parameters, at least one.
/// @param max_iter Nonnegative maximum full updates, default 100. Zero evaluates the initial state
/// only.
/// @param tol Finite positive absolute proposed-step norm tolerance, default 1e-12.
/// @return Owned final parameters, matching SSE/residuals, completed-update count and trace.
/// @throws std::invalid_argument Invalid sizes/controls/derivatives or nonfinite arithmetic.
/// @throws std::runtime_error A visited state's scaled Jacobian is numerically rank deficient.
/// @note User callback exceptions propagate unchanged. Rank failure can occur after a full step.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& p,double){return p[0]*p[0];};
/// const auto j=[](const std::vector<double>& p,double){return std::vector<double>{2*p[0]};};
/// const auto fit=dm::algorithms::gauss_newton(f,j,{0},{2},{1});
/// std::print("{:.6f}\n",fit.params[0]); // 1.414214
/// @endcode
/// @par References
/// - B. W. Rust (2002), Fitting Nature's Basic Functions Part III: Exponentials,
///   Sinusoids and Nonlinear Least Squares. Computing in Science & Engineering 4(4).
///   https://www.nist.gov/publications/fitting-natures-basic-functions-part-iii-exponentials-sinusoids-and-nonlinear-least
/// @see gauss_newton_linearize(), GaussNewtonResult::trace()
template<class Model, class Jacobian>
GaussNewtonResult gauss_newton(
    Model                      model,
    Jacobian                   jacobian,
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    std::vector<double>        params0,
    int                        max_iter = 100,
    double                     tol      = 1e-12) {
  /// @brief Builds the local analytic least-squares problem at one proposed state.
  /// @param params Parameter state to evaluate.
  /// @return Owned checked local problem.
  const auto linearize = [&](const std::vector<double>& params) {
    return gauss_newton_linearize(model, jacobian, xs, ys, params);
  };
  return detail::gn_run(linearize, std::move(params0), max_iter, tol);
}
/// @brief Fits unweighted nonlinear least squares using central-difference full QR steps.
/// Same full-step method, absolute proposed-step stopping rule and numerical rank policy
/// as the analytic overload. Jacobians use gauss_newton_linearize() central differences;
/// the model must be defined on both perturbations of each visited parameter. Finite
/// differences can limit accuracy even at a small residual. No analytic derivatives,
/// line search or damping are silently substituted. Per evaluated state: m*(1+2*n)
/// model calls, O(m*n^2+m^2*n) QR work, O(m^2+m*n) temporary storage. The trace adds
/// O((iterations+1)*n) storage. All input arrays remain unchanged.
/// @tparam Model Callable double(const std::vector<double>&,double).
/// @param model Finite model callable, passed by value; callback exceptions propagate.
/// @param xs Nonempty finite coordinates, matching ys and at least params0.size().
/// @param ys Finite observed responses; no missing-value omission or weighting.
/// @param params0 Owned nonempty finite initial parameters.
/// @param max_iter Nonnegative maximum completed updates, default 100. Zero evaluates the initial
/// state.
/// @param tol Finite positive absolute proposed-step norm tolerance, default 1e-12.
/// @return Owned final parameters, matching SSE/residuals, update count, stopping flag and trace.
/// @throws std::invalid_argument Invalid input sizes/controls or nonfinite arithmetic/model values.
/// @throws std::runtime_error Zero or numerically rank-deficient scaled Jacobian at a visited
/// state.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& p,double x){return p[0]+p[1]*x;};
/// const auto fit=dm::algorithms::gauss_newton(f,{-1,0,1},{0,2,4},{0,0});
/// std::print("{:.6f} {:.6f}\n",fit.params[0],fit.params[1]); // 2.000000 2.000000
/// @endcode
/// @par References
/// - B. W. Rust (2002), Fitting Nature's Basic Functions Part III: Exponentials,
///   Sinusoids and Nonlinear Least Squares. Computing in Science & Engineering 4(4).
///   https://www.nist.gov/publications/fitting-natures-basic-functions-part-iii-exponentials-sinusoids-and-nonlinear-least
/// @see gauss_newton_linearize(), GaussNewtonResult::trace()
template<class Model>
GaussNewtonResult gauss_newton(
    Model                      model,
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    std::vector<double>        params0,
    int                        max_iter = 100,
    double                     tol      = 1e-12) {
  /// @brief Builds the central-difference least-squares problem at one proposed state.
  /// @param params Parameter state to evaluate.
  /// @return Owned checked local problem.
  const auto linearize = [&](const std::vector<double>& params) {
    return gauss_newton_linearize(model, xs, ys, params);
  };
  return detail::gn_run(linearize, std::move(params0), max_iter, tol);
}
} // namespace datamunge::algorithms
