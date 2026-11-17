/// @file halley.cpp
/// @brief Shared checked Halley arithmetic and online actual-inspection capture.
#include <cmath>
#include <datamunge/algorithms/halley.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private callback validation and producing kernel.
namespace halley_detail {
/// @brief Invoke one callback and reject nonfinite output, preserving callback exceptions.
/// @param f Borrowed nonempty callback.
/// @param x Finite input.
/// @return Actual finite signed output.
/// @throws std::domain_error On nonfinite callback output.
double evaluate(const std::function<double(double)>& f, double x) {
  const double y = f(x);
  if (!std::isfinite(y))
    throw std::domain_error("Halley callback is nonfinite");
  return y;
}
/// @brief Execute the legacy control flow once with optional admitted inspection records.
/// @param f Borrowed scalar callback.
/// @param df Borrowed first derivative.
/// @param d2f Borrowed second derivative.
/// @param x0 Initial iterate, validated here.
/// @param tol Strict residual tolerance, validated here.
/// @param max_iter Update budget, validated here.
/// @param steps Optional borrowed initially empty trace sink.
/// @param cap Online record cap, zero disables policy cap.
/// @return Owned actual summary, stop and work.
/// @throws std::invalid_argument On invalid controls or empty callbacks.
/// @throws std::domain_error On nonfinite callback results.
/// @throws std::overflow_error On nonfinite update arithmetic.
/// @throws std::length_error On online capture/container limits.
/// @throws std::bad_alloc On record allocation failure.
HalleyRun execute(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    const std::function<double(double)>& d2f,
    double                               x0,
    double                               tol,
    int                                  max_iter,
    std::vector<HalleyStep>*             steps,
    std::size_t                          cap) {
  if (!f || !df || !d2f || !std::isfinite(x0) || !std::isfinite(tol) || tol < 0 || max_iter < 0 ||
      max_iter == std::numeric_limits<int>::max())
    throw std::invalid_argument("Halley controls/callbacks invalid");
  HalleyRun run;
  double    x = x0;
  for (int j = 0; j <= max_iter; ++j) {
    if (steps) {
      if ((cap && steps->size() >= cap) || steps->size() == steps->max_size())
        throw std::length_error("Halley online capture limit");
      // Allocate/admit the whole record before this callback can have side effects.
      steps->emplace_back();
    }
    HalleyStep record;
    record.inspection = j + 1;
    record.x = record.next = x;
    record.fx              = evaluate(f, x);
    ++run.work.f_calls;
    run.result.root     = x;
    run.result.residual = std::fabs(record.fx);
    if (j < max_iter)
      run.result.iterations = j + 1;
    if (run.result.residual < tol) {
      run.result.converged = true;
      run.stop             = HalleyStop::Residual;
      if (steps)
        steps->back() = record;
      return run;
    }
    if (j == max_iter) {
      run.stop = HalleyStop::IterationLimit;
      if (steps)
        steps->back() = record;
      return run;
    }
    record.derivative = evaluate(df, x);
    ++run.work.df_calls;
    record.second_derivative = evaluate(d2f, x);
    ++run.work.d2f_calls;
    record.derivatives_evaluated = true;
    record.denominator =
        2.0 * record.derivative * record.derivative - record.fx * record.second_derivative;
    if (!std::isfinite(record.denominator))
      throw std::overflow_error("Halley denominator nonfinite");
    if (record.denominator == 0) {
      run.stop = HalleyStop::ZeroDenominator;
      if (steps)
        steps->back() = record;
      return run;
    }
    const double correction = 2.0 * record.fx * record.derivative / record.denominator;
    record.next             = x - correction;
    if (!std::isfinite(correction) || !std::isfinite(record.next))
      throw std::overflow_error("Halley update nonfinite");
    record.updated = true;
    x              = record.next;
    ++run.work.updates;
    if (steps)
      steps->back() = record;
  }
  return run;
}
} // namespace halley_detail
RootResult halley(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    const std::function<double(double)>& d2f,
    double                               x0,
    double                               tol,
    int                                  max_iter) {
  return halley_detail::execute(f, df, d2f, x0, tol, max_iter, nullptr, 0).result;
}
HalleyRun halley_run(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    const std::function<double(double)>& d2f,
    double                               x0,
    double                               tol,
    int                                  max_iter) {
  return halley_detail::execute(f, df, d2f, x0, tol, max_iter, nullptr, 0);
}
HalleyTrace halley_trace(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    const std::function<double(double)>& d2f,
    double                               x0,
    double                               tol,
    int                                  max_iter,
    std::size_t                          max_entries) {
  HalleyTrace t;
  t.run = halley_detail::execute(f, df, d2f, x0, tol, max_iter, &t.steps, max_entries);
  return t;
}
} // namespace datamunge::algorithms
