/// @file goldschmidt_division.cpp
/// @brief Shared scalar/work/online-history Goldschmidt execution in preserved arithmetic order.
#include <cmath>
#include <datamunge/algorithms/goldschmidt_division.hpp>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private execution shared by scalar and recorded interfaces.
namespace goldschmidt_detail {
/// @brief Execute normalization, original ordered products and final sign handling once.
/// @param numerator Finite dividend, checked before arithmetic.
/// @param denominator Finite nonzero divisor, checked before arithmetic.
/// @param max_updates Nonnegative update budget.
/// @param steps Optional borrowed owned-history sink; null means no capture.
/// @param max_records Optional whole-update quota; zero disables it.
/// @return Actual output, normalization state, completed work and stop.
/// @throws std::invalid_argument For invalid inputs/budget.
/// @throws std::length_error Before an inadmissible next record's arithmetic.
/// @throws std::bad_alloc If admitted storage allocation fails.
GoldschmidtRun execute(
    double                        numerator,
    double                        denominator,
    int                           max_updates,
    std::vector<GoldschmidtStep>* steps,
    std::size_t                   max_records) {
  if (!std::isfinite(numerator) || !std::isfinite(denominator) || denominator == 0 ||
      max_updates < 0)
    throw std::invalid_argument(
        "goldschmidt_division: finite operands, nonzero divisor and nonnegative budget required");
  const double   s = denominator < 0.0 ? -1.0 : 1.0;
  const double   d = std::fabs(denominator);
  int            e = 0;
  const double   m = std::frexp(d, &e);
  double         N = std::ldexp(numerator, -e);
  double         D = m;
  GoldschmidtRun run;
  run.sign     = s;
  run.exponent = e;
  run.initial  = {N, D};
  for (int i = 0; i < max_updates; ++i) {
    if (steps &&
        ((max_records && steps->size() >= max_records) || steps->size() >= steps->max_size()))
      throw std::length_error("goldschmidt_trace: complete update-record quota");
    const GoldschmidtState before{N, D};
    const double           f = 2.0 - D;
    N *= f;
    D *= f;
    ++run.updates;
    run.loop_multiplications += 2;
    if (steps)
      steps->push_back({run.updates, before, f, {N, D}});
    if (std::fabs(D - 1.0) < 1e-16)
      break;
  }
  run.quotient    = s * N;
  run.final_state = {N, D};
  run.stop =
      std::fabs(D - 1.0) < 1e-16 ? GoldschmidtStop::tolerance : GoldschmidtStop::iteration_limit;
  return run;
}
} // namespace goldschmidt_detail

double goldschmidt_division(double numerator, double denominator) {
  return goldschmidt_detail::execute(numerator, denominator, 60, nullptr, 0).quotient;
}
GoldschmidtRun goldschmidt_run(double numerator, double denominator, int max_updates) {
  return goldschmidt_detail::execute(numerator, denominator, max_updates, nullptr, 0);
}
GoldschmidtTrace
goldschmidt_trace(double numerator, double denominator, int max_updates, std::size_t max_records) {
  GoldschmidtTrace trace;
  trace.run =
      goldschmidt_detail::execute(numerator, denominator, max_updates, &trace.steps, max_records);
  return trace;
}
} // namespace datamunge::algorithms
