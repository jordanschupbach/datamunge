/// @file gram_schmidt.cpp
/// @brief Shared validated modified projection loop with optional actual floating-state capture.
#include <cmath>
#include <datamunge/algorithms/gram_schmidt.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Checked work, validation and the shared producing Gram–Schmidt loop.
namespace gram_schmidt_detail {
/// @brief Nonwrapping logical count addition.
/// @param a Current count.
/// @param b Increment.
/// @return Exact size_t sum.
/// @throws std::length_error On overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Gram-Schmidt count overflow");
  return a + b;
}
/// @brief Nonwrapping logical count multiplication.
/// @param a First count.
/// @param b Second count.
/// @return Exact size_t product.
/// @throws std::length_error On overflow.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("Gram-Schmidt count product overflow");
  return a * b;
}
/// @brief Validate all rows/values and options before cap checks or internal copies.
/// @param v Borrowed rectangular finite rows.
/// @param o Arithmetic options and complete input quota.
/// @return Common width, zero for empty input.
/// @throws std::invalid_argument For malformed shape/values/options.
/// @throws std::length_error On complete input size overflow or quota violation.
std::size_t validate(const std::vector<std::vector<double>>& v, const GramSchmidtOptions& o) {
  const auto d = v.empty() ? 0 : v[0].size();
  for (const auto& row : v) {
    if (row.size() != d)
      throw std::invalid_argument("Gram-Schmidt needs equal row widths");
    for (double x : row)
      if (!std::isfinite(x))
        throw std::invalid_argument("Gram-Schmidt needs finite inputs");
  }
  if (!std::isfinite(o.tolerance) || o.tolerance < 0 || (o.passes != 1 && o.passes != 2))
    throw std::invalid_argument("Gram-Schmidt tolerance/passes invalid");
  const auto n = add(v.size(), multiply(v.size(), d));
  if (o.max_input_entries && n > o.max_input_entries)
    throw std::length_error("Gram-Schmidt complete input quota");
  return d;
}
/// @brief Admit actual retained floating-value units before a producing dot or copies.
/// @param t Optional borrowed capture under construction; null disables capture.
/// @param count Complete next projection/row floating-value count.
/// @param cap Optional policy limit, zero unlimited.
/// @throws std::length_error If checked cumulative units exceed cap or overflow.
void admit(GramSchmidtTrace* t, std::size_t count, std::size_t cap) {
  if (!t)
    return;
  auto n = add(t->captured_entries, count);
  if (cap && n > cap)
    throw std::length_error("Gram-Schmidt floating-state quota");
  t->captured_entries = n;
}
/// @brief Compute the original sequential double dot and count its actual terms.
/// @param a Borrowed first row.
/// @param b Borrowed equally sized second row.
/// @param w Producing selected work, incremented before arithmetic.
/// @return Actual rounded sequential dot.
/// @throws std::overflow_error If the completed dot is nonfinite.
/// @throws std::length_error On checked counter overflow.
double dot(const std::vector<double>& a, const std::vector<double>& b, GramSchmidtWork& w) {
  w.dot_products = add(w.dot_products, 1);
  w.dot_terms    = add(w.dot_terms, a.size());
  double s       = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    s += a[i] * b[i];
  if (!std::isfinite(s))
    throw std::overflow_error("Gram-Schmidt dot became nonfinite");
  return s;
}
/// @brief Execute the original ordered projections with optional second pass and actual capture.
/// @param v Borrowed original rows, never mutated or retained.
/// @param o Validated arithmetic/capture policies, checked by this kernel.
/// @param trace Optional borrowed capture sink, initially empty.
/// @return Actual accepted basis, source rows and selected work.
/// @throws std::invalid_argument On invalid rows/options before caps/copies.
/// @throws std::length_error On input/capture/counter/container limits.
/// @throws std::overflow_error On nonfinite intermediate arithmetic.
/// @throws std::bad_alloc On allocation failure.
GramSchmidtRun
execute(const std::vector<std::vector<double>>& v, GramSchmidtOptions o, GramSchmidtTrace* trace) {
  const auto     d = validate(v, o);
  GramSchmidtRun run;
  auto&          w = run.work;
  w.rows           = v.size();
  w.dimension      = d;
  for (std::size_t j = 0; j < v.size(); ++j) {
    auto u = v[j];
    for (unsigned pass = 0; pass < o.passes; ++pass)
      for (std::size_t k = 0; k < run.basis.size(); ++k) {
        if (trace)
          admit(trace, add(1, multiply(2, d)), o.max_trace_entries);
        GramSchmidtProjection p;
        if (trace) {
          p.input_row = j;
          p.basis_row = k;
          p.pass      = pass;
          p.before    = u;
        }
        const double coefficient = dot(u, run.basis[k], w);
        w.projections            = add(w.projections, 1);
        w.update_terms           = add(w.update_terms, d);
        for (std::size_t i = 0; i < d; ++i) {
          u[i] -= coefficient * run.basis[k][i];
          if (!std::isfinite(u[i]))
            throw std::overflow_error("Gram-Schmidt update became nonfinite");
        }
        if (trace) {
          p.coefficient = coefficient;
          p.after       = u;
          trace->projections.push_back(std::move(p));
        }
      }
    if (trace)
      admit(trace, add(1, d), o.max_trace_entries);
    const double norm     = std::sqrt(dot(u, u, w));
    const bool   accepted = norm > o.tolerance;
    if (trace)
      trace->rows.push_back({j, u, norm, accepted, add(run.basis.size(), accepted ? 1 : 0)});
    if (accepted) {
      w.divisions = add(w.divisions, d);
      for (double& x : u)
        x /= norm;
      run.basis.push_back(u);
      run.input_rows.push_back(j);
    }
  }
  return run;
}
} // namespace gram_schmidt_detail
std::vector<std::vector<double>> gram_schmidt(const std::vector<std::vector<double>>& vectors) {
  GramSchmidtOptions o;
  o.max_input_entries = 0;
  o.max_trace_entries = 0;
  return gram_schmidt_detail::execute(vectors, o, nullptr).basis;
}
GramSchmidtRun gram_schmidt_run(
    const std::vector<std::vector<double>>& vectors,
    GramSchmidtOptions                      options) {
  return gram_schmidt_detail::execute(vectors, options, nullptr);
}
GramSchmidtTrace gram_schmidt_trace(
    const std::vector<std::vector<double>>& vectors,
    GramSchmidtOptions                      options) {
  GramSchmidtTrace t;
  t.run = gram_schmidt_detail::execute(vectors, options, &t);
  return t;
}
} // namespace datamunge::algorithms
