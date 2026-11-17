/// @file lll.cpp
/// @brief Bounded shared floating-decision LLL with checked exact integer row transforms and actual GS/operation evidence.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/lll.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private safe geometry, actual floating GS and checked exact row-operation producer.
namespace lll_detail {
/// @brief Actual64-bit integer row basis/transform storage type.
using Basis = std::vector<std::vector<long long>>;
/// @brief Signed widened exact product/subtraction for two64-bit row coordinates.
using Wide = __int128;
/// @brief Actual floating row/Gram-Schmidt matrix type.
using Matrix = std::vector<std::vector<double>>;
/// @brief Require the original API's signed long long coordinates to be exactly64-bit.
static_assert(std::numeric_limits<long long>::digits == 63);
/// @brief Add selected work/count without size_t overflow.
/// @param count Mutable count.
/// @param amount Actual increment, default1.
/// @throws std::length_error On count overflow.
void add(std::size_t& count, std::size_t amount = 1) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("LLL count geometry");
  count += amount;
}
/// @brief Multiply full logical dimensions safely.
/// @param a First count.
/// @param b Second count.
/// @return Exact representable product.
/// @throws std::length_error On size_t overflow.
std::size_t product(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("LLL complete matrix geometry");
  return a * b;
}
/// @brief Require a finite actual produced floating scalar.
/// @param x Actual arithmetic result.
/// @return Unchanged finite double.
/// @throws std::overflow_error On nonfinite produced floating arithmetic.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("LLL floating arithmetic range");
  return x;
}
/// @brief Validate all rows, dimension, delta and bounded counter/container geometry before GS.
/// @param basis Complete original integer rows.
/// @param delta Actual reduction parameter.
/// @param max_steps Explicit actual outer-step budget.
/// @return Original coordinate dimension,0 for empty basis.
/// @throws std::invalid_argument On complete ragged/dimension/parameter violations.
/// @throws std::length_error On full matrix/work/step geometry.
std::size_t validate(const Basis& basis, double delta, std::size_t max_steps) {
  if (!std::isfinite(delta) || delta <= .25 || delta >= 1)
    throw std::invalid_argument("LLL delta strictly in(.25,1)");
  if (max_steps == std::numeric_limits<std::size_t>::max())
    throw std::length_error("LLL outer-step geometry");
  if (basis.empty())
    return 0;
  const auto n = basis.size(), d = basis.front().size();
  if (!d || n > d)
    throw std::invalid_argument("LLL nonempty independent-row shape required");
  for (auto& r : basis)
    if (r.size() != d)
      throw std::invalid_argument("LLL complete rectangular basis");
  if (product(n, d) > std::vector<double>{}.max_size() ||
      product(n, n) > std::vector<long long>{}.max_size())
    throw std::length_error("LLL complete matrix container geometry");
  return d;
}
/// @brief Count and compute one actual finite ordered floating dot product.
/// @param a Actual matched first row.
/// @param b Actual matched second row.
/// @param work Mutable actual GS coordinate-product count.
/// @return Actual finite double sum.
/// @throws std::overflow_error On nonfinite product/sum.
/// @throws std::length_error On actual count overflow.
double dot(const std::vector<double>& a, const std::vector<double>& b, LLLWork& work) {
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum = finite(sum + finite(a[i] * b[i]));
    add(work.dot_terms);
  }
  return sum;
}
/// @brief One actual full floating Gram-Schmidt state.
struct GS {
  Matrix              orthogonal; ///< Actual sequentially computed orthogonal rows.
  Matrix              mu;         ///< Actual projection coefficients.
  std::vector<double> norms;      ///< Actual positive computed squared orthogonal row norms.
};
/// @brief Evaluate actual full original-order floating Gram-Schmidt with cached identical norm sums.
/// @param basis Actual current integer rows.
/// @param work Mutable actual GS run/product/update counts.
/// @return Owned actual finite computed orthogonal/projection/norm arrays.
/// @throws std::domain_error On zero computed orthogonal norm, not an exact rank certificate.
/// @throws std::overflow_error On nonfinite produced arithmetic.
/// @throws std::bad_alloc On actual floating matrix allocations.
GS gram_schmidt(const Basis& basis, LLLWork& work) {
  add(work.gram_schmidt_runs);
  const auto n = basis.size(), d = n ? basis.front().size() : 0;
  Matrix     original(n, std::vector<double>(d));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < d; ++j)
      original[i][j] = static_cast<double>(basis[i][j]);
  GS out;
  out.orthogonal = original;
  out.mu.assign(n, std::vector<double>(n));
  out.norms.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      out.mu[i][j] = finite(dot(original[i], out.orthogonal[j], work) / out.norms[j]);
      for (std::size_t k = 0; k < d; ++k) {
        out.orthogonal[i][k] =
            finite(out.orthogonal[i][k] - finite(out.mu[i][j] * out.orthogonal[j][k]));
        add(work.orthogonal_updates);
      }
    }
    out.norms[i] = dot(out.orthogonal[i], out.orthogonal[i], work);
    if (out.norms[i] == 0)
      throw std::domain_error("LLL zero computed GS norm; possible dependence/numerical breakdown");
  }
  return out;
}
/// @brief Round an actual finite projection to an exactly representable signed64 multiplier.
/// @param mu Actual computed projection coefficient.
/// @return Round-half-away-from-zero integer.
/// @throws std::overflow_error On nonfinite/out-of-int64 rounded multiplier.
long long multiplier(double mu) {
  const double rounded = std::round(finite(mu));
  if (rounded < -0x1p63 || rounded >= 0x1p63)
    throw std::overflow_error("LLL rounded integer multiplier range");
  return static_cast<long long>(rounded);
}
/// @brief Prepare a whole checked exact integer row subtraction before committing any coordinate.
/// @param target Actual original target row.
/// @param source Actual original source row.
/// @param q Actual signed64 integer multiplier.
/// @return Owned exact target-q*source row if all coordinates fit signed64.
/// @throws std::overflow_error On any unrepresentable exact coordinate.
/// @throws std::bad_alloc On temporary row allocation failure.
std::vector<long long>
subtract(const std::vector<long long>& target, const std::vector<long long>& source, long long q) {
  std::vector<long long> out;
  out.reserve(target.size());
  for (std::size_t i = 0; i < target.size(); ++i) {
    const Wide value = static_cast<Wide>(target[i]) - static_cast<Wide>(q) * source[i];
    if (value < std::numeric_limits<long long>::min() ||
        value > std::numeric_limits<long long>::max())
      throw std::overflow_error("LLL exact basis/transform coordinate range");
    out.push_back(static_cast<long long>(value));
  }
  return out;
}
/// @brief Admit actual complete logical capture records before copying/appending.
/// @param trace Optional mutable complete capture.
/// @param amount Known actual next record geometry.
/// @param cap Actual optional cap,0 off.
/// @throws std::length_error On count/quota overflow.
void admit(LLLTrace* trace, std::size_t amount, std::size_t cap) {
  if (!trace)
    return;
  if (cap && (trace->captured_entries > cap || amount > cap - trace->captured_entries))
    throw std::length_error("LLL actual capture quota");
  add(trace->captured_entries, amount);
}
/// @brief Capture one actual computed GS/basis/transform state, not a fresh reduction replay.
/// @param trace Optional complete capture.
/// @param basis Actual current integer basis.
/// @param transform Actual current checked unimodular transform.
/// @param gs Actual just-computed GS arrays.
/// @param cap Actual logical cap.
void state(
    LLLTrace*    trace,
    const Basis& basis,
    const Basis& transform,
    const GS&    gs,
    std::size_t  cap) {
  if (!trace)
    return;
  const auto  n = basis.size(), d = n ? basis.front().size() : 0;
  std::size_t records = 1;
  add(records, product(2, product(n, d)));
  add(records, product(2, product(n, n)));
  add(records, n);
  admit(trace, records, cap);
  trace->states.push_back({basis, transform, gs.orthogonal, gs.mu, gs.norms});
}
/// @brief Capture an actual producing event linked to the current owned real state.
/// @param trace Optional complete capture.
/// @param event Actual complete operation/check fields.
/// @param cap Actual logical cap.
void event(LLLTrace* trace, LLLEvent event, std::size_t cap) {
  if (!trace)
    return;
  admit(trace, 1, cap);
  event.state = trace->states.size() - 1;
  trace->events.push_back(event);
}
/// @brief Observe actual integer row squared norms with widened accumulation, separately counted.
/// @param basis Actual integer rows.
/// @param work Mutable additional row-norm coordinate count.
/// @return Owned widened finite norm observations, not exact integer determinants.
std::vector<long double> row_norms(const Basis& basis, LLLWork& work) {
  std::vector<long double> out;
  out.reserve(basis.size());
  for (auto& r : basis) {
    long double sum = 0;
    for (long long x : r) {
      sum += static_cast<long double>(x) * x;
      add(work.row_norm_terms);
    }
    out.push_back(sum);
  }
  return out;
}
/// @brief Evaluate actual finite Lovasz right expression in the historical operation order.
/// @param gs Actual current computed Gram-Schmidt state.
/// @param k Actual adjacent target row>=1.
/// @param delta Actual finite reduction parameter.
/// @return Actual finite double right-hand expression.
/// @throws std::overflow_error On nonfinite expression arithmetic.
double lovasz_right(const GS& gs, std::size_t k, double delta) {
  return finite(finite(delta - finite(gs.mu[k][k - 1] * gs.mu[k][k - 1])) * gs.norms[k - 1]);
}
/// @brief Produce bounded LLL decisions and checked exact row transformations once with optional actual evidence.
/// @param original Validated complete original integer rows.
/// @param delta Validated finite reduction parameter.
/// @param max_steps Actual finite outer budget.
/// @param trace Optional original-input-admitted complete capture.
/// @param cap Actual logical cap.
/// @return Owned actual complete/partial equivalent basis/transform and floating observations/work.
LLLAnalysis
run(const Basis& original, double delta, std::size_t max_steps, LLLTrace* trace, std::size_t cap) {
  LLLAnalysis out;
  out.basis    = original;
  const auto n = original.size(), d = n ? original.front().size() : 0;
  out.transform.assign(n, std::vector<long long>(n));
  for (std::size_t i = 0; i < n; ++i)
    out.transform[i][i] = 1;
  out.initial_squared_norms = row_norms(original, out.work);
  if (!n) {
    out.final_squared_norms = out.initial_squared_norms;
    return out;
  }
  auto gs = gram_schmidt(out.basis, out.work);
  state(trace, out.basis, out.transform, gs, cap);
  std::size_t k = 1;
  while (k < n && out.work.outer_steps < max_steps) {
    for (std::size_t j = k; j-- > 0;) {
      const double projection = gs.mu[k][j];
      if (std::abs(projection) > .5) {
        const auto q     = multiplier(projection);
        auto       row   = subtract(out.basis[k], out.basis[j], q),
             transform   = subtract(out.transform[k], out.transform[j], q);
        out.basis[k]     = std::move(row);
        out.transform[k] = std::move(transform);
        add(out.work.size_reductions);
        add(out.work.basis_updates, d);
        add(out.work.transform_updates, n);
        gs = gram_schmidt(out.basis, out.work);
        state(trace, out.basis, out.transform, gs, cap);
        event(trace, {LLLEventKind::SizeReduce, k, j, q, projection}, cap);
      }
      if (j == 0)
        break;
    }
    const double lhs = gs.norms[k], rhs = lovasz_right(gs, k, delta);
    const bool   passed = lhs >= rhs;
    add(out.work.lovasz_tests);
    event(trace, {LLLEventKind::Lovasz, k, k - 1, 0, gs.mu[k][k - 1], lhs, rhs, passed}, cap);
    if (passed)
      ++k;
    else {
      std::swap(out.basis[k], out.basis[k - 1]);
      std::swap(out.transform[k], out.transform[k - 1]);
      add(out.work.swaps);
      gs = gram_schmidt(out.basis, out.work);
      state(trace, out.basis, out.transform, gs, cap);
      event(trace, {LLLEventKind::Swap, k, k - 1}, cap);
      k = k > 1 ? k - 1 : 1;
    }
    add(out.work.outer_steps);
  }
  out.stop                     = k < n ? LLLStop::StepLimit : LLLStop::Completed;
  out.mu                       = gs.mu;
  out.orthogonal_squared_norms = gs.norms;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j)
      if (std::abs(gs.mu[i][j]) > .5)
        out.observed_size_reduced = false;
    if (i && gs.norms[i] < lovasz_right(gs, i, delta))
      out.observed_lovasz = false;
  }
  out.final_squared_norms = row_norms(out.basis, out.work);
  return out;
}
} // namespace lll_detail
std::vector<std::vector<long long>> lll_reduce(
    std::vector<std::vector<long long>> basis,
    double                              delta) {
  auto result = lll_analyze(basis, delta);
  if (result.stop == LLLStop::StepLimit)
    throw std::length_error("LLL default outer-step budget exhausted");
  return std::move(result.basis);
}
LLLAnalysis
lll_analyze(const std::vector<std::vector<long long>>& basis, double delta, std::size_t max_steps) {
  lll_detail::validate(basis, delta, max_steps);
  return lll_detail::run(basis, delta, max_steps, nullptr, 0);
}
LLLTrace lll_trace(
    const std::vector<std::vector<long long>>& basis,
    double                                     delta,
    std::size_t                                max_steps,
    std::size_t                                max_entries) {
  const auto d = lll_detail::validate(basis, delta, max_steps);
  LLLTrace   out;
  out.delta = delta;
  lll_detail::admit(&out, lll_detail::product(basis.size(), d), max_entries);
  out.original = basis;
  out.analysis = lll_detail::run(basis, delta, max_steps, &out, max_entries);
  return out;
}
} // namespace datamunge::algorithms
