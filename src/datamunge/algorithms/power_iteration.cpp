/// @file power_iteration.cpp
/// @brief Single checked power producer and actual iterate-boundary capture.
#include <datamunge/algorithms/power_iteration.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Local checked producer and optional actual-history implementation.
namespace power_detail {
/// @brief Dense row-major finite matrix type.
using Matrix = std::vector<std::vector<double>>;
/// @brief Owned real coordinate vector.
using Vector = std::vector<double>;
/// @brief Historical local SplitMix64 initialization word.
/// @param s Mutable local unsigned seed state, wrapping modulo2^64.
/// @return Actual mixed word.
std::uint64_t splitmix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
/// @brief Check a producing floating-point scalar.
/// @param v Actual computed value.
/// @return Unchanged finite value.
/// @throws std::overflow_error If v is nonfinite.
double finite(double v) {
  if (!std::isfinite(v))
    throw std::overflow_error("power iteration nonfinite arithmetic");
  return v;
}
/// @brief Historical left-to-right double dot product.
/// @param a Matching finite first vector.
/// @param b Matching finite second vector.
/// @return Actual finite dot product.
/// @throws std::overflow_error If a producing product, dot, norm or selected counter is not representable.
double dot(const Vector& a, const Vector& b) {
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    sum = finite(sum + a[i] * b[i]);
  return sum;
}
/// @brief Execute one actual dense product and count its scalar terms.
/// @param matrix Validated square matrix.
/// @param v Matching finite vector.
/// @param r Mutable selected work.
/// @return Owned actual product.
/// @throws std::overflow_error If a producing product, dot, norm or selected counter is not representable.
Vector matvec(const Matrix& matrix, const Vector& v, PowerIterationRun& r) {
  if (r.matvec_calls == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("power product count");
  ++r.matvec_calls;
  Vector out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    for (std::size_t j = 0; j < v.size(); ++j) {
      if (r.scalar_products == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("power scalar count");
      ++r.scalar_products;
      out[i] = finite(out[i] + matrix[i][j] * v[j]);
    }
  return out;
}
/// @brief Admit actual retained logical entries before vector copies.
/// @param used Mutable admitted entry count.
/// @param amount Next actual entry amount.
/// @param cap Positive policy or zero for no policy.
/// @throws std::length_error If the requested-iteration or actual-history policy is exceeded.
void retain(std::size_t& used, std::size_t amount, std::size_t cap) {
  if (amount > std::numeric_limits<std::size_t>::max() - used ||
      (cap && (used > cap || amount > cap - used)))
    throw std::length_error("power history cap");
  used += amount;
}
/// @brief Validate and execute the single seeded or explicit-vector producer.
/// @param matrix Borrowed complete matrix.
/// @param start Borrowed explicit vector or null for seeded initialization.
/// @param iterations Requested updates.
/// @param seed Local SplitMix seed.
/// @param options Policies.
/// @param capture Whether actual histories are kept.
/// @return Owned actual output with optional histories.
/// @throws std::invalid_argument If the matrix is ragged/nonsquare/nonfinite, iterations is negative, or an explicit start is malformed/nonfinite/zero.
/// @throws std::length_error If the requested-iteration or actual-history policy is exceeded.
/// @throws std::overflow_error If a producing product, dot, norm or selected counter is not representable.
PowerIterationTrace produce(
    const Matrix&         matrix,
    const Vector*         start,
    int                   iterations,
    std::uint64_t         seed,
    PowerIterationOptions options,
    bool                  capture) {
  const auto n = matrix.size();
  if (iterations < 0)
    throw std::invalid_argument("power negative iterations");
  for (const auto& row : matrix) {
    if (row.size() != n)
      throw std::invalid_argument("power nonsquare");
    for (auto v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("power nonfinite input");
  }
  if (start) {
    if (start->size() != n)
      throw std::invalid_argument("power initial shape");
    for (auto v : *start)
      if (!std::isfinite(v))
        throw std::invalid_argument("power initial finite");
  }
  if (options.max_iterations && std::size_t(iterations) > options.max_iterations)
    throw std::length_error("power iteration cap");
  PowerIterationTrace trace;
  auto&               r = trace.run;
  if (!n)
    return trace;
  Vector v(n);
  if (start)
    v = *start;
  else
    for (auto& x : v)
      x = double(splitmix(seed) % 2000) / 1000. - 1.;
  double norm = std::sqrt(dot(v, v));
  if (norm < 1e-18) {
    if (start)
      throw std::invalid_argument("power zero initial vector");
    v.assign(n, 0);
    v[0] = 1;
    norm = 1;
  }
  for (auto& x : v)
    x /= norm;
  std::size_t entries = 0;
  if (capture) {
    retain(entries, n, options.max_trace_entries);
    trace.initial = v;
  }
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (capture) {
      if (n > (std::numeric_limits<std::size_t>::max() - 1) / 3)
        throw std::length_error("power history size");
      retain(entries, 3 * n + 1, options.max_trace_entries);
    }
    Vector product = matvec(matrix, v, r);
    ++r.attempts;
    double norm = std::sqrt(dot(product, product)), ray = dot(v, product), residual = 0;
    for (std::size_t i = 0; i < n; ++i)
      residual = finite(std::hypot(residual, finite(product[i] - ray * v[i])));
    bool   accepted = norm >= 1e-18, flip = false;
    Vector after = accepted ? product : v;
    if (accepted) {
      for (auto& x : after)
        x /= norm;
      flip = dot(after, v) < 0;
      if (flip)
        for (auto& x : after)
          x = -x;
      ++r.accepted;
    } else
      r.tiny_product = true;
    if (capture)
      trace.steps.push_back(
          {std::size_t(iteration), v, product, after, norm, ray, residual, flip, accepted});
    v = std::move(after);
    if (!accepted)
      break;
  }
  r.pair.vector      = v;
  auto final_product = matvec(matrix, v, r);
  r.pair.value       = dot(v, final_product);
  for (std::size_t i = 0; i < n; ++i)
    r.residual_norm =
        finite(std::hypot(r.residual_norm, finite(final_product[i] - r.pair.value * v[i])));
  return trace;
}
} // namespace power_detail
EigenPair power_iteration(
    const std::vector<std::vector<double>>& matrix,
    int                                     iterations,
    std::uint64_t                           seed) {
  return power_detail::produce(matrix, nullptr, iterations, seed, {}, false).run.pair;
}
PowerIterationRun power_iteration_run(
    const std::vector<std::vector<double>>& matrix,
    int                                     iterations,
    std::uint64_t                           seed,
    PowerIterationOptions                   options) {
  return power_detail::produce(matrix, nullptr, iterations, seed, options, false).run;
}
PowerIterationRun power_iteration_from(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>&              start,
    int                                     iterations,
    PowerIterationOptions                   options) {
  return power_detail::produce(matrix, &start, iterations, 0, options, false).run;
}
PowerIterationTrace power_iteration_trace(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>&              start,
    int                                     iterations,
    PowerIterationOptions                   options) {
  return power_detail::produce(matrix, &start, iterations, 0, options, true);
}
} // namespace datamunge::algorithms
