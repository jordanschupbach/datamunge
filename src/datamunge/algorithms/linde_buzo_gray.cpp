/// @file linde_buzo_gray.cpp
/// @brief Shared checked multiplicative LBG producer with actual Lloyd records and final nearest-codebook consistency.
#include <cmath>
#include <datamunge/algorithms/linde_buzo_gray.hpp>
#include <limits>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete geometry, producing means/distances and capture admission.
namespace lbg_detail {
/// @brief Check an addition used in selected work or capture geometry.
/// @param value Mutable count.
/// @param amount Actual additional count, default1.
/// @throws std::length_error On size_t wrap.
void add(std::size_t& value, std::size_t amount = 1) {
  if (amount > std::numeric_limits<std::size_t>::max() - value)
    throw std::length_error("LBG count overflow");
  value += amount;
}
/// @brief Multiply a complete logical geometry without wrap.
/// @param a First dimension/count.
/// @param b Second dimension/count.
/// @return Exact representable product.
/// @throws std::length_error On size_t overflow.
std::size_t product(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("LBG complete geometry");
  return a * b;
}
/// @brief Validate every row/coordinate of a complete finite rectangular point or codeword matrix.
/// @param data Borrowed complete original matrix.
/// @return Positive column dimension when nonempty,0 when empty.
/// @throws std::invalid_argument On zero dimension, ragged rows or nonfinite coordinates.
/// @throws std::length_error On complete coordinate count.
std::size_t shape(const std::vector<std::vector<double>>& data) {
  if (data.empty())
    return 0;
  const auto d = data.front().size();
  if (!d)
    throw std::invalid_argument("LBG positive point dimension");
  product(data.size(), d);
  for (auto& x : data) {
    if (x.size() != d)
      throw std::invalid_argument("LBG complete rectangular data");
    for (double v : x)
      if (!std::isfinite(v))
        throw std::invalid_argument("LBG finite data");
  }
  return d;
}
/// @brief Require an actual finite produced double without changing ordinary arithmetic order.
/// @param x Actual producing result.
/// @return Unchanged finite double.
/// @throws std::overflow_error On nonfinite actual arithmetic.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("LBG produced arithmetic range");
  return x;
}
/// @brief Convert a widened actual aggregate into finite double range.
/// @param x Actual widened mean/distortion aggregate.
/// @return Rounded finite double.
/// @throws std::overflow_error On nonfinite/out-of-double-range aggregate.
double narrow(long double x) {
  if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
    throw std::overflow_error("LBG widened aggregate range");
  return static_cast<double>(x);
}
/// @brief Admit a complete actual record before copying any of its captured fields.
/// @param trace Optional mutable complete capture.
/// @param amount Known actual next record geometry.
/// @param cap Actual logical cap,0 off.
/// @throws std::length_error On count or quota limits.
void admit(LBGTrace* trace, std::size_t amount, std::size_t cap) {
  if (!trace)
    return;
  if (cap && (trace->captured_entries > cap || amount > cap - trace->captured_entries))
    throw std::length_error("LBG actual capture quota");
  add(trace->captured_entries, amount);
}
/// @brief Evaluate one actual squared Euclidean distance in historical finite double order.
/// @param x Complete finite point.
/// @param c Complete matched finite codeword.
/// @param work Mutable actual distance/coordinate counts.
/// @return Actual finite nonnegative squared distance.
/// @throws std::overflow_error On nonfinite difference/product/sum.
/// @throws std::length_error On actual selected count overflow.
double distance(const std::vector<double>& x, const std::vector<double>& c, LBGWork& work) {
  double sum = 0;
  for (std::size_t j = 0; j < x.size(); ++j) {
    const double d = finite(x[j] - c[j]);
    sum            = finite(sum + finite(d * d));
    add(work.coordinate_squares);
  }
  add(work.distance_evaluations);
  return sum;
}
/// @brief Perform one actual nearest-codeword pass with earliest-index ties and consistent observed objective.
/// @param data Validated complete finite points.
/// @param codebook Validated/generated matched finite nonempty codebook.
/// @param work Mutable actual assignment/distance counts.
/// @return Actual owned assignments and mean squared distance; returned work field unused privately.
/// @throws std::overflow_error On produced distance/objective range.
/// @throws std::length_error On actual selected count overflow.
/// @throws std::bad_alloc On assignment allocation failure.
LBGQuantization assign(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::vector<double>>& codebook,
    LBGWork&                                work) {
  LBGQuantization out;
  out.assignment.resize(data.size());
  double      sum  = 0;
  long double wide = 0;
  add(work.assignment_passes);
  for (std::size_t i = 0; i < data.size(); ++i) {
    double      best  = std::numeric_limits<double>::infinity();
    std::size_t index = 0;
    for (std::size_t k = 0; k < codebook.size(); ++k) {
      const auto d = distance(data[i], codebook[k], work);
      if (d < best) {
        best  = d;
        index = k;
      }
    }
    out.assignment[i] = index;
    sum += best;
    wide += best;
    add(work.point_assignments);
  }
  if (!data.empty())
    out.distortion =
        std::isfinite(sum) ? sum / static_cast<double>(data.size()) : narrow(wide / data.size());
  return out;
}
/// @brief Actual assigned mean update and cluster counts from the shared producing coordinate loop.
struct Means {
  std::vector<std::vector<double>>
                           codebook; ///< Actual new finite means, retaining empty old codewords.
  std::vector<std::size_t> counts;   ///< Actual original point counts per codeword.
};
/// @brief Produce assigned means in ordinary double order, with parallel widened sum fallback.
/// @param data Validated complete original points.
/// @param assignment Actual current point-to-codeword indices.
/// @param old Actual old codebook; retained for empty clusters.
/// @param work Mutable actual coordinate-read/update counts.
/// @return Actual new codebook and observed assigned counts.
/// @throws std::overflow_error On unrepresentable widened means.
/// @throws std::length_error On actual selected count overflow.
/// @throws std::bad_alloc On runtime aggregate allocation failure.
Means means(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::size_t>&         assignment,
    const std::vector<std::vector<double>>& old,
    LBGWork&                                work) {
  Means out;
  out.codebook = old;
  out.counts.resize(old.size());
  const auto                            d = old.front().size();
  std::vector<std::vector<double>>      sum(old.size(), std::vector<double>(d));
  std::vector<std::vector<long double>> wide(old.size(), std::vector<long double>(d));
  for (std::size_t i = 0; i < data.size(); ++i) {
    const auto k = assignment[i];
    ++out.counts[k];
    for (std::size_t j = 0; j < d; ++j) {
      sum[k][j] += data[i][j];
      wide[k][j] += data[i][j];
      add(work.centroid_reads);
    }
  }
  for (std::size_t k = 0; k < old.size(); ++k)
    if (out.counts[k])
      for (std::size_t j = 0; j < d; ++j) {
        out.codebook[k][j] = std::isfinite(sum[k][j])
                                 ? sum[k][j] / static_cast<double>(out.counts[k])
                                 : narrow(wide[k][j] / out.counts[k]);
        add(work.centroid_updates);
      }
  return out;
}
/// @brief Validate full data/numeric settings and representable rounded target/selected-work geometry.
/// @param data Original complete finite point matrix.
/// @param requested Actual requested size.
/// @param epsilon Actual finite nonnegative split fraction.
/// @param iterations Actual positive Lloyd budget per stage.
/// @param tol Actual finite nonnegative relative threshold.
/// @return Actual rounded-up power-of-two target,0 for empty/zero-target no-op.
/// @throws std::invalid_argument On complete data/numeric/iteration violations.
/// @throws std::length_error On rounded target/container/work limits.
std::size_t validate(
    const std::vector<std::vector<double>>& data,
    std::size_t                             requested,
    double                                  epsilon,
    std::size_t                             iterations,
    double                                  tol) {
  const auto d = shape(data);
  if (!std::isfinite(epsilon) || epsilon < 0 || iterations == 0 || !std::isfinite(tol) || tol < 0)
    throw std::invalid_argument("LBG split/iteration/tolerance contract");
  if (data.empty() || requested == 0)
    return 0;
  std::size_t target = 1;
  while (target < requested) {
    if (target > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("LBG rounded target overflow");
    target *= 2;
  }
  if (target > std::vector<std::vector<double>>{}.max_size() ||
      product(target, d) > std::vector<double>{}.max_size())
    throw std::length_error("LBG complete codebook geometry");
  std::size_t passes = iterations;
  add(passes);
  const auto stage_sum = target - 1 + target;
  product(product(product(stage_sum, passes), data.size()), d);
  return target;
}
/// @brief Produce the historical split/stopping path with corrected final-stage nearest consistency and optional real capture.
/// @param data Validated complete original points.
/// @param requested Original requested count.
/// @param target Validated rounded target,0 no-op.
/// @param epsilon Actual finite nonnegative split fraction.
/// @param iterations Actual positive per-stage update budget.
/// @param tol Actual finite nonnegative relative pre-update threshold.
/// @param trace Optional complete owned capture.
/// @param cap Actual record cap,0 off.
/// @return Owned actual final result and selected work.
/// @throws std::overflow_error On produced numerical range failures.
/// @throws std::length_error On selected count or capture limits.
/// @throws std::bad_alloc On runtime/output/capture allocation failure.
LBGAnalysis run(
    const std::vector<std::vector<double>>& data,
    std::size_t                             requested,
    std::size_t                             target,
    double                                  epsilon,
    std::size_t                             iterations,
    double                                  tol,
    LBGTrace*                               trace,
    std::size_t                             cap) {
  LBGAnalysis out;
  out.requested_size = requested;
  out.rounded_size   = target;
  if (!target)
    return out;
  const auto                       n = data.size(), d = data.front().size();
  std::vector<std::vector<double>> codebook(1, std::vector<double>(d));
  codebook                = means(data, std::vector<std::size_t>(n), codebook, out.work).codebook;
  std::size_t stage_index = 0;
  while (true) {
    const auto  k = codebook.size(), coordinates = product(k, d);
    std::size_t stage_count = 1;
    add(stage_count, product(2, coordinates));
    add(stage_count, n);
    admit(trace, stage_count, cap);
    LBGStage stage;
    stage.size = k;
    if (trace)
      stage.initial = codebook;
    double previous      = 0;
    bool   have_previous = false;
    for (std::size_t it = 0; it < iterations; ++it) {
      std::size_t iteration_count = 1;
      add(iteration_count, product(2, coordinates));
      add(iteration_count, n);
      add(iteration_count, k);
      admit(trace, iteration_count, cap);
      LBGIteration record;
      record.stage     = stage_index;
      record.iteration = it;
      if (trace)
        record.before = codebook;
      auto assigned = assign(data, codebook, out.work);
      auto updated  = means(data, assigned.assignment, codebook, out.work);
      codebook      = std::move(updated.codebook);
      add(out.work.lloyd_iterations);
      ++stage.iterations;
      const bool stop =
          have_previous && (previous - assigned.distortion < tol * (previous + 1e-12));
      if (trace) {
        record.assignment        = std::move(assigned.assignment);
        record.before_distortion = assigned.distortion;
        record.counts            = std::move(updated.counts);
        record.after             = codebook;
        record.tolerance_stop    = stop;
        trace->iterations.push_back(std::move(record));
      }
      if (stop) {
        stage.tolerance_stop = true;
        break;
      }
      previous      = assigned.distortion;
      have_previous = true;
    }
    auto final            = assign(data, codebook, out.work);
    out.result.distortion = final.distortion;
    out.result.distortion_history.push_back(final.distortion);
    out.result.assignment = std::move(final.assignment);
    if (trace) {
      stage.final      = codebook;
      stage.assignment = out.result.assignment;
      stage.distortion = out.result.distortion;
      trace->stages.push_back(std::move(stage));
    }
    if (codebook.size() == target)
      break;
    std::vector<std::vector<double>> next;
    next.reserve(2 * k);
    for (auto& c : codebook) {
      auto lo = c, hi = c;
      for (std::size_t j = 0; j < d; ++j) {
        lo[j] = finite(lo[j] * (1. - epsilon));
        hi[j] = finite(hi[j] * (1. + epsilon));
        add(out.work.split_writes, 2);
      }
      next.push_back(std::move(lo));
      next.push_back(std::move(hi));
    }
    codebook = std::move(next);
    ++stage_index;
  }
  out.result.codebook = std::move(codebook);
  return out;
}
} // namespace lbg_detail
LBGResult linde_buzo_gray(
    const std::vector<std::vector<double>>& data,
    std::size_t                             size,
    double                                  epsilon,
    std::size_t                             iterations,
    double                                  tol) {
  return linde_buzo_gray_analyze(data, size, epsilon, iterations, tol).result;
}
LBGAnalysis linde_buzo_gray_analyze(
    const std::vector<std::vector<double>>& data,
    std::size_t                             size,
    double                                  epsilon,
    std::size_t                             iterations,
    double                                  tol) {
  const auto target = lbg_detail::validate(data, size, epsilon, iterations, tol);
  return lbg_detail::run(data, size, target, epsilon, iterations, tol, nullptr, 0);
}
LBGTrace linde_buzo_gray_trace(
    const std::vector<std::vector<double>>& data,
    std::size_t                             size,
    double                                  epsilon,
    std::size_t                             iterations,
    double                                  tol,
    std::size_t                             max_entries) {
  const auto target = lbg_detail::validate(data, size, epsilon, iterations, tol);
  LBGTrace   out;
  const auto coordinates = data.empty() ? 0 : lbg_detail::product(data.size(), data.front().size());
  lbg_detail::admit(&out, coordinates, max_entries);
  out.data     = data;
  out.analysis = lbg_detail::run(data, size, target, epsilon, iterations, tol, &out, max_entries);
  return out;
}
LBGQuantization lbg_quantize(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::vector<double>>& codebook) {
  const auto d = lbg_detail::shape(data), cd = lbg_detail::shape(codebook);
  if (codebook.empty() || (!data.empty() && d != cd))
    throw std::invalid_argument("LBG matched nonempty codebook required");
  lbg_detail::product(lbg_detail::product(data.size(), codebook.size()), cd);
  LBGWork work;
  auto    out = lbg_detail::assign(data, codebook, work);
  out.work    = work;
  return out;
}
std::vector<std::vector<double>> lbg_demo_mixture() {
  std::mt19937_64                  rng(12);
  std::normal_distribution<double> noise(0, .6);
  std::vector<std::vector<double>> data;
  for (const auto& c : std::vector<std::vector<double>>{{0, 0}, {6, 1}, {2, 6}, {7, 7}, {3, 3}})
    for (int i = 0; i < 120; ++i)
      data.push_back({c[0] + noise(rng), c[1] + noise(rng)});
  return data;
}
} // namespace datamunge::algorithms
