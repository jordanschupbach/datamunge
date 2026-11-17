/// @file lanczos_resampling.cpp
/// @brief Shared checked raw/normalized windowed-sinc tap producer with exact numeric integer zeros.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/lanczos_resampling.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Complete geometry and finite arithmetic helpers for actual bounded-index reconstruction.
namespace resample_detail {
/// @brief Require a produced double arithmetic value to be finite.
/// @param value Actual product/sum/result.
/// @return Unchanged finite value.
/// @throws std::overflow_error On nonfinite produced arithmetic.
double finite(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Lanczos resampling nonfinite arithmetic");
  return value;
}
/// @brief Validate complete samples/lobes and exact representable integer window geometry.
/// @param samples Borrowed complete original finite nonempty samples.
/// @param a Actual positive half-width.
/// @return Complete2*a candidate count.
/// @throws std::invalid_argument On empty/nonfinite samples or nonpositive lobes.
/// @throws std::length_error On integer coordinate or tap-count geometry limits.
std::size_t validate(const std::vector<double>& samples, int a) {
  if (samples.empty() || a <= 0)
    throw std::invalid_argument("Lanczos resampling samples/lobes");
  for (double x : samples)
    if (!std::isfinite(x))
      throw std::invalid_argument("Lanczos resampling finite samples");
  const auto width = static_cast<std::uintmax_t>(a);
  const auto coordinate_limit =
      std::min<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max(), 9007199254740991ULL);
  if (width > std::numeric_limits<std::size_t>::max() / 2 || width > coordinate_limit ||
      samples.size() - 1 > coordinate_limit - width)
    throw std::length_error("Lanczos resampling coordinate geometry");
  return 2 * static_cast<std::size_t>(a);
}
/// @brief Evaluate the original tiny-center sinc approximation plus explicit exact integer zeros.
/// @param x Known finite bounded kernel argument.
/// @return Actual finite sinc value.
double sinc(double x) {
  if (std::abs(x) < 1e-12)
    return 1;
  if (x == std::trunc(x))
    return 0;
  const double px = std::numbers::pi * x;
  return std::sin(px) / px;
}
/// @brief Evaluate one bounded raw finite window weight without repeated input validation.
/// @param x Actual finite distance.
/// @param a Validated positive half-width.
/// @return Actual finite raw weight, zero at/outside support.
double weight(double x, int a) {
  if (x <= -static_cast<double>(a) || x >= a)
    return 0;
  return sinc(x) * sinc(x / a);
}
/// @brief Produce one complete actual raw tap sum, with explicit optional final normalization.
/// @param samples Validated original finite nonempty samples.
/// @param p Validated finite original query.
/// @param a Validated positive half-width/index geometry.
/// @param normalize Actual final normalization mode.
/// @param trace Optional preadmitted/preallocated producing capture.
/// @return Owned actual sums/value and selected tap counts.
/// @throws std::overflow_error On nonfinite produced arithmetic.
/// @throws std::domain_error On zero normalized denominator.
LanczosResampleAnalysis run(
    const std::vector<double>& samples,
    double                     p,
    int                        a,
    bool                       normalize,
    LanczosResampleTrace*      trace) {
  LanczosResampleAnalysis out;
  out.position           = p;
  out.effective_position = std::clamp(p, 0., static_cast<double>(samples.size() - 1));
  out.lobes              = a;
  out.normalized         = normalize;
  const auto floor       = static_cast<std::ptrdiff_t>(std::floor(out.effective_position));
  const auto first       = floor - static_cast<std::ptrdiff_t>(a) + 1,
             last        = floor + static_cast<std::ptrdiff_t>(a);
  for (auto k = first;; ++k) {
    const auto   kc      = k < 0                                           ? std::size_t{0}
                           : static_cast<std::size_t>(k) >= samples.size() ? samples.size() - 1
                                                                           : static_cast<std::size_t>(k);
    const double w       = weight(out.effective_position - static_cast<double>(k), a),
                 product = finite(samples[kc] * w), before = out.raw_sum;
    out.raw_sum    = finite(out.raw_sum + product);
    out.weight_sum = finite(out.weight_sum + w);
    ++out.work.taps;
    out.work.clamped_taps += k < 0 || static_cast<std::size_t>(k) >= samples.size();
    out.work.negative_weights += w < 0;
    if (trace)
      trace->taps.push_back({k, kc, samples[kc], w, product, before, out.raw_sum, out.weight_sum});
    if (k == last)
      break;
  }
  if (normalize && out.weight_sum == 0)
    throw std::domain_error("Lanczos resampling zero normalization weight");
  out.value = normalize ? finite(out.raw_sum / out.weight_sum) : out.raw_sum;
  return out;
}
} // namespace resample_detail
double lanczos_kernel_weight(double distance, int lobes) {
  if (!std::isfinite(distance) || lobes <= 0)
    throw std::invalid_argument("Lanczos kernel finite distance/positive lobes");
  return resample_detail::weight(distance, lobes);
}
double lanczos_resample_at(const std::vector<double>& samples, double p, int a) {
  resample_detail::validate(samples, a);
  if (!std::isfinite(p))
    throw std::invalid_argument("Lanczos resampling finite query");
  return resample_detail::run(samples, p, a, false, nullptr).value;
}
std::vector<double>
lanczos_resample(const std::vector<double>& samples, const std::vector<double>& positions, int a) {
  resample_detail::validate(samples, a);
  for (double p : positions)
    if (!std::isfinite(p))
      throw std::invalid_argument("Lanczos resampling finite positions");
  std::vector<double> out;
  out.reserve(positions.size());
  for (double p : positions)
    out.push_back(resample_detail::run(samples, p, a, false, nullptr).value);
  return out;
}
LanczosResampleAnalysis
lanczos_resample_analyze(const std::vector<double>& samples, double p, int a, bool normalize) {
  resample_detail::validate(samples, a);
  if (!std::isfinite(p))
    throw std::invalid_argument("Lanczos resampling finite query");
  return resample_detail::run(samples, p, a, normalize, nullptr);
}
LanczosResampleTrace lanczos_resample_trace(
    const std::vector<double>& samples,
    double                     p,
    int                        a,
    bool                       normalize,
    std::size_t                max_entries) {
  const auto taps = resample_detail::validate(samples, a);
  if (!std::isfinite(p))
    throw std::invalid_argument("Lanczos resampling finite query");
  if (taps > std::numeric_limits<std::size_t>::max() - samples.size())
    throw std::length_error("Lanczos resampling capture count");
  const auto capture = samples.size() + taps;
  if (max_entries && capture > max_entries)
    throw std::length_error("Lanczos resampling trace quota");
  LanczosResampleTrace out;
  out.captured_entries = capture;
  out.samples          = samples;
  out.taps.reserve(taps);
  out.analysis = resample_detail::run(samples, p, a, normalize, &out);
  return out;
}
} // namespace datamunge::algorithms
