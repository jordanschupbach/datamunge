/// @file filtered_back_projection.cpp
/// @brief Checked finite projection geometry and shared Ram-Lak/filter/gather kernels.
#include <cmath>
#include <datamunge/algorithms/filtered_back_projection.hpp>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Geometry validation and actual finite tomography execution without shared mutation.
namespace tomography_detail {
/// @brief Canonical double pi, matching the former M_PI binary constant.
constexpr double pi = std::numbers::pi_v<double>;
/// @brief Check a positive output side and the complete row-major pixel geometry.
/// @param N Requested positive square side.
/// @return N*N fitting size_t and vector<double> geometry.
/// @throws std::invalid_argument If N<=0.
/// @throws std::length_error Before count multiplication/container limits can be exceeded.
std::size_t pixels(int N) {
  if (N <= 0)
    throw std::invalid_argument("tomography image side must be positive");
  const auto n = static_cast<std::size_t>(N);
  if (n > std::vector<double>{}.max_size() / n)
    throw std::length_error("tomography pixel count exceeds container limits");
  return n * n;
}
/// @brief Check nested sinogram dimensions and aggregate scalar geometry before allocation.
/// @param views Number of rows, possibly zero for standalone filtering.
/// @param detectors Positive row width.
/// @throws std::invalid_argument If detector count is zero.
/// @throws std::length_error On row/container or aggregate scalar count limits.
void geometry(std::size_t views, std::size_t detectors) {
  if (!detectors)
    throw std::invalid_argument("tomography detector count must be positive");
  if (views > std::vector<std::vector<double>>{}.max_size() ||
      detectors > std::vector<double>{}.max_size() ||
      (views && detectors > std::vector<double>{}.max_size() / views))
    throw std::length_error("tomography sinogram geometry exceeds container limits");
}
/// @brief Validate detector count and nonempty finite angle values.
/// @param angles Borrowed view angles in radians.
/// @param detectors Positive detector count.
/// @throws std::invalid_argument On nonpositive detectors, empty angles or nonfinite angles.
/// @throws std::length_error On sinogram geometry limits.
void views(const std::vector<double>& angles, int detectors) {
  if (detectors <= 0 || angles.empty())
    throw std::invalid_argument("tomography requires detectors and nonempty angles");
  geometry(angles.size(), static_cast<std::size_t>(detectors));
  for (double theta : angles)
    if (!std::isfinite(theta))
      throw std::invalid_argument("tomography angles must be finite");
}
/// @brief Validate a complete rectangular finite sinogram against explicit dimensions.
/// @param s Borrowed projection rows.
/// @param rows Expected row count.
/// @param columns Expected positive row width.
/// @throws std::invalid_argument On shape mismatch or nonfinite entries.
void rectangle(const std::vector<std::vector<double>>& s, std::size_t rows, std::size_t columns) {
  if (s.size() != rows)
    throw std::invalid_argument("tomography view count mismatch");
  for (const auto& row : s) {
    if (row.size() != columns)
      throw std::invalid_argument("tomography detector row width mismatch");
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument("tomography samples must be finite");
  }
}
/// @brief A supported detector interpolation location, allowing a lower bin of minus one.
struct Weights {
  int    lower    = 0; ///< Floor of supported s, in[-1,D-1].
  double fraction = 0; ///< Actual s-floor(s), between zero and one.
};
/// @brief Compute the original detector coordinate, rejecting unsupported values before int cast.
/// @param r Pixel row.
/// @param c Pixel column.
/// @param center Pixel-grid center.
/// @param dcen Detector-grid center.
/// @param ct Cosine of the finite angle.
/// @param st Sine of the finite angle.
/// @param detectors Positive detector count.
/// @return Floor/fraction for s in[-1,D), otherwise nullopt for zero outside support.
/// @throws std::overflow_error If coordinate arithmetic is nonfinite.
std::optional<Weights>
weights(int r, int c, double center, double dcen, double ct, double st, int detectors) {
  const double x = c - center, y = r - center, s = x * ct + y * st + dcen;
  if (!std::isfinite(s))
    throw std::overflow_error("tomography detector coordinate is nonfinite");
  if (s < -1.0 || s >= detectors)
    return std::nullopt;
  const int lower = static_cast<int>(std::floor(s));
  return Weights{lower, s - lower};
}
/// @brief Reject nonfinite results of finite-input arithmetic.
/// @param value Actual updated sum or product.
/// @throws std::overflow_error If value is NaN or infinite.
void finite_result(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("tomography finite-input accumulation overflow");
}
/// @brief Compute the full finite unit-spacing ramp after checked kernel geometry.
/// @param width Positive detector row width.
/// @return Owned symmetric lags-(D-1)..D-1 in original coefficient arithmetic order.
/// @throws std::length_error On kernel count/container limits.
/// @throws std::bad_alloc On kernel allocation failure.
std::vector<double> kernel(std::size_t width) {
  const auto half = width - 1;
  if (half > (std::vector<double>{}.max_size() - 1) / 2)
    throw std::length_error("tomography filter kernel exceeds vector limits");
  std::vector<double> h(2 * half + 1, 0.0);
  const auto          signed_half = static_cast<std::ptrdiff_t>(half);
  for (std::ptrdiff_t n = -signed_half; n <= signed_half; ++n) {
    double value = 0;
    if (!n)
      value = .25;
    else if (n % 2 != 0)
      value = -1.0 / (pi * pi * static_cast<double>(n) * n);
    h[static_cast<std::size_t>(n + signed_half)] = value;
  }
  return h;
}
/// @brief Filter validated finite rows with the actual original direct-convolution order.
/// @param s Validated rectangular nonempty finite projection data.
/// @return Owned filtered rows.
/// @throws std::length_error On kernel geometry limits.
/// @throws std::overflow_error On nonfinite filtering arithmetic.
/// @throws std::bad_alloc On output/kernel storage failure.
std::vector<std::vector<double>> filter(const std::vector<std::vector<double>>& s) {
  const auto                       width = s.front().size(), half = width - 1;
  const auto                       h = kernel(width);
  std::vector<std::vector<double>> filtered(s.size(), std::vector<double>(width, 0.0));
  for (std::size_t ai = 0; ai < s.size(); ++ai)
    for (std::size_t d = 0; d < width; ++d) {
      double acc = 0;
      for (std::size_t dp = 0; dp < width; ++dp) {
        const auto index = d >= dp ? half + (d - dp) : half - (dp - d);
        acc += s[ai][dp] * h[index];
        finite_result(acc);
      }
      filtered[ai][d] = acc;
    }
  return filtered;
}
/// @brief Gather already validated rows with actual shared weights and fixed pi/M scaling.
/// @param s Validated projection rows, filtered or raw.
/// @param N Positive output side.
/// @param angles Nonempty finite angles matched to rows.
/// @param detectors Positive row width.
/// @param count Checked N*N output count.
/// @return Owned actual backprojected pixels.
/// @throws std::overflow_error On nonfinite coordinate/accumulation arithmetic.
/// @throws std::bad_alloc On output allocation failure.
std::vector<double> gather(
    const std::vector<std::vector<double>>& s,
    int                                     N,
    const std::vector<double>&              angles,
    int                                     detectors,
    std::size_t                             count) {
  std::vector<double> image(count, 0);
  const double        center = (N - 1) / 2.0, dcen = (detectors - 1) / 2.0,
               scale = pi / static_cast<double>(angles.size());
  for (std::size_t ai = 0; ai < angles.size(); ++ai) {
    const double ct = std::cos(angles[ai]), st = std::sin(angles[ai]);
    const auto&  fv = s[ai];
    for (int r = 0; r < N; ++r)
      for (int c = 0; c < N; ++c) {
        const auto w = weights(r, c, center, dcen, ct, st, detectors);
        if (!w)
          continue;
        double value = 0;
        if (w->lower >= 0 && w->lower < detectors)
          value += fv[w->lower] * (1 - w->fraction);
        if (w->lower + 1 >= 0 && w->lower + 1 < detectors)
          value += fv[w->lower + 1] * w->fraction;
        finite_result(value);
        auto& pixel = image[static_cast<std::size_t>(r) * N + c];
        pixel += scale * value;
        finite_result(pixel);
      }
  }
  return image;
}
} // namespace tomography_detail
std::vector<double> disk_phantom(int N, double radius, double intensity) {
  const auto count = tomography_detail::pixels(N);
  if (!std::isfinite(radius) || radius < 0 || !std::isfinite(intensity))
    throw std::invalid_argument(
        "disk phantom needs finite nonnegative radius and finite intensity");
  std::vector<double> image(count, 0);
  const double        center = (N - 1) / 2.0;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c)
      if (std::hypot(c - center, r - center) <= radius)
        image[static_cast<std::size_t>(r) * N + c] = intensity;
  return image;
}
std::vector<double> parallel_projection_angles(std::size_t count) {
  if (!count)
    throw std::invalid_argument("projection angle count must be positive");
  if (count > std::vector<double>{}.max_size())
    throw std::length_error("projection angles exceed vector limits");
  std::vector<double> a(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double value =
        tomography_detail::pi * static_cast<double>(i) / static_cast<double>(count);
    a[i] = value < tomography_detail::pi ? value : std::nextafter(tomography_detail::pi, 0.0);
  }
  return a;
}
std::vector<std::vector<double>> radon_transform(
    const std::vector<double>& image,
    int                        N,
    const std::vector<double>& angles,
    int                        detectors) {
  const auto count = tomography_detail::pixels(N);
  tomography_detail::views(angles, detectors);
  if (image.size() != count)
    throw std::invalid_argument("tomography pixel input must have N*N values");
  for (double value : image)
    if (!std::isfinite(value))
      throw std::invalid_argument("tomography pixels must be finite");
  std::vector<std::vector<double>> s(angles.size(), std::vector<double>(detectors, 0));
  const double                     center = (N - 1) / 2.0, dcen = (detectors - 1) / 2.0;
  for (std::size_t ai = 0; ai < angles.size(); ++ai) {
    const double ct = std::cos(angles[ai]), st = std::sin(angles[ai]);
    auto&        row = s[ai];
    for (int r = 0; r < N; ++r)
      for (int c = 0; c < N; ++c) {
        const double value = image[static_cast<std::size_t>(r) * N + c];
        if (value == 0.0)
          continue;
        const auto w = tomography_detail::weights(r, c, center, dcen, ct, st, detectors);
        if (!w)
          continue;
        if (w->lower >= 0 && w->lower < detectors) {
          row[w->lower] += value * (1 - w->fraction);
          tomography_detail::finite_result(row[w->lower]);
        }
        if (w->lower + 1 >= 0 && w->lower + 1 < detectors) {
          row[w->lower + 1] += value * w->fraction;
          tomography_detail::finite_result(row[w->lower + 1]);
        }
      }
  }
  return s;
}
std::vector<std::vector<double>> ram_lak_filter(const std::vector<std::vector<double>>& s) {
  if (s.empty())
    return {};
  tomography_detail::geometry(s.size(), s.front().size());
  tomography_detail::rectangle(s, s.size(), s.front().size());
  return tomography_detail::filter(s);
}
std::vector<double> back_projection(
    const std::vector<std::vector<double>>& s,
    int                                     N,
    const std::vector<double>&              angles,
    int                                     detectors) {
  const auto count = tomography_detail::pixels(N);
  tomography_detail::views(angles, detectors);
  tomography_detail::rectangle(s, angles.size(), static_cast<std::size_t>(detectors));
  return tomography_detail::gather(s, N, angles, detectors, count);
}
std::vector<double> filtered_back_projection(
    const std::vector<std::vector<double>>& s,
    int                                     N,
    const std::vector<double>&              angles,
    int                                     detectors) {
  const auto count = tomography_detail::pixels(N);
  tomography_detail::views(angles, detectors);
  tomography_detail::rectangle(s, angles.size(), static_cast<std::size_t>(detectors));
  const auto filtered = tomography_detail::filter(s);
  return tomography_detail::gather(filtered, N, angles, detectors, count);
}
} // namespace datamunge::algorithms
