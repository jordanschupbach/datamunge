/// @file kabsch.cpp
/// @brief One checked normalized Horn-quaternion fit with optional actual observations.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/kabsch.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Finite conversions and same-execution normalized alignment details.
namespace kabsch_detail {
/// @brief Convert one produced output to finite double, rejecting overflow.
/// @param x Produced widened output.
/// @return Rounded finite double; finite underflow can round to zero.
/// @throws std::overflow_error On nonfinite or out-of-double-range value.
double output(long double x) {
  if (!std::isfinite(x) || std::fabs(x) > std::numeric_limits<double>::max())
    throw std::overflow_error("Kabsch finite output overflow");
  return static_cast<double>(x);
}
/// @brief Validate and fit corresponding points once, optionally copying actual output observations.
/// @param source Borrowed original source points.
/// @param target Borrowed same-sized target points.
/// @param capture Whether to own original/aligned points and residual norms.
/// @param cap Complete optional pair ceiling, zero off.
/// @return Actual scalar fit and optional owned observations.
/// @throws std::invalid_argument On invalid paired inputs.
/// @throws std::length_error On enabled pair cap/container limits.
/// @throws std::overflow_error On produced arithmetic/output overflow.
/// @throws std::runtime_error On failed eigen/quaternion diagnostics.
/// @throws std::bad_alloc On allocation failure.
KabschAnalysis run(
    const std::vector<Point3>& source,
    const std::vector<Point3>& target,
    bool                       capture,
    std::size_t                cap) {
  const auto n = source.size();
  if (!n || target.size() != n)
    throw std::invalid_argument("Kabsch requires equal nonempty sets");
  KabschAnalysis a;
  for (std::size_t i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(source[i][d]) || !std::isfinite(target[i][d]))
        throw std::invalid_argument("Kabsch nonfinite coordinate");
      a.source_scale = std::max(a.source_scale, std::fabs(static_cast<long double>(source[i][d])));
      a.target_scale = std::max(a.target_scale, std::fabs(static_cast<long double>(target[i][d])));
    }
  if (cap && n > cap)
    throw std::length_error("Kabsch complete pair ceiling");
  if (capture) {
    a.source = source;
    a.target = target;
    a.aligned.reserve(n);
    a.residual_norms.reserve(n);
  }
  std::array<long double, 3> cs{}, ct{};
  for (std::size_t i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) {
      cs[d] += (static_cast<long double>(source[i][d]) / a.source_scale) / n;
      ct[d] += (static_cast<long double>(target[i][d]) / a.target_scale) / n;
    }
  for (int d = 0; d < 3; ++d) {
    a.source_centroid[d] = cs[d] * a.source_scale;
    a.target_centroid[d] = ct[d] * a.target_scale;
  }
  for (std::size_t i = 0; i < n; ++i)
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        a.covariance[r][c] += ((static_cast<long double>(source[i][r]) / a.source_scale - cs[r]) *
                               (static_cast<long double>(target[i][c]) / a.target_scale - ct[c])) /
                              n;
  for (const auto& row : a.covariance)
    for (auto x : row) {
      if (!std::isfinite(x))
        throw std::overflow_error("Kabsch covariance overflow");
      a.covariance_scale = std::max(a.covariance_scale, std::fabs(x));
    }
  std::array<std::array<long double, 3>, 3> h = a.covariance;
  if (a.covariance_scale)
    for (auto& row : h)
      for (auto& x : row)
        x /= a.covariance_scale;
  const auto xx = h[0][0], xy = h[0][1], xz = h[0][2], yx = h[1][0], yy = h[1][1], yz = h[1][2],
             zx = h[2][0], zy = h[2][1], zz = h[2][2];
  a.key_matrix = {
      {{xx + yy + zz, yz - zy, zx - xz, xy - yx},
       {yz - zy, xx - yy - zz, xy + yx, zx + xz},
       {zx - xz, xy + yx, -xx + yy - zz, yz + zy},
       {xy - yx, zx + xz, yz + zy, -xx - yy + zz}}};
  linalg::DenseMatrix<long double> key(4, 4, 0);
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      key(r, c) = a.key_matrix[r][c];
  const auto  eigen = linalg::jacobi_eigen(key, 1e-15L, 100);
  long double norm  = 0;
  for (int i = 0; i < 4; ++i) {
    a.eigenvalues[i] = eigen.eigenvalues[i];
    if(!std::isfinite(a.eigenvalues[i]))throw std::runtime_error("Kabsch nonfinite eigenvalue");
    a.quaternion[i]  = eigen.eigenvectors(i, 0);
    norm             = std::hypot(norm, a.quaternion[i]);
  }
  if (!std::isfinite(norm) || norm == 0)
    throw std::runtime_error("Kabsch invalid quaternion");
  for (auto& x : a.quaternion)
    x /= norm;
  for (int r = 0; r < 4; ++r) {
    long double e = -a.eigenvalues[0] * a.quaternion[r];
    for (int c = 0; c < 4; ++c)
      e += key(r, c) * a.quaternion[c];
    a.eigen_residual = std::hypot(a.eigen_residual, e);
  }
  a.eigen_gap = a.eigenvalues[0] - a.eigenvalues[1];
  if (!std::isfinite(a.eigen_residual) || a.eigen_residual > 1e-11L)
    throw std::runtime_error("Kabsch eigensolver residual");
  const auto w = a.quaternion[0], x = a.quaternion[1], y = a.quaternion[2], z = a.quaternion[3];
  const std::array<std::array<long double, 3>, 3> rotation = {
      {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
       {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
       {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
  for (int r = 0; r < 3; ++r) {
    long double translation = a.target_centroid[r];
    for (int c = 0; c < 3; ++c) {
      a.result.rotation[r][c] = output(rotation[r][c]);
      translation -= static_cast<long double>(a.result.rotation[r][c]) * a.source_centroid[c];
    }
    a.result.translation[r] = output(translation);
  }
  long double rms = 0;
  for (std::size_t i = 0; i < n; ++i) {
    Point3      aligned;
    long double residual = 0;
    for (int r = 0; r < 3; ++r) {
      long double value = a.result.translation[r];
      for (int c = 0; c < 3; ++c)
        value += static_cast<long double>(a.result.rotation[r][c]) * source[i][c];
      aligned[r] = output(value);
      residual   = std::hypot(residual, value - target[i][r]);
    }
    const long double weight = static_cast<long double>(i) / (i + 1);
    rms =
        std::hypot(rms * std::sqrt(weight), residual / std::sqrt(static_cast<long double>(i + 1)));
    if (capture) {
      a.aligned.push_back(aligned);
      a.residual_norms.push_back(residual);
    }
  }
  a.result.rmsd = output(rms);
  return a;
}
} // namespace kabsch_detail
KabschResult kabsch(const std::vector<Point3>& s, const std::vector<Point3>& t) {
  return kabsch_detail::run(s, t, false, 0).result;
}
KabschAnalysis
kabsch_analyze(const std::vector<Point3>& s, const std::vector<Point3>& t, std::size_t cap) {
  return kabsch_detail::run(s, t, true, cap);
}
} // namespace datamunge::algorithms
