/// @file lanczos_analysis.cpp
/// @brief Original fixture and explicit widened Ritz-vector/full-residual/Gram/projection observations.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/lanczos_analysis.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Local finite diagnostic conversion and complete input/geometry checks.
namespace ritz_detail {
/// @brief Require a widened diagnostic value to be finite.
/// @param x Actual widened arithmetic result.
/// @return Unchanged finite widened value.
/// @throws std::overflow_error On nonfinite value.
long double finite(long double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Ritz nonfinite diagnostic");
  return x;
}
/// @brief Round one finite widened formed coordinate to a representable finite double.
/// @param x Actual widened coordinate.
/// @return Rounded finite double.
/// @throws std::overflow_error On nonfinite/out-of-double-range value.
double coordinate(long double x) {
  finite(x);
  if (std::abs(x) > std::numeric_limits<double>::max())
    throw std::overflow_error("Ritz coordinate range");
  return static_cast<double>(x);
}
/// @brief Validate complete symmetric input, compatible finite result and full mode/matrix geometry.
/// @param A Borrowed original finite symmetric matrix.
/// @param r Borrowed complete result, no provenance inferred.
/// @param cap Optional complete mode ceiling.
/// @throws std::invalid_argument On matrix/result shape/finite/symmetry violations.
/// @throws std::length_error On complete mode quota or multiplication geometry.
void validate(const std::vector<std::vector<double>>& A, const LanczosResult& r, std::size_t cap) {
  const auto n = A.size(), m = r.alpha.size();
  if (r.V.size() != m || m > n || r.beta.size() < (m ? m - 1 : 0))
    throw std::invalid_argument("Ritz result shape");
  for (const auto& row : A) {
    if (row.size() != n)
      throw std::invalid_argument("Ritz matrix shape");
    for (double x : row)
      if (!std::isfinite(x))
        throw std::invalid_argument("Ritz finite matrix");
  }
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (A[i][j] != A[j][i])
        throw std::invalid_argument("Ritz exact symmetry");
  for (const auto& v : r.V) {
    if (v.size() != n)
      throw std::invalid_argument("Ritz basis shape");
    for (double x : v)
      if (!std::isfinite(x))
        throw std::invalid_argument("Ritz finite basis");
  }
  for (double x : r.alpha)
    if (!std::isfinite(x))
      throw std::invalid_argument("Ritz finite alpha");
  for (double x : r.beta)
    if (!std::isfinite(x))
      throw std::invalid_argument("Ritz finite beta");
  if (cap && m > cap)
    throw std::length_error("Ritz mode quota");
  if (m && (m > std::numeric_limits<std::size_t>::max() / m ||
            n > std::numeric_limits<std::size_t>::max() / m))
    throw std::length_error("Ritz matrix geometry");
}
} // namespace ritz_detail
std::vector<std::vector<double>> lanczos_demo_matrix() {
  std::vector<std::vector<double>> A(30, std::vector<double>(30));
  std::uint32_t                    state = 2024;
  for (std::size_t i = 0; i < 30; ++i)
    for (std::size_t j = i; j < 30; ++j) {
      state              = state * 1103515245U + 12345U;
      const double value = ((state >> 16) % 1000) / 500. - 1.;
      A[i][j] = A[j][i] = value;
    }
  return A;
}
LanczosRitzAnalysis lanczos_ritz_analysis(
    const std::vector<std::vector<double>>& A,
    const LanczosResult&                    r,
    std::size_t                             max_modes) {
  ritz_detail::validate(A, r, max_modes);
  const auto          n = A.size(), m = r.alpha.size();
  LanczosRitzAnalysis out;
  if (m == 0)
    return out;
  out.tridiagonal.assign(m, std::vector<double>(m, 0));
  linalg::DenseMatrix<double> T(m, m, 0);
  double                      scale = 1;
  for (std::size_t i = 0; i < m; ++i) {
    T(i, i) = out.tridiagonal[i][i] = r.alpha[i];
    scale                           = std::max(scale, std::abs(r.alpha[i]));
    if (i + 1 < m) {
      T(i, i + 1) = T(i + 1, i) = r.beta[i];
      out.tridiagonal[i][i + 1] = out.tridiagonal[i + 1][i] = r.beta[i];
      scale                                                 = std::max(scale, std::abs(r.beta[i]));
    }
  }
  const auto e = linalg::jacobi_eigen(T, 1e-13, 100);
  out.values   = e.eigenvalues;
  out.coefficients.assign(m, std::vector<double>(m));
  out.vectors.assign(m, std::vector<double>(n));
  out.vector_norms.resize(m);
  out.residual_norms.resize(m);
  for (std::size_t mode = 0; mode < m; ++mode) {
    ritz_detail::finite(out.values[mode]);
    for (std::size_t j = 0; j < m; ++j)
      out.coefficients[j][mode] = ritz_detail::coordinate(e.eigenvectors(j, mode));
    long double eigres = 0;
    for (std::size_t i = 0; i < m; ++i) {
      long double v = 0;
      for (std::size_t j = 0; j < m; ++j)
        v += static_cast<long double>(T(i, j)) * e.eigenvectors(j, mode);
      v -= static_cast<long double>(out.values[mode]) * e.eigenvectors(i, mode);
      eigres = std::hypot(eigres, ritz_detail::finite(v));
    }
    out.projected_eigen_residual = std::max(out.projected_eigen_residual, eigres);
    for (std::size_t i = 0; i < n; ++i) {
      long double y = 0;
      for (std::size_t j = 0; j < m; ++j)
        y += static_cast<long double>(r.V[j][i]) * e.eigenvectors(j, mode);
      out.vectors[mode][i] = ritz_detail::coordinate(y);
      out.vector_norms[mode] =
          std::hypot(out.vector_norms[mode], static_cast<long double>(out.vectors[mode][i]));
    }
    if (out.vector_norms[mode] == 0)
      throw std::invalid_argument("Ritz zero formed vector");
    for (std::size_t i = 0; i < n; ++i) {
      long double v = 0;
      for (std::size_t j = 0; j < n; ++j)
        v += static_cast<long double>(A[i][j]) * out.vectors[mode][j];
      v -= static_cast<long double>(out.values[mode]) * out.vectors[mode][i];
      out.residual_norms[mode] = std::hypot(out.residual_norms[mode], ritz_detail::finite(v));
    }
    ritz_detail::finite(out.vector_norms[mode]);
    ritz_detail::finite(out.residual_norms[mode]);
  }
  if (out.projected_eigen_residual > 1e-9L * scale * std::sqrt(static_cast<long double>(m)))
    throw std::runtime_error("Ritz projected eigensystem residual");
  std::vector<std::vector<long double>> AV(m, std::vector<long double>(n));
  for (std::size_t j = 0; j < m; ++j)
    for (std::size_t i = 0; i < n; ++i) {
      long double v = 0;
      for (std::size_t k = 0; k < n; ++k)
        v += static_cast<long double>(A[i][k]) * r.V[j][k];
      AV[j][i] = ritz_detail::finite(v);
    }
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < m; ++j) {
      long double gram = 0, projection = 0;
      for (std::size_t k = 0; k < n; ++k) {
        gram += static_cast<long double>(r.V[i][k]) * r.V[j][k];
        projection += static_cast<long double>(r.V[i][k]) * AV[j][k];
      }
      out.orthogonality_error = std::max(
          out.orthogonality_error, std::abs(ritz_detail::finite(gram) - (i == j ? 1.L : 0.L)));
      out.projection_error =
          std::max(out.projection_error, std::abs(ritz_detail::finite(projection) - T(i, j)));
    }
  return out;
}
} // namespace datamunge::algorithms
