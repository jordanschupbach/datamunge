// datamunge::nn port of tflow_examples src/common/ml/linear.h.
// Linear least squares and proximal regularization built with nn ops.
// Cholesky and thin QR require a full-column-rank design with rows >= columns.
// Neither forms an explicit inverse. QR avoids squaring the condition number.
// ElasticNetStep minimizes ||Xw-y||^2/(2N) + l1*sum(mask*|w|)
// + l2*sum(mask*w^2)/2. A zero mask entry leaves an intercept unpenalized.
// The proximal step assumes nonnegative penalties/mask and a
// rate <= 1/(||X||_2^2/N + l2*max(mask)). It returns the next value, not an
// assignment, so callers control parameter lifetime and update ordering.
#pragma once

/// @file linear.hpp
/// @brief Tensor least-squares solvers and proximal elastic-net steps.

#include <cmath>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

/// @brief Small tensor-based statistical workflows and validation utilities.
namespace datamunge::nn::ml {
namespace nn = datamunge::nn;

/// @brief Solve full-column-rank least squares by Cholesky normal equations.
///
/// No explicit inverse is formed. QR avoids squaring the condition number.
/// Rank deficiency is not reliably diagnosed by a dedicated rank test; callers must
/// check their design. Decomposition operations retain their usual autograd behavior.
/// @param x Floating design [N,p], N >= p, full column rank.
/// @param y Floating response [N,r], same dtype; multiple right-hand sides allowed.
/// @return Coefficient tensor [p,r]; inputs unchanged.
/// @throws std::invalid_argument On incompatible tensor shapes; std::runtime_error on a failed
/// factorisation.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto x=n::tensor({1,0,1,1,1,2},{3,2},n::DType::Float64);
/// const auto y=n::tensor({1,3,5},{3,1},n::DType::Float64);
/// const auto w=m::LeastSquaresCholesky(x,y); // [1,2]
/// @endcode
///
/// @par References
/// - LAPACK Users Guide, https://www.netlib.org/lapack/lug/node27.html
/// @see LeastSquaresQr(), LeastSquaresCholesky()
inline nn::Tensor LeastSquaresCholesky(const nn::Tensor& x, const nn::Tensor& y) {
  const auto gram  = nn::matmul(x, x, true, false);
  const auto rhs   = nn::matmul(x, y, true, false);
  const auto lower = nn::cholesky(gram);
  const auto z     = nn::triangular_solve(lower, rhs);
  return nn::triangular_solve(lower, z, /*lower=*/true, /*adjoint=*/true);
}

/// @brief Solve full-column-rank least squares by thin QR.
///
/// No explicit inverse is formed. QR avoids squaring the condition number.
/// Rank deficiency is not reliably diagnosed by a dedicated rank test; callers must
/// check their design. Decomposition operations retain their usual autograd behavior.
/// @param x Floating design [N,p], N >= p, full column rank.
/// @param y Floating response [N,r], same dtype; multiple right-hand sides allowed.
/// @return Coefficient tensor [p,r]; inputs unchanged.
/// @throws std::invalid_argument On incompatible tensor shapes; std::runtime_error on a failed
/// factorisation.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto x=n::tensor({1,0,1,1,1,2},{3,2},n::DType::Float64);
/// const auto y=n::tensor({1,3,5},{3,1},n::DType::Float64);
/// const auto w=m::LeastSquaresQr(x,y); // [1,2]
/// @endcode
///
/// @par References
/// - LAPACK Users Guide, https://www.netlib.org/lapack/lug/node27.html
/// @see LeastSquaresQr(), LeastSquaresCholesky()
inline nn::Tensor LeastSquaresQr(const nn::Tensor& x, const nn::Tensor& y) {
  const auto qr  = nn::qr(x);
  const auto rhs = nn::matmul(qr.q, y, true, false);
  return nn::triangular_solve(qr.r, rhs, /*lower=*/false);
}

/// @brief Return one proximal-gradient elastic-net iterate without assigning it.
///
/// The smooth gradient is X-transpose residual / N + l2*mask*w. The
/// threshold is rate*l1*mask. Callers ensure nonnegative mask and a stable
/// rate; these spectral/mask preconditions are not measured by the helper.
/// @param x Floating design [N,p], N positive.
/// @param y Response [N,r].
/// @param w Current coefficients [p,r].
/// @param mask Nonnegative broadcastable penalty weights; zero excludes an intercept.
/// @param l1 Finite nonnegative absolute penalty.
/// @param l2 Finite nonnegative squared penalty.
/// @param rate Finite positive step, at most reciprocal smooth Lipschitz constant for descent.
/// @return Soft-thresholded next coefficient tensor.
/// @throws std::invalid_argument On invalid penalties/rate or incompatible tensor operations.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto x=n::tensor({1,0,1,1,1,2},{3,2});
/// const auto
/// next=m::ElasticNetStep(x,n::ones({3,1}),n::zeros({2,1}),n::tensor({0,1},{2,1}),.1f,.2f,.1f);
/// @endcode
///
/// @par References
/// - Parikh and Boyd (2014), Proximal Algorithms,
/// https://web.stanford.edu/~boyd/papers/prox_algs.html
/// @see LeastSquaresQr()
inline nn::Tensor ElasticNetStep(
    const nn::Tensor& x,
    const nn::Tensor& y,
    const nn::Tensor& w,
    const nn::Tensor& mask,
    float             l1,
    float             l2,
    float             rate) {
  if (l1 < 0 || l2 < 0 || rate <= 0 || !std::isfinite(l1 + l2 + rate))
    throw std::invalid_argument("invalid elastic-net step parameters");
  const double n         = static_cast<double>(x.size(0));
  const auto   residual  = nn::matmul(x, w) - y;
  const auto   gradient  = nn::matmul(x, residual, true, false) / n + l2 * (mask * w);
  const auto   z         = w - rate * gradient;
  const auto   threshold = (static_cast<double>(l1) * rate) * mask;
  return nn::sign(z) * nn::maximum(nn::abs(z) - threshold, nn::scalar_like(0.0, z));
}

} // namespace datamunge::nn::ml
