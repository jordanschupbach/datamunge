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

#include <cmath>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>

namespace ml {
namespace nn = datamunge::nn;

inline nn::Tensor LeastSquaresCholesky(const nn::Tensor& x, const nn::Tensor& y) {
  const auto gram  = nn::matmul(x, x, true, false);
  const auto rhs   = nn::matmul(x, y, true, false);
  const auto lower = nn::cholesky(gram);
  const auto z     = nn::triangular_solve(lower, rhs);
  return nn::triangular_solve(lower, z, /*lower=*/true, /*adjoint=*/true);
}

inline nn::Tensor LeastSquaresQr(const nn::Tensor& x, const nn::Tensor& y) {
  const auto qr  = nn::qr(x);
  const auto rhs = nn::matmul(qr.q, y, true, false);
  return nn::triangular_solve(qr.r, rhs, /*lower=*/false);
}

inline nn::Tensor ElasticNetStep(const nn::Tensor& x, const nn::Tensor& y, const nn::Tensor& w, const nn::Tensor& mask,
                                 float l1, float l2, float rate) {
  if (l1 < 0 || l2 < 0 || rate <= 0 || !std::isfinite(l1 + l2 + rate))
    throw std::invalid_argument("invalid elastic-net step parameters");
  const double n        = static_cast<double>(x.size(0));
  const auto   residual = nn::matmul(x, w) - y;
  const auto   gradient = nn::matmul(x, residual, true, false) / n + l2 * (mask * w);
  const auto   z        = w - rate * gradient;
  const auto   threshold = (l1 * rate) * mask;
  return nn::sign(z) * nn::maximum(nn::abs(z) - threshold, nn::scalar_like(0.0, z));
}

}  // namespace ml
