// CMD: just run-nn linear_regression
// Linear regression by Cholesky, QR and gradient descent.
//
// Port of tflow's linear_regression. Fit an intercept and two Fourier
// features to 64 noisy observations. The noise is orthogonal to the design,
// so the exact least-squares coefficients and residual MSE are known. The
// direct solvers use datamunge's float64 Cholesky and Householder QR; the
// gradient-descent path runs in float32 through nn autograd. Nearly
// collinear features then show why QR beats the normal equations.

#include <algorithm>
#include <cmath>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/qr.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <vector>

#include "common/checks.hpp"

namespace nn     = datamunge::nn;
namespace linalg = datamunge::linalg;
using nn::DType;
using nn::Tensor;

namespace {

linalg::DenseMatrix<double> ToDense(const Tensor& t) {
  return linalg::DenseMatrix<double>(static_cast<std::size_t>(t.size(0)), static_cast<std::size_t>(t.size(1)),
                                     t.to_vector<double>());
}

std::vector<double> LeastSquaresCholesky(const Tensor& x, const Tensor& y) {
  // Normal equations: (X^T X) w = X^T y.
  const Tensor gram = nn::matmul(x, x, true, false);
  const Tensor rhs  = nn::matmul(x, y, true, false);
  return linalg::cholesky(ToDense(gram)).solve(rhs.to_vector<double>());
}

std::vector<double> LeastSquaresQr(const Tensor& x, const Tensor& y) {
  return linalg::qr(ToDense(x)).solve(y.to_vector<double>());
}

}  // namespace

int main() {
  try {
    constexpr int  n  = 64;
    const double   pi = std::acos(-1.);
    const double   truth[]{.7, -1.2, 2.5};
    Tensor         x = nn::empty({n, 3}, DType::Float64), y = nn::empty({n, 1}, DType::Float64);
    double*        xp = x.data_ptr<double>();
    double*        yp = y.data_ptr<double>();
    for (int i = 0; i < n; ++i) {
      const double t = 2 * pi * i / n;
      xp[i * 3 + 0]  = 1;
      xp[i * 3 + 1]  = std::sqrt(2.) * std::cos(t);
      xp[i * 3 + 2]  = std::sqrt(2.) * std::sin(t);
      yp[i]          = .1 * std::cos(2 * t);
      for (int j = 0; j < 3; ++j) yp[i] += xp[i * 3 + j] * truth[j];
    }

    const auto   chol     = LeastSquaresCholesky(x, y);
    const auto   qr       = LeastSquaresQr(x, y);
    const Tensor qr_w     = nn::from_vector(qr, {3, 1});
    const Tensor residual = nn::matmul(x, qr_w) - y;
    const auto   normal   = nn::matmul(x, residual, true, false).to_vector<double>();
    std::cout << "== Linear regression\n  Cholesky / QR weights:";
    for (int j = 0; j < 3; ++j) {
      check::Near(chol[j], truth[j], 1e-12, "Cholesky exact coefficients");
      check::Near(qr[j], truth[j], 1e-12, "QR exact coefficients");
      check::Near(normal[j], 0, 1e-11, "normal-equation stationarity");
      std::cout << ' ' << qr[j];
    }

    // Gradient descent in float32: loss = 0.5 * mean((xw - y)^2).
    const Tensor xf = nn::cast(x, DType::Float32), yf = nn::cast(y, DType::Float32);
    Tensor       w  = nn::zeros({3, 1}).requires_grad_();
    for (int step = 0; step < 180; ++step) {
      const Tensor loss = 0.5 * nn::mean(nn::square(nn::matmul(xf, w) - yf), {0, 1});
      const Tensor g    = nn::grad(loss, {w})[0];
      nn::NoGradGuard guard;
      w.sub_(g, .1);
    }
    const double mse = nn::mean(nn::square(nn::matmul(xf, w) - yf)).item();
    const auto   wv  = w.to_vector<double>();
    for (int j = 0; j < 3; ++j) check::Near(wv[j], truth[j], 2e-6, "SGD coefficient agreement");
    double test_mse = 0;
    for (int i = 0; i < n; ++i) {
      const double t = 2 * pi * (i + .5) / n, features[]{1, std::sqrt(2.) * std::cos(t), std::sqrt(2.) * std::sin(t)};
      double       error = -.1 * std::cos(2 * t);
      for (int j = 0; j < 3; ++j) error += features[j] * (wv[j] - truth[j]);
      test_mse += error * error / n;
    }
    check::Near(mse, .005, 1e-8, "known residual energy");
    check::Near(test_mse, .005, 1e-9, "held-out residual energy");
    std::cout << "\n  180 SGD steps: train MSE=" << mse << "; test MSE=" << test_mse << '\n';

    // An ill-conditioned full-rank design: forming the Gram matrix loses precision.
    for (int i = 0; i < n; ++i) {
      const double a = xp[i * 3 + 1], b = xp[i * 3 + 2];
      xp[i * 3 + 2]  = a + 1e-5 * b;
      yp[i]          = 1 + 2 * a - 2 * xp[i * 3 + 2];
    }
    const auto   chol2 = LeastSquaresCholesky(x, y);
    const auto   qr2   = LeastSquaresQr(x, y);
    double       error_chol = 0, error_qr = 0;
    const double near_truth[]{1, 2, -2};
    for (int j = 0; j < 3; ++j) {
      error_chol = std::max(error_chol, std::abs(chol2[j] - near_truth[j]));
      error_qr   = std::max(error_qr, std::abs(qr2[j] - near_truth[j]));
    }
    std::cout << "  collinear coefficient max error: Cholesky=" << error_chol << "; QR=" << error_qr << '\n';
    check::Require(error_qr < 1e-8 && error_chol < .01, "collinear solutions remain accurate");
    std::cout << "All linear regression checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
