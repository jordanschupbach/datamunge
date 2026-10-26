// CMD: just run-nn linear_regression
// Linear regression by Cholesky, QR and gradient descent.
//
// Port of tflow's linear_regression. Fit an intercept and two Fourier
// features to 64 noisy observations. The noise is orthogonal to the design,
// so the exact least-squares coefficients and residual MSE are known. The
// direct solvers use nn::cholesky / nn::qr and triangular solves in float64,
// like TF's Cholesky, Qr and MatrixTriangularSolve; the gradient-descent path runs in float32 through nn autograd. Nearly
// collinear features then show why QR beats the normal equations.

#include <algorithm>
#include <cmath>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <vector>

#include "common/checks.hpp"

namespace nn     = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

std::vector<double> LeastSquaresCholesky(const Tensor& x, const Tensor& y) {
  // Normal equations: (X^T X) w = X^T y, by two triangular solves with L.
  const Tensor gram  = nn::matmul(x, x, true, false);
  const Tensor rhs   = nn::matmul(x, y, true, false);
  const Tensor lower = nn::cholesky(gram);
  const Tensor z     = nn::triangular_solve(lower, rhs);
  return nn::triangular_solve(lower, z, true, /*adjoint=*/true).to_vector<double>();
}

std::vector<double> LeastSquaresQr(const Tensor& x, const Tensor& y) {
  // Thin QR: R w = Q^T y.
  const auto   qr  = nn::qr(x);
  const Tensor rhs = nn::matmul(qr.q, y, true, false);
  return nn::triangular_solve(qr.r, rhs, /*lower=*/false).to_vector<double>();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int steps = 180;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) steps = std::stoi(argument.substr(7));
      if (argument.rfind("steps=", 0) == 0) steps = std::stoi(argument.substr(6));
    }
    check::Require(steps > 0, "positive training budget required");
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
    for (int step = 0; step < steps; ++step) {
      const Tensor loss = 0.5 * nn::mean(nn::square(nn::matmul(xf, w) - yf), {0, 1});
      const Tensor g    = nn::grad(loss, {w})[0];
      nn::NoGradGuard guard;
      w.sub_(g, .1);
    }
    const double mse = nn::mean(nn::square(nn::matmul(xf, w) - yf)).item();
    const auto   wv  = w.to_vector<double>();
    for (int j = 0; j < 3; ++j)
      check::Near(wv[j], truth[j] * (1 - std::pow(.9, steps)), 2e-6, "finite-step SGD coefficient agreement");
    double test_mse = 0;
    for (int i = 0; i < n; ++i) {
      const double t = 2 * pi * (i + .5) / n, features[]{1, std::sqrt(2.) * std::cos(t), std::sqrt(2.) * std::sin(t)};
      double       error = -.1 * std::cos(2 * t);
      for (int j = 0; j < 3; ++j) error += features[j] * (wv[j] - truth[j]);
      test_mse += error * error / n;
    }
    const double expected_mse = .005 + (.7 * .7 + 1.2 * 1.2 + 2.5 * 2.5) * std::pow(.9, 2 * steps);
    check::Near(mse, expected_mse, 3e-6, "finite-step residual energy");
    check::Near(test_mse, expected_mse, 3e-6, "finite-step held-out residual energy");
    if (steps >= 180) {
      check::Near(mse, .005, 1e-8, "known converged residual energy");
      check::Near(test_mse, .005, 1e-9, "converged held-out residual energy");
    }
    std::cout << "\n  " << steps << " SGD steps: train MSE=" << mse << "; test MSE=" << test_mse << '\n';

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
