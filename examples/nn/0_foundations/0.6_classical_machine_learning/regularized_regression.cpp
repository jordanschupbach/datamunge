// CMD: just run-nn regularized_regression
// Ridge, lasso and elastic net with proximal-gradient updates.
//
// Port of tflow's regularized_regression (float32, like TF).
// Objective: mean squared error / 2 + lambda1*||w||_1 + lambda2*||w||^2/2,
// excluding the intercept from both penalties. On an orthonormal design the
// exact solution is soft_threshold(X'y/N, lambda1)/(1+lambda2). Compare all
// three methods with that formula and ridge with a Cholesky solve
// (nn::cholesky + nn::triangular_solve, TF's Cholesky and
// MatrixTriangularSolve). Repeat on correlated columns, verify the objective
// never increases, and check the KKT conditions for both zero and nonzero
// coefficients. Lasso needs a proximal soft-threshold step to produce exact zeros.

#include <array>
#include <cmath>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <vector>

#include "common/checks.hpp"
#include "common/linear.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main() {
  try {
    constexpr int n = 96, d = 5;
    Tensor        mask = nn::ones({d, 1});
    mask.data_ptr<float>()[0] = 0;
    const std::array<const char*, 3> names{"ridge", "lasso", "elastic_net"};
    const std::array<float, 3>       l1{0.f, .2f, .2f}, l2{.5f, 0.f, .5f};
    const auto objective = [&](const Tensor& x, const Tensor& y, const Tensor& w, int method) {
      const Tensor residual = nn::matmul(x, w) - y;
      return .5f * nn::mean(nn::square(residual), {0, 1}) + l1[method] * nn::sum(mask * nn::abs(w), {0, 1}) +
             (.5f * l2[method]) * nn::sum(mask * nn::square(w), {0, 1});
    };
    const auto smooth_gradient = [&](const Tensor& x, const Tensor& y, const Tensor& w, int method) {
      const Tensor residual = nn::matmul(x, w) - y;
      return nn::matmul(x, residual, true, false) / float(n) + l2[method] * (mask * w);
    };
    const float truth[]{.7f, 2.f, -1.f, .12f, 0.f};
    std::cout << "== Ridge / lasso / elastic net\n";
    for (bool correlated : {false, true}) {
      Tensor x = nn::empty({n, d}), y = nn::empty({n, 1});
      float* xp = x.data_ptr<float>();
      float* yp = y.data_ptr<float>();
      for (int i = 0; i < n; ++i) {
        const double t = 2 * std::acos(-1.) * i / n;
        xp[i * d]      = 1;
        for (int j = 1; j < d; ++j) xp[i * d + j] = std::sqrt(2.) * std::cos(j * t);
        if (correlated) xp[i * d + 2] = .8f * xp[i * d + 1] + .6f * xp[i * d + 2];
        yp[i] = .05 * std::cos(5 * t);
        for (int j = 0; j < d; ++j) yp[i] += xp[i * d + j] * truth[j];
      }
      std::vector<Tensor>   weights(3, nn::zeros({d, 1}));
      std::array<double, 3> previous{1e100, 1e100, 1e100};
      for (int step = 0; step < 240; ++step) {
        // One "train" run: every method steps from its current value.
        for (int j = 0; j < 3; ++j) weights[j] = ml::ElasticNetStep(x, y, weights[j], mask, l1[j], l2[j], .4f);
        for (int j = 0; j < 3; ++j) {
          const float value = objective(x, y, weights[j], j).item<float>();
          check::Require(std::isfinite(value) && value <= previous[j] + 5e-7, "monotone proximal objective");
          previous[j] = value;
        }
      }
      // Ridge by a direct Cholesky solve of (X'X/N + l2 diag(mask)) w = X'y/N.
      const Tensor gram  = nn::matmul(x, x, true, false) / float(n) + l2[0] * nn::matrix_diag(nn::reshape(mask, {d}));
      const Tensor lower = nn::cholesky(gram);
      const Tensor rhs   = nn::matmul(x, y, true, false) / float(n);
      const auto   ridge = nn::triangular_solve(lower, nn::triangular_solve(lower, rhs), true, true).to_vector<float>();
      std::cout << "  " << (correlated ? "correlated" : "orthonormal") << " design:\n";
      for (int method = 0; method < 3; ++method) {
        std::cout << "    " << names[method] << " weights:";
        const auto w_values = weights[method].to_vector<float>();
        const auto g_values = smooth_gradient(x, y, weights[method], method).to_vector<float>();
        int        zeros    = 0;
        double     max_kkt  = 0;
        for (int j = 0; j < d; ++j) {
          const float  w = w_values[j], g = g_values[j];
          const float  penalty = j ? l1[method] : 0;
          const double kkt = w == 0 ? std::max(0.f, std::abs(g) - penalty) : std::abs(g + std::copysign(penalty, w));
          max_kkt          = std::max(max_kkt, kkt);
          if (j && w == 0) ++zeros;
          if (!correlated) {
            const float expected =
                j ? std::copysign(std::max(0.f, std::abs(truth[j]) - l1[method]) / (1 + l2[method]), truth[j]) : truth[0];
            check::Near(w, expected, 2e-6, "orthonormal soft-threshold solution");
          }
          if (method == 0) check::Near(w, ridge[j], 3e-6, "ridge direct solve agreement");
          std::cout << ' ' << w;
        }
        check::Require(max_kkt < 3e-6, "regularized KKT stationarity");
        if (method > 0) check::Require(zeros >= 2, "L1 regularization creates exact zeros");
        std::cout << "; objective=" << objective(x, y, weights[method], method).item<float>() << "; KKT=" << max_kkt
                  << '\n';
      }
    }
    std::cout << "All regularized regression checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
