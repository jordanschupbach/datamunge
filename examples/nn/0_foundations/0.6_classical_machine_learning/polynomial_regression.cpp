// CMD: just run-nn polynomial_regression
// Polynomial regression and a Monte Carlo bias-variance decomposition.
//
// Port of tflow's polynomial_regression (Legendre features in
// common/polynomial.hpp, QR least squares in common/linear.hpp).
// Fit degrees 1, 3 and 15 to 20 fixed x locations with 400 independent noisy
// responses from sin(pi*x), using a Legendre basis and float64 thin QR. One
// multiple-right-hand-side solve fits all replicas. Evaluate on a 201-point
// grid. Check the exact finite-ensemble identity MSE=bias^2+variance and add
// the known irreducible noise variance to predict noisy-test risk. Training
// error falls with degree while degree 15 has high prediction variance.
// Save mean/standard-deviation curves and per-degree statistics as TSV files
// under output/nn/metrics/polynomial_regression. Data come from the same
// seeded stream as the TF version.

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/checks.hpp"
#include "common/linear.hpp"
#include "common/paths.hpp"
#include "common/polynomial.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main() {
  try {
    constexpr int                    n = 20, replicas = 400, grid = 201;
    constexpr double                 sigma = .3;
    std::vector<double>              xs(n), ys(n * replicas), qs(grid);
    std::mt19937                     rng(113);
    std::normal_distribution<double> noise(0, sigma);
    for (int i = 0; i < n; ++i) {
      xs[i] = -1 + 2. * i / (n - 1);
      for (int r = 0; r < replicas; ++r) ys[i * replicas + r] = std::sin(std::acos(-1.) * xs[i]) + noise(rng);
    }
    for (int i = 0; i < grid; ++i) qs[i] = -1 + 2. * i / (grid - 1);
    const Tensor x = nn::from_vector(xs, {n, 1}), y = nn::from_vector(ys, {n, replicas});
    const Tensor query = nn::from_vector(qs, {grid, 1});
    const auto   dir   = paths::OutputDir("metrics/polynomial_regression");
    std::ofstream curves(dir / "curves.tsv"), summary(dir / "summary.tsv");
    check::Require(bool(curves) && bool(summary), "open polynomial metrics");
    curves << "degree\tx\ttruth\tmean\tstddev\n";
    summary << "degree\ttrain_mse\tbias_squared\tvariance\tnoise_variance\texpected_test_mse\n";
    std::array<double, 3>    train_errors{}, risks{}, variances{}, biases{};
    const std::array<int, 3> degrees{1, 3, 15};
    std::cout << "== Polynomial bias / variance (400 training replicas)\n";
    for (int model = 0; model < 3; ++model) {
      const int  degree  = degrees[model];
      const auto design  = ml::PolynomialFeatures(x, degree), query_design = ml::PolynomialFeatures(query, degree);
      const auto weights = ml::LeastSquaresQr(design, y);
      const auto prediction = nn::matmul(query_design, weights).to_vector<double>();
      const auto fit        = nn::matmul(design, weights).to_vector<double>();
      for (int i = 0; i < n * replicas; ++i) train_errors[model] += std::pow(fit[i] - ys[i], 2) / (n * replicas);
      double direct_mse = 0;
      for (int i = 0; i < grid; ++i) {
        const double truth = std::sin(std::acos(-1.) * qs[i]);
        double       mean = 0, variance = 0;
        for (int r = 0; r < replicas; ++r) mean += prediction[i * replicas + r] / replicas;
        for (int r = 0; r < replicas; ++r) {
          variance += std::pow(prediction[i * replicas + r] - mean, 2) / replicas;
          direct_mse += std::pow(prediction[i * replicas + r] - truth, 2) / (replicas * grid);
        }
        biases[model] += std::pow(mean - truth, 2) / grid;
        variances[model] += variance / grid;
        curves << degree << '\t' << qs[i] << '\t' << truth << '\t' << mean << '\t' << std::sqrt(variance) << '\n';
      }
      check::Near(direct_mse, biases[model] + variances[model], 1e-11, "finite-ensemble bias-variance identity");
      risks[model] = direct_mse + sigma * sigma;
      summary << degree << '\t' << train_errors[model] << '\t' << biases[model] << '\t' << variances[model] << '\t'
              << sigma * sigma << '\t' << risks[model] << '\n';
      std::cout << "  degree " << degree << ": train MSE=" << train_errors[model] << "; bias^2=" << biases[model]
                << "; variance=" << variances[model] << "; noisy-test risk=" << risks[model] << '\n';
    }
    check::Require(train_errors[0] > train_errors[1] && train_errors[1] > train_errors[2],
                   "nested polynomial training error decreases");
    check::Require(biases[0] > biases[1] * 10 && variances[2] > variances[1] * 5, "underfit bias and overfit variance");
    check::Require(risks[1] < risks[0] && risks[1] < risks[2], "moderate degree generalizes best");
    check::Require(bool(curves) && bool(summary), "write polynomial metrics");
    std::cout << "  curves/statistics: " << dir.string() << "/\nAll polynomial regression checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
