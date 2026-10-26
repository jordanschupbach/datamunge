// CMD: just run-nn cross_validation
// A reusable k-fold/stratified cross-validation harness with leakage checks.
//
// Port of tflow's cross_validation (splits in common/validation.hpp, Legendre
// features in common/polynomial.hpp, QR least squares in common/linear.hpp).
// Select polynomial degree and ridge strength using five folds of 67 noisy
// development observations. Each fold independently fits feature centering
// and scaling, builds Legendre features, and solves augmented ridge least
// squares with nn::qr (float64). Refit the selected setting on all
// development data, then evaluate a separate 401-point test sample once.
// Check balanced folds, reproducible indices, exact coverage, training-only
// statistics and sample-weighted scoring when validation folds have unequal
// sizes. Data and folds come from the same seeded streams as the TF version.
// No files are written.

#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/checks.hpp"
#include "common/linear.hpp"
#include "common/polynomial.hpp"
#include "common/validation.hpp"

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

/// Fits centering/scaling and ridge-regularized Legendre weights on (x, y)
/// only, then predicts at `query`. All [N,1] float64.
class PolynomialFit {
 public:
  PolynomialFit(int degree, double ridge) : degree_(degree), ridge_(ridge) {}

  Tensor Predict(const Tensor& x, const Tensor& y, const Tensor& query) const {
    const Tensor mean   = nn::mean(x, {0, 1});
    const Tensor radius = nn::maximum(nn::max(nn::abs(x - mean), {0, 1}), nn::scalar_like(1e-12, x));
    const Tensor design = ml::PolynomialFeatures((x - mean) / radius, degree_);
    const Tensor query_design = ml::PolynomialFeatures((query - mean) / radius, degree_);
    Tensor       penalty      = nn::zeros({degree_ + 1, degree_ + 1}, DType::Float64);
    for (int j = 1; j <= degree_; ++j) penalty.data_ptr<double>()[j * (degree_ + 1) + j] = std::sqrt(ridge_);
    const Tensor augmented_x = nn::concat({design, penalty}, 0);
    const Tensor augmented_y = nn::concat({y, nn::zeros({degree_ + 1, 1}, DType::Float64)}, 0);
    const Tensor weights     = ml::LeastSquaresQr(augmented_x, augmented_y);
    const auto   xs          = x.to_vector<double>();
    double       host_mean = 0, host_radius = 0;
    for (double v : xs) host_mean += v / xs.size();
    for (double v : xs) host_radius = std::max(host_radius, std::abs(v - host_mean));
    check::Near(mean.item(), host_mean, 1e-12, "centering uses only training rows");
    check::Near(radius.item(), std::max(host_radius, 1e-12), 1e-12, "scaling uses only training rows");
    return nn::matmul(query_design, weights);
  }

 private:
  int    degree_;
  double ridge_;
};

double Mse(const Tensor& prediction, const Tensor& target) {
  check::Require(prediction.shape() == target.shape(), "prediction/target shapes");
  return nn::mean(nn::square(prediction - target)).item();
}

Tensor GatherRows(const Tensor& x, const std::vector<size_t>& rows) {
  return nn::gather(x, nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())), 0);
}

void SplitChecks() {
  const auto       folds = ml::KFold(67, 5, 127), repeat = ml::KFold(67, 5, 127);
  std::vector<int> counts(67);
  for (size_t f = 0; f < folds.size(); ++f) {
    check::Require(folds[f].validation == repeat[f].validation && folds[f].training == repeat[f].training,
                   "seeded reproducible folds");
    check::Require(folds[f].validation.size() == (f < 2 ? 14u : 13u), "balanced remainder folds");
    std::vector<bool> training(67);
    for (auto row : folds[f].training) training[row] = true;
    for (auto row : folds[f].validation) {
      check::Require(!training[row], "disjoint fold partitions");
      ++counts[row];
    }
  }
  for (int count : counts) check::Require(count == 1, "each row held out once");
  std::vector<int> labels;
  for (int i = 0; i < 18; ++i) labels.push_back(i % 3);
  for (const auto& fold : ml::StratifiedKFold(labels, 3, 131)) {
    std::array<int, 3> histogram{};
    for (auto row : fold.validation) ++histogram[labels[row]];
    for (int count : histogram) check::Require(count == 2, "stratified class balance");
  }
  const auto score =
      ml::CrossValidate(folds, [](const auto&, const auto& validation) { return double(validation.size()); });
  check::Near(score.mean, (2 * 14. * 14 + 3 * 13. * 13) / 67, 1e-12, "sample-weighted unequal folds");
  bool rejected = false;
  try {
    ml::StratifiedKFold({0, 0, 0, 1}, 3, 0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected, "insufficient class count rejected");
  std::cout << "  seeded coverage, disjointness, stratification, uneven weighting and invalid splits passed\n";
}

}  // namespace

int main() {
  try {
    std::cout << "== Cross-validation\n";
    SplitChecks();
    std::vector<double>                    xv(67), yv(67);
    std::mt19937                           rng(137);
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::normal_distribution<double>       noise(0, .15);
    for (int i = 0; i < 67; ++i) {
      xv[i] = uniform(rng);
      yv[i] = std::sin(std::acos(-1.) * xv[i]) + noise(rng);
    }
    const Tensor x = nn::from_vector(xv, {67, 1}), y = nn::from_vector(yv, {67, 1});
    const auto   folds      = ml::KFold(67, 5, 127);
    double       best_score = std::numeric_limits<double>::infinity(), best_ridge = 0, linear_score = 0;
    int          best_degree = -1;
    for (int degree : {1, 3, 5, 9})
      for (double ridge : {0., .01, 1.}) {
        PolynomialFit model(degree, ridge);
        const auto    score = ml::CrossValidate(folds, [&](const auto& train, const auto& validation) {
          return Mse(model.Predict(GatherRows(x, train), GatherRows(y, train), GatherRows(x, validation)),
                        GatherRows(y, validation));
        });
        if (degree == 1 && ridge == 0) linear_score = score.mean;
        if (score.mean < best_score) best_score = score.mean, best_ridge = ridge, best_degree = degree;
        std::cout << "  degree=" << degree << ", ridge=" << ridge << ": CV MSE=" << score.mean << '\n';
      }
    check::Require(best_score < linear_score * .4 && best_degree > 1, "CV selects a useful nonlinear model");
    // This sample is created only after selection; its responses never enter CV.
    std::vector<double> txv(401), tyv(401);
    for (int i = 0; i < 401; ++i) {
      txv[i] = -1 + 2. * i / 400;
      tyv[i] = std::sin(std::acos(-1.) * txv[i]) + noise(rng);
    }
    PolynomialFit refit(best_degree, best_ridge);
    const double  test_error = Mse(refit.Predict(x, y, nn::from_vector(txv, {401, 1})), nn::from_vector(tyv, {401, 1}));
    check::Require(test_error < .06, "selected model generalizes to final test set");
    std::cout << "  selected degree=" << best_degree << ", ridge=" << best_ridge << "; refit test MSE=" << test_error
              << "\nAll cross-validation checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
