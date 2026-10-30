/// @file kernel_ridge_regression.cpp
/// @brief Centered-kernel ridge solves, exact fixed-metric deletions and fold-local validation.
#include <algorithm>
#include <cmath>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/kernel_ridge_regression.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Internal centering, spectral solves and deletion calculations.
namespace {
/// @brief Dense matrix type used for Gram matrices.
using Matrix = linalg::DenseMatrix<double>;
/// @brief Stable arithmetic mean of a nonempty finite vector.
/// @param x Nonempty finite entries.
/// @return Arithmetic mean accumulated in extended precision.
double mean(const std::vector<double>& x) {
  long double s = 0;
  for (double v : x)
    s += v;
  return static_cast<double>(s / x.size());
}
/// @brief Eigenbasis of an optionally centered kernel and response.
struct Basis {
  Matrix              gram;     ///< Raw kernel matrix.
  Matrix              centered; ///< Optionally centered Gram matrix.
  Matrix              vectors;  ///< Eigenvectors, projected off the constant direction when needed.
  std::vector<double> values;   ///< PSD eigenvalues, with numerical zeros discarded.
  std::vector<double> projections;     ///< Eigenvector projections of the centered/raw response.
  std::vector<double> means;           ///< Raw Gram column means, or zero without intercept.
  double              grand{0};        ///< Raw Gram grand mean, or zero without intercept.
  double              y_mean{0};       ///< Response offset, zero without intercept.
  double              tolerance{0};    ///< Relative spectral rank threshold.
  bool                intercept{true}; ///< Whether the constant direction is unpenalized.
  std::size_t         rank{0};         ///< Positive numerical kernel rank, excluding the intercept.
};
/// @brief Center and diagonalize a finite PSD Gram matrix.
/// @param gram Raw n-by-n kernel matrix.
/// @param y n finite outcomes.
/// @param intercept Whether to center both Gram and response.
/// @return Owned reusable basis; eigenvalues below 64*n*epsilon*max(abs(K)) are numerical zero.
/// @throws std::runtime_error Nonfinite kernel/eigendecomposition or materially negative
/// eigenvalues.
Basis decompose(const Matrix& gram, const std::vector<double>& y, bool intercept) {
  const auto n = y.size();
  Basis      b;
  b.gram      = gram;
  b.centered  = gram;
  b.intercept = intercept;
  b.means.assign(n, 0);
  double scale = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      if (!std::isfinite(gram(i, j)))
        throw std::runtime_error("KRR: nonfinite kernel");
      scale = std::max(scale, std::abs(gram(i, j)));
    }
  b.tolerance = 64 * std::numeric_limits<double>::epsilon() * n * scale;
  if (intercept) {
    b.y_mean = mean(y);
    for (std::size_t j = 0; j < n; ++j) {
      long double s = 0;
      for (std::size_t i = 0; i < n; ++i)
        s += gram(i, j);
      b.means[j] = static_cast<double>(s / n);
    }
    b.grand = mean(b.means);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i; j < n; ++j) {
        const double v = static_cast<double>(
            static_cast<long double>(gram(i, j)) - b.means[i] - b.means[j] + b.grand);
        b.centered(i, j) = b.centered(j, i) = v;
      }
  }
  auto eig  = linalg::symmetric_eigen(b.centered);
  b.values  = eig.eigenvalues;
  b.vectors = std::move(eig.eigenvectors);
  b.projections.assign(n, 0);
  for (std::size_t k = 0; k < n; ++k) {
    auto& d = b.values[k];
    if (!std::isfinite(d) || d < -b.tolerance)
      throw std::runtime_error("KRR: kernel is not numerically positive semidefinite");
    if (d <= b.tolerance)
      d = 0;
    else
      ++b.rank;
    if (intercept) {
      long double s = 0;
      for (std::size_t i = 0; i < n; ++i)
        s += b.vectors(i, k);
      const double a = static_cast<double>(s / n);
      for (std::size_t i = 0; i < n; ++i)
        b.vectors(i, k) -= a;
    }
    long double p = 0;
    for (std::size_t i = 0; i < n; ++i)
      p += b.vectors(i, k) * (static_cast<long double>(y[i]) - b.y_mean);
    b.projections[k] = static_cast<double>(p);
  }
  return b;
}
/// @brief One spectral solution and its training/deletion diagnostics.
struct Solution {
  std::vector<double> alpha;    ///< Penalized centered/raw dual solution.
  std::vector<double> fitted;   ///< Full-fit predictions.
  std::vector<double> leverage; ///< Full smoother diagonal.
  std::vector<double> loo;      ///< Fixed-metric deletion predictions.
  double              dof{0};   ///< Full smoother trace including intercept.
  double              mse{0};   ///< Fixed-metric LOO mean squared error, if requested.
};
/// @brief Fit at a nonnegative penalty in a reusable basis.
/// @param b Matching kernel/response basis.
/// @param y Finite outcomes.
/// @param lambda Fixed nonnegative penalty.
/// @return Dual coefficients, full-fit predictions and leverages (no deletion recursion).
/// @throws std::runtime_error Unrepresentable reciprocal/coefficient/prediction.
Solution solve(const Basis& b, const std::vector<double>& y, double lambda) {
  const auto n = y.size();
  Solution   s;
  s.alpha.assign(n, 0);
  s.fitted.assign(n, b.y_mean);
  s.leverage.assign(n, b.intercept ? 1. / n : 0);
  std::vector<long double> alpha(n, 0), fitted(n, b.y_mean), leverage(n, b.intercept ? 1.L / n : 0);
  for (std::size_t k = 0; k < n; ++k) {
    const double d = b.values[k], denom = d + lambda;
    const double inv = lambda == 0 && d == 0 ? 0 : 1 / denom;
    if (!std::isfinite(inv))
      throw std::runtime_error("KRR: unrepresentable penalty reciprocal");
    const double shrink = d * inv, coefficient = b.projections[k] * inv;
    for (std::size_t i = 0; i < n; ++i) {
      const double u = b.vectors(i, k);
      alpha[i] += u * static_cast<long double>(coefficient);
      fitted[i] += u * shrink * static_cast<long double>(b.projections[k]);
      leverage[i] += u * u * static_cast<long double>(shrink);
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    s.alpha[i]    = static_cast<double>(alpha[i]);
    s.fitted[i]   = static_cast<double>(fitted[i]);
    s.leverage[i] = static_cast<double>(leverage[i]);
    s.dof += s.leverage[i];
    if (!std::isfinite(s.alpha[i]) || !std::isfinite(s.fitted[i]))
      throw std::runtime_error("KRR: unrepresentable spectral fit");
  }
  if (b.intercept) {
    const double a = mean(s.alpha);
    for (auto& v : s.alpha)
      v -= a;
  }
  return s;
}
/// @brief Predict from a raw query-kernel row using the matching fitted centering.
/// @param b Fitting basis and means.
/// @param s Fitted solution.
/// @param kernel Raw query kernels in fitting-row order.
/// @return Scalar prediction accumulated in extended precision.
double evaluate(const Basis& b, const Solution& s, const std::vector<double>& kernel) {
  const double offset = b.intercept ? mean(kernel) : 0;
  long double  value  = b.y_mean;
  for (std::size_t j = 0; j < kernel.size(); ++j)
    value += (static_cast<long double>(kernel[j]) - b.means[j] - offset + b.grand) * s.alpha[j];
  return static_cast<double>(value);
}
/// @brief Compute fixed-metric deletions with intercept refitting and no leverage floors.
/// @param b Full-fitting kernel and basis.
/// @param y Full-fitting responses.
/// @param lambda Fixed nonnegative penalty.
/// @return Full fit plus deletion predictions/loss. Positive penalties use alpha_i/B_ii;
/// zero uses explicit deleted spectral fits with the same predictor metric.
/// @throws std::runtime_error Unrepresentable deletion diagonal, prediction or loss.
Solution with_loo(const Basis& b, const std::vector<double>& y, double lambda) {
  auto       s = solve(b, y, lambda);
  const auto n = y.size();
  s.loo.resize(n);
  long double error = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (lambda > 0) {
      long double diagonal = 0;
      for (std::size_t k = 0; k < n; ++k)
        diagonal +=
            static_cast<long double>(b.vectors(i, k)) * b.vectors(i, k) / (b.values[k] + lambda);
      if (!(diagonal > 0) || !std::isfinite(static_cast<double>(diagonal)))
        throw std::runtime_error("KRR: invalid deletion diagonal");
      s.loo[i] = y[i] - static_cast<double>(s.alpha[i] / diagonal);
    } else {
      Matrix                   sub(n - 1, n - 1, 0);
      std::vector<double>      sub_y, query;
      std::vector<std::size_t> indices;
      for (std::size_t j = 0; j < n; ++j)
        if (j != i) {
          indices.push_back(j);
          sub_y.push_back(y[j]);
          query.push_back(b.gram(i, j));
        }
      for (std::size_t j = 0; j < n - 1; ++j)
        for (std::size_t k = 0; k < n - 1; ++k)
          sub(j, k) = b.gram(indices[j], indices[k]);
      const auto basis = decompose(sub, sub_y, b.intercept);
      s.loo[i]         = evaluate(basis, solve(basis, sub_y, 0), query);
    }
    const long double residual = static_cast<long double>(y[i]) - s.loo[i];
    error += residual * residual;
    if (!std::isfinite(s.loo[i]))
      throw std::runtime_error("KRR: nonfinite deletion prediction");
  }
  s.mse = static_cast<double>(error / n);
  if (!std::isfinite(s.mse))
    throw std::runtime_error("KRR: nonfinite deletion loss");
  return s;
}
/// @brief Validate finite controls and PSD polynomial conventions.
/// @param o Caller-supplied options, unchanged.
/// @throws std::invalid_argument Invalid kernel, controls or used explicit candidate values.
void validate_options(const KernelRidgeRegressionOptions& o) {
  if (o.kernel != "rbf" && o.kernel != "linear" && o.kernel != "polynomial")
    throw std::invalid_argument("KRR: kernel must be rbf, linear or polynomial");
  if (!std::isfinite(o.lambda) || !std::isfinite(o.gamma) || !std::isfinite(o.degree) ||
      !std::isfinite(o.coef0) || !o.n_lambda_grid || !o.n_gamma_grid)
    throw std::invalid_argument("KRR: finite controls and nonempty grids required");
  if (o.kernel != "linear" && o.gamma == 0)
    throw std::invalid_argument("KRR: gamma must be positive or negative for selection");
  if (o.kernel == "polynomial" && (o.degree < 1 || o.degree != std::trunc(o.degree) || o.coef0 < 0))
    throw std::invalid_argument(
        "KRR: polynomial needs positive integer degree and nonnegative offset");
  if (o.lambda < 0)
    for (double v : o.lambda_candidates)
      if (!std::isfinite(v) || v < 0)
        throw std::invalid_argument("KRR: invalid lambda candidate");
  if (o.kernel != "linear" && o.gamma < 0)
    for (double v : o.gamma_candidates)
      if (!std::isfinite(v) || v <= 0)
        throw std::invalid_argument("KRR: invalid gamma candidate");
}
} // namespace
KernelRidgeRegression::KernelRidgeRegression(
    const dstruct::DataFrame&    data,
    const std::string&           formula,
    KernelRidgeRegressionOptions options)
    : formula_(formula), options_(std::move(options)) {
  validate_options(options_);
  fit(data);
}
double KernelRidgeRegression::kernel_scalar(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double                     gamma) const {
  long double dot = 0, distance = 0;
  for (std::size_t j = 0; j < a.size(); ++j) {
    dot += static_cast<long double>(a[j]) * b[j];
    const long double d = static_cast<long double>(a[j]) - b[j];
    distance += d * d;
  }
  const long double value = options_.kernel == "linear" ? dot
                            : options_.kernel == "polynomial"
                                ? std::pow(gamma * dot + options_.coef0, options_.degree)
                                : std::exp(-gamma * distance);
  if (!std::isfinite(static_cast<double>(value)))
    throw std::runtime_error("KRR: kernel arithmetic overflow");
  return static_cast<double>(value);
}
std::vector<double> KernelRidgeRegression::kernel_vector(const std::vector<double>& query) const {
  std::vector<double> k(observations_);
  for (std::size_t j = 0; j < observations_; ++j)
    k[j] = kernel_scalar(query, training_X_std_.row(j), gamma_used_);
  const double offset = options_.fit_intercept ? mean(k) : 0;
  for (std::size_t j = 0; j < k.size(); ++j)
    k[j] = static_cast<double>(
        static_cast<long double>(k[j]) - kernel_means_[j] - offset + kernel_grand_mean_);
  return k;
}
void KernelRidgeRegression::fit(const dstruct::DataFrame& data) {
  design_       = formula_.resolve(data, ResponseKind::Numeric, true);
  const auto dm = design_.build_matrix(data, true);
  const auto n = dm.X.rows(), p = dm.X.cols();
  if (n < 2 || p == 0)
    throw std::invalid_argument("KRR: at least two complete rows and one predictor required");
  observations_    = n;
  predictor_names_ = design_.coefficient_names;
  training_y_      = dm.y;
  kept_rows_       = dm.used_row_indices;
  y_mean_          = mean(training_y_);
  for (double v : training_y_)
    if (!std::isfinite(v))
      throw std::invalid_argument("KRR: nonfinite response");
  feature_mean_.assign(p, 0);
  feature_scale_.assign(p, 1);
  feature_min_.assign(p, INFINITY);
  feature_max_.assign(p, -INFINITY);
  for (std::size_t j = 0; j < p; ++j) {
    std::vector<double> column = dm.X.col(j);
    double              scale  = 0;
    for (double v : column) {
      if (!std::isfinite(v))
        throw std::invalid_argument("KRR: nonfinite design");
      feature_min_[j] = std::min(feature_min_[j], v);
      feature_max_[j] = std::max(feature_max_[j], v);
    }
    if (options_.standardize) {
      feature_mean_[j] = mean(column);
      for (double v : column)
        scale = std::max(scale, std::abs(v - feature_mean_[j]));
      if (scale > 0) {
        long double ss = 0;
        for (double v : column) {
          const long double d = (static_cast<long double>(v) - feature_mean_[j]) / scale;
          ss += d * d;
        }
        feature_scale_[j] = scale * std::sqrt(static_cast<double>(ss / n));
      }
    }
    if (!std::isfinite(feature_mean_[j]) || !std::isfinite(feature_scale_[j]) ||
        feature_scale_[j] <= 0)
      throw std::invalid_argument("KRR: unrepresentable predictor scaling");
  }
  training_X_std_ = Matrix(n, p, 0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < p; ++j) {
      training_X_std_(i, j) = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];
      if (!std::isfinite(training_X_std_(i, j)))
        throw std::invalid_argument("KRR: nonfinite standardized design");
    }
  std::vector<double> gammas, lambdas;
  gamma_was_selected_  = options_.kernel != "linear" && options_.gamma < 0;
  lambda_was_selected_ = options_.lambda < 0;
  if (options_.kernel == "linear")
    gammas = {1};
  else if (!gamma_was_selected_)
    gammas = {options_.gamma};
  else if (!options_.gamma_candidates.empty())
    gammas = options_.gamma_candidates;
  else {
    std::vector<double> distances;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        long double d = 0;
        for (std::size_t c = 0; c < p; ++c) {
          const long double v =
              static_cast<long double>(training_X_std_(i, c)) - training_X_std_(j, c);
          d += v * v;
        }
        if (d > 0 && std::isfinite(static_cast<double>(d)))
          distances.push_back(static_cast<double>(d));
      }
    std::sort(distances.begin(), distances.end());
    const double base = distances.empty() ? 1 : 1 / distances[distances.size() / 2];
    for (std::size_t i = 0; i < options_.n_gamma_grid; ++i) {
      const double t = options_.n_gamma_grid == 1 ? 0 : double(i) / (options_.n_gamma_grid - 1);
      gammas.push_back(
          options_.n_gamma_grid == 1 ? base : base * std::exp(std::log(.03) + t * std::log(1000.)));
    }
  }
  if (!lambda_was_selected_)
    lambdas = {options_.lambda};
  else if (!options_.lambda_candidates.empty())
    lambdas = options_.lambda_candidates;
  else
    for (std::size_t i = 0; i < options_.n_lambda_grid; ++i) {
      const double t = options_.n_lambda_grid == 1 ? 0 : double(i) / (options_.n_lambda_grid - 1);
      lambdas.push_back(std::exp(std::log(1e-4) + t * std::log(1e7)));
    }
  for (double v : gammas)
    if (!std::isfinite(v) || v <= 0)
      throw std::invalid_argument("KRR: unrepresentable automatic gamma grid");
  double              best = INFINITY;
  Basis               selected;
  Solution            final;
  std::vector<double> candidate_id, candidate_gamma, candidate_lambda, candidate_mse, candidate_df,
      chosen;
  std::vector<double> selected_curve;
  std::size_t         best_index = 0;
  for (double g : gammas) {
    Matrix K(n, n, 0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i; j < n; ++j)
        K(i, j) = K(j, i) = kernel_scalar(training_X_std_.row(i), training_X_std_.row(j), g);
    const auto          basis = decompose(K, training_y_, options_.fit_intercept);
    std::vector<double> curve;
    bool                this_gamma = false;
    for (double lambda : lambdas) {
      const auto result = with_loo(basis, training_y_, lambda);
      const auto id     = candidate_id.size();
      candidate_id.push_back(id);
      candidate_gamma.push_back(g);
      candidate_lambda.push_back(lambda);
      candidate_mse.push_back(result.mse);
      candidate_df.push_back(result.dof);
      curve.push_back(result.mse);
      if (result.mse < best) {
        best         = result.mse;
        best_index   = id;
        gamma_used_  = g;
        lambda_used_ = lambda;
        selected     = basis;
        final        = result;
        this_gamma   = true;
      }
    }
    if (this_gamma)
      selected_curve = curve;
  }
  if (!std::isfinite(best))
    throw std::runtime_error("KRR: no usable candidate");
  alpha_                = std::move(final.alpha);
  fitted_               = std::move(final.loo);
  training_predictions_ = std::move(final.fitted);
  leverage_             = std::move(final.leverage);
  effective_dof_        = final.dof;
  gram_                 = std::move(selected.gram);
  centered_gram_        = std::move(selected.centered);
  vectors_              = std::move(selected.vectors);
  eigenvalues_          = std::move(selected.values);
  projections_          = std::move(selected.projections);
  kernel_means_         = std::move(selected.means);
  kernel_grand_mean_    = selected.grand;
  rank_                 = selected.rank;
  rank_tolerance_       = selected.tolerance;
  intercept_            = options_.fit_intercept ? y_mean_ : 0;
  if (options_.fit_intercept) {
    long double b = y_mean_;
    for (std::size_t i = 0; i < n; ++i)
      b -= static_cast<long double>(kernel_means_[i]) * alpha_[i];
    intercept_ = static_cast<double>(b);
  }
  if (lambda_was_selected_) {
    lambda_grid_ = lambdas;
    cv_mse_      = selected_curve;
  }
  chosen.assign(candidate_id.size(), 0);
  chosen[best_index] = 1;
  candidates_.add_column("candidate", candidate_id);
  candidates_.add_column("gamma", candidate_gamma);
  candidates_.add_column("lambda", candidate_lambda);
  candidates_.add_column("conditional_loo_mse", candidate_mse);
  candidates_.add_column("df", candidate_df);
  candidates_.add_column("selected", chosen);
}
Matrix KernelRidgeRegression::query_design(const dstruct::DataFrame& data) const {
  design_.validate_categorical_levels(data, "KernelRidgeRegression::predict");
  Matrix x(data.nrows(), predictor_names_.size(), NAN);
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    std::vector<double> row;
    bool                valid = true;
    for (const auto& term : design_.columns) {
      double v = 0;
      if (!term.evaluate(data, i, v) || !std::isfinite(v)) {
        valid = false;
        break;
      }
      row.push_back(v);
    }
    if (valid)
      for (std::size_t j = 0; j < row.size(); ++j) {
        const double v = (row[j] - feature_mean_[j]) / feature_scale_[j];
        if (!std::isfinite(v)) {
          valid = false;
          break;
        }
        x(i, j) = v;
      }
    if (!valid)
      for (std::size_t j = 0; j < x.cols(); ++j)
        x(i, j) = NAN;
  }
  return x;
}
std::vector<double> KernelRidgeRegression::predict(const dstruct::DataFrame& data) const {
  const auto          x = query_design(data);
  std::vector<double> result(data.nrows(), NAN);
  for (std::size_t i = 0; i < x.rows(); ++i) {
    if (!std::isfinite(x(i, 0)))
      continue;
    const auto  k     = kernel_vector(x.row(i));
    long double value = options_.fit_intercept ? y_mean_ : 0;
    for (std::size_t j = 0; j < observations_; ++j)
      value += static_cast<long double>(k[j]) * alpha_[j];
    result[i] = static_cast<double>(value);
    if (!std::isfinite(result[i]))
      throw std::runtime_error("KRR: nonfinite prediction");
  }
  return result;
}
Matrix KernelRidgeRegression::prediction_weights(const dstruct::DataFrame& data) const {
  const auto x = query_design(data);
  Matrix     weights(data.nrows(), observations_, NAN);
  for (std::size_t i = 0; i < x.rows(); ++i) {
    if (!std::isfinite(x(i, 0)))
      continue;
    const auto               k = kernel_vector(x.row(i));
    std::vector<long double> values(
        observations_, options_.fit_intercept ? 1.L / observations_ : 0);
    for (std::size_t a = 0; a < observations_; ++a) {
      const double d = eigenvalues_[a];
      if (lambda_used_ == 0 && d == 0)
        continue;
      long double projection = 0;
      for (std::size_t j = 0; j < observations_; ++j)
        projection += static_cast<long double>(k[j]) * vectors_(j, a);
      const long double coefficient = projection / (d + lambda_used_);
      for (std::size_t j = 0; j < observations_; ++j)
        values[j] += coefficient * vectors_(j, a);
    }
    for (std::size_t j = 0; j < observations_; ++j)
      weights(i, j) = static_cast<double>(values[j]);
  }
  return weights;
}
Matrix KernelRidgeRegression::smoothing_matrix() const {
  Matrix s(observations_, observations_, options_.fit_intercept ? 1. / observations_ : 0);
  for (std::size_t a = 0; a < observations_; ++a) {
    const double d = eigenvalues_[a], shrink = d == 0 ? 0 : d / (d + lambda_used_);
    for (std::size_t i = 0; i < observations_; ++i)
      for (std::size_t j = 0; j < observations_; ++j)
        s(i, j) += vectors_(i, a) * shrink * vectors_(j, a);
  }
  return s;
}
double KernelRidgeRegression::rmse() const {
  long double ss = 0;
  for (std::size_t i = 0; i < observations_; ++i) {
    const long double r = static_cast<long double>(training_y_[i]) - fitted_[i];
    ss += r * r;
  }
  return std::sqrt(static_cast<double>(ss / observations_));
}
double KernelRidgeRegression::r_squared() const {
  long double total = 0, error = 0;
  for (std::size_t i = 0; i < observations_; ++i) {
    const long double d = static_cast<long double>(training_y_[i]) - y_mean_,
                      r = static_cast<long double>(training_y_[i]) - fitted_[i];
    total += d * d;
    error += r * r;
  }
  return total > 0 ? 1 - static_cast<double>(error / total) : 0;
}
dstruct::DataFrame KernelRidgeRegression::training_diagnostics() const {
  std::vector<double> residual, loo_residual;
  for (std::size_t i = 0; i < observations_; ++i) {
    residual.push_back(training_y_[i] - training_predictions_[i]);
    loo_residual.push_back(training_y_[i] - fitted_[i]);
  }
  dstruct::DataFrame out;
  out.add_column("row", std::vector<double>(kept_rows_.begin(), kept_rows_.end()));
  out.add_column("response", training_y_);
  out.add_column("training_prediction", training_predictions_);
  out.add_column("loo_prediction", fitted_);
  out.add_column("leverage", leverage_);
  out.add_column("dual", alpha_);
  out.add_column("training_residual", residual);
  out.add_column("loo_residual", loo_residual);
  return out;
}
dstruct::DataFrame KernelRidgeRegression::spectrum() const {
  std::vector<double> index, shrink;
  for (std::size_t i = 0; i < observations_; ++i) {
    index.push_back(i);
    shrink.push_back(eigenvalues_[i] == 0 ? 0 : eigenvalues_[i] / (eigenvalues_[i] + lambda_used_));
  }
  dstruct::DataFrame out;
  out.add_column("mode", index);
  out.add_column("eigenvalue", eigenvalues_);
  out.add_column("shrinkage", shrink);
  out.add_column("response_projection", projections_);
  return out;
}
dstruct::DataFrame KernelRidgeRegression::diagnostics() const {
  long double ss = 0, norm = 0, residual = 0, alpha_sum = 0;
  for (std::size_t k = 0; k < observations_; ++k) {
    const double d = eigenvalues_[k];
    if (d > 0) {
      const long double a = projections_[k] / (d + lambda_used_);
      norm += d * a * a;
    }
  }
  for (std::size_t i = 0; i < observations_; ++i) {
    const long double r = static_cast<long double>(training_y_[i]) - training_predictions_[i];
    ss += r * r;
    alpha_sum += alpha_[i];
    long double v = lambda_used_ * static_cast<long double>(alpha_[i]) -
                    (training_y_[i] - (options_.fit_intercept ? y_mean_ : 0));
    for (std::size_t j = 0; j < observations_; ++j)
      v += static_cast<long double>(centered_gram_(i, j)) * alpha_[j];
    residual = std::max(residual, std::abs(v));
  }
  dstruct::DataFrame out;
  for (const auto& [name, value] : std::vector<std::pair<std::string, double>>{
           {"observations", double(observations_)},
           {"lambda", lambda_used_},
           {"gamma", gamma_used_},
           {"intercept", intercept_},
           {"rank", double(rank_)},
           {"df", effective_dof_},
           {"training_rmse", std::sqrt(static_cast<double>(ss / observations_))},
           {"loo_rmse", rmse()},
           {"rkhs_norm_squared", static_cast<double>(norm)},
           {"objective", static_cast<double>(ss + lambda_used_ * norm)},
           {"system_residual", static_cast<double>(residual)},
           {"alpha_sum", static_cast<double>(alpha_sum)}})
    out.add_column(name, std::vector<double>{value});
  return out;
}
dstruct::DataFrame KernelRidgeRegression::selection_table() const {
  return candidates_;
}
dstruct::DataFrame KernelRidgeRegression::score(const dstruct::DataFrame& data) const {
  if (data.nrows() == 0 || !data.has_column(design_.response_name) ||
      data.column_type(design_.response_name) != dstruct::DataFrame::ColumnType::Numeric)
    throw std::invalid_argument("KRR score: labelled nonempty frame required");
  const auto          prediction = predict(data);
  std::vector<double> y;
  long double         error = 0, base = 0;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    if (data.is_null(design_.response_name, i))
      throw std::invalid_argument("KRR score: missing response");
    const double v = data.double_at(design_.response_name, i);
    if (!std::isfinite(v) || !std::isfinite(prediction[i]))
      throw std::invalid_argument("KRR score: nonfinite response/prediction");
    y.push_back(v);
    error += std::pow(static_cast<long double>(v) - prediction[i], 2);
    base += std::pow(static_cast<long double>(v) - y_mean_, 2);
  }
  const double center = mean(y);
  long double  total  = 0;
  for (double v : y)
    total += std::pow(static_cast<long double>(v) - center, 2);
  dstruct::DataFrame out;
  out.add_column("rows", std::vector<double>{double(y.size())});
  out.add_column("rmse", std::vector<double>{std::sqrt(static_cast<double>(error / y.size()))});
  out.add_column(
      "r_squared", std::vector<double>{total > 0 ? 1 - static_cast<double>(error / total) : 0});
  out.add_column(
      "baseline_rmse", std::vector<double>{std::sqrt(static_cast<double>(base / y.size()))});
  return out;
}
std::string KernelRidgeRegression::summary() const {
  std::ostringstream out;
  print_summary(out);
  return out.str();
}
void KernelRidgeRegression::print_summary() const {
  print_summary(std::cout);
}
void KernelRidgeRegression::print_summary(std::ostream& out) const {
  out << "Kernel Ridge Regression\nFormula: " << formula_.text() << "\nKernel: " << options_.kernel
      << ", observations: " << observations_ << "\n"
      << std::setprecision(6) << "lambda: " << lambda_used_ << ", gamma: " << gamma_used_
      << ", free intercept: " << options_.fit_intercept
      << "\nEffective df (including intercept): " << effective_dof_
      << "\nFixed-metric LOO R^2: " << r_squared() << ", RMSE: " << rmse()
      << "\nLOO does not refit predictor scaling or hyperparameter selection.\n";
}
graphics::RPlot KernelRidgeRegression::plot_fit(
    const dstruct::DataFrame& data,
    std::size_t               resolution) const {
  if (predictor_names_.size() != 1 || design_.columns[0].kind != ResolvedColumnKind::Numeric ||
      resolution < 2)
    throw std::invalid_argument(
        "KRR plot_fit: plain numeric predictor and at least two locations required");
  const auto&         feature = design_.columns[0].source_column;
  const double        pad     = std::max(1e-6, (feature_max_[0] - feature_min_[0]) * .05);
  std::vector<double> x(resolution);
  for (std::size_t i = 0; i < resolution; ++i)
    x[i] = feature_min_[0] - pad +
           (feature_max_[0] - feature_min_[0] + 2 * pad) * i / (resolution - 1);
  dstruct::DataFrame query;
  query.add_column(feature, x);
  std::vector<double> observed_x, observed_y;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    const auto a = data.optional_double_at(feature, i),
               b = data.optional_double_at(design_.response_name, i);
    if (a && b && std::isfinite(*a) && std::isfinite(*b)) {
      observed_x.push_back(*a);
      observed_y.push_back(*b);
    }
  }
  auto plot = graphics::RPlot::create();
  plot.points(observed_x, observed_y, "observed", {145, 150, 160}, 4)
      .line(x, predict(query), "KRR", {40, 100, 190}, 2);
  return plot.title("Kernel ridge fit").x_label(feature).y_label(design_.response_name);
}
graphics::RPlot KernelRidgeRegression::plot_predicted_vs_actual() const {
  return graphics::RPlot::create()
      .points(training_y_, fitted_, "LOO")
      .title("Fixed-metric LOO predictions")
      .x_label("response")
      .y_label("LOO prediction");
}
graphics::RPlot KernelRidgeRegression::plot_residuals_vs_fitted() const {
  std::vector<double> residual;
  for (std::size_t i = 0; i < observations_; ++i)
    residual.push_back(training_y_[i] - fitted_[i]);
  return graphics::RPlot::create()
      .points(fitted_, residual, "LOO")
      .title("Fixed-metric LOO residuals")
      .x_label("LOO prediction")
      .y_label("LOO residual");
}
graphics::RPlot KernelRidgeRegression::plot_cv_curve() const {
  if (!lambda_was_selected_)
    throw std::invalid_argument("KRR plot_cv_curve: lambda was fixed");
  std::vector<double> axis;
  double              low = INFINITY;
  for (double v : lambda_grid_)
    if (v > 0)
      low = std::min(low, std::log10(v));
  if (!std::isfinite(low))
    low = -4;
  for (double v : lambda_grid_)
    axis.push_back(v > 0 ? std::log10(v) : low - 1);
  const auto best = std::min_element(cv_mse_.begin(), cv_mse_.end()) - cv_mse_.begin();
  return graphics::RPlot::create()
      .line(axis, cv_mse_, "conditional LOO", {40, 100, 190})
      .points({axis[best]}, {cv_mse_[best]}, "selected", {190, 80, 40}, 7)
      .title("Fixed-metric LOO penalty search")
      .x_label("log10 lambda (zero placed at left)")
      .y_label("mean squared error");
}
KernelRidgeCVResult kernel_ridge_cross_validate(
    const dstruct::DataFrame&                        data,
    const std::string&                               formula,
    const std::vector<KernelRidgeRegressionOptions>& candidates,
    const std::vector<std::size_t>&                  ids) {
  if (candidates.empty() || ids.size() != data.nrows())
    throw std::invalid_argument("KRR CV: candidates and original-row fold IDs required");
  const auto            design   = Formula(formula).resolve(data, ResponseKind::Numeric, true);
  const auto            complete = design.build_matrix(data, true);
  const auto&           kept     = complete.used_row_indices;
  std::set<std::size_t> levels;
  for (auto i : kept)
    levels.insert(ids[i]);
  const auto F = levels.size();
  if (F < 2 || *levels.begin() != 0 || *levels.rbegin() != F - 1)
    throw std::invalid_argument("KRR CV: contiguous retained fold IDs required");
  std::vector<double> ci, lambda, gamma, mse, rmse, selected, fc, fold, ntrain, ntest, fold_mse;
  std::vector<double> pc, pf, pr, py, pp;
  std::vector<std::string>         kernels;
  std::vector<std::vector<double>> means, scales;
  double                           best = INFINITY;
  KernelRidgeCVResult              out;
  for (std::size_t c = 0; c < candidates.size(); ++c) {
    const auto& o = candidates[c];
    validate_options(o);
    if (o.lambda < 0 || (o.kernel != "linear" && o.gamma <= 0))
      throw std::invalid_argument("KRR CV: every candidate must have fixed lambda and gamma");
    long double total = 0;
    for (std::size_t f = 0; f < F; ++f) {
      std::vector<std::size_t> training, validation;
      for (auto row : kept)
        (ids[row] == f ? validation : training).push_back(row);
      if (training.size() < 2 || validation.empty())
        throw std::invalid_argument("KRR CV: insufficient rows in a fold");
      const KernelRidgeRegression model(data.take_rows(training), formula, o);
      const auto                  held        = data.take_rows(validation);
      const auto                  predictions = model.predict(held);
      for (std::size_t i = 0; i < validation.size(); ++i) {
        pc.push_back(c);
        pf.push_back(f);
        pr.push_back(validation[i]);
        py.push_back(held.double_at(design.response_name, i));
        pp.push_back(predictions[i]);
      }
      const auto   score = model.score(held);
      const double loss  = score.double_at("rmse", 0) * score.double_at("rmse", 0);
      total += loss * validation.size();
      fc.push_back(c);
      fold.push_back(f);
      ntrain.push_back(model.observations());
      ntest.push_back(validation.size());
      fold_mse.push_back(loss);
      means.push_back(model.feature_mean());
      scales.push_back(model.feature_scale());
      if (model.predictor_names() != design.coefficient_names)
        throw std::invalid_argument("KRR CV: fold design schema differs from complete design");
    }
    const double loss = static_cast<double>(total / kept.size());
    ci.push_back(c);
    kernels.push_back(o.kernel);
    lambda.push_back(o.lambda);
    gamma.push_back(o.kernel == "linear" ? 1 : o.gamma);
    mse.push_back(loss);
    rmse.push_back(std::sqrt(loss));
    if (loss < best) {
      best               = loss;
      out.best_candidate = c;
    }
  }
  selected.assign(candidates.size(), 0);
  selected[out.best_candidate] = 1;
  out.candidates.add_column("candidate", ci);
  out.candidates.add_column("kernel", kernels);
  out.candidates.add_column("lambda", lambda);
  out.candidates.add_column("gamma", gamma);
  out.candidates.add_column("mse", mse);
  out.candidates.add_column("rmse", rmse);
  out.candidates.add_column("selected", selected);
  out.folds.add_column("candidate", fc);
  out.folds.add_column("fold", fold);
  out.folds.add_column("training_rows", ntrain);
  out.folds.add_column("validation_rows", ntest);
  out.folds.add_column("mse", fold_mse);
  for (std::size_t j = 0; j < design.coefficient_names.size(); ++j) {
    std::vector<double> a, b;
    for (std::size_t i = 0; i < means.size(); ++i) {
      a.push_back(means[i][j]);
      b.push_back(scales[i][j]);
    }
    out.folds.add_column("mean_" + std::to_string(j), a);
    out.folds.add_column("scale_" + std::to_string(j), b);
  }
  out.predictions.add_column("candidate", pc);
  out.predictions.add_column("fold", pf);
  out.predictions.add_column("row", pr);
  out.predictions.add_column("observed", py);
  out.predictions.add_column("prediction", pp);
  return out;
}
dstruct::DataFrame kernel_ridge_wave(std::size_t n, std::uint64_t seed, double noise_sd) {
  if (n < 2 || !std::isfinite(noise_sd) || noise_sd <= 0)
    throw std::invalid_argument(
        "KRR wave: at least two rows and finite positive noise SD required");
  std::mt19937_64                        rng(seed);
  std::uniform_real_distribution<double> uniform(0, 1);
  std::normal_distribution<double>       normal(0, noise_sd);
  std::vector<double>                    x, y, truth;
  for (std::size_t i = 0; i < n; ++i) {
    const double v = uniform(rng), m = std::sin(2 * std::acos(-1.) * v);
    x.push_back(v);
    truth.push_back(m);
    y.push_back(m + normal(rng));
  }
  dstruct::DataFrame out;
  out.add_column("x", x);
  out.add_column("y", y);
  out.add_column("truth", truth);
  return out;
}
} // namespace datamunge::stats
