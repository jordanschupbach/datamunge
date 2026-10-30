/// @file elastic_net.cpp
/// @brief Cyclic penalized least squares with fold-local preprocessing and KKT evidence.
#include <algorithm>
#include <cmath>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/elastic_net.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Private preprocessing, optimization and row-alignment helpers.
namespace {
/// @brief Converts extended arithmetic to a finite double, permitting underflow.
/// @param x Intermediate arithmetic.
/// @return Representable finite value.
/// @throws std::invalid_argument Nonfinite/unrepresentable arithmetic.
double finite_en(long double x) {
  const double v = static_cast<double>(x);
  if (!std::isfinite(v))
    throw std::invalid_argument("ElasticNet: nonfinite/unrepresentable arithmetic");
  return v;
}
/// @brief Owns the numerical preprocessing fitted on one training subset.
struct EnPrepared {
  linalg::DenseMatrix<double> x;     ///< Centred/optionally scaled predictors.
  std::vector<double>         y;     ///< Response centred only when fitting an intercept.
  std::vector<double>         mean;  ///< Fitted offsets, or zeros without an intercept.
  std::vector<double>         scale; ///< Population RMS divisors, or ones if disabled/zero.
  double response_mean{0};           ///< Training response offset, or zero without an intercept.
};
/// @brief Fits preprocessing on exactly the supplied training rows.
/// @param x Finite nonempty raw predictor matrix, possibly zero columns.
/// @param y Matching finite response vector.
/// @param intercept Whether to centre predictors/response.
/// @param standardize Whether to use population RMS about the selected origin.
/// @return Owned preprocessed data and frozen offsets/divisors.
/// @throws std::invalid_argument Unrepresentable centring, scale or normalized entry.
EnPrepared prepare_en(
    const linalg::DenseMatrix<double>& x,
    const std::vector<double>&         y,
    bool                               intercept,
    bool                               standardize) {
  const auto n = x.rows(), p = x.cols();
  EnPrepared out;
  out.x = x;
  out.y = y;
  out.mean.assign(p, 0);
  out.scale.assign(p, 1);
  long double sy = 0;
  for (auto v : y)
    sy += finite_en(v);
  out.response_mean = intercept ? finite_en(sy / n) : 0;
  for (auto& v : out.y)
    v = finite_en(static_cast<long double>(v) - out.response_mean);
  for (std::size_t j = 0; j < p; ++j) {
    long double sum = 0;
    for (std::size_t i = 0; i < n; ++i)
      sum += finite_en(x(i, j));
    out.mean[j]         = intercept ? finite_en(sum / n) : 0;
    long double squares = 0;
    for (std::size_t i = 0; i < n; ++i) {
      out.x(i, j) = finite_en(static_cast<long double>(x(i, j)) - out.mean[j]);
      squares += static_cast<long double>(out.x(i, j)) * out.x(i, j);
    }
    if (standardize) {
      const double rms = finite_en(std::sqrt(squares / n));
      if (rms > 0)
        out.scale[j] = rms;
    }
    for (std::size_t i = 0; i < n; ++i)
      out.x(i, j) = finite_en(static_cast<long double>(out.x(i, j)) / out.scale[j]);
  }
  return out;
}
/// @brief Owns coefficients and measured numerical stopping/optimality evidence.
struct EnSolution {
  std::vector<double> beta;             ///< Penalized-coordinate coefficients.
  std::size_t         iterations{0};    ///< Completed cyclic sweeps.
  bool                converged{false}; ///< Change and scaled KKT conditions both satisfied.
  double              max_change{0};    ///< Largest absolute coefficient change in the final sweep.
  double              kkt{0};           ///< Largest subgradient stationarity residual.
  double              loss{0};          ///< Squared residual sum divided by 2*n.
  double              l1{0};            ///< Sum of absolute penalized-coordinate coefficients.
  double              l2_squared{0};    ///< Sum of squared penalized-coordinate coefficients.
  double              objective{0};     ///< Loss plus the requested elastic-net penalty.
};
/// @brief Recomputes residuals, objective and subgradient violation from current coefficients.
/// @param X Preprocessed predictors.
/// @param y Preprocessed responses.
/// @param alpha L1 mixing proportion.
/// @param lambda Penalty strength.
/// @param s Solution state updated in place, preserving its sweep status.
/// @return Fresh residual vector for drift correction.
/// @throws std::invalid_argument Unrepresentable residual/objective/diagnostic.
std::vector<double> measure_en(
    const linalg::DenseMatrix<double>& X,
    const std::vector<double>&         y,
    double                             alpha,
    double                             lambda,
    EnSolution&                        s) {
  const auto          n = X.rows(), p = X.cols();
  std::vector<double> r(n);
  long double         rss = 0, l1 = 0, l2 = 0, kkt = 0;
  for (std::size_t i = 0; i < n; ++i) {
    long double pred = 0;
    for (std::size_t j = 0; j < p; ++j)
      pred += static_cast<long double>(X(i, j)) * s.beta[j];
    r[i] = finite_en(static_cast<long double>(y[i]) - pred);
    rss += static_cast<long double>(r[i]) * r[i];
  }
  for (std::size_t j = 0; j < p; ++j) {
    const long double b = s.beta[j];
    l1 += std::abs(b);
    l2 += b * b;
    long double score = 0;
    for (std::size_t i = 0; i < n; ++i)
      score += static_cast<long double>(X(i, j)) * r[i];
    const long double gradient  = static_cast<long double>(lambda) * (1 - alpha) * b - score / n;
    const long double threshold = static_cast<long double>(lambda) * alpha;
    const long double violation = b == 0 ? std::max(0.L, std::abs(gradient) - threshold)
                                         : std::abs(gradient + (b > 0 ? threshold : -threshold));
    kkt                         = std::max(kkt, violation);
  }
  s.loss       = finite_en(rss / (2 * n));
  s.l1         = finite_en(l1);
  s.l2_squared = finite_en(l2);
  s.kkt        = finite_en(kkt);
  s.objective  = finite_en(
      rss / (2 * n) + static_cast<long double>(lambda) * (alpha * l1 + (1 - alpha) * l2 / 2));
  return r;
}
/// @brief Cyclic exact-coordinate minimization with actual per-column squared norms.
/// @param X Finite preprocessed predictors with nonempty row set.
/// @param y Matching preprocessed responses.
/// @param alpha L1 mixing proportion in [0,1].
/// @param lambda Finite nonnegative penalty strength.
/// @param beta Warm-start coefficients, owned and updated.
/// @param max_iter Positive sweep limit.
/// @param tol Positive coefficient-change and scale-relative KKT tolerance.
/// @return Owned coefficients/status; finite exhaustion is returned with converged=false.
/// @throws std::invalid_argument Unrepresentable intermediate/result.
/// @par References
/// - Friedman, Hastie and Tibshirani (2010). https://doi.org/10.18637/jss.v033.i01
EnSolution coordinate_descent(
    const linalg::DenseMatrix<double>& X,
    const std::vector<double>&         y,
    double                             alpha,
    double                             lambda,
    std::vector<double>                beta,
    std::size_t                        max_iter,
    double                             tol) {
  const auto n = X.rows(), p = X.cols();
  EnSolution out;
  out.beta              = std::move(beta);
  auto                r = measure_en(X, y, alpha, lambda, out);
  std::vector<double> q(p);
  long double         gradient_scale = 1;
  for (std::size_t j = 0; j < p; ++j) {
    long double norm = 0, score = 0;
    for (std::size_t i = 0; i < n; ++i) {
      norm += static_cast<long double>(X(i, j)) * X(i, j);
      score += static_cast<long double>(X(i, j)) * y[i];
    }
    q[j]           = finite_en(norm / n);
    gradient_scale = std::max(gradient_scale, std::abs(score / n));
  }
  if (p == 0) {
    out.converged = true;
    return out;
  }
  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    out.max_change = 0;
    for (std::size_t j = 0; j < p; ++j) {
      const double old   = out.beta[j];
      long double  score = 0;
      for (std::size_t i = 0; i < n; ++i) {
        r[i] = finite_en(static_cast<long double>(r[i]) + static_cast<long double>(X(i, j)) * old);
        score += static_cast<long double>(X(i, j)) * r[i];
      }
      score /= n;
      const long double threshold = static_cast<long double>(lambda) * alpha;
      const long double soft =
          score > threshold ? score - threshold : (score < -threshold ? score + threshold : 0);
      const long double denom = q[j] + static_cast<long double>(lambda) * (1 - alpha);
      const double      next  = denom > 0 ? finite_en(soft / denom) : 0;
      out.beta[j]             = next;
      for (std::size_t i = 0; i < n; ++i)
        r[i] = finite_en(static_cast<long double>(r[i]) - static_cast<long double>(X(i, j)) * next);
      out.max_change = std::max(out.max_change, std::abs(next - old));
    }
    out.iterations = iter + 1;
    if (out.max_change <= tol) {
      r = measure_en(X, y, alpha, lambda, out);
      if (out.kkt <= tol * gradient_scale) {
        out.converged = true;
        break;
      }
    }
  }
  (void)measure_en(X, y, alpha, lambda, out);
  return out;
}
/// @brief Formats one scalar for the plain-text summary.
/// @param v Finite statistic or NaN.
/// @param precision Fixed decimal count, default six.
/// @return Owned formatted text.
std::string format_stat(double v, int precision = 6) {
  if (std::isnan(v))
    return "NaN";
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision) << v;
  return os.str();
}
/// @brief Aligns compact predictions with the original query rows.
/// @param used_rows Complete-row indices.
/// @param total_rows Original row count.
/// @param compact Predictions in complete-row order.
/// @return Owned vector with NaNs at omitted rows.
std::vector<double> expand_to_full(
    const std::vector<std::size_t>& used_rows,
    std::size_t                     total_rows,
    const std::vector<double>&      compact) {
  std::vector<double> out(total_rows, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < used_rows.size(); ++i)
    out[used_rows[i]] = compact[i];
  return out;
}
/// @brief Copies native numerical evidence into a one-row table.
/// @param s Measured solution.
/// @param alpha Mixing proportion.
/// @param lambda Penalty strength.
/// @return Owned row of numerical diagnostics.
dstruct::DataFrame solution_table(const EnSolution& s, double alpha, double lambda) {
  dstruct::DataFrame out;
  out.add_column("alpha", std::vector<double>{alpha});
  out.add_column("lambda", std::vector<double>{lambda});
  out.add_column("iterations", std::vector<double>{double(s.iterations)});
  out.add_column("converged", std::vector<double>{s.converged ? 1. : 0.});
  out.add_column("max_change", std::vector<double>{s.max_change});
  out.add_column("max_kkt_violation", std::vector<double>{s.kkt});
  out.add_column("loss", std::vector<double>{s.loss});
  out.add_column("l1_norm", std::vector<double>{s.l1});
  out.add_column("l2_squared", std::vector<double>{s.l2_squared});
  out.add_column("objective", std::vector<double>{s.objective});
  return out;
}
} // namespace
ElasticNet::ElasticNet(
    const dstruct::DataFrame& data,
    const std::string&        formula,
    ElasticNetOptions         options)
    : formula_(formula), options_(options) {
  fit(data);
}
ElasticNet::ElasticNet(
    const dstruct::DataFrame& data,
    const std::string&        formula,
    double                    fixed_alpha,
    ElasticNetOptions         options)
    : formula_(formula), options_(options) {
  options_.alpha = fixed_alpha;
  fit(data);
}
void ElasticNet::fit(const dstruct::DataFrame& data) {
  if (!std::isfinite(options_.alpha) || options_.alpha < 0 || options_.alpha > 1 ||
      !std::isfinite(options_.lambda) || options_.max_iter == 0 || !std::isfinite(options_.tol) ||
      options_.tol <= 0)
    throw std::invalid_argument(
        "ElasticNet: finite valid alpha/lambda and positive iteration/tolerance controls required");
  if (!std::isfinite(options_.lambda_min_ratio) || options_.lambda_min_ratio < 0 ||
      options_.lambda_min_ratio > 1)
    throw std::invalid_argument("ElasticNet: lambda_min_ratio must be in [0,1]");
  design_            = formula_.resolve(data, ResponseKind::Numeric);
  const auto matrix  = design_.build_matrix(data, true);
  const auto n       = matrix.X.rows();
  const bool has_int = design_.has_intercept;
  const auto p       = matrix.X.cols() - (has_int ? 1 : 0);
  predictor_names_.assign(
      design_.coefficient_names.begin() + (has_int ? 1 : 0), design_.coefficient_names.end());
  observations_     = n;
  training_y_       = matrix.y;
  kept_row_indices_ = matrix.used_row_indices;
  linalg::DenseMatrix<double> raw(n, p);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < p; ++j)
      raw(i, j) = matrix.X(i, j + (has_int ? 1 : 0));
  const auto prepared = prepare_en(raw, training_y_, has_int, options_.standardize);
  feature_mean_       = prepared.mean;
  feature_scale_      = prepared.scale;
  y_mean_             = prepared.response_mean;
  // Convert only the slope coordinates; the intercept is recovered separately.
  auto unstandardize = [&](const std::vector<double>& b) {
    std::vector<double> out(p);
    for (std::size_t j = 0; j < p; ++j)
      out[j] = finite_en(static_cast<long double>(b[j]) / feature_scale_[j]);
    return out;
  };
  EnSolution selected;
  lambda_was_selected_ = options_.lambda < 0;
  if (!lambda_was_selected_) {
    lambda_used_ = options_.lambda;
    selected     = coordinate_descent(
        prepared.x,
        prepared.y,
        options_.alpha,
        lambda_used_,
        std::vector<double>(p, 0),
        options_.max_iter,
        options_.tol);
  } else {
    if (options_.n_lambda == 0 || options_.cv_folds < 2 || options_.cv_folds > n)
      throw std::invalid_argument(
          "ElasticNet: auto path needs n_lambda>=1 and 2<=cv_folds<=complete rows");
    long double largest = 0;
    for (std::size_t j = 0; j < p; ++j) {
      long double score = 0;
      for (std::size_t i = 0; i < n; ++i)
        score += static_cast<long double>(prepared.x(i, j)) * prepared.y[i];
      largest = std::max(largest, std::abs(score / n));
    }
    const double maximum =
        largest > 0 ? finite_en(largest / (options_.alpha > 0 ? options_.alpha : 1e-3)) : 1;
    const double ratio =
        options_.lambda_min_ratio > 0 ? options_.lambda_min_ratio : (n < p ? .01 : .0001);
    lambda_path_.resize(options_.n_lambda);
    for (std::size_t k = 0; k < options_.n_lambda; ++k) {
      const double t  = options_.n_lambda == 1 ? 0 : double(k) / (options_.n_lambda - 1);
      lambda_path_[k] = finite_en(std::exp(std::log(maximum) + t * std::log(ratio)));
      if (!(lambda_path_[k] > 0))
        throw std::invalid_argument("ElasticNet: unrepresentable positive lambda path");
    }
    std::vector<EnSolution> full;
    full.reserve(options_.n_lambda);
    path_coefficients_ = linalg::DenseMatrix<double>(options_.n_lambda, p);
    std::vector<double> warm(p, 0), path_iterations, path_converged, path_kkt, path_objective;
    for (std::size_t k = 0; k < options_.n_lambda; ++k) {
      full.push_back(coordinate_descent(
          prepared.x,
          prepared.y,
          options_.alpha,
          lambda_path_[k],
          std::move(warm),
          options_.max_iter,
          options_.tol));
      warm                = full.back().beta;
      const auto original = unstandardize(warm);
      for (std::size_t j = 0; j < p; ++j)
        path_coefficients_(k, j) = original[j];
      path_iterations.push_back(full.back().iterations);
      path_converged.push_back(full.back().converged ? 1 : 0);
      path_kkt.push_back(full.back().kkt);
      path_objective.push_back(full.back().objective);
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    random::SplitMix64 rng(options_.seed);
    for (std::size_t i = n; i-- > 1;) {
      const auto j = static_cast<std::size_t>(rng.next_u64() % (i + 1));
      std::swap(order[i], order[j]);
    }
    fold_ids_.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      fold_ids_[order[i]] = i % options_.cv_folds;
    std::vector<long double> sse(options_.n_lambda, 0);
    std::vector<double>      f_id, f_lambda, f_train, f_test, f_mse, f_iter, f_conv, f_kkt;
    for (std::size_t f = 0; f < options_.cv_folds; ++f) {
      std::vector<std::size_t> train, test;
      for (std::size_t i = 0; i < n; ++i)
        (fold_ids_[i] == f ? test : train).push_back(i);
      linalg::DenseMatrix<double> tx(train.size(), p);
      std::vector<double>         ty(train.size());
      for (std::size_t i = 0; i < train.size(); ++i) {
        ty[i] = training_y_[train[i]];
        for (std::size_t j = 0; j < p; ++j)
          tx(i, j) = raw(train[i], j);
      }
      const auto          prep = prepare_en(tx, ty, has_int, options_.standardize);
      std::vector<double> b(p, 0);
      for (std::size_t k = 0; k < options_.n_lambda; ++k) {
        const auto fit = coordinate_descent(
            prep.x,
            prep.y,
            options_.alpha,
            lambda_path_[k],
            std::move(b),
            options_.max_iter,
            options_.tol);
        b                 = fit.beta;
        long double error = 0;
        for (auto row : test) {
          long double prediction = prep.response_mean;
          for (std::size_t j = 0; j < p; ++j)
            prediction +=
                (static_cast<long double>(raw(row, j)) - prep.mean[j]) / prep.scale[j] * b[j];
          const long double e = training_y_[row] - prediction;
          error += e * e;
        }
        sse[k] += error;
        f_id.push_back(f);
        f_lambda.push_back(lambda_path_[k]);
        f_train.push_back(train.size());
        f_test.push_back(test.size());
        f_mse.push_back(finite_en(error / test.size()));
        f_iter.push_back(fit.iterations);
        f_conv.push_back(fit.converged ? 1 : 0);
        f_kkt.push_back(fit.kkt);
      }
    }
    cv_mse_path_.resize(options_.n_lambda);
    for (std::size_t k = 0; k < options_.n_lambda; ++k)
      cv_mse_path_[k] = finite_en(sse[k] / n);
    best_lambda_index_ = static_cast<std::size_t>(
        std::min_element(cv_mse_path_.begin(), cv_mse_path_.end()) - cv_mse_path_.begin());
    lambda_used_ = lambda_path_[best_lambda_index_];
    selected     = full[best_lambda_index_];
    path_diagnostics_.add_column("lambda", lambda_path_);
    path_diagnostics_.add_column("cv_mse", cv_mse_path_);
    path_diagnostics_.add_column("iterations", path_iterations);
    path_diagnostics_.add_column("converged", path_converged);
    path_diagnostics_.add_column("max_kkt_violation", path_kkt);
    path_diagnostics_.add_column("objective", path_objective);
    cv_fold_results_.add_column("fold", f_id);
    cv_fold_results_.add_column("lambda", f_lambda);
    cv_fold_results_.add_column("n_train", f_train);
    cv_fold_results_.add_column("n_test", f_test);
    cv_fold_results_.add_column("mse", f_mse);
    cv_fold_results_.add_column("iterations", f_iter);
    cv_fold_results_.add_column("converged", f_conv);
    cv_fold_results_.add_column("max_kkt_violation", f_kkt);
  }
  coefficients_      = unstandardize(selected.beta);
  iterations_        = selected.iterations;
  converged_         = selected.converged;
  diagnostics_       = solution_table(selected, options_.alpha, lambda_used_);
  long double offset = y_mean_;
  for (std::size_t j = 0; j < p; ++j)
    offset -= static_cast<long double>(feature_mean_[j]) * coefficients_[j];
  intercept_ = has_int ? finite_en(offset) : 0;
  fitted_.resize(n);
  residuals_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    long double v = intercept_;
    for (std::size_t j = 0; j < p; ++j)
      v += static_cast<long double>(raw(i, j)) * coefficients_[j];
    fitted_[i]    = finite_en(v);
    residuals_[i] = finite_en(static_cast<long double>(training_y_[i]) - v);
  }
}
dstruct::DataFrame ElasticNet::coefficient_table() const {
  std::vector<double> penalized(coefficients_.size());
  for (std::size_t j = 0; j < penalized.size(); ++j)
    penalized[j] = finite_en(static_cast<long double>(coefficients_[j]) * feature_scale_[j]);
  dstruct::DataFrame out;
  out.add_column("predictor", predictor_names_);
  out.add_column("coefficient", coefficients_);
  out.add_column("penalty_coefficient", penalized);
  out.add_column("center", feature_mean_);
  out.add_column("scale", feature_scale_);
  return out;
}
std::size_t ElasticNet::non_zero_coefficients() const {
  return static_cast<std::size_t>(
      std::count_if(coefficients_.begin(), coefficients_.end(), [](const double c) {
        return std::fabs(c) > 1e-10;
      }));
}

double ElasticNet::r_squared() const {
  long double mean = 0;
  for (auto y : training_y_)
    mean += y;
  mean /= training_y_.size();
  long double tss = 0, rss = 0;
  for (std::size_t i = 0; i < training_y_.size(); ++i) {
    const long double d = training_y_[i] - mean;
    tss += d * d;
    rss += static_cast<long double>(residuals_[i]) * residuals_[i];
  }
  return tss > 0 ? finite_en(1 - rss / tss)
                 : (rss == 0 ? 1 : std::numeric_limits<double>::quiet_NaN());
}
double ElasticNet::rmse() const {
  long double sse = 0;
  for (auto r : residuals_)
    sse += static_cast<long double>(r) * r;
  return finite_en(std::sqrt(sse / residuals_.size()));
}

std::string ElasticNet::summary() const {
  std::ostringstream out;
  out << "Call:\nelasticNet(formula = " << formula_.text()
      << ", alpha = " << format_stat(options_.alpha, 3) << ")\n\n";

  if (lambda_was_selected_)
    out << "Lambda: " << format_stat(lambda_used_, 6) << "  (selected by " << options_.cv_folds
        << "-fold cross-validation over " << lambda_path_.size() << " values)\n";
  else
    out << "Lambda: " << format_stat(lambda_used_, 6) << "  (fixed)\n";

  out << "Converged: " << (converged_ ? "yes" : "no") << " after " << iterations_
      << " sweeps; maximum KKT violation: " << diagnostics_.double_at("max_kkt_violation", 0)
      << "\n";
  out << "Non-zero coefficients: " << non_zero_coefficients() << " of " << predictor_names_.size()
      << "\n\n";

  out << "Coefficients:\n";
  if (design_.has_intercept)
    out << "  (Intercept): " << format_stat(intercept_) << "\n";
  for (std::size_t j = 0; j < predictor_names_.size(); ++j)
    out << "  " << predictor_names_[j] << ": " << format_stat(coefficients_[j]) << "\n";

  out << "\nR-squared: " << format_stat(r_squared(), 4) << "\n";
  out << "RMSE: " << format_stat(rmse(), 4) << "  (" << observations_ << " observations)\n";

  return out.str();
}

void ElasticNet::print_summary(std::ostream& os) const {
  os << summary();
}

void ElasticNet::print_summary() const {
  print_summary(std::cout);
}

std::vector<double> ElasticNet::predict(const dstruct::DataFrame& newdata) const {
  if (newdata.nrows() == 0)
    return {};
  design_.validate_categorical_levels(newdata, "ElasticNet::predict");
  const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

  const bool        has_int = design_.has_intercept;
  const std::size_t p       = predictor_names_.size();

  std::vector<double> compact(dm.X.rows());
  for (std::size_t i = 0; i < dm.X.rows(); ++i) {
    double s = intercept_;
    for (std::size_t j = 0; j < p; ++j)
      s = finite_en(
          static_cast<long double>(s) +
          finite_en(dm.X(i, j + (has_int ? 1 : 0))) * static_cast<long double>(coefficients_[j]));
    compact[i] = s;
  }
  return expand_to_full(dm.used_row_indices, newdata.nrows(), compact);
}

dstruct::DataFrame ElasticNet::score(const dstruct::DataFrame& newdata) const {
  design_.validate_categorical_levels(newdata, "ElasticNet::score");
  const auto  m    = design_.build_matrix(newdata, true);
  const auto  n    = m.X.rows();
  long double mean = 0, baseline = 0;
  for (auto y : training_y_)
    baseline += y;
  baseline /= training_y_.size();
  for (auto y : m.y)
    mean += finite_en(y);
  mean /= n;
  long double ss = 0, absolute = 0, tss = 0, base = 0;
  for (std::size_t i = 0; i < n; ++i) {
    long double prediction = intercept_;
    for (std::size_t j = 0; j < coefficients_.size(); ++j)
      prediction += static_cast<long double>(finite_en(m.X(i, j + (has_intercept() ? 1 : 0)))) *
                    coefficients_[j];
    const long double e = finite_en(static_cast<long double>(m.y[i]) - prediction),
                      d = m.y[i] - mean, b = m.y[i] - baseline;
    ss += e * e;
    absolute += std::abs(e);
    tss += d * d;
    base += b * b;
  }
  const double r2 =
      tss > 0 ? finite_en(1 - ss / tss) : (ss == 0 ? 1 : std::numeric_limits<double>::quiet_NaN());
  dstruct::DataFrame out;
  out.add_column("n_input", std::vector<double>{double(newdata.nrows())});
  out.add_column("n_scored", std::vector<double>{double(n)});
  out.add_column("rmse", std::vector<double>{finite_en(std::sqrt(ss / n))});
  out.add_column("mae", std::vector<double>{finite_en(absolute / n)});
  out.add_column("r_squared", std::vector<double>{r2});
  out.add_column("mean_baseline_rmse", std::vector<double>{finite_en(std::sqrt(base / n))});
  return out;
}

graphics::RPlot ElasticNet::plot_coefficient_path() const {
  if (!lambda_was_selected_)
    throw std::invalid_argument(
        "ElasticNet::plot_coefficient_path: no path available for a fixed-lambda fit");

  std::vector<double> log_lambda(lambda_path_.size());
  for (std::size_t k = 0; k < lambda_path_.size(); ++k)
    log_lambda[k] = std::log10(lambda_path_[k]);

  /// @brief Recycled categorical colours for full-data predictor paths.
  const graphics::RGB colors[] = {
      {37, 99, 235},
      {220, 38, 38},
      {22, 163, 74},
      {217, 119, 6},
      {124, 58, 237},
      {8, 145, 178},
  };

  auto plot = graphics::RPlot::create();
  for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
    std::vector<double> trace(lambda_path_.size());
    for (std::size_t k = 0; k < lambda_path_.size(); ++k)
      trace[k] = path_coefficients_(k, j);
    plot.line(
        log_lambda, trace, predictor_names_[j], colors[j % (sizeof(colors) / sizeof(colors[0]))]);
  }
  plot.title("Coefficient Path").x_label("log10(lambda)").y_label("Coefficient");
  return plot;
}

graphics::RPlot ElasticNet::plot_cv_curve() const {
  if (!lambda_was_selected_)
    throw std::invalid_argument(
        "ElasticNet::plot_cv_curve: no cross-validation curve for a fixed-lambda fit");

  std::vector<double> log_lambda(lambda_path_.size());
  for (std::size_t k = 0; k < lambda_path_.size(); ++k)
    log_lambda[k] = std::log10(lambda_path_[k]);
  const auto& finite_mse = cv_mse_path_;

  auto plot = graphics::RPlot::create();
  plot.line(log_lambda, finite_mse, "CV mean squared error", {37, 99, 235});
  if (std::isfinite(cv_mse_path_[best_lambda_index_]))
    plot.points(
        {std::log10(lambda_path_[best_lambda_index_])},
        {cv_mse_path_[best_lambda_index_]},
        "selected lambda",
        {220, 38, 38},
        8.0);
  plot.title("Cross-Validation Curve").x_label("log10(lambda)").y_label("CV mean squared error");
  return plot;
}

graphics::RPlot ElasticNet::plot_predicted_vs_actual() const {
  auto plot = graphics::RPlot::create();
  plot.points(training_y_, fitted_, "predictions");
  plot.title("Predicted vs Actual").x_label("Actual").y_label("Predicted");
  return plot;
}

graphics::RPlot ElasticNet::plot_residuals_vs_fitted() const {
  auto plot = graphics::RPlot::create();
  plot.points(fitted_, residuals_, "residuals");
  plot.title("Residuals vs Fitted").x_label("Fitted values").y_label("Residuals");
  return plot;
}

Ridge::Ridge(const dstruct::DataFrame& data, const std::string& formula, RidgeOptions options)
    // Forward shared controls while forcing the ridge mixing proportion.
    : ElasticNet(data, formula, 0.0, [&] {
        ElasticNetOptions o;
        o.lambda           = options.lambda;
        o.n_lambda         = options.n_lambda;
        o.cv_folds         = options.cv_folds;
        o.max_iter         = options.max_iter;
        o.tol              = options.tol;
        o.seed             = options.seed;
        o.standardize      = options.standardize;
        o.lambda_min_ratio = options.lambda_min_ratio;
        return o;
      }()) {}

Lasso::Lasso(const dstruct::DataFrame& data, const std::string& formula, LassoOptions options)
    // Forward shared controls while forcing the lasso mixing proportion.
    : ElasticNet(data, formula, 1.0, [&] {
        ElasticNetOptions o;
        o.lambda           = options.lambda;
        o.n_lambda         = options.n_lambda;
        o.cv_folds         = options.cv_folds;
        o.max_iter         = options.max_iter;
        o.tol              = options.tol;
        o.seed             = options.seed;
        o.standardize      = options.standardize;
        o.lambda_min_ratio = options.lambda_min_ratio;
        return o;
      }()) {}

} // namespace datamunge::stats
