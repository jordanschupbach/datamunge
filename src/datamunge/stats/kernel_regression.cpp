/// @file kernel_regression.cpp
/// @brief Stable radial local means, complete-coverage bandwidth selection and diagnostics.
#include <algorithm>
#include <cmath>
#include <datamunge/stats/kernel_regression.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Internal metric summaries and finite-curve plotting.
namespace {
/// @brief Quiet missing-result marker used for unsupported predictions.
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
/// @brief Positive infinity marks ineligible cross-validation objectives.
constexpr double infinity = std::numeric_limits<double>::infinity();
/// @brief Conditional regression errors on finite-prediction rows.
struct Errors {
  std::size_t n;             ///< Covered response count.
  double      mse;           ///< Covered mean square error, NaN for no coverage.
  double      rmse;          ///< Covered root mean square error.
  double      mae;           ///< Covered mean absolute error.
  double      r2;            ///< Covered R-squared; NaN for constant responses/no coverage.
  double      baseline_rmse; ///< Error of the training-only response mean on the same rows.
};
/// @brief Computes errors and a baseline on exactly the same covered evaluation subset.
/// @param actual Finite responses.
/// @param predicted Same-length predictions, NaN permitted.
/// @param baseline Fixed training-only mean.
/// @return Owned errors, with NaN unavailable metrics.
Errors
errors(const std::vector<double>& actual, const std::vector<double>& predicted, double baseline) {
  std::size_t count = 0;
  long double mean  = 0;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::isfinite(predicted[i])) {
      mean += actual[i];
      ++count;
    }
  if (!count)
    return {0, missing, missing, missing, missing, missing};
  mean /= count;
  bool   constant   = true;
  double first      = 0;
  bool   have_first = false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::isfinite(predicted[i])) {
      if (!have_first) {
        first      = actual[i];
        have_first = true;
      } else if (actual[i] != first)
        constant = false;
    }
  long double sse = 0, absolute = 0, tss = 0, baseline_sse = 0;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::isfinite(predicted[i])) {
      const long double e = static_cast<long double>(actual[i]) - predicted[i],
                        d = actual[i] - mean, b = static_cast<long double>(actual[i]) - baseline;
      sse += e * e;
      absolute += std::abs(e);
      tss += d * d;
      baseline_sse += b * b;
    }
  return {
      count,
      static_cast<double>(sse / count),
      static_cast<double>(std::sqrt(sse / count)),
      static_cast<double>(absolute / count),
      !constant && tss > 0 ? static_cast<double>(1 - sse / tss) : missing,
      static_cast<double>(std::sqrt(baseline_sse / count))};
}
/// @brief Validates a source column independently of any particular row's null values.
/// @param data Query schema.
/// @param name Required source name.
/// @param categorical Whether a string column is required, otherwise numeric.
void require_source(
    const dstruct::DataFrame& data,
    const std::string&        name,
    bool                      categorical = false) {
  const auto expected = categorical ? dstruct::DataFrame::ColumnType::String
                                    : dstruct::DataFrame::ColumnType::Numeric;
  if (!data.has_column(name) || data.column_type(name) != expected)
    throw std::invalid_argument(
        "KernelRegression: missing or incompatible predictor column: " + name);
}
/// @brief Validates numeric columns referenced inside an arithmetic expression.
/// @param data Query schema.
/// @param expression Expression node; null nodes are ignored.
void require_expression(
    const dstruct::DataFrame&        data,
    const std::shared_ptr<ExprNode>& expression) {
  if (!expression)
    return;
  if (expression->op == ExprOp::Column)
    require_source(data, expression->column);
  require_expression(data, expression->left);
  require_expression(data, expression->right);
}
/// @brief Recursively validates the source schema of a frozen resolved design column.
/// @param data Query schema.
/// @param column Resolved training design column.
void require_design_column(const dstruct::DataFrame& data, const ResolvedColumn& column) {
  switch (column.kind) {
    case ResolvedColumnKind::Numeric:
    case ResolvedColumnKind::Function:
      require_source(data, column.source_column);
      break;
    case ResolvedColumnKind::CategoricalDummy:
      require_source(data, column.source_column, true);
      break;
    case ResolvedColumnKind::BSpline:
      for (const auto& name : column.bspline_columns)
        require_source(data, name);
      break;
    case ResolvedColumnKind::Interaction:
      for (const auto& part : column.interaction_parts)
        require_design_column(data, part);
      break;
    case ResolvedColumnKind::Expression:
      require_expression(data, column.expression);
      break;
  }
}
/// @brief Human-readable kernel label.
/// @param kernel Valid kernel.
/// @return Static borrowed label.
const char* kernel_name(KernelRegressionKernel kernel) {
  switch (kernel) {
    case KernelRegressionKernel::Gaussian:
      return "gaussian";
    case KernelRegressionKernel::Epanechnikov:
      return "epanechnikov";
    case KernelRegressionKernel::Uniform:
      return "uniform";
    case KernelRegressionKernel::Triangular:
      return "triangular";
  }
  return "invalid";
}
/// @brief Adds finite contiguous curve runs so unsupported windows remain gaps.
/// @param plot Editable destination.
/// @param x Ordered abscissae.
/// @param y Matching finite/NaN ordinates.
/// @param label Label attached to the first finite run only.
void add_curve(
    graphics::RPlot&           plot,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::string&         label) {
  bool named = false;
  for (std::size_t first = 0; first < x.size();) {
    if (!std::isfinite(y[first])) {
      ++first;
      continue;
    }
    std::size_t last = first + 1;
    while (last < x.size() && std::isfinite(y[last]))
      ++last;
    std::vector<double> xx(x.begin() + first, x.begin() + last),
        yy(y.begin() + first, y.begin() + last);
    if (last - first > 1)
      plot.lines(xx, yy, named ? "" : label, {220, 38, 38});
    else
      plot.points(xx, yy, named ? "" : label, {220, 38, 38});
    named = true;
    first = last;
  }
}
} // namespace
KernelRegression::KernelRegression(
    const dstruct::DataFrame& data,
    const std::string&        formula,
    KernelRegressionOptions   options)
    : formula_(formula), options_(std::move(options)) {
  fit(data);
}

void KernelRegression::fit(const dstruct::DataFrame& data) {
  if (!std::isfinite(options_.bandwidth) || options_.bandwidth == 0 || options_.n_bandwidth == 0)
    throw std::invalid_argument(
        "KernelRegression: finite nonzero bandwidth and positive grid size required");
  switch (options_.kernel) {
    case KernelRegressionKernel::Gaussian:
    case KernelRegressionKernel::Epanechnikov:
    case KernelRegressionKernel::Uniform:
    case KernelRegressionKernel::Triangular:
      break;
    default:
      throw std::invalid_argument("KernelRegression: invalid kernel");
  }
  design_           = formula_.resolve(data, ResponseKind::Numeric, true);
  const auto matrix = design_.build_matrix(data, true);
  const auto n = matrix.X.rows(), p = matrix.X.cols();
  if (n < 2 || p == 0)
    throw std::invalid_argument(
        "KernelRegression: at least two complete rows and one predictor required");
  observations_             = n;
  predictor_names_          = design_.coefficient_names;
  kept_rows_                = matrix.used_row_indices;
  training_y_               = matrix.y;
  long double response_mean = 0;
  for (double y : training_y_) {
    if (!std::isfinite(y))
      throw std::invalid_argument("KernelRegression: retained responses must be finite");
    response_mean += static_cast<long double>(y) / n;
  }
  training_mean_ = static_cast<double>(response_mean);
  feature_mean_.assign(p, 0);
  feature_scale_.assign(p, 1);
  feature_min_.assign(p, infinity);
  feature_max_.assign(p, -infinity);
  for (std::size_t j = 0; j < p; ++j) {
    long double mean = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const double x = matrix.X(i, j);
      if (!std::isfinite(x))
        throw std::invalid_argument("KernelRegression: retained predictors must be finite");
      mean += static_cast<long double>(x) / n;
      feature_min_[j] = std::min(feature_min_[j], x);
      feature_max_[j] = std::max(feature_max_[j], x);
    }
    if (options_.standardize) {
      if (feature_min_[j] == feature_max_[j]) {
        feature_mean_[j] = feature_min_[j];
        continue;
      }
      long double ss = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const long double d = matrix.X(i, j) - mean;
        ss += d * d;
      }
      feature_mean_[j]  = static_cast<double>(mean);
      feature_scale_[j] = static_cast<double>(std::sqrt(ss / n));
      if (!std::isfinite(feature_scale_[j]) || feature_scale_[j] <= 0)
        throw std::invalid_argument("KernelRegression: unrepresentable predictor scale");
    }
  }
  training_X_std_ = linalg::DenseMatrix<double>(n, p);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < p; ++j) {
      const double x = static_cast<double>(
          (static_cast<long double>(matrix.X(i, j)) - feature_mean_[j]) / feature_scale_[j]);
      if (!std::isfinite(x))
        throw std::invalid_argument("KernelRegression: unrepresentable standardized design");
      training_X_std_(i, j) = x;
    }
  bandwidth_was_selected_ = options_.bandwidth < 0;
  if (bandwidth_was_selected_) {
    bandwidth_grid_ = options_.bandwidth_candidates;
    if (bandwidth_grid_.empty()) {
      const double low  = .01 * std::sqrt(static_cast<double>(p)),
                   high = 2 * std::sqrt(static_cast<double>(p));
      for (std::size_t k = 0; k < options_.n_bandwidth; ++k) {
        const double t =
            options_.n_bandwidth == 1 ? 0 : static_cast<double>(k) / (options_.n_bandwidth - 1);
        bandwidth_grid_.push_back(std::exp(std::log(low) + t * (std::log(high) - std::log(low))));
      }
    }
    for (double h : bandwidth_grid_)
      if (!std::isfinite(h) || h <= 0)
        throw std::invalid_argument("KernelRegression: CV candidates must be positive and finite");
    cv_mse_.assign(bandwidth_grid_.size(), infinity);
    cv_conditional_mse_.assign(bandwidth_grid_.size(), missing);
    cv_counts_.assign(bandwidth_grid_.size(), 0);
    std::vector<double> prediction(n);
    std::vector<double> query(p);
    for (std::size_t k = 0; k < bandwidth_grid_.size(); ++k) {
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j)
          query[j] = training_X_std_(i, j);
        prediction[i] = local_fit(query, bandwidth_grid_[k], i).prediction;
      }
      const auto e           = errors(training_y_, prediction, training_mean_);
      cv_counts_[k]          = e.n;
      cv_conditional_mse_[k] = e.mse;
      if (e.n == n && std::isfinite(e.mse))
        cv_mse_[k] = e.mse;
    }
    const auto best = std::min_element(cv_mse_.begin(), cv_mse_.end());
    if (!std::isfinite(*best))
      throw std::runtime_error(
          "KernelRegression: no bandwidth has complete finite leave-one-out coverage; supply a "
          "broader candidate grid or a fixed bandwidth and inspect coverage");
    bandwidth_used_ = bandwidth_grid_[static_cast<std::size_t>(best - cv_mse_.begin())];
  } else
    bandwidth_used_ = options_.bandwidth;
  std::vector<double> query(p);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < p; ++j)
      query[j] = training_X_std_(i, j);
    const auto full = local_fit(query, bandwidth_used_), loo = local_fit(query, bandwidth_used_, i);
    full_fitted_.push_back(full.prediction);
    fitted_.push_back(loo.prediction);
    leverage_.push_back(full.weights[i]);
    effective_n_.push_back(full.effective_n);
    loo_effective_n_.push_back(loo.effective_n);
  }
}
KernelRegression::LocalFit
KernelRegression::local_fit(const std::vector<double>& query, double h, std::size_t exclude) const {
  const auto               n = observations_, p = predictor_names_.size();
  LocalFit                 out{std::vector<double>(n, missing), missing, missing, infinity, 0};
  std::vector<long double> distance(n), weight(n, 0);
  std::size_t              anchor  = n;
  long double              nearest = std::numeric_limits<long double>::infinity();
  for (std::size_t i = 0; i < n; ++i)
    if (i != exclude) {
      long double d = 0;
      for (std::size_t j = 0; j < p; ++j)
        d = std::hypot(d, static_cast<long double>(query[j]) - training_X_std_(i, j));
      distance[i] = d;
      if (d < nearest) {
        nearest = d;
        anchor  = i;
      }
    }
  out.nearest_distance = static_cast<double>(nearest);
  if (options_.kernel == KernelRegressionKernel::Gaussian) {
    long double maximum = -std::numeric_limits<long double>::infinity();
    for (std::size_t i = 0; i < n; ++i)
      if (i != exclude) {
        long double norm_delta = 0, dot_delta = 0;
        for (std::size_t j = 0; j < p; ++j) {
          const long double a = training_X_std_(anchor, j), b = training_X_std_(i, j),
                            delta = b - a;
          norm_delta += delta * (a + b);
          dot_delta += delta * query[j];
        }
        const long double log_weight =
            -(norm_delta - 2 * dot_delta) / static_cast<long double>(h) / h / 2;
        if (!std::isfinite(log_weight))
          throw std::invalid_argument(
              "KernelRegression: relative Gaussian weight geometry is not representable");
        weight[i] = log_weight;
        maximum   = std::max(maximum, log_weight);
      }
    for (std::size_t i = 0; i < n; ++i)
      if (i != exclude)
        weight[i] = std::exp(weight[i] - maximum);
  } else {
    for (std::size_t i = 0; i < n; ++i)
      if (i != exclude) {
        const long double u = distance[i] / h;
        if (u > 1)
          continue;
        switch (options_.kernel) {
          case KernelRegressionKernel::Epanechnikov:
            weight[i] = .75L * (1 - u * u);
            break;
          case KernelRegressionKernel::Uniform:
            weight[i] = 1;
            break;
          case KernelRegressionKernel::Triangular:
            weight[i] = 1 - u;
            break;
          default:
            break;
        }
      }
  }
  long double total = 0;
  for (auto w : weight)
    total += w;
  if (!(total > 0))
    return out;
  long double prediction = 0, square_sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const long double w = weight[i] / total;
    out.weights[i]      = static_cast<double>(w);
    prediction += w * training_y_[i];
    square_sum += w * w;
    if (out.weights[i] > 0)
      ++out.positive_weights;
  }
  out.prediction  = static_cast<double>(prediction);
  out.effective_n = static_cast<double>(1 / square_sum);
  return out;
}
std::vector<double> KernelRegression::query_row(
    const dstruct::DataFrame& data,
    std::size_t               row,
    std::string&              status) const {
  std::vector<double> query;
  status = "ok";
  for (std::size_t j = 0; j < design_.columns.size(); ++j) {
    double value;
    if (!design_.columns[j].evaluate(data, row, value)) {
      status = "missing_predictor";
      return {};
    }
    if (!std::isfinite(value)) {
      status = "nonfinite_predictor";
      return {};
    }
    const double x = static_cast<double>(
        (static_cast<long double>(value) - feature_mean_[j]) / feature_scale_[j]);
    if (!std::isfinite(x)) {
      status = "nonfinite_predictor";
      return {};
    }
    query.push_back(x);
  }
  return query;
}
std::vector<double> KernelRegression::predict(const dstruct::DataFrame& data) const {
  if (data.nrows() == 0)
    return {};
  for (const auto& col : design_.columns)
    require_design_column(data, col);
  design_.validate_categorical_levels(data, "KernelRegression::predict");
  std::vector<double> out(data.nrows(), missing);
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    std::string status;
    const auto  q = query_row(data, i, status);
    if (!q.empty())
      out[i] = local_fit(q, bandwidth_used_).prediction;
  }
  return out;
}
dstruct::DataFrame KernelRegression::prediction_details(const dstruct::DataFrame& data) const {
  if (data.nrows())
    for (const auto& col : design_.columns)
      require_design_column(data, col);
  design_.validate_categorical_levels(data, "KernelRegression::prediction_details");
  std::vector<double>      row, prediction, count, effective, nearest;
  std::vector<std::string> status;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    std::string state;
    const auto  q = query_row(data, i, state);
    LocalFit    fit{{}, missing, missing, missing, 0};
    if (!q.empty()) {
      fit = local_fit(q, bandwidth_used_);
      if (!std::isfinite(fit.prediction))
        state = "no_support";
    }
    row.push_back(i);
    prediction.push_back(fit.prediction);
    count.push_back(fit.positive_weights);
    effective.push_back(fit.effective_n);
    nearest.push_back(fit.nearest_distance);
    status.push_back(state);
  }
  dstruct::DataFrame out;
  out.add_column("query_row", row);
  out.add_column("prediction", prediction);
  out.add_column("positive_weights", count);
  out.add_column("effective_n", effective);
  out.add_column("nearest_distance", nearest);
  out.add_column("status", status);
  return out;
}
linalg::DenseMatrix<double> KernelRegression::query_weights(const dstruct::DataFrame& data) const {
  if (data.nrows())
    for (const auto& col : design_.columns)
      require_design_column(data, col);
  design_.validate_categorical_levels(data, "KernelRegression::query_weights");
  linalg::DenseMatrix<double> out(data.nrows(), observations_, missing);
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    std::string status;
    const auto  q = query_row(data, i, status);
    if (q.empty())
      continue;
    const auto fit = local_fit(q, bandwidth_used_);
    for (std::size_t j = 0; j < observations_; ++j)
      out(i, j) = fit.weights[j];
  }
  return out;
}
linalg::DenseMatrix<double> KernelRegression::smoothing_matrix() const {
  linalg::DenseMatrix<double> out(observations_, observations_);
  std::vector<double>         query(predictor_names_.size());
  for (std::size_t i = 0; i < observations_; ++i) {
    for (std::size_t j = 0; j < query.size(); ++j)
      query[j] = training_X_std_(i, j);
    const auto fit = local_fit(query, bandwidth_used_);
    for (std::size_t j = 0; j < observations_; ++j)
      out(i, j) = fit.weights[j];
  }
  return out;
}
dstruct::DataFrame KernelRegression::training_details() const {
  std::vector<double> row, residual;
  for (std::size_t i = 0; i < observations_; ++i) {
    row.push_back(kept_rows_[i]);
    residual.push_back(training_y_[i] - fitted_[i]);
  }
  dstruct::DataFrame out;
  out.add_column("row", row);
  out.add_column("actual", training_y_);
  out.add_column("full_fit", full_fitted_);
  out.add_column("loo_fit", fitted_);
  out.add_column("loo_residual", residual);
  out.add_column("leverage", leverage_);
  out.add_column("effective_n", effective_n_);
  out.add_column("loo_effective_n", loo_effective_n_);
  return out;
}
dstruct::DataFrame KernelRegression::bandwidth_diagnostics() const {
  std::vector<double> h = bandwidth_grid_, count = cv_counts_, conditional = cv_conditional_mse_,
                      objective = cv_mse_, coverage, eligible, selected;
  if (!bandwidth_was_selected_) {
    const auto e = errors(training_y_, fitted_, training_mean_);
    h            = {bandwidth_used_};
    count        = {static_cast<double>(e.n)};
    conditional  = {e.mse};
    objective    = {e.n == observations_ && std::isfinite(e.mse) ? e.mse : infinity};
  }
  bool marked = false;
  for (std::size_t k = 0; k < h.size(); ++k) {
    coverage.push_back(count[k] / observations_);
    eligible.push_back(std::isfinite(objective[k]));
    const bool chosen = !marked && h[k] == bandwidth_used_;
    selected.push_back(chosen);
    if (chosen)
      marked = true;
  }
  dstruct::DataFrame out;
  out.add_column("bandwidth", h);
  out.add_column("loo_n", count);
  out.add_column("coverage", coverage);
  out.add_column("conditional_mse", conditional);
  out.add_column("objective_mse", objective);
  out.add_column("eligible", eligible);
  out.add_column("selected", selected);
  return out;
}
double KernelRegression::r_squared() const {
  return errors(training_y_, fitted_, training_mean_).r2;
}
double KernelRegression::rmse() const {
  return errors(training_y_, fitted_, training_mean_).rmse;
}
dstruct::DataFrame KernelRegression::model_summary() const {
  const auto loo  = errors(training_y_, fitted_, training_mean_),
             full = errors(training_y_, full_fitted_, training_mean_);
  const double df = std::accumulate(leverage_.begin(), leverage_.end(), 0.0);
  const bool   boundary =
      bandwidth_was_selected_ &&
      (bandwidth_used_ == *std::min_element(bandwidth_grid_.begin(), bandwidth_grid_.end()) ||
       bandwidth_used_ == *std::max_element(bandwidth_grid_.begin(), bandwidth_grid_.end()));
  dstruct::DataFrame out;
  out.add_column("n", std::vector<double>{double(observations_)});
  out.add_column("p", std::vector<double>{double(predictor_names_.size())});
  out.add_column("bandwidth", std::vector<double>{bandwidth_used_});
  out.add_column("loo_n", std::vector<double>{double(loo.n)});
  out.add_column("loo_coverage", std::vector<double>{double(loo.n) / observations_});
  out.add_column("loo_rmse", std::vector<double>{loo.rmse});
  out.add_column("loo_r_squared", std::vector<double>{loo.r2});
  out.add_column("apparent_rmse", std::vector<double>{full.rmse});
  out.add_column("effective_df", std::vector<double>{df});
  out.add_column("cv_selected", std::vector<double>{double(bandwidth_was_selected_)});
  out.add_column("search_boundary", std::vector<double>{double(boundary)});
  return out;
}
dstruct::DataFrame KernelRegression::score(const dstruct::DataFrame& data) const {
  if (!data.has_column(design_.response_name) ||
      data.column_type(design_.response_name) != dstruct::DataFrame::ColumnType::Numeric)
    throw std::invalid_argument("KernelRegression::score: numeric response column required");
  std::vector<std::size_t> rows;
  std::vector<double>      actual;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    const auto y = data.optional_double_at(design_.response_name, i);
    if (!y)
      continue;
    if (!std::isfinite(*y))
      throw std::invalid_argument("KernelRegression::score: observed responses must be finite");
    rows.push_back(i);
    actual.push_back(*y);
  }
  const auto         predicted = predict(data.take_rows(rows));
  const auto         e         = errors(actual, predicted, training_mean_);
  dstruct::DataFrame out;
  out.add_column("n", std::vector<double>{double(data.nrows())});
  out.add_column("n_response", std::vector<double>{double(rows.size())});
  out.add_column("n_predicted", std::vector<double>{double(e.n)});
  out.add_column(
      "coverage", std::vector<double>{rows.empty() ? missing : double(e.n) / rows.size()});
  out.add_column("rmse", std::vector<double>{e.rmse});
  out.add_column("mae", std::vector<double>{e.mae});
  out.add_column("r_squared", std::vector<double>{e.r2});
  out.add_column("mean_baseline_rmse", std::vector<double>{e.baseline_rmse});
  return out;
}
std::string KernelRegression::summary() const {
  const auto         d = model_summary();
  std::ostringstream out;
  out << "Kernel Regression (Nadaraya-Watson)\nFormula: " << formula_.text()
      << "\nKernel: " << kernel_name(options_.kernel) << "\nBandwidth: " << std::setprecision(6)
      << bandwidth_used_
      << (bandwidth_was_selected_ ? " (selected by conditional leave-one-out CV)" : " (fixed)")
      << "\nLeave-one-out coverage: " << d.double_at("loo_n", 0) << " / " << observations_
      << "\nLeave-one-out R-squared: " << r_squared() << "\nLeave-one-out RMSE: " << rmse() << "\n";
  return out.str();
}
void KernelRegression::print_summary(std::ostream& os) const {
  os << summary();
}
void KernelRegression::print_summary() const {
  print_summary(std::cout);
}
graphics::RPlot KernelRegression::plot_fit(const dstruct::DataFrame& data, std::size_t resolution)
    const {
  if (design_.columns.size() != 1 || design_.columns[0].kind != ResolvedColumnKind::Numeric ||
      resolution < 2)
    throw std::invalid_argument(
        "KernelRegression::plot_fit: one direct numeric predictor and resolution>=2 required");
  const auto&  name  = design_.columns[0].source_column;
  const double range = feature_max_[0] - feature_min_[0],
               pad   = range > 0 ? .05 * range : std::max(1.0, .05 * std::abs(feature_min_[0]));
  const double low = feature_min_[0] - pad, high = feature_max_[0] + pad;
  if (!std::isfinite(low) || !std::isfinite(high))
    throw std::invalid_argument("KernelRegression::plot_fit: unrepresentable plotting range");
  std::vector<double> x;
  for (std::size_t i = 0; i < resolution; ++i) {
    const double t = double(i) / (resolution - 1);
    x.push_back((1 - t) * low + t * high);
  }
  dstruct::DataFrame grid;
  grid.add_column(name, x);
  const auto          y    = predict(grid);
  auto                plot = graphics::RPlot::create();
  std::vector<double> actual_x, actual_y;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    const auto a = data.optional_double_at(name, i),
               b = data.optional_double_at(design_.response_name, i);
    if (a && b && std::isfinite(*a) && std::isfinite(*b)) {
      actual_x.push_back(*a);
      actual_y.push_back(*b);
    }
  }
  if (!actual_x.empty())
    plot.points(actual_x, actual_y, "observed", {148, 163, 184});
  add_curve(plot, x, y, "kernel fit");
  return plot.title("Kernel Regression Fit").x_label(name).y_label(design_.response_name);
}
graphics::RPlot KernelRegression::plot_predicted_vs_actual() const {
  std::vector<double> actual, prediction;
  for (std::size_t i = 0; i < observations_; ++i)
    if (std::isfinite(fitted_[i])) {
      actual.push_back(training_y_[i]);
      prediction.push_back(fitted_[i]);
    }
  if (actual.empty())
    throw std::invalid_argument("KernelRegression: no finite deletion predictions to plot");
  auto plot = graphics::RPlot::create();
  plot.points(actual, prediction, "LOO predictions").abline(0, 1);
  return plot.title("Predicted vs Actual (leave-one-out)").x_label("Actual").y_label("Predicted");
}
graphics::RPlot KernelRegression::plot_residuals_vs_fitted() const {
  std::vector<double> prediction, residual;
  for (std::size_t i = 0; i < observations_; ++i)
    if (std::isfinite(fitted_[i])) {
      prediction.push_back(fitted_[i]);
      residual.push_back(training_y_[i] - fitted_[i]);
    }
  if (prediction.empty())
    throw std::invalid_argument("KernelRegression: no finite deletion predictions to plot");
  auto plot = graphics::RPlot::create();
  plot.points(prediction, residual, "LOO residuals").abline_h(0);
  return plot.title("Residuals vs Fitted (leave-one-out)")
      .x_label("LOO prediction")
      .y_label("Residual");
}
graphics::RPlot KernelRegression::plot_cv_curve() const {
  if (!bandwidth_was_selected_)
    throw std::invalid_argument(
        "KernelRegression::plot_cv_curve: fixed bandwidth has no selection curve");
  std::vector<std::size_t> order(bandwidth_grid_.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
    return bandwidth_grid_[a] < bandwidth_grid_[b];
  });
  std::vector<double> x, y;
  double              chosen = missing;

  auto plot = graphics::RPlot::create();
  for (auto k : order) {
    if (!std::isfinite(cv_mse_[k])) {
      x.push_back(std::log10(bandwidth_grid_[k]));
      y.push_back(missing);
      continue;
    }
    x.push_back(std::log10(bandwidth_grid_[k]));
    y.push_back(cv_mse_[k]);
    if (bandwidth_grid_[k] == bandwidth_used_)
      chosen = cv_mse_[k];
  }

  add_curve(plot, x, y, "complete LOO MSE");
  plot.points({std::log10(bandwidth_used_)}, {chosen}, "chosen h", {37, 99, 235}, 6);
  return plot.title("Conditional Bandwidth Cross-Validation")
      .x_label("log10 bandwidth")
      .y_label("complete LOO MSE");
}
} // namespace datamunge::stats
