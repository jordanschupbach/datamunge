#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <datamunge/datamunge.hpp>
#include <datamunge/datasets/datasets.hpp>

namespace datamunge {
void hello() {
  std::cout << "Hello datamunge" << std::endl;
}

double call_with_callback(double x, Callback* cb) {
  return cb ? cb->call(x) : x;
}

std::vector<double> map_dvector_with_callback(const std::vector<double>& values, Callback* cb) {
  if (!cb) {
    return values;
  }
  std::vector<double> out;
  out.reserve(values.size());
  for (double v : values) {
    out.push_back(cb->call(v));
  }
  return out;
}

std::vector<double> make_dvector(double a, double b, double c) {
  return {a, b, c};
}

double sum_dvector(const std::vector<double>& values) {
  double sum = 0.0;
  for (double v : values) {
    sum += v;
  }
  return sum;
}

std::pair<double, double> make_dpair(double a, double b) {
  return {a, b};
}

double sum_dpair(const std::pair<double, double>& values) {
  return values.first + values.second;
}

DataFrame::DataFrame(dstruct::DataFrame frame) : frame_(std::move(frame)) {}

std::size_t DataFrame::nrows() const { return frame_.nrows(); }

std::size_t DataFrame::ncols() const { return frame_.ncols(); }

std::vector<std::size_t> DataFrame::shape() const {
  const auto [rows, cols] = frame_.shape();
  return {rows, cols};
}

std::vector<std::string> DataFrame::columns() const { return frame_.columns(); }

void DataFrame::add_numeric_column(const std::string& column_name, const std::vector<double>& values,
                                   const std::vector<int>& valid_mask) {
  frame_.add_column(column_name, apply_valid_mask(values, valid_mask));
}

void DataFrame::add_string_column(const std::string& column_name, const std::vector<std::string>& values,
                                  const std::vector<int>& valid_mask) {
  frame_.add_column(column_name, apply_valid_mask(values, valid_mask));
}

void DataFrame::add_string_column_encoded(const std::string& column_name, const std::string& encoded_values,
                                          const std::vector<int>& valid_mask) {
  frame_.add_column(column_name, apply_valid_mask(split_encoded_strings(encoded_values), valid_mask));
}

void DataFrame::fill_null_numeric(const std::string& column_name, const double value) { frame_.fill_null(column_name, value); }

void DataFrame::fill_null_string(const std::string& column_name, const std::string& value) {
  frame_.fill_null(column_name, value);
}

DataFrame* DataFrame::select(const std::vector<std::string>& selected_columns) const {
  return new DataFrame(frame_.select(selected_columns));
}

DataFrame* DataFrame::select_encoded(const std::string& encoded_columns) const {
  return new DataFrame(frame_.select(split_encoded_strings(encoded_columns)));
}

DataFrame* DataFrame::sort_by(const std::string& column_name, const bool ascending) const {
  return new DataFrame(frame_.sort_by(column_name, ascending));
}

DataFrame* DataFrame::drop_duplicates(const std::vector<std::string>& subset) const {
  return new DataFrame(frame_.drop_duplicates(subset));
}

DataFrame* DataFrame::drop_duplicates_encoded(const std::string& encoded_subset) const {
  return new DataFrame(frame_.drop_duplicates(split_encoded_strings(encoded_subset)));
}

DataFrame* DataFrame::group_by_sum(const std::vector<std::string>& key_columns,
                                   const std::vector<std::string>& value_columns) const {
  return new DataFrame(frame_.group_by_sum(key_columns, value_columns));
}

DataFrame* DataFrame::group_by_sum_encoded(const std::string& encoded_key_columns,
                                           const std::string& encoded_value_columns) const {
  return new DataFrame(
      frame_.group_by_sum(split_encoded_strings(encoded_key_columns), split_encoded_strings(encoded_value_columns)));
}

DataFrame* DataFrame::join(const DataFrame& right, const std::string& left_key, const std::string& right_key,
                           const bool left_join) const {
  return new DataFrame(frame_.join(right.frame_, left_key, right_key,
                                   left_join ? dstruct::DataFrame::JoinType::Left : dstruct::DataFrame::JoinType::Inner));
}

std::size_t DataFrame::numeric_count(const std::string& column_name) const {
  return frame_.describe_numeric(column_name).count;
}

std::size_t DataFrame::numeric_null_count(const std::string& column_name) const {
  return frame_.describe_numeric(column_name).null_count;
}

double DataFrame::numeric_sum(const std::string& column_name) const { return frame_.describe_numeric(column_name).sum; }

double DataFrame::numeric_mean(const std::string& column_name) const { return frame_.describe_numeric(column_name).mean; }

double DataFrame::numeric_min(const std::string& column_name) const { return frame_.describe_numeric(column_name).min; }

double DataFrame::numeric_max(const std::string& column_name) const { return frame_.describe_numeric(column_name).max; }

std::string DataFrame::to_string(const std::size_t max_rows) const { return frame_.to_string(max_rows); }

bool DataFrame::is_numeric_column(const std::string& column_name) const {
  return frame_.column_type(column_name) == dstruct::DataFrame::ColumnType::Numeric;
}

bool DataFrame::is_null(const std::string& column_name, const std::size_t row_index) const {
  return frame_.is_null(column_name, row_index);
}

double DataFrame::numeric_at(const std::string& column_name, const std::size_t row_index) const {
  return frame_.double_at(column_name, row_index);
}

std::string DataFrame::string_at(const std::string& column_name, const std::size_t row_index) const {
  return frame_.string_at(column_name, row_index);
}

DataFrame* DataFrame::empty() { return new DataFrame(); }

DataFrame* DataFrame::iris() { return new DataFrame(datasets::iris()); }

DataFrame* DataFrame::penguins() { return new DataFrame(datasets::penguins()); }

template <typename T>
std::vector<std::optional<T>> DataFrame::apply_valid_mask(const std::vector<T>& values, const std::vector<int>& valid_mask) {
  if (!valid_mask.empty() && valid_mask.size() != values.size()) {
    throw std::invalid_argument("DataFrame valid mask length mismatch");
  }

  std::vector<std::optional<T>> result;
  result.reserve(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!valid_mask.empty() && valid_mask[index] == 0) {
      result.emplace_back(std::nullopt);
    } else {
      result.emplace_back(values[index]);
    }
  }
  return result;
}

std::vector<std::string> DataFrame::split_encoded_strings(const std::string& encoded_values) {
  const auto header_end = encoded_values.find('\x1e');
  if (header_end == std::string::npos) {
    std::vector<std::string> values;
    if (encoded_values.empty()) {
      return values;
    }
    std::stringstream stream(encoded_values);
    std::string item;
    while (std::getline(stream, item, '\x1f')) {
      values.push_back(item);
    }
    return values;
  }

  const auto expected_count = static_cast<std::size_t>(std::stoull(encoded_values.substr(0, header_end)));
  const auto payload = encoded_values.substr(header_end + 1);

  std::vector<std::string> values;
  values.reserve(expected_count);

  std::size_t start = 0;
  while (start <= payload.size() && values.size() < expected_count) {
    const auto delimiter = payload.find('\x1f', start);
    if (delimiter == std::string::npos) {
      values.push_back(payload.substr(start));
      break;
    }
    values.push_back(payload.substr(start, delimiter - start));
    start = delimiter + 1;
  }

  while (values.size() < expected_count) {
    values.push_back("");
  }
  return values;
}

LM::LM(const DataFrame& data, const std::string& formula, const std::string& weights_column)
    : lm_(data.frame_, formula,
          stats::LmOptions{weights_column.empty() ? std::nullopt : std::optional<std::string>(weights_column)}) {}

std::string LM::formula_text() const { return lm_.formula_text(); }

bool LM::has_intercept() const { return lm_.has_intercept(); }

std::size_t LM::observations() const { return lm_.observations(); }

std::size_t LM::rank() const { return lm_.rank(); }

std::size_t LM::degrees_of_freedom() const { return lm_.degrees_of_freedom(); }

std::vector<double> LM::coefficients() const { return lm_.coefficients(); }

std::vector<std::string> LM::coefficient_names() const { return lm_.coefficient_names(); }

std::vector<double> LM::fitted_values() const { return lm_.fitted_values(); }

std::vector<double> LM::residuals() const { return lm_.residuals(); }

std::vector<double> LM::standard_errors() const { return lm_.standard_errors(); }

std::vector<double> LM::t_values() const { return lm_.t_values(); }

std::vector<double> LM::p_values() const { return lm_.p_values(); }

double LM::r_squared() const { return lm_.r_squared(); }

double LM::adjusted_r_squared() const { return lm_.adjusted_r_squared(); }

double LM::sigma() const { return lm_.sigma(); }

double LM::f_statistic() const { return lm_.f_statistic(); }

double LM::f_p_value() const { return lm_.f_p_value(); }

std::vector<double> LM::confidence_interval_lower(const double level) const {
  std::vector<double> out;
  out.reserve(lm_.rank());
  for (const auto& interval : lm_.confidence_intervals(level)) out.push_back(interval.lower);
  return out;
}

std::vector<double> LM::confidence_interval_upper(const double level) const {
  std::vector<double> out;
  out.reserve(lm_.rank());
  for (const auto& interval : lm_.confidence_intervals(level)) out.push_back(interval.upper);
  return out;
}

std::vector<double> LM::leverage() const { return lm_.leverage(); }

std::vector<double> LM::standardized_residuals() const { return lm_.standardized_residuals(); }

std::vector<double> LM::studentized_residuals() const { return lm_.studentized_residuals(); }

std::vector<double> LM::cooks_distance() const { return lm_.cooks_distance(); }

std::string LM::summary() const { return lm_.summary(); }

void LM::print_summary() const { lm_.print_summary(); }

std::vector<double> LM::predict(const DataFrame& newdata) const { return lm_.predict(newdata.frame_); }

DataFrame* LM::predict_frame(const DataFrame& newdata, const std::string& interval_kind, const double level) const {
  auto kind = stats::PredictionInterval::None;
  if (interval_kind == "confidence") {
    kind = stats::PredictionInterval::Confidence;
  } else if (interval_kind == "prediction") {
    kind = stats::PredictionInterval::Prediction;
  } else if (interval_kind != "none" && !interval_kind.empty()) {
    throw std::invalid_argument("LM::predict_frame: interval_kind must be 'none', 'confidence', or 'prediction'");
  }
  return new DataFrame(lm_.predict_frame(newdata.frame_, kind, level));
}

DataFrame* LM::anova() const {
  std::vector<std::string> terms;
  std::vector<double> degrees_of_freedom, sum_sq, mean_sq, f_value, p_value;
  for (const auto& row : lm_.anova()) {
    terms.push_back(row.term);
    degrees_of_freedom.push_back(static_cast<double>(row.degrees_of_freedom));
    sum_sq.push_back(row.sum_sq);
    mean_sq.push_back(row.mean_sq);
    f_value.push_back(row.f_value);
    p_value.push_back(row.p_value);
  }

  dstruct::DataFrame frame;
  frame.add_column("term", terms);
  frame.add_column("df", degrees_of_freedom);
  frame.add_column("sum_sq", sum_sq);
  frame.add_column("mean_sq", mean_sq);
  frame.add_column("f_value", f_value);
  frame.add_column("p_value", p_value);
  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot LM::plot_residuals_vs_fitted() const { return lm_.plot_residuals_vs_fitted(); }

datamunge::plot::ScatterPlot LM::plot_normal_qq() const { return lm_.plot_normal_qq(); }

datamunge::plot::ScatterPlot LM::plot_scale_location() const { return lm_.plot_scale_location(); }

datamunge::plot::ScatterPlot LM::plot_residuals_vs_leverage() const { return lm_.plot_residuals_vs_leverage(); }

void LM::save_diagnostic_plots(const std::string& path_prefix) const { lm_.save_diagnostic_plots(path_prefix); }

LMM::LMM(const DataFrame& data, const std::string& formula, const bool reml, const std::size_t de_population_size,
        const std::size_t de_max_generations, const double theta_bound, const std::size_t seed)
    : lmm_(data.frame_, formula, [&] {
        stats::LMMOptions options;
        options.reml               = reml;
        options.de_population_size = de_population_size;
        options.de_max_generations = de_max_generations;
        options.theta_bound        = theta_bound;
        options.seed                = seed;
        return options;
      }()) {}

std::string LMM::formula_text() const { return lmm_.formula_text(); }

std::string LMM::group_variable() const { return lmm_.group_variable(); }

bool LMM::has_random_intercept() const { return lmm_.has_random_intercept(); }

std::vector<std::string> LMM::random_effect_names() const { return lmm_.random_effect_names(); }

bool LMM::is_reml() const { return lmm_.is_reml(); }

std::size_t LMM::observations() const { return lmm_.observations(); }

std::size_t LMM::num_groups() const { return lmm_.num_groups(); }

std::size_t LMM::rank() const { return lmm_.rank(); }

std::vector<double> LMM::coefficients() const { return lmm_.coefficients(); }

std::vector<std::string> LMM::coefficient_names() const { return lmm_.coefficient_names(); }

std::vector<double> LMM::standard_errors() const { return lmm_.standard_errors(); }

std::vector<double> LMM::z_values() const { return lmm_.z_values(); }

std::vector<double> LMM::p_values() const { return lmm_.p_values(); }

std::vector<double> LMM::fitted_values() const { return lmm_.fitted_values(); }

std::vector<double> LMM::residuals() const { return lmm_.residuals(); }

double LMM::residual_variance() const { return lmm_.residual_variance(); }

double LMM::residual_std_dev() const { return lmm_.residual_std_dev(); }

std::vector<double> LMM::random_effect_std_devs() const { return lmm_.random_effect_std_devs(); }

double LMM::random_effect_correlation(const std::size_t i, const std::size_t j) const {
  return lmm_.random_effect_correlation(i, j);
}

std::vector<std::string> LMM::group_labels() const { return lmm_.group_labels(); }

std::vector<double> LMM::random_effects_for_group(const std::size_t group_index) const {
  const auto& effects = lmm_.random_effects();
  if (group_index >= effects.size()) throw std::invalid_argument("LMM::random_effects_for_group: group_index out of range");
  return effects[group_index];
}

double LMM::log_likelihood() const { return lmm_.log_likelihood(); }

double LMM::deviance() const { return lmm_.deviance(); }

double LMM::aic() const { return lmm_.aic(); }

double LMM::bic() const { return lmm_.bic(); }

std::string LMM::summary() const { return lmm_.summary(); }

void LMM::print_summary() const { lmm_.print_summary(); }

std::vector<double> LMM::predict(const DataFrame& newdata) const { return lmm_.predict(newdata.frame_); }

LDA::LDA(const DataFrame& data, const std::string& formula, const std::vector<double>& priors)
    : lda_(data.frame_, formula,
          stats::LDAOptions{priors.empty() ? std::nullopt : std::optional<std::vector<double>>(priors)}) {}

std::vector<std::string> LDA::classes() const { return lda_.classes(); }

std::vector<std::string> LDA::predictor_names() const { return lda_.predictor_names(); }

std::size_t LDA::observations() const { return lda_.observations(); }

std::size_t LDA::num_discriminants() const { return lda_.num_discriminants(); }

std::vector<double> LDA::priors() const { return lda_.priors(); }

DataFrame* LDA::group_means() const {
  dstruct::DataFrame frame;
  frame.add_column("class", lda_.classes());
  const auto& names = lda_.predictor_names();
  for (std::size_t j = 0; j < names.size(); ++j) {
    std::vector<double> column;
    for (const auto& row : lda_.group_means()) column.push_back(row[j]);
    frame.add_column(names[j], column);
  }
  return new DataFrame(std::move(frame));
}

DataFrame* LDA::scaling() const {
  dstruct::DataFrame frame;
  frame.add_column("predictor", lda_.predictor_names());
  const auto& coefficients = lda_.scaling();
  for (std::size_t ld = 0; ld < coefficients.cols(); ++ld) {
    std::vector<double> column;
    for (std::size_t j = 0; j < coefficients.rows(); ++j) column.push_back(coefficients(j, ld));
    frame.add_column("LD" + std::to_string(ld + 1), column);
  }
  return new DataFrame(std::move(frame));
}

std::vector<double> LDA::proportion_of_trace() const { return lda_.proportion_of_trace(); }

double LDA::training_accuracy() const { return lda_.training_accuracy(); }

DataFrame* LDA::confusion_matrix() const {
  dstruct::DataFrame frame;
  const auto& classes = lda_.classes();
  frame.add_column("actual", classes);
  const auto counts = lda_.confusion_matrix();
  for (std::size_t predicted = 0; predicted < classes.size(); ++predicted) {
    std::vector<double> column;
    for (std::size_t actual = 0; actual < classes.size(); ++actual) column.push_back(counts(actual, predicted));
    frame.add_column(classes[predicted], column);
  }
  return new DataFrame(std::move(frame));
}

std::string LDA::summary() const { return lda_.summary(); }

void LDA::print_summary() const { lda_.print_summary(); }

std::vector<std::string> LDA::predict(const DataFrame& newdata) const { return lda_.predict(newdata.frame_); }

DataFrame* LDA::predict_frame(const DataFrame& newdata) const {
  const auto detail = lda_.predict_detail(newdata.frame_);

  std::vector<std::optional<std::string>> labels;
  labels.reserve(detail.class_label.size());
  for (const auto& label : detail.class_label)
    labels.emplace_back(label.empty() ? std::nullopt : std::optional<std::string>(label));

  dstruct::DataFrame frame;
  frame.add_column("class", labels);

  for (std::size_t ld = 0; ld < lda_.num_discriminants(); ++ld) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.discriminants.size());
    for (const auto& row : detail.discriminants)
      column.emplace_back(row.size() > ld ? std::optional<double>(row[ld]) : std::nullopt);
    frame.add_column("LD" + std::to_string(ld + 1), column);
  }

  const auto& classes = lda_.classes();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.posterior.size());
    for (const auto& row : detail.posterior)
      column.emplace_back(row.size() > c ? std::optional<double>(row[c]) : std::nullopt);
    frame.add_column("posterior_" + classes[c], column);
  }

  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot LDA::plot_discriminants() const { return lda_.plot_discriminants(); }

void LDA::save_discriminant_plot(const std::string& path) const { lda_.save_discriminant_plot(path); }

namespace {

stats::SVMKernel parse_svm_kernel(const std::string& kernel) {
  if (kernel == "linear") return stats::SVMKernel::Linear;
  if (kernel == "polynomial") return stats::SVMKernel::Polynomial;
  if (kernel == "radial") return stats::SVMKernel::Radial;
  if (kernel == "sigmoid") return stats::SVMKernel::Sigmoid;
  throw std::invalid_argument("SVM: unknown kernel '" + kernel + "' (expected linear, polynomial, radial, or sigmoid)");
}

} // namespace

SVM::SVM(const DataFrame& data, const std::string& formula, const std::string& kernel, const double cost,
         const double gamma, const double coef0, const int degree, const bool scale)
    : svm_(data.frame_, formula, [&] {
        stats::SVMOptions options;
        options.kernel = parse_svm_kernel(kernel);
        options.cost   = cost;
        options.gamma  = gamma;
        options.coef0  = coef0;
        options.degree = degree;
        options.scale  = scale;
        return options;
      }()) {}

std::vector<std::string> SVM::classes() const { return svm_.classes(); }

std::vector<std::string> SVM::predictor_names() const { return svm_.predictor_names(); }

std::size_t SVM::observations() const { return svm_.observations(); }

std::size_t SVM::num_support_vectors() const { return svm_.num_support_vectors(); }

double SVM::training_accuracy() const { return svm_.training_accuracy(); }

DataFrame* SVM::confusion_matrix() const {
  dstruct::DataFrame frame;
  const auto& classes = svm_.classes();
  frame.add_column("actual", classes);
  const auto counts = svm_.confusion_matrix();
  for (std::size_t predicted = 0; predicted < classes.size(); ++predicted) {
    std::vector<double> column;
    for (std::size_t actual = 0; actual < classes.size(); ++actual) column.push_back(counts(actual, predicted));
    frame.add_column(classes[predicted], column);
  }
  return new DataFrame(std::move(frame));
}

std::string SVM::summary() const { return svm_.summary(); }

void SVM::print_summary() const { svm_.print_summary(); }

std::vector<std::string> SVM::predict(const DataFrame& newdata) const { return svm_.predict(newdata.frame_); }

DataFrame* SVM::predict_frame(const DataFrame& newdata) const {
  const auto detail = svm_.predict_detail(newdata.frame_);

  std::vector<std::optional<std::string>> labels;
  labels.reserve(detail.class_label.size());
  for (const auto& label : detail.class_label)
    labels.emplace_back(label.empty() ? std::nullopt : std::optional<std::string>(label));

  dstruct::DataFrame frame;
  frame.add_column("class", labels);

  const auto& classes = svm_.classes();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.votes.size());
    for (const auto& row : detail.votes)
      column.emplace_back(row.size() > c ? std::optional<double>(row[c]) : std::nullopt);
    frame.add_column("votes_" + classes[c], column);
  }

  return new DataFrame(std::move(frame));
}

namespace {

stats::SplitCriterion parse_split_criterion(const std::string& criterion) {
  if (criterion == "gini") return stats::SplitCriterion::Gini;
  if (criterion == "entropy") return stats::SplitCriterion::Entropy;
  throw std::invalid_argument("DecisionTreeClassifier: unknown criterion '" + criterion + "' (expected gini or entropy)");
}

stats::DistanceMetric parse_distance_metric(const std::string& metric) {
  if (metric == "euclidean") return stats::DistanceMetric::Euclidean;
  if (metric == "manhattan") return stats::DistanceMetric::Manhattan;
  throw std::invalid_argument("KNN: unknown metric '" + metric + "' (expected euclidean or manhattan)");
}

stats::LinkageCriterion parse_linkage_criterion(const std::string& linkage) {
  if (linkage == "single") return stats::LinkageCriterion::Single;
  if (linkage == "complete") return stats::LinkageCriterion::Complete;
  if (linkage == "average") return stats::LinkageCriterion::Average;
  if (linkage == "ward") return stats::LinkageCriterion::Ward;
  throw std::invalid_argument("AgglomerativeClustering: unknown linkage '" + linkage +
                              "' (expected single, complete, average, or ward)");
}

stats::GLMFamily parse_glm_family(const std::string& family) {
  if (family == "gaussian") return stats::GLMFamily::Gaussian;
  if (family == "binomial") return stats::GLMFamily::Binomial;
  if (family == "poisson") return stats::GLMFamily::Poisson;
  if (family == "Gamma") return stats::GLMFamily::Gamma;
  throw std::invalid_argument("GLM: unknown family '" + family + "' (expected gaussian, binomial, poisson, or Gamma)");
}

stats::GLMMFamily parse_glmm_family(const std::string& family) {
  if (family == "binomial") return stats::GLMMFamily::Binomial;
  if (family == "poisson") return stats::GLMMFamily::Poisson;
  throw std::invalid_argument("GLMM: unknown family '" + family + "' (expected binomial or poisson)");
}

stats::INLAMixedModelFamily parse_inla_family(const std::string& family) {
  if (family == "gaussian") return stats::INLAMixedModelFamily::Gaussian;
  if (family == "binomial") return stats::INLAMixedModelFamily::Binomial;
  if (family == "poisson") return stats::INLAMixedModelFamily::Poisson;
  throw std::invalid_argument("INLAMixedModel: unknown family '" + family + "' (expected gaussian, binomial, or poisson)");
}

bayes::INLAIntegrationStrategy parse_inla_strategy(const std::string& strategy) {
  if (strategy == "grid") return bayes::INLAIntegrationStrategy::Grid;
  if (strategy == "eb") return bayes::INLAIntegrationStrategy::EmpiricalBayes;
  throw std::invalid_argument("INLAMixedModel: unknown strategy '" + strategy + "' (expected grid or eb)");
}

linalg::TensorDType parse_tensor_dtype(const std::string& dtype) {
  if (dtype == "float64") return linalg::TensorDType::Float64;
  if (dtype == "bool") return linalg::TensorDType::Bool;
  if (dtype == "string") return linalg::TensorDType::String;
  throw std::invalid_argument("Tensor: unknown dtype '" + dtype + "' (expected float64, bool, or string)");
}

stats::KernelRegressionKernel parse_kernel_regression_kernel(const std::string& kernel) {
  if (kernel == "gaussian") return stats::KernelRegressionKernel::Gaussian;
  if (kernel == "epanechnikov") return stats::KernelRegressionKernel::Epanechnikov;
  if (kernel == "uniform") return stats::KernelRegressionKernel::Uniform;
  if (kernel == "triangular") return stats::KernelRegressionKernel::Triangular;
  throw std::invalid_argument("KernelRegression: unknown kernel '" + kernel
                              + "' (expected gaussian, epanechnikov, uniform, or triangular)");
}

} // namespace

DecisionTreeClassifier::DecisionTreeClassifier(const DataFrame& data, const std::string& formula,
                                               const std::size_t max_depth, const std::size_t min_samples_split,
                                               const std::size_t min_samples_leaf, const std::string& criterion)
    : tree_(data.frame_, formula, [&] {
        stats::DecisionTreeClassifierOptions options;
        options.max_depth             = max_depth;
        options.min_samples_split     = min_samples_split;
        options.min_samples_leaf      = min_samples_leaf;
        options.criterion             = parse_split_criterion(criterion);
        return options;
      }()) {}

std::vector<std::string> DecisionTreeClassifier::classes() const { return tree_.classes(); }

std::vector<std::string> DecisionTreeClassifier::predictor_names() const { return tree_.predictor_names(); }

std::size_t DecisionTreeClassifier::observations() const { return tree_.observations(); }

std::size_t DecisionTreeClassifier::node_count() const { return tree_.node_count(); }

std::size_t DecisionTreeClassifier::leaf_count() const { return tree_.leaf_count(); }

std::size_t DecisionTreeClassifier::depth() const { return tree_.depth(); }

std::vector<double> DecisionTreeClassifier::feature_importance() const { return tree_.feature_importance(); }

double DecisionTreeClassifier::training_accuracy() const { return tree_.training_accuracy(); }

DataFrame* DecisionTreeClassifier::confusion_matrix() const {
  dstruct::DataFrame frame;
  const auto& classes = tree_.classes();
  frame.add_column("actual", classes);
  const auto counts = tree_.confusion_matrix();
  for (std::size_t predicted = 0; predicted < classes.size(); ++predicted) {
    std::vector<double> column;
    for (std::size_t actual = 0; actual < classes.size(); ++actual) column.push_back(counts(actual, predicted));
    frame.add_column(classes[predicted], column);
  }
  return new DataFrame(std::move(frame));
}

std::string DecisionTreeClassifier::summary() const { return tree_.summary(); }

void DecisionTreeClassifier::print_summary() const { tree_.print_summary(); }

std::vector<std::string> DecisionTreeClassifier::predict(const DataFrame& newdata) const {
  return tree_.predict(newdata.frame_);
}

DataFrame* DecisionTreeClassifier::predict_frame(const DataFrame& newdata) const {
  const auto detail = tree_.predict_detail(newdata.frame_);

  std::vector<std::optional<std::string>> labels;
  labels.reserve(detail.class_label.size());
  for (const auto& label : detail.class_label)
    labels.emplace_back(label.empty() ? std::nullopt : std::optional<std::string>(label));

  dstruct::DataFrame frame;
  frame.add_column("class", labels);

  const auto& classes = tree_.classes();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.probability.size());
    for (const auto& row : detail.probability)
      column.emplace_back(row.size() > c ? std::optional<double>(row[c]) : std::nullopt);
    frame.add_column("prob_" + classes[c], column);
  }

  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot DecisionTreeClassifier::plot_classification(const DataFrame& data,
                                                                         const std::string& x_feature,
                                                                         const std::string& y_feature) const {
  return tree_.plot_classification(data.frame_, x_feature, y_feature);
}

datamunge::plot::ScatterPlot DecisionTreeClassifier::plot_decision_regions(const std::string& x_feature,
                                                                           const std::string& y_feature,
                                                                           const std::size_t grid_resolution) const {
  return tree_.plot_decision_regions(x_feature, y_feature, grid_resolution);
}

DecisionTreeRegressor::DecisionTreeRegressor(const DataFrame& data, const std::string& formula,
                                             const std::size_t max_depth, const std::size_t min_samples_split,
                                             const std::size_t min_samples_leaf)
    : tree_(data.frame_, formula, [&] {
        stats::DecisionTreeRegressorOptions options;
        options.max_depth         = max_depth;
        options.min_samples_split = min_samples_split;
        options.min_samples_leaf  = min_samples_leaf;
        return options;
      }()) {}

std::vector<std::string> DecisionTreeRegressor::predictor_names() const { return tree_.predictor_names(); }

std::size_t DecisionTreeRegressor::observations() const { return tree_.observations(); }

std::size_t DecisionTreeRegressor::node_count() const { return tree_.node_count(); }

std::size_t DecisionTreeRegressor::leaf_count() const { return tree_.leaf_count(); }

std::size_t DecisionTreeRegressor::depth() const { return tree_.depth(); }

std::vector<double> DecisionTreeRegressor::feature_importance() const { return tree_.feature_importance(); }

std::vector<double> DecisionTreeRegressor::fitted_values() const { return tree_.fitted_values(); }

double DecisionTreeRegressor::r_squared() const { return tree_.r_squared(); }

double DecisionTreeRegressor::rmse() const { return tree_.rmse(); }

std::string DecisionTreeRegressor::summary() const { return tree_.summary(); }

void DecisionTreeRegressor::print_summary() const { tree_.print_summary(); }

std::vector<double> DecisionTreeRegressor::predict(const DataFrame& newdata) const {
  return tree_.predict(newdata.frame_);
}

datamunge::plot::ScatterPlot DecisionTreeRegressor::plot_predicted_vs_actual() const {
  return tree_.plot_predicted_vs_actual();
}

datamunge::plot::ScatterPlot DecisionTreeRegressor::plot_residuals_vs_fitted() const {
  return tree_.plot_residuals_vs_fitted();
}

RandomForestClassifier::RandomForestClassifier(const DataFrame& data, const std::string& formula,
                                               const std::size_t n_trees, const std::size_t max_depth,
                                               const std::size_t min_samples_split, const std::size_t min_samples_leaf,
                                               const std::size_t max_features, const std::string& criterion,
                                               const bool bootstrap, const double sample_fraction,
                                               const std::uint64_t seed)
    : forest_(data.frame_, formula, [&] {
        stats::RandomForestClassifierOptions options;
        options.n_trees           = n_trees;
        options.max_depth         = max_depth;
        options.min_samples_split = min_samples_split;
        options.min_samples_leaf  = min_samples_leaf;
        options.max_features      = max_features;
        options.criterion         = parse_split_criterion(criterion);
        options.bootstrap         = bootstrap;
        options.sample_fraction   = sample_fraction;
        options.seed              = seed;
        return options;
      }()) {}

std::vector<std::string> RandomForestClassifier::classes() const { return forest_.classes(); }

std::vector<std::string> RandomForestClassifier::predictor_names() const { return forest_.predictor_names(); }

std::size_t RandomForestClassifier::observations() const { return forest_.observations(); }

std::size_t RandomForestClassifier::n_trees() const { return forest_.n_trees(); }

std::size_t RandomForestClassifier::max_features_used() const { return forest_.max_features_used(); }

std::vector<double> RandomForestClassifier::feature_importance() const { return forest_.feature_importance(); }

double RandomForestClassifier::training_accuracy() const { return forest_.training_accuracy(); }

double RandomForestClassifier::oob_accuracy() const { return forest_.oob_accuracy(); }

DataFrame* RandomForestClassifier::confusion_matrix() const {
  dstruct::DataFrame frame;
  const auto& classes = forest_.classes();
  frame.add_column("actual", classes);
  const auto counts = forest_.confusion_matrix();
  for (std::size_t predicted = 0; predicted < classes.size(); ++predicted) {
    std::vector<double> column;
    for (std::size_t actual = 0; actual < classes.size(); ++actual) column.push_back(counts(actual, predicted));
    frame.add_column(classes[predicted], column);
  }
  return new DataFrame(std::move(frame));
}

std::string RandomForestClassifier::summary() const { return forest_.summary(); }

void RandomForestClassifier::print_summary() const { forest_.print_summary(); }

std::vector<std::string> RandomForestClassifier::predict(const DataFrame& newdata) const {
  return forest_.predict(newdata.frame_);
}

DataFrame* RandomForestClassifier::predict_frame(const DataFrame& newdata) const {
  const auto detail = forest_.predict_detail(newdata.frame_);

  std::vector<std::optional<std::string>> labels;
  labels.reserve(detail.class_label.size());
  for (const auto& label : detail.class_label)
    labels.emplace_back(label.empty() ? std::nullopt : std::optional<std::string>(label));

  dstruct::DataFrame frame;
  frame.add_column("class", labels);

  const auto& classes = forest_.classes();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.vote_share.size());
    for (const auto& row : detail.vote_share)
      column.emplace_back(row.size() > c ? std::optional<double>(row[c]) : std::nullopt);
    frame.add_column("votes_" + classes[c], column);
  }

  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot RandomForestClassifier::plot_classification(const DataFrame& data,
                                                                          const std::string& x_feature,
                                                                          const std::string& y_feature) const {
  return forest_.plot_classification(data.frame_, x_feature, y_feature);
}

datamunge::plot::ScatterPlot RandomForestClassifier::plot_decision_regions(const std::string& x_feature,
                                                                            const std::string& y_feature,
                                                                            const std::size_t grid_resolution) const {
  return forest_.plot_decision_regions(x_feature, y_feature, grid_resolution);
}

RandomForestRegressor::RandomForestRegressor(const DataFrame& data, const std::string& formula,
                                             const std::size_t n_trees, const std::size_t max_depth,
                                             const std::size_t min_samples_split, const std::size_t min_samples_leaf,
                                             const std::size_t max_features, const bool bootstrap,
                                             const double sample_fraction, const std::uint64_t seed)
    : forest_(data.frame_, formula, [&] {
        stats::RandomForestRegressorOptions options;
        options.n_trees           = n_trees;
        options.max_depth         = max_depth;
        options.min_samples_split = min_samples_split;
        options.min_samples_leaf  = min_samples_leaf;
        options.max_features      = max_features;
        options.bootstrap         = bootstrap;
        options.sample_fraction   = sample_fraction;
        options.seed              = seed;
        return options;
      }()) {}

std::vector<std::string> RandomForestRegressor::predictor_names() const { return forest_.predictor_names(); }

std::size_t RandomForestRegressor::observations() const { return forest_.observations(); }

std::size_t RandomForestRegressor::n_trees() const { return forest_.n_trees(); }

std::size_t RandomForestRegressor::max_features_used() const { return forest_.max_features_used(); }

std::vector<double> RandomForestRegressor::feature_importance() const { return forest_.feature_importance(); }

std::vector<double> RandomForestRegressor::fitted_values() const { return forest_.fitted_values(); }

double RandomForestRegressor::r_squared() const { return forest_.r_squared(); }

double RandomForestRegressor::rmse() const { return forest_.rmse(); }

double RandomForestRegressor::oob_r_squared() const { return forest_.oob_r_squared(); }

double RandomForestRegressor::oob_rmse() const { return forest_.oob_rmse(); }

std::string RandomForestRegressor::summary() const { return forest_.summary(); }

void RandomForestRegressor::print_summary() const { forest_.print_summary(); }

std::vector<double> RandomForestRegressor::predict(const DataFrame& newdata) const {
  return forest_.predict(newdata.frame_);
}

datamunge::plot::ScatterPlot RandomForestRegressor::plot_predicted_vs_actual() const {
  return forest_.plot_predicted_vs_actual();
}

datamunge::plot::ScatterPlot RandomForestRegressor::plot_residuals_vs_fitted() const {
  return forest_.plot_residuals_vs_fitted();
}

ElasticNet::ElasticNet(const DataFrame& data, const std::string& formula, const double alpha, const double lambda,
                       const std::size_t n_lambda, const std::size_t cv_folds, const bool standardize,
                       const std::uint64_t seed)
    : net_(data.frame_, formula, [&] {
        stats::ElasticNetOptions options;
        options.alpha       = alpha;
        options.lambda      = lambda;
        options.n_lambda     = n_lambda;
        options.cv_folds     = cv_folds;
        options.standardize  = standardize;
        options.seed         = seed;
        return options;
      }()) {}

std::string ElasticNet::formula_text() const { return net_.formula_text(); }

bool ElasticNet::has_intercept() const { return net_.has_intercept(); }

std::vector<std::string> ElasticNet::predictor_names() const { return net_.predictor_names(); }

std::size_t ElasticNet::observations() const { return net_.observations(); }

double ElasticNet::alpha() const { return net_.alpha(); }

double ElasticNet::lambda() const { return net_.lambda(); }

bool ElasticNet::lambda_was_selected() const { return net_.lambda_was_selected(); }

std::vector<double> ElasticNet::lambda_path() const { return net_.lambda_path(); }

std::vector<double> ElasticNet::cv_mean_squared_error() const { return net_.cv_mean_squared_error(); }

std::vector<double> ElasticNet::coefficients() const { return net_.coefficients(); }

double ElasticNet::intercept() const { return net_.intercept(); }

std::size_t ElasticNet::non_zero_coefficients() const { return net_.non_zero_coefficients(); }

std::vector<double> ElasticNet::fitted_values() const { return net_.fitted_values(); }

std::vector<double> ElasticNet::residuals() const { return net_.residuals(); }

double ElasticNet::r_squared() const { return net_.r_squared(); }

double ElasticNet::rmse() const { return net_.rmse(); }

std::string ElasticNet::summary() const { return net_.summary(); }

void ElasticNet::print_summary() const { net_.print_summary(); }

std::vector<double> ElasticNet::predict(const DataFrame& newdata) const { return net_.predict(newdata.frame_); }

datamunge::plot::ScatterPlot ElasticNet::plot_coefficient_path() const { return net_.plot_coefficient_path(); }

datamunge::plot::ScatterPlot ElasticNet::plot_cv_curve() const { return net_.plot_cv_curve(); }

datamunge::plot::ScatterPlot ElasticNet::plot_predicted_vs_actual() const { return net_.plot_predicted_vs_actual(); }

datamunge::plot::ScatterPlot ElasticNet::plot_residuals_vs_fitted() const { return net_.plot_residuals_vs_fitted(); }

Ridge::Ridge(const DataFrame& data, const std::string& formula, const double lambda, const std::size_t n_lambda,
            const std::size_t cv_folds, const bool standardize, const std::uint64_t seed)
    : ridge_(data.frame_, formula, [&] {
        stats::RidgeOptions options;
        options.lambda      = lambda;
        options.n_lambda     = n_lambda;
        options.cv_folds     = cv_folds;
        options.standardize  = standardize;
        options.seed         = seed;
        return options;
      }()) {}

std::string Ridge::formula_text() const { return ridge_.formula_text(); }

bool Ridge::has_intercept() const { return ridge_.has_intercept(); }

std::vector<std::string> Ridge::predictor_names() const { return ridge_.predictor_names(); }

std::size_t Ridge::observations() const { return ridge_.observations(); }

double Ridge::lambda() const { return ridge_.lambda(); }

bool Ridge::lambda_was_selected() const { return ridge_.lambda_was_selected(); }

std::vector<double> Ridge::lambda_path() const { return ridge_.lambda_path(); }

std::vector<double> Ridge::cv_mean_squared_error() const { return ridge_.cv_mean_squared_error(); }

std::vector<double> Ridge::coefficients() const { return ridge_.coefficients(); }

double Ridge::intercept() const { return ridge_.intercept(); }

std::vector<double> Ridge::fitted_values() const { return ridge_.fitted_values(); }

std::vector<double> Ridge::residuals() const { return ridge_.residuals(); }

double Ridge::r_squared() const { return ridge_.r_squared(); }

double Ridge::rmse() const { return ridge_.rmse(); }

std::string Ridge::summary() const { return ridge_.summary(); }

void Ridge::print_summary() const { ridge_.print_summary(); }

std::vector<double> Ridge::predict(const DataFrame& newdata) const { return ridge_.predict(newdata.frame_); }

datamunge::plot::ScatterPlot Ridge::plot_coefficient_path() const { return ridge_.plot_coefficient_path(); }

datamunge::plot::ScatterPlot Ridge::plot_cv_curve() const { return ridge_.plot_cv_curve(); }

datamunge::plot::ScatterPlot Ridge::plot_predicted_vs_actual() const { return ridge_.plot_predicted_vs_actual(); }

datamunge::plot::ScatterPlot Ridge::plot_residuals_vs_fitted() const { return ridge_.plot_residuals_vs_fitted(); }

Lasso::Lasso(const DataFrame& data, const std::string& formula, const double lambda, const std::size_t n_lambda,
            const std::size_t cv_folds, const bool standardize, const std::uint64_t seed)
    : lasso_(data.frame_, formula, [&] {
        stats::LassoOptions options;
        options.lambda      = lambda;
        options.n_lambda     = n_lambda;
        options.cv_folds     = cv_folds;
        options.standardize  = standardize;
        options.seed         = seed;
        return options;
      }()) {}

std::string Lasso::formula_text() const { return lasso_.formula_text(); }

bool Lasso::has_intercept() const { return lasso_.has_intercept(); }

std::vector<std::string> Lasso::predictor_names() const { return lasso_.predictor_names(); }

std::size_t Lasso::observations() const { return lasso_.observations(); }

double Lasso::lambda() const { return lasso_.lambda(); }

bool Lasso::lambda_was_selected() const { return lasso_.lambda_was_selected(); }

std::vector<double> Lasso::lambda_path() const { return lasso_.lambda_path(); }

std::vector<double> Lasso::cv_mean_squared_error() const { return lasso_.cv_mean_squared_error(); }

std::vector<double> Lasso::coefficients() const { return lasso_.coefficients(); }

double Lasso::intercept() const { return lasso_.intercept(); }

std::size_t Lasso::non_zero_coefficients() const { return lasso_.non_zero_coefficients(); }

std::vector<double> Lasso::fitted_values() const { return lasso_.fitted_values(); }

std::vector<double> Lasso::residuals() const { return lasso_.residuals(); }

double Lasso::r_squared() const { return lasso_.r_squared(); }

double Lasso::rmse() const { return lasso_.rmse(); }

std::string Lasso::summary() const { return lasso_.summary(); }

void Lasso::print_summary() const { lasso_.print_summary(); }

std::vector<double> Lasso::predict(const DataFrame& newdata) const { return lasso_.predict(newdata.frame_); }

datamunge::plot::ScatterPlot Lasso::plot_coefficient_path() const { return lasso_.plot_coefficient_path(); }

datamunge::plot::ScatterPlot Lasso::plot_cv_curve() const { return lasso_.plot_cv_curve(); }

datamunge::plot::ScatterPlot Lasso::plot_predicted_vs_actual() const { return lasso_.plot_predicted_vs_actual(); }

datamunge::plot::ScatterPlot Lasso::plot_residuals_vs_fitted() const { return lasso_.plot_residuals_vs_fitted(); }

KNNClassifier::KNNClassifier(const DataFrame& data, const std::string& formula, const std::size_t k,
                             const std::string& metric, const bool weighted, const bool standardize)
    : knn_(data.frame_, formula, [&] {
        stats::KNNClassifierOptions options;
        options.k           = k;
        options.metric       = parse_distance_metric(metric);
        options.weighted     = weighted;
        options.standardize  = standardize;
        return options;
      }()) {}

std::vector<std::string> KNNClassifier::classes() const { return knn_.classes(); }

std::vector<std::string> KNNClassifier::predictor_names() const { return knn_.predictor_names(); }

std::size_t KNNClassifier::observations() const { return knn_.observations(); }

std::size_t KNNClassifier::k() const { return knn_.k(); }

double KNNClassifier::training_accuracy() const { return knn_.training_accuracy(); }

DataFrame* KNNClassifier::confusion_matrix() const {
  dstruct::DataFrame frame;
  const auto& classes = knn_.classes();
  frame.add_column("actual", classes);
  const auto counts = knn_.confusion_matrix();
  for (std::size_t predicted = 0; predicted < classes.size(); ++predicted) {
    std::vector<double> column;
    for (std::size_t actual = 0; actual < classes.size(); ++actual) column.push_back(counts(actual, predicted));
    frame.add_column(classes[predicted], column);
  }
  return new DataFrame(std::move(frame));
}

std::string KNNClassifier::summary() const { return knn_.summary(); }

void KNNClassifier::print_summary() const { knn_.print_summary(); }

std::vector<std::string> KNNClassifier::predict(const DataFrame& newdata) const {
  return knn_.predict(newdata.frame_);
}

DataFrame* KNNClassifier::predict_frame(const DataFrame& newdata) const {
  const auto detail = knn_.predict_detail(newdata.frame_);

  std::vector<std::optional<std::string>> labels;
  labels.reserve(detail.class_label.size());
  for (const auto& label : detail.class_label)
    labels.emplace_back(label.empty() ? std::nullopt : std::optional<std::string>(label));

  dstruct::DataFrame frame;
  frame.add_column("class", labels);

  const auto& classes = knn_.classes();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.vote_share.size());
    for (const auto& row : detail.vote_share)
      column.emplace_back(row.size() > c ? std::optional<double>(row[c]) : std::nullopt);
    frame.add_column("votes_" + classes[c], column);
  }

  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot KNNClassifier::plot_classification(const DataFrame& data, const std::string& x_feature,
                                                                 const std::string& y_feature) const {
  return knn_.plot_classification(data.frame_, x_feature, y_feature);
}

datamunge::plot::ScatterPlot KNNClassifier::plot_decision_regions(const std::string& x_feature,
                                                                   const std::string& y_feature,
                                                                   const std::size_t grid_resolution) const {
  return knn_.plot_decision_regions(x_feature, y_feature, grid_resolution);
}

KNNRegressor::KNNRegressor(const DataFrame& data, const std::string& formula, const std::size_t k,
                           const std::string& metric, const bool weighted, const bool standardize)
    : knn_(data.frame_, formula, [&] {
        stats::KNNRegressorOptions options;
        options.k           = k;
        options.metric       = parse_distance_metric(metric);
        options.weighted     = weighted;
        options.standardize  = standardize;
        return options;
      }()) {}

std::vector<std::string> KNNRegressor::predictor_names() const { return knn_.predictor_names(); }

std::size_t KNNRegressor::observations() const { return knn_.observations(); }

std::size_t KNNRegressor::k() const { return knn_.k(); }

std::vector<double> KNNRegressor::fitted_values() const { return knn_.fitted_values(); }

double KNNRegressor::r_squared() const { return knn_.r_squared(); }

double KNNRegressor::rmse() const { return knn_.rmse(); }

std::string KNNRegressor::summary() const { return knn_.summary(); }

void KNNRegressor::print_summary() const { knn_.print_summary(); }

std::vector<double> KNNRegressor::predict(const DataFrame& newdata) const { return knn_.predict(newdata.frame_); }

datamunge::plot::ScatterPlot KNNRegressor::plot_predicted_vs_actual() const { return knn_.plot_predicted_vs_actual(); }

datamunge::plot::ScatterPlot KNNRegressor::plot_residuals_vs_fitted() const { return knn_.plot_residuals_vs_fitted(); }

KMeans::KMeans(const DataFrame& data, const std::vector<std::string>& feature_columns, const std::size_t n_clusters,
              const std::size_t max_iterations, const std::size_t n_init, const double tolerance, const std::size_t seed)
    : kmeans_(data.frame_, feature_columns, [&] {
        stats::KMeansOptions options;
        options.n_clusters     = n_clusters;
        options.max_iterations = max_iterations;
        options.n_init          = n_init;
        options.tolerance       = tolerance;
        options.seed             = seed;
        return options;
      }()) {}

KMeans::KMeans(const DataFrame& data, const std::string& encoded_feature_columns, const std::size_t n_clusters,
              const std::size_t max_iterations, const std::size_t n_init, const double tolerance, const std::size_t seed)
    : KMeans(data, DataFrame::split_encoded_strings(encoded_feature_columns), n_clusters, max_iterations, n_init,
             tolerance, seed) {}

std::vector<std::string> KMeans::feature_names() const { return kmeans_.feature_names(); }

std::size_t KMeans::n_clusters() const { return kmeans_.n_clusters(); }

std::size_t KMeans::observations() const { return kmeans_.observations(); }

std::size_t KMeans::iterations_used() const { return kmeans_.iterations_used(); }

std::vector<std::size_t> KMeans::labels() const { return kmeans_.labels(); }

double KMeans::inertia() const { return kmeans_.inertia(); }

std::vector<double> KMeans::cluster_center(const std::size_t cluster_index) const {
  const auto& centers = kmeans_.cluster_centers();
  if (cluster_index >= centers.rows()) throw std::invalid_argument("KMeans::cluster_center: cluster_index out of range");
  std::vector<double> center(centers.cols());
  for (std::size_t j = 0; j < centers.cols(); ++j) center[j] = centers(cluster_index, j);
  return center;
}

std::vector<std::size_t> KMeans::predict(const DataFrame& newdata) const { return kmeans_.predict(newdata.frame_); }

std::string KMeans::summary() const { return kmeans_.summary(); }

void KMeans::print_summary() const { kmeans_.print_summary(); }

AgglomerativeClustering::AgglomerativeClustering(const DataFrame& data, const std::vector<std::string>& feature_columns,
                                                  const std::size_t n_clusters, const std::string& linkage,
                                                  const std::string& metric)
    : clustering_(data.frame_, feature_columns, [&] {
        stats::AgglomerativeClusteringOptions options;
        options.n_clusters = n_clusters;
        options.linkage     = parse_linkage_criterion(linkage);
        options.metric       = parse_distance_metric(metric);
        return options;
      }()) {}

AgglomerativeClustering::AgglomerativeClustering(const DataFrame& data, const std::string& encoded_feature_columns,
                                                  const std::size_t n_clusters, const std::string& linkage,
                                                  const std::string& metric)
    : AgglomerativeClustering(data, DataFrame::split_encoded_strings(encoded_feature_columns), n_clusters, linkage,
                              metric) {}

std::vector<std::string> AgglomerativeClustering::feature_names() const { return clustering_.feature_names(); }

std::size_t AgglomerativeClustering::observations() const { return clustering_.observations(); }

std::vector<std::size_t> AgglomerativeClustering::labels() const { return clustering_.labels(); }

std::vector<std::size_t> AgglomerativeClustering::cut(const std::size_t n_clusters) const {
  return clustering_.cut(n_clusters);
}

std::size_t AgglomerativeClustering::num_merges() const { return clustering_.merge_history().size(); }

std::size_t AgglomerativeClustering::merge_cluster_a(const std::size_t merge_index) const {
  return clustering_.merge_history().at(merge_index).cluster_a;
}

std::size_t AgglomerativeClustering::merge_cluster_b(const std::size_t merge_index) const {
  return clustering_.merge_history().at(merge_index).cluster_b;
}

double AgglomerativeClustering::merge_distance(const std::size_t merge_index) const {
  return clustering_.merge_history().at(merge_index).distance;
}

std::size_t AgglomerativeClustering::merge_size(const std::size_t merge_index) const {
  return clustering_.merge_history().at(merge_index).size;
}

std::string AgglomerativeClustering::summary() const { return clustering_.summary(); }

void AgglomerativeClustering::print_summary() const { clustering_.print_summary(); }

DBSCAN::DBSCAN(const DataFrame& data, const std::vector<std::string>& feature_columns, const double eps,
              const std::size_t min_samples, const std::string& metric)
    : dbscan_(data.frame_, feature_columns, [&] {
        stats::DBSCANOptions options;
        options.eps         = eps;
        options.min_samples = min_samples;
        options.metric       = parse_distance_metric(metric);
        return options;
      }()) {}

DBSCAN::DBSCAN(const DataFrame& data, const std::string& encoded_feature_columns, const double eps,
              const std::size_t min_samples, const std::string& metric)
    : DBSCAN(data, DataFrame::split_encoded_strings(encoded_feature_columns), eps, min_samples, metric) {}

std::vector<std::string> DBSCAN::feature_names() const { return dbscan_.feature_names(); }

std::size_t DBSCAN::observations() const { return dbscan_.observations(); }

std::size_t DBSCAN::n_clusters() const { return dbscan_.n_clusters(); }

std::size_t DBSCAN::n_noise() const { return dbscan_.n_noise(); }

std::vector<int> DBSCAN::labels() const { return dbscan_.labels(); }

std::string DBSCAN::summary() const { return dbscan_.summary(); }

void DBSCAN::print_summary() const { dbscan_.print_summary(); }

GBMClassifier::GBMClassifier(const DataFrame& data, const std::string& formula, const std::size_t n_trees,
                             const double learning_rate, const std::size_t max_depth,
                             const std::size_t min_samples_split, const std::size_t min_samples_leaf,
                             const double subsample, const std::uint64_t seed)
    : gbm_(data.frame_, formula, [&] {
        stats::GBMClassifierOptions options;
        options.n_trees           = n_trees;
        options.learning_rate     = learning_rate;
        options.max_depth         = max_depth;
        options.min_samples_split = min_samples_split;
        options.min_samples_leaf  = min_samples_leaf;
        options.subsample         = subsample;
        options.seed              = seed;
        return options;
      }()) {}

std::vector<std::string> GBMClassifier::classes() const { return gbm_.classes(); }

std::vector<std::string> GBMClassifier::predictor_names() const { return gbm_.predictor_names(); }

std::size_t GBMClassifier::observations() const { return gbm_.observations(); }

std::size_t GBMClassifier::n_trees() const { return gbm_.n_trees(); }

std::vector<double> GBMClassifier::feature_importance() const { return gbm_.feature_importance(); }

double GBMClassifier::training_accuracy() const { return gbm_.training_accuracy(); }

DataFrame* GBMClassifier::confusion_matrix() const {
  dstruct::DataFrame frame;
  const auto& classes = gbm_.classes();
  frame.add_column("actual", classes);
  const auto counts = gbm_.confusion_matrix();
  for (std::size_t predicted = 0; predicted < classes.size(); ++predicted) {
    std::vector<double> column;
    for (std::size_t actual = 0; actual < classes.size(); ++actual) column.push_back(counts(actual, predicted));
    frame.add_column(classes[predicted], column);
  }
  return new DataFrame(std::move(frame));
}

std::vector<double> GBMClassifier::training_deviance() const { return gbm_.training_deviance(); }

std::string GBMClassifier::summary() const { return gbm_.summary(); }

void GBMClassifier::print_summary() const { gbm_.print_summary(); }

std::vector<std::string> GBMClassifier::predict(const DataFrame& newdata) const {
  return gbm_.predict(newdata.frame_);
}

DataFrame* GBMClassifier::predict_frame(const DataFrame& newdata) const {
  const auto detail = gbm_.predict_detail(newdata.frame_);

  std::vector<std::optional<std::string>> labels;
  labels.reserve(detail.class_label.size());
  for (const auto& label : detail.class_label)
    labels.emplace_back(label.empty() ? std::nullopt : std::optional<std::string>(label));

  dstruct::DataFrame frame;
  frame.add_column("class", labels);

  const auto& classes = gbm_.classes();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.probability.size());
    for (const auto& row : detail.probability)
      column.emplace_back(row.size() > c ? std::optional<double>(row[c]) : std::nullopt);
    frame.add_column("prob_" + classes[c], column);
  }

  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot GBMClassifier::plot_classification(const DataFrame& data, const std::string& x_feature,
                                                                 const std::string& y_feature) const {
  return gbm_.plot_classification(data.frame_, x_feature, y_feature);
}

datamunge::plot::ScatterPlot GBMClassifier::plot_decision_regions(const std::string& x_feature,
                                                                   const std::string& y_feature,
                                                                   const std::size_t grid_resolution) const {
  return gbm_.plot_decision_regions(x_feature, y_feature, grid_resolution);
}

datamunge::plot::ScatterPlot GBMClassifier::plot_training_deviance() const { return gbm_.plot_training_deviance(); }

GBMRegressor::GBMRegressor(const DataFrame& data, const std::string& formula, const std::size_t n_trees,
                           const double learning_rate, const std::size_t max_depth,
                           const std::size_t min_samples_split, const std::size_t min_samples_leaf,
                           const double subsample, const std::uint64_t seed)
    : gbm_(data.frame_, formula, [&] {
        stats::GBMRegressorOptions options;
        options.n_trees           = n_trees;
        options.learning_rate     = learning_rate;
        options.max_depth         = max_depth;
        options.min_samples_split = min_samples_split;
        options.min_samples_leaf  = min_samples_leaf;
        options.subsample         = subsample;
        options.seed              = seed;
        return options;
      }()) {}

std::vector<std::string> GBMRegressor::predictor_names() const { return gbm_.predictor_names(); }

std::size_t GBMRegressor::observations() const { return gbm_.observations(); }

std::size_t GBMRegressor::n_trees() const { return gbm_.n_trees(); }

std::vector<double> GBMRegressor::feature_importance() const { return gbm_.feature_importance(); }

std::vector<double> GBMRegressor::fitted_values() const { return gbm_.fitted_values(); }

double GBMRegressor::r_squared() const { return gbm_.r_squared(); }

double GBMRegressor::rmse() const { return gbm_.rmse(); }

std::vector<double> GBMRegressor::training_deviance() const { return gbm_.training_deviance(); }

std::string GBMRegressor::summary() const { return gbm_.summary(); }

void GBMRegressor::print_summary() const { gbm_.print_summary(); }

std::vector<double> GBMRegressor::predict(const DataFrame& newdata) const { return gbm_.predict(newdata.frame_); }

datamunge::plot::ScatterPlot GBMRegressor::plot_predicted_vs_actual() const { return gbm_.plot_predicted_vs_actual(); }

datamunge::plot::ScatterPlot GBMRegressor::plot_residuals_vs_fitted() const { return gbm_.plot_residuals_vs_fitted(); }

datamunge::plot::ScatterPlot GBMRegressor::plot_training_deviance() const { return gbm_.plot_training_deviance(); }

XGBoostClassifier::XGBoostClassifier(const DataFrame& data, const std::string& formula, const std::size_t n_trees,
                                     const double learning_rate, const std::size_t max_depth, const double lambda,
                                     const double alpha, const double gamma, const double min_child_weight,
                                     const std::size_t min_samples_leaf, const double subsample,
                                     const double colsample_bytree, const std::uint64_t seed)
    : xgb_(data.frame_, formula, [&] {
        stats::XGBoostClassifierOptions options;
        options.n_trees           = n_trees;
        options.learning_rate     = learning_rate;
        options.max_depth         = max_depth;
        options.lambda            = lambda;
        options.alpha             = alpha;
        options.gamma             = gamma;
        options.min_child_weight  = min_child_weight;
        options.min_samples_leaf  = min_samples_leaf;
        options.subsample         = subsample;
        options.colsample_bytree  = colsample_bytree;
        options.seed              = seed;
        return options;
      }()) {}

std::vector<std::string> XGBoostClassifier::classes() const { return xgb_.classes(); }

std::vector<std::string> XGBoostClassifier::predictor_names() const { return xgb_.predictor_names(); }

std::size_t XGBoostClassifier::observations() const { return xgb_.observations(); }

std::size_t XGBoostClassifier::n_trees() const { return xgb_.n_trees(); }

std::vector<double> XGBoostClassifier::feature_importance() const { return xgb_.feature_importance(); }

double XGBoostClassifier::training_accuracy() const { return xgb_.training_accuracy(); }

DataFrame* XGBoostClassifier::confusion_matrix() const {
  dstruct::DataFrame frame;
  const auto& classes = xgb_.classes();
  frame.add_column("actual", classes);
  const auto counts = xgb_.confusion_matrix();
  for (std::size_t predicted = 0; predicted < classes.size(); ++predicted) {
    std::vector<double> column;
    for (std::size_t actual = 0; actual < classes.size(); ++actual) column.push_back(counts(actual, predicted));
    frame.add_column(classes[predicted], column);
  }
  return new DataFrame(std::move(frame));
}

std::vector<double> XGBoostClassifier::training_deviance() const { return xgb_.training_deviance(); }

std::string XGBoostClassifier::summary() const { return xgb_.summary(); }

void XGBoostClassifier::print_summary() const { xgb_.print_summary(); }

std::vector<std::string> XGBoostClassifier::predict(const DataFrame& newdata) const {
  return xgb_.predict(newdata.frame_);
}

DataFrame* XGBoostClassifier::predict_frame(const DataFrame& newdata) const {
  const auto detail = xgb_.predict_detail(newdata.frame_);

  std::vector<std::optional<std::string>> labels;
  labels.reserve(detail.class_label.size());
  for (const auto& label : detail.class_label)
    labels.emplace_back(label.empty() ? std::nullopt : std::optional<std::string>(label));

  dstruct::DataFrame frame;
  frame.add_column("class", labels);

  const auto& classes = xgb_.classes();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.probability.size());
    for (const auto& row : detail.probability)
      column.emplace_back(row.size() > c ? std::optional<double>(row[c]) : std::nullopt);
    frame.add_column("prob_" + classes[c], column);
  }

  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot XGBoostClassifier::plot_classification(const DataFrame& data,
                                                                     const std::string& x_feature,
                                                                     const std::string& y_feature) const {
  return xgb_.plot_classification(data.frame_, x_feature, y_feature);
}

datamunge::plot::ScatterPlot XGBoostClassifier::plot_decision_regions(const std::string& x_feature,
                                                                       const std::string& y_feature,
                                                                       const std::size_t grid_resolution) const {
  return xgb_.plot_decision_regions(x_feature, y_feature, grid_resolution);
}

datamunge::plot::ScatterPlot XGBoostClassifier::plot_training_deviance() const { return xgb_.plot_training_deviance(); }

XGBoostRegressor::XGBoostRegressor(const DataFrame& data, const std::string& formula, const std::size_t n_trees,
                                   const double learning_rate, const std::size_t max_depth, const double lambda,
                                   const double alpha, const double gamma, const double min_child_weight,
                                   const std::size_t min_samples_leaf, const double subsample,
                                   const double colsample_bytree, const std::uint64_t seed)
    : xgb_(data.frame_, formula, [&] {
        stats::XGBoostRegressorOptions options;
        options.n_trees           = n_trees;
        options.learning_rate     = learning_rate;
        options.max_depth         = max_depth;
        options.lambda            = lambda;
        options.alpha             = alpha;
        options.gamma             = gamma;
        options.min_child_weight  = min_child_weight;
        options.min_samples_leaf  = min_samples_leaf;
        options.subsample         = subsample;
        options.colsample_bytree  = colsample_bytree;
        options.seed              = seed;
        return options;
      }()) {}

std::vector<std::string> XGBoostRegressor::predictor_names() const { return xgb_.predictor_names(); }

std::size_t XGBoostRegressor::observations() const { return xgb_.observations(); }

std::size_t XGBoostRegressor::n_trees() const { return xgb_.n_trees(); }

std::vector<double> XGBoostRegressor::feature_importance() const { return xgb_.feature_importance(); }

std::vector<double> XGBoostRegressor::fitted_values() const { return xgb_.fitted_values(); }

double XGBoostRegressor::r_squared() const { return xgb_.r_squared(); }

double XGBoostRegressor::rmse() const { return xgb_.rmse(); }

std::vector<double> XGBoostRegressor::training_deviance() const { return xgb_.training_deviance(); }

std::string XGBoostRegressor::summary() const { return xgb_.summary(); }

void XGBoostRegressor::print_summary() const { xgb_.print_summary(); }

std::vector<double> XGBoostRegressor::predict(const DataFrame& newdata) const { return xgb_.predict(newdata.frame_); }

datamunge::plot::ScatterPlot XGBoostRegressor::plot_predicted_vs_actual() const { return xgb_.plot_predicted_vs_actual(); }

datamunge::plot::ScatterPlot XGBoostRegressor::plot_residuals_vs_fitted() const { return xgb_.plot_residuals_vs_fitted(); }

datamunge::plot::ScatterPlot XGBoostRegressor::plot_training_deviance() const { return xgb_.plot_training_deviance(); }

KernelRegression::KernelRegression(const DataFrame& data, const std::string& formula, const std::string& kernel,
                                   const double bandwidth, const std::size_t n_bandwidth, const bool standardize)
    : kernel_regression_(data.frame_, formula, [&] {
        stats::KernelRegressionOptions options;
        options.kernel       = parse_kernel_regression_kernel(kernel);
        options.bandwidth     = bandwidth;
        options.n_bandwidth   = n_bandwidth;
        options.standardize   = standardize;
        return options;
      }()) {}

std::vector<std::string> KernelRegression::predictor_names() const { return kernel_regression_.predictor_names(); }

std::size_t KernelRegression::observations() const { return kernel_regression_.observations(); }

double KernelRegression::bandwidth() const { return kernel_regression_.bandwidth(); }

bool KernelRegression::bandwidth_was_selected() const { return kernel_regression_.bandwidth_was_selected(); }

std::vector<double> KernelRegression::bandwidth_grid() const { return kernel_regression_.bandwidth_grid(); }

std::vector<double> KernelRegression::cv_mean_squared_error() const {
  return kernel_regression_.cv_mean_squared_error();
}

std::vector<double> KernelRegression::fitted_values() const { return kernel_regression_.fitted_values(); }

double KernelRegression::r_squared() const { return kernel_regression_.r_squared(); }

double KernelRegression::rmse() const { return kernel_regression_.rmse(); }

std::string KernelRegression::summary() const { return kernel_regression_.summary(); }

void KernelRegression::print_summary() const { kernel_regression_.print_summary(); }

std::vector<double> KernelRegression::predict(const DataFrame& newdata) const {
  return kernel_regression_.predict(newdata.frame_);
}

datamunge::plot::ScatterPlot KernelRegression::plot_fit(const DataFrame& data,
                                                        const std::size_t grid_resolution) const {
  return kernel_regression_.plot_fit(data.frame_, grid_resolution);
}

datamunge::plot::ScatterPlot KernelRegression::plot_predicted_vs_actual() const {
  return kernel_regression_.plot_predicted_vs_actual();
}

datamunge::plot::ScatterPlot KernelRegression::plot_residuals_vs_fitted() const {
  return kernel_regression_.plot_residuals_vs_fitted();
}

datamunge::plot::ScatterPlot KernelRegression::plot_cv_curve() const { return kernel_regression_.plot_cv_curve(); }

GaussianProcessRegression::GaussianProcessRegression(const DataFrame& data, const std::string& formula,
                                                      const double length_scale, const double noise_ratio,
                                                      const std::size_t n_length_scale_grid,
                                                      const std::size_t n_noise_grid, const bool standardize)
    : gpr_(data.frame_, formula, [&] {
        stats::GaussianProcessRegressionOptions options;
        options.length_scale         = length_scale;
        options.noise_ratio           = noise_ratio;
        options.n_length_scale_grid   = n_length_scale_grid;
        options.n_noise_grid          = n_noise_grid;
        options.standardize           = standardize;
        return options;
      }()) {}

std::vector<std::string> GaussianProcessRegression::predictor_names() const { return gpr_.predictor_names(); }

std::size_t GaussianProcessRegression::observations() const { return gpr_.observations(); }

double GaussianProcessRegression::length_scale() const { return gpr_.length_scale(); }

double GaussianProcessRegression::signal_variance() const { return gpr_.signal_variance(); }

double GaussianProcessRegression::noise_variance() const { return gpr_.noise_variance(); }

double GaussianProcessRegression::log_marginal_likelihood() const { return gpr_.log_marginal_likelihood(); }

bool GaussianProcessRegression::length_scale_was_selected() const { return gpr_.length_scale_was_selected(); }

bool GaussianProcessRegression::noise_ratio_was_selected() const { return gpr_.noise_ratio_was_selected(); }

std::vector<double> GaussianProcessRegression::length_scale_grid() const { return gpr_.length_scale_grid(); }

std::vector<double> GaussianProcessRegression::length_scale_profile_log_likelihood() const {
  return gpr_.length_scale_profile_log_likelihood();
}

std::vector<double> GaussianProcessRegression::fitted_values() const { return gpr_.fitted_values(); }

double GaussianProcessRegression::r_squared() const { return gpr_.r_squared(); }

double GaussianProcessRegression::rmse() const { return gpr_.rmse(); }

std::string GaussianProcessRegression::summary() const { return gpr_.summary(); }

void GaussianProcessRegression::print_summary() const { gpr_.print_summary(); }

std::vector<double> GaussianProcessRegression::predict(const DataFrame& newdata) const {
  return gpr_.predict(newdata.frame_);
}

DataFrame* GaussianProcessRegression::predict_frame(const DataFrame& newdata, const std::string& interval_kind,
                                                    const double level) const {
  const bool want_interval = (interval_kind == "confidence");
  if (!want_interval && interval_kind != "none" && !interval_kind.empty())
    throw std::invalid_argument("GaussianProcessRegression::predict_frame: interval_kind must be 'none' or "
                                "'confidence'");

  const auto detail = gpr_.predict_detail(newdata.frame_, level);
  dstruct::DataFrame frame;
  frame.add_column("fit", detail.fit);
  if (want_interval) {
    frame.add_column("se_fit", detail.se_fit);
    frame.add_column("lwr", detail.lower);
    frame.add_column("upr", detail.upper);
  }
  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot GaussianProcessRegression::plot_fit(const DataFrame& data,
                                                                  const std::size_t grid_resolution,
                                                                  const double level) const {
  return gpr_.plot_fit(data.frame_, grid_resolution, level);
}

datamunge::plot::ScatterPlot GaussianProcessRegression::plot_predicted_vs_actual() const {
  return gpr_.plot_predicted_vs_actual();
}

datamunge::plot::ScatterPlot GaussianProcessRegression::plot_residuals_vs_fitted() const {
  return gpr_.plot_residuals_vs_fitted();
}

datamunge::plot::ScatterPlot GaussianProcessRegression::plot_length_scale_profile() const {
  return gpr_.plot_length_scale_profile();
}

NaiveBayesClassifier::NaiveBayesClassifier(const DataFrame& data, const std::string& formula,
                                           const double laplace_smoothing, const double var_smoothing)
    : nb_(data.frame_, formula, [&] {
        stats::NaiveBayesClassifierOptions options;
        options.laplace_smoothing = laplace_smoothing;
        options.var_smoothing     = var_smoothing;
        return options;
      }()) {}

std::vector<std::string> NaiveBayesClassifier::classes() const { return nb_.classes(); }

std::vector<std::string> NaiveBayesClassifier::predictor_names() const { return nb_.predictor_names(); }

std::size_t NaiveBayesClassifier::observations() const { return nb_.observations(); }

std::vector<double> NaiveBayesClassifier::class_priors() const { return nb_.class_priors(); }

double NaiveBayesClassifier::training_accuracy() const { return nb_.training_accuracy(); }

DataFrame* NaiveBayesClassifier::confusion_matrix() const {
  dstruct::DataFrame frame;
  const auto& classes = nb_.classes();
  frame.add_column("actual", classes);
  const auto counts = nb_.confusion_matrix();
  for (std::size_t predicted = 0; predicted < classes.size(); ++predicted) {
    std::vector<double> column;
    for (std::size_t actual = 0; actual < classes.size(); ++actual) column.push_back(counts(actual, predicted));
    frame.add_column(classes[predicted], column);
  }
  return new DataFrame(std::move(frame));
}

std::string NaiveBayesClassifier::summary() const { return nb_.summary(); }

void NaiveBayesClassifier::print_summary() const { nb_.print_summary(); }

std::vector<std::string> NaiveBayesClassifier::predict(const DataFrame& newdata) const {
  return nb_.predict(newdata.frame_);
}

DataFrame* NaiveBayesClassifier::predict_frame(const DataFrame& newdata) const {
  const auto detail = nb_.predict_detail(newdata.frame_);

  std::vector<std::optional<std::string>> labels;
  labels.reserve(detail.class_label.size());
  for (const auto& label : detail.class_label)
    labels.emplace_back(label.empty() ? std::nullopt : std::optional<std::string>(label));

  dstruct::DataFrame frame;
  frame.add_column("class", labels);

  const auto& classes = nb_.classes();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<std::optional<double>> column;
    column.reserve(detail.probability.size());
    for (const auto& row : detail.probability)
      column.emplace_back(row.size() > c ? std::optional<double>(row[c]) : std::nullopt);
    frame.add_column("prob_" + classes[c], column);
  }

  return new DataFrame(std::move(frame));
}

datamunge::plot::ScatterPlot NaiveBayesClassifier::plot_classification(const DataFrame& data,
                                                                        const std::string& x_feature,
                                                                        const std::string& y_feature) const {
  return nb_.plot_classification(data.frame_, x_feature, y_feature);
}

datamunge::plot::ScatterPlot NaiveBayesClassifier::plot_decision_regions(const std::string& x_feature,
                                                                          const std::string& y_feature,
                                                                          const std::size_t grid_resolution) const {
  return nb_.plot_decision_regions(x_feature, y_feature, grid_resolution);
}

GLM::GLM(const DataFrame& data, const std::string& formula, const std::string& family,
        const std::string& weights_column, const std::size_t max_iter, const double tol)
    : glm_(data.frame_, formula, [&] {
        stats::GLMOptions options;
        options.family = parse_glm_family(family);
        if (!weights_column.empty()) options.weights_column = weights_column;
        options.max_iter = max_iter;
        options.tol       = tol;
        return options;
      }()) {}

std::string GLM::formula_text() const { return glm_.formula_text(); }

std::string GLM::family() const {
  switch (glm_.family()) {
    case stats::GLMFamily::Gaussian: return "gaussian";
    case stats::GLMFamily::Binomial: return "binomial";
    case stats::GLMFamily::Poisson: return "poisson";
    case stats::GLMFamily::Gamma: return "Gamma";
  }
  return "unknown";
}

bool GLM::has_intercept() const { return glm_.has_intercept(); }

std::size_t GLM::observations() const { return glm_.observations(); }

std::size_t GLM::rank() const { return glm_.rank(); }

std::size_t GLM::degrees_of_freedom() const { return glm_.degrees_of_freedom(); }

std::vector<double> GLM::coefficients() const { return glm_.coefficients(); }

std::vector<std::string> GLM::coefficient_names() const { return glm_.coefficient_names(); }

std::vector<double> GLM::fitted_values() const { return glm_.fitted_values(); }

std::vector<double> GLM::linear_predictors() const { return glm_.linear_predictors(); }

std::vector<double> GLM::residuals() const { return glm_.residuals(); }

std::vector<double> GLM::pearson_residuals() const { return glm_.pearson_residuals(); }

std::vector<double> GLM::standardized_residuals() const { return glm_.standardized_residuals(); }

std::vector<double> GLM::leverage() const { return glm_.leverage(); }

std::vector<double> GLM::standard_errors() const { return glm_.standard_errors(); }

std::vector<double> GLM::test_statistics() const { return glm_.test_statistics(); }

std::vector<double> GLM::p_values() const { return glm_.p_values(); }

double GLM::deviance() const { return glm_.deviance(); }

double GLM::null_deviance() const { return glm_.null_deviance(); }

double GLM::dispersion() const { return glm_.dispersion(); }

double GLM::aic() const { return glm_.aic(); }

std::vector<double> GLM::confidence_interval_lower(const double level) const {
  std::vector<double> out;
  out.reserve(glm_.rank());
  for (const auto& interval : glm_.confidence_intervals(level)) out.push_back(interval.lower);
  return out;
}

std::vector<double> GLM::confidence_interval_upper(const double level) const {
  std::vector<double> out;
  out.reserve(glm_.rank());
  for (const auto& interval : glm_.confidence_intervals(level)) out.push_back(interval.upper);
  return out;
}

std::string GLM::summary() const { return glm_.summary(); }

void GLM::print_summary() const { glm_.print_summary(); }

std::vector<double> GLM::predict(const DataFrame& newdata) const { return glm_.predict(newdata.frame_); }

DataFrame* GLM::predict_frame(const DataFrame& newdata, const std::string& interval_kind, const double level) const {
  auto kind = stats::GLMPredictionInterval::None;
  if (interval_kind == "confidence") {
    kind = stats::GLMPredictionInterval::Confidence;
  } else if (interval_kind != "none" && !interval_kind.empty()) {
    throw std::invalid_argument("GLM::predict_frame: interval_kind must be 'none' or 'confidence'");
  }
  return new DataFrame(glm_.predict_frame(newdata.frame_, kind, level));
}

datamunge::plot::ScatterPlot GLM::plot_residuals_vs_fitted() const { return glm_.plot_residuals_vs_fitted(); }

datamunge::plot::ScatterPlot GLM::plot_normal_qq() const { return glm_.plot_normal_qq(); }

datamunge::plot::ScatterPlot GLM::plot_scale_location() const { return glm_.plot_scale_location(); }

datamunge::plot::ScatterPlot GLM::plot_residuals_vs_leverage() const { return glm_.plot_residuals_vs_leverage(); }

void GLM::save_diagnostic_plots(const std::string& path_prefix) const { glm_.save_diagnostic_plots(path_prefix); }

Tensor::Tensor(const std::vector<std::size_t>& shape, const std::string& dtype)
    : tensor_(shape, parse_tensor_dtype(dtype)) {}

Tensor::Tensor(linalg::Tensor tensor) : tensor_(std::move(tensor)) {}

Tensor* Tensor::zeros(const std::vector<std::size_t>& shape) { return new Tensor(linalg::Tensor::zeros(shape)); }

Tensor* Tensor::ones(const std::vector<std::size_t>& shape) { return new Tensor(linalg::Tensor::ones(shape)); }

Tensor* Tensor::full(const std::vector<std::size_t>& shape, const double value) {
  return new Tensor(linalg::Tensor::full(shape, value));
}

Tensor* Tensor::from_values(const std::vector<std::size_t>& shape, const std::vector<double>& values) {
  return new Tensor(linalg::Tensor::from_values(shape, values));
}

Tensor* Tensor::from_bool_values(const std::vector<std::size_t>& shape, const std::vector<int>& values) {
  return new Tensor(linalg::Tensor::from_bool_values(shape, values));
}

Tensor* Tensor::from_string_values(const std::vector<std::size_t>& shape, const std::vector<std::string>& values) {
  return new Tensor(linalg::Tensor::from_string_values(shape, values));
}

Tensor* Tensor::arange(const double start, const double stop, const double step) {
  return new Tensor(linalg::Tensor::arange(start, stop, step));
}

Tensor* Tensor::eye(const std::size_t n) { return new Tensor(linalg::Tensor::eye(n)); }

std::size_t Tensor::ndim() const { return tensor_.ndim(); }

std::vector<std::size_t> Tensor::shape() const { return tensor_.shape(); }

std::size_t Tensor::size() const { return tensor_.size(); }

std::string Tensor::dtype_name() const { return tensor_.dtype_name(); }

double Tensor::at(const std::vector<std::size_t>& index) const { return tensor_.at(index); }

void Tensor::set(const std::vector<std::size_t>& index, const double value) { tensor_.set(index, value); }

std::string Tensor::string_at(const std::vector<std::size_t>& index) const { return tensor_.string_at(index); }

void Tensor::set_string(const std::vector<std::size_t>& index, const std::string& value) {
  tensor_.set_string(index, value);
}

double Tensor::at_flat(const std::size_t i) const { return tensor_.at_flat(i); }

void Tensor::set_flat(const std::size_t i, const double value) { tensor_.set_flat(i, value); }

std::string Tensor::string_at_flat(const std::size_t i) const { return tensor_.string_at_flat(i); }

void Tensor::set_string_flat(const std::size_t i, const std::string& value) { tensor_.set_string_flat(i, value); }

Tensor* Tensor::reshape(const std::vector<std::size_t>& new_shape) const {
  return new Tensor(tensor_.reshape(new_shape));
}

Tensor* Tensor::flatten() const { return new Tensor(tensor_.flatten()); }

Tensor* Tensor::transpose(const std::vector<std::size_t>& permutation) const {
  return new Tensor(tensor_.transpose(permutation));
}

Tensor* Tensor::squeeze() const { return new Tensor(tensor_.squeeze()); }

Tensor* Tensor::squeeze_axis(const std::size_t axis) const { return new Tensor(tensor_.squeeze_axis(axis)); }

Tensor* Tensor::expand_dims(const std::size_t axis) const { return new Tensor(tensor_.expand_dims(axis)); }

Tensor* Tensor::slice(const std::size_t axis, const std::size_t start, const std::size_t stop,
                      const std::size_t step) const {
  return new Tensor(tensor_.slice(axis, start, stop, step));
}

Tensor* Tensor::index_select(const std::size_t axis, const std::vector<std::size_t>& indices) const {
  return new Tensor(tensor_.index_select(axis, indices));
}

Tensor* Tensor::concatenate2(const Tensor& a, const Tensor& b, const std::size_t axis) {
  return new Tensor(linalg::Tensor::concatenate({a.tensor_, b.tensor_}, axis));
}

Tensor* Tensor::stack2(const Tensor& a, const Tensor& b, const std::size_t axis) {
  return new Tensor(linalg::Tensor::stack({a.tensor_, b.tensor_}, axis));
}

Tensor* Tensor::add(const Tensor& other) const { return new Tensor(tensor_.add(other.tensor_)); }

Tensor* Tensor::subtract(const Tensor& other) const { return new Tensor(tensor_.subtract(other.tensor_)); }

Tensor* Tensor::multiply(const Tensor& other) const { return new Tensor(tensor_.multiply(other.tensor_)); }

Tensor* Tensor::divide(const Tensor& other) const { return new Tensor(tensor_.divide(other.tensor_)); }

Tensor* Tensor::power(const Tensor& other) const { return new Tensor(tensor_.power(other.tensor_)); }

Tensor* Tensor::add_scalar(const double scalar) const { return new Tensor(tensor_.add_scalar(scalar)); }

Tensor* Tensor::subtract_scalar(const double scalar) const { return new Tensor(tensor_.subtract_scalar(scalar)); }

Tensor* Tensor::multiply_scalar(const double scalar) const { return new Tensor(tensor_.multiply_scalar(scalar)); }

Tensor* Tensor::divide_scalar(const double scalar) const { return new Tensor(tensor_.divide_scalar(scalar)); }

Tensor* Tensor::power_scalar(const double exponent) const { return new Tensor(tensor_.power_scalar(exponent)); }

Tensor* Tensor::negate() const { return new Tensor(tensor_.negate()); }

Tensor* Tensor::abs() const { return new Tensor(tensor_.abs()); }

Tensor* Tensor::sqrt() const { return new Tensor(tensor_.sqrt()); }

Tensor* Tensor::exp() const { return new Tensor(tensor_.exp()); }

Tensor* Tensor::log() const { return new Tensor(tensor_.log()); }

Tensor* Tensor::from_image(const datamunge::image::Image& img) { return new Tensor(cv::image_to_tensor(img)); }

Tensor* Tensor::conv2d(const Tensor& input, const Tensor& kernel, const Tensor& bias, int stride, int padding) {
  return new Tensor(cv::conv2d(input.tensor_, kernel.tensor_, bias.tensor_, stride, padding));
}

Tensor* Tensor::max_pool2d(int pool_size, int stride) const { return new Tensor(cv::max_pool2d(tensor_, pool_size, stride)); }

Tensor* Tensor::avg_pool2d(int pool_size, int stride) const { return new Tensor(cv::avg_pool2d(tensor_, pool_size, stride)); }

Tensor* Tensor::relu() const { return new Tensor(cv::relu(tensor_)); }

Tensor* Tensor::sigmoid() const { return new Tensor(cv::sigmoid(tensor_)); }

Tensor* Tensor::softmax() const { return new Tensor(cv::softmax(tensor_)); }

Tensor* Tensor::apply(Callback* callback) const {
  if (!callback) throw std::invalid_argument("Tensor::apply: callback must not be null");
  return new Tensor(tensor_.apply([callback](const double x) { return callback->call(x); }));
}

Tensor* Tensor::equal(const Tensor& other) const { return new Tensor(tensor_.equal(other.tensor_)); }

Tensor* Tensor::not_equal(const Tensor& other) const { return new Tensor(tensor_.not_equal(other.tensor_)); }

Tensor* Tensor::less(const Tensor& other) const { return new Tensor(tensor_.less(other.tensor_)); }

Tensor* Tensor::less_equal(const Tensor& other) const { return new Tensor(tensor_.less_equal(other.tensor_)); }

Tensor* Tensor::greater(const Tensor& other) const { return new Tensor(tensor_.greater(other.tensor_)); }

Tensor* Tensor::greater_equal(const Tensor& other) const { return new Tensor(tensor_.greater_equal(other.tensor_)); }

double Tensor::sum() const { return tensor_.sum(); }

double Tensor::mean() const { return tensor_.mean(); }

double Tensor::max() const { return tensor_.max(); }

double Tensor::min() const { return tensor_.min(); }

double Tensor::prod() const { return tensor_.prod(); }

std::size_t Tensor::argmax() const { return tensor_.argmax(); }

std::size_t Tensor::argmin() const { return tensor_.argmin(); }

bool Tensor::all() const { return tensor_.all(); }

bool Tensor::any() const { return tensor_.any(); }

Tensor* Tensor::sum_axis(const std::size_t axis, const bool keepdims) const {
  return new Tensor(tensor_.sum_axis(axis, keepdims));
}

Tensor* Tensor::mean_axis(const std::size_t axis, const bool keepdims) const {
  return new Tensor(tensor_.mean_axis(axis, keepdims));
}

Tensor* Tensor::max_axis(const std::size_t axis, const bool keepdims) const {
  return new Tensor(tensor_.max_axis(axis, keepdims));
}

Tensor* Tensor::min_axis(const std::size_t axis, const bool keepdims) const {
  return new Tensor(tensor_.min_axis(axis, keepdims));
}

Tensor* Tensor::prod_axis(const std::size_t axis, const bool keepdims) const {
  return new Tensor(tensor_.prod_axis(axis, keepdims));
}

Tensor* Tensor::argmax_axis(const std::size_t axis, const bool keepdims) const {
  return new Tensor(tensor_.argmax_axis(axis, keepdims));
}

Tensor* Tensor::argmin_axis(const std::size_t axis, const bool keepdims) const {
  return new Tensor(tensor_.argmin_axis(axis, keepdims));
}

Tensor* Tensor::matmul(const Tensor& other) const { return new Tensor(tensor_.matmul(other.tensor_)); }

double Tensor::dot(const Tensor& other) const { return tensor_.dot(other.tensor_); }

Tensor* Tensor::outer(const Tensor& other) const { return new Tensor(tensor_.outer(other.tensor_)); }

std::string Tensor::to_string(const std::size_t max_elements) const { return tensor_.to_string(max_elements); }

Dual::Dual(const double value, const double derivative) : dual_(value, derivative) {}

Dual::Dual(autodiff::Dual dual) : dual_(dual) {}

double Dual::value() const { return dual_.value(); }

double Dual::derivative() const { return dual_.derivative(); }

Dual Dual::add(const Dual& other) const { return Dual(dual_ + other.dual_); }

Dual Dual::subtract(const Dual& other) const { return Dual(dual_ - other.dual_); }

Dual Dual::multiply(const Dual& other) const { return Dual(dual_ * other.dual_); }

Dual Dual::divide(const Dual& other) const { return Dual(dual_ / other.dual_); }

Dual Dual::negate() const { return Dual(-dual_); }

Dual Dual::add_scalar(const double scalar) const { return Dual(dual_ + scalar); }

Dual Dual::subtract_scalar(const double scalar) const { return Dual(dual_ - scalar); }

Dual Dual::multiply_scalar(const double scalar) const { return Dual(dual_ * scalar); }

Dual Dual::divide_scalar(const double scalar) const { return Dual(dual_ / scalar); }

Dual Dual::pow(const double exponent) const { return Dual(autodiff::pow(dual_, exponent)); }

Dual Dual::exp() const { return Dual(autodiff::exp(dual_)); }

Dual Dual::log() const { return Dual(autodiff::log(dual_)); }

Dual Dual::sqrt() const { return Dual(autodiff::sqrt(dual_)); }

Dual Dual::sin() const { return Dual(autodiff::sin(dual_)); }

Dual Dual::cos() const { return Dual(autodiff::cos(dual_)); }

Dual Dual::tan() const { return Dual(autodiff::tan(dual_)); }

Dual Dual::tanh() const { return Dual(autodiff::tanh(dual_)); }

Dual Dual::abs() const { return Dual(autodiff::abs(dual_)); }

HyperDual::HyperDual(const double value, const double eps1, const double eps2, const double eps1eps2)
    : dual_(value, eps1, eps2, eps1eps2) {}

HyperDual::HyperDual(autodiff::HyperDual dual) : dual_(dual) {}

double HyperDual::value() const { return dual_.value(); }

double HyperDual::eps1() const { return dual_.eps1(); }

double HyperDual::eps2() const { return dual_.eps2(); }

double HyperDual::eps1eps2() const { return dual_.eps1eps2(); }

HyperDual HyperDual::add(const HyperDual& other) const { return HyperDual(dual_ + other.dual_); }

HyperDual HyperDual::subtract(const HyperDual& other) const { return HyperDual(dual_ - other.dual_); }

HyperDual HyperDual::multiply(const HyperDual& other) const { return HyperDual(dual_ * other.dual_); }

HyperDual HyperDual::divide(const HyperDual& other) const { return HyperDual(dual_ / other.dual_); }

HyperDual HyperDual::negate() const { return HyperDual(-dual_); }

HyperDual HyperDual::add_scalar(const double scalar) const { return HyperDual(dual_ + scalar); }

HyperDual HyperDual::subtract_scalar(const double scalar) const { return HyperDual(dual_ - scalar); }

HyperDual HyperDual::multiply_scalar(const double scalar) const { return HyperDual(dual_ * scalar); }

HyperDual HyperDual::divide_scalar(const double scalar) const { return HyperDual(dual_ / scalar); }

HyperDual HyperDual::pow(const double exponent) const { return HyperDual(autodiff::pow(dual_, exponent)); }

HyperDual HyperDual::exp() const { return HyperDual(autodiff::exp(dual_)); }

HyperDual HyperDual::log() const { return HyperDual(autodiff::log(dual_)); }

HyperDual HyperDual::sqrt() const { return HyperDual(autodiff::sqrt(dual_)); }

HyperDual HyperDual::sin() const { return HyperDual(autodiff::sin(dual_)); }

HyperDual HyperDual::cos() const { return HyperDual(autodiff::cos(dual_)); }

HyperDual HyperDual::tan() const { return HyperDual(autodiff::tan(dual_)); }

HyperDual HyperDual::tanh() const { return HyperDual(autodiff::tanh(dual_)); }

std::size_t Tape::size() const { return tape_.size(); }

double Tape::value_at(const std::size_t index) const { return tape_.value_at(index); }

std::vector<double> Tape::backward(const Var& output) const { return tape_.backward(output.var_.index()); }

Var::Var(Tape& tape, const double value) : var_(tape.tape_, value) {}

Var::Var(autodiff::Var var) : var_(var) {}

double Var::value() const { return var_.value(); }

std::size_t Var::index() const { return var_.index(); }

Var Var::add(const Var& other) const { return Var(var_ + other.var_); }

Var Var::subtract(const Var& other) const { return Var(var_ - other.var_); }

Var Var::multiply(const Var& other) const { return Var(var_ * other.var_); }

Var Var::divide(const Var& other) const { return Var(var_ / other.var_); }

Var Var::negate() const { return Var(-var_); }

Var Var::add_scalar(const double scalar) const { return Var(var_ + scalar); }

Var Var::subtract_scalar(const double scalar) const { return Var(var_ - scalar); }

Var Var::multiply_scalar(const double scalar) const { return Var(var_ * scalar); }

Var Var::divide_scalar(const double scalar) const { return Var(var_ / scalar); }

Var Var::pow(const double exponent) const { return Var(autodiff::pow(var_, exponent)); }

Var Var::exp() const { return Var(autodiff::exp(var_)); }

Var Var::log() const { return Var(autodiff::log(var_)); }

Var Var::sqrt() const { return Var(autodiff::sqrt(var_)); }

Var Var::sin() const { return Var(autodiff::sin(var_)); }

Var Var::cos() const { return Var(autodiff::cos(var_)); }

Var Var::tan() const { return Var(autodiff::tan(var_)); }

Var Var::tanh() const { return Var(autodiff::tanh(var_)); }

Var Var::abs() const { return Var(autodiff::abs(var_)); }

GLMM::GLMM(const DataFrame& data, const std::string& formula, const std::string& family, const std::size_t max_iterations,
          const double tol, const std::size_t de_population_size, const std::size_t de_max_generations,
          const double theta_bound, const std::size_t seed)
    : glmm_(data.frame_, formula, [&] {
        stats::GLMMOptions options;
        options.family             = parse_glmm_family(family);
        options.max_iterations     = max_iterations;
        options.tol                 = tol;
        options.de_population_size = de_population_size;
        options.de_max_generations = de_max_generations;
        options.theta_bound        = theta_bound;
        options.seed                 = seed;
        return options;
      }()) {}

std::string GLMM::formula_text() const { return glmm_.formula_text(); }

std::string GLMM::family() const { return glmm_.family(); }

std::string GLMM::group_variable() const { return glmm_.group_variable(); }

bool GLMM::has_random_intercept() const { return glmm_.has_random_intercept(); }

std::vector<std::string> GLMM::random_effect_names() const { return glmm_.random_effect_names(); }

std::size_t GLMM::observations() const { return glmm_.observations(); }

std::size_t GLMM::num_groups() const { return glmm_.num_groups(); }

std::size_t GLMM::rank() const { return glmm_.rank(); }

std::size_t GLMM::iterations() const { return glmm_.iterations(); }

std::vector<double> GLMM::coefficients() const { return glmm_.coefficients(); }

std::vector<std::string> GLMM::coefficient_names() const { return glmm_.coefficient_names(); }

std::vector<double> GLMM::standard_errors() const { return glmm_.standard_errors(); }

std::vector<double> GLMM::z_values() const { return glmm_.z_values(); }

std::vector<double> GLMM::p_values() const { return glmm_.p_values(); }

std::vector<double> GLMM::fitted_values() const { return glmm_.fitted_values(); }

std::vector<double> GLMM::random_effect_std_devs() const { return glmm_.random_effect_std_devs(); }

double GLMM::random_effect_correlation(const std::size_t i, const std::size_t j) const {
  return glmm_.random_effect_correlation(i, j);
}

std::vector<std::string> GLMM::group_labels() const { return glmm_.group_labels(); }

std::vector<double> GLMM::random_effects_for_group(const std::size_t group_index) const {
  const auto& effects = glmm_.random_effects();
  if (group_index >= effects.size()) throw std::invalid_argument("GLMM::random_effects_for_group: group_index out of range");
  return effects[group_index];
}

double GLMM::deviance() const { return glmm_.deviance(); }

double GLMM::aic() const { return glmm_.aic(); }

double GLMM::bic() const { return glmm_.bic(); }

std::string GLMM::summary() const { return glmm_.summary(); }

void GLMM::print_summary() const { glmm_.print_summary(); }

std::vector<double> GLMM::predict(const DataFrame& newdata) const { return glmm_.predict(newdata.frame_); }

INLAMixedModel::INLAMixedModel(const DataFrame& data, const std::string& formula, const std::string& family,
          const std::string& strategy, const double fixed_effect_prior_sd, const std::size_t grid_points_per_dim,
          const double grid_span, const std::size_t mode_population_size, const std::size_t mode_max_generations,
          const std::size_t seed)
    : model_(data.frame_, formula, [&] {
        stats::INLAMixedModelOptions options;
        options.family                 = parse_inla_family(family);
        options.strategy               = parse_inla_strategy(strategy);
        options.fixed_effect_prior_sd  = fixed_effect_prior_sd;
        options.grid_points_per_dim    = grid_points_per_dim;
        options.grid_span              = grid_span;
        options.mode_population_size   = mode_population_size;
        options.mode_max_generations   = mode_max_generations;
        options.seed                   = seed;
        return options;
      }()) {}

std::string INLAMixedModel::formula_text() const { return model_.formula_text(); }

std::string INLAMixedModel::family() const { return model_.family(); }

std::string INLAMixedModel::group_variable() const { return model_.group_variable(); }

std::vector<std::string> INLAMixedModel::random_effect_names() const { return model_.random_effect_names(); }

std::size_t INLAMixedModel::observations() const { return model_.observations(); }

std::size_t INLAMixedModel::num_groups() const { return model_.num_groups(); }

std::vector<double> INLAMixedModel::fixed_effects_mean() const { return model_.fixed_effects_mean(); }

std::vector<double> INLAMixedModel::fixed_effects_sd() const { return model_.fixed_effects_sd(); }

std::vector<std::string> INLAMixedModel::coefficient_names() const { return model_.coefficient_names(); }

std::vector<double> INLAMixedModel::random_effect_std_devs() const { return model_.random_effect_std_devs(); }

double INLAMixedModel::residual_std_dev() const { return model_.residual_std_dev(); }

std::vector<std::string> INLAMixedModel::group_labels() const { return model_.group_labels(); }

std::vector<double> INLAMixedModel::random_effects_mean_for_group(const std::size_t group_index) const {
  const auto& effects = model_.random_effects_mean();
  if (group_index >= effects.size())
    throw std::invalid_argument("INLAMixedModel::random_effects_mean_for_group: group_index out of range");
  return effects[group_index];
}

std::vector<double> INLAMixedModel::random_effects_sd_for_group(const std::size_t group_index) const {
  const auto& effects = model_.random_effects_sd();
  if (group_index >= effects.size())
    throw std::invalid_argument("INLAMixedModel::random_effects_sd_for_group: group_index out of range");
  return effects[group_index];
}

double INLAMixedModel::log_marginal_likelihood() const { return model_.log_marginal_likelihood(); }

std::string INLAMixedModel::summary() const { return model_.summary(); }

void INLAMixedModel::print_summary() const { model_.print_summary(); }

std::vector<double> INLAMixedModel::predict(const DataFrame& newdata) const { return model_.predict(newdata.frame_); }

} // namespace datamunge
