/// @file gbm_classifier.cpp
/// @brief Plain residual-tree multinomial boosting with inspectable stages and stable evaluation.
#include <algorithm>
#include <cmath>
#include <datamunge/stats/gbm_classifier.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Private finite arithmetic, encoding and multinomial metric helpers.
namespace {

/// @brief Formats one summary statistic with fixed decimal precision.
/// @param v Statistic.
/// @param precision Decimal places, default four.
/// @return Owned formatted text.
std::string format_stat(double v, int precision = 4) {
  if (std::isnan(v))
    return "NaN";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << v;
  return oss.str();
}

/// @brief Recycled colours for sorted categorical classes.
const graphics::RGB kSeriesColors[] = {
    {37, 99, 235},
    {220, 38, 38},
    {22, 163, 74},
    {217, 119, 6},
    {124, 58, 237},
    {8, 145, 178},
};

/// @brief Copies finite encoded design columns to formula-safe numeric names.
/// @param X Predictor matrix.
/// @return Owned x0 through x(p-1) frame in row/feature order.
dstruct::DataFrame build_predictor_frame(const linalg::DenseMatrix<double>& X) {
  dstruct::DataFrame frame;
  for (std::size_t j = 0; j < X.cols(); ++j) {
    std::vector<double> column(X.rows());
    for (std::size_t i = 0; i < X.rows(); ++i)
      column[i] = X(i, j);
    frame.add_column("x" + std::to_string(j), std::move(column));
  }
  return frame;
}

/// @brief Constructs a synthetic response formula for every encoded predictor.
/// @param p Positive encoded feature count.
/// @return Formula y ~ x0 + ... .
std::string predictor_formula(std::size_t p) {
  std::ostringstream oss;
  oss << "y ~ x0";
  for (std::size_t j = 1; j < p; ++j)
    oss << " + x" << j;
  return oss.str();
}

/// @brief Computes a max-subtracted softmax, allowing probability underflow to zero.
/// @param scores Finite score matrix.
/// @param row Zero-based row to normalize.
/// @param K Positive class count matching scores.cols().
/// @return Owned normalized K-vector in fitted class order.
std::vector<double>
softmax_row(const linalg::DenseMatrix<double>& scores, std::size_t row, std::size_t K) {
  double max_score = scores(row, 0);
  for (std::size_t k = 1; k < K; ++k)
    max_score = std::max(max_score, scores(row, k));
  std::vector<double> p(K);
  double              sum_exp = 0.0;
  for (std::size_t k = 0; k < K; ++k) {
    p[k] = std::exp(scores(row, k) - max_score);
    sum_exp += p[k];
  }
  for (double& v : p)
    v /= sum_exp;
  return p;
}

/// @brief Converts extended arithmetic to a finite double.
/// @param x Intermediate value.
/// @return Representable finite double, permitting underflow.
/// @throws std::invalid_argument Nonfinite or unrepresentable result.
double finite_gbm(long double x) {
  const double v = static_cast<double>(x);
  if (!std::isfinite(v))
    throw std::invalid_argument("GBMClassifier: nonfinite/unrepresentable arithmetic");
  return v;
}
/// @brief Validates every retained transformed design entry without changing it.
/// @param X Borrowed encoded matrix.
/// @throws std::invalid_argument A retained value is nonfinite.
void check_gbm_matrix(const linalg::DenseMatrix<double>& X) {
  for (std::size_t i = 0; i < X.rows(); ++i)
    for (std::size_t j = 0; j < X.cols(); ++j)
      (void)finite_gbm(X(i, j));
}
/// @brief Maps complete response labels to the frozen sorted training classes.
/// @param labels Evaluation labels.
/// @param classes Sorted fitted classes.
/// @return Owned zero-based class indices.
/// @throws std::invalid_argument An evaluation label is absent from training.
std::vector<std::size_t> gbm_truth(
    const std::vector<std::string>& labels,
    const std::vector<std::string>& classes) {
  std::vector<std::size_t> out;
  for (const auto& label : labels) {
    auto it = std::lower_bound(classes.begin(), classes.end(), label);
    if (it == classes.end() || *it != label)
      throw std::invalid_argument("GBMClassifier: unseen response class");
    out.push_back(static_cast<std::size_t>(it - classes.begin()));
  }
  return out;
}
/// @brief Stable unfloored negative log-softmax for an observed class.
/// @param scores Finite row-by-class score matrix.
/// @param row Evaluated row.
/// @param truth True fitted class index.
/// @return Negative log probability from log-sum-exp, even when softmax underflows.
/// @throws std::invalid_argument Unrepresentable log loss.
double gbm_log_loss(const linalg::DenseMatrix<double>& scores, std::size_t row, std::size_t truth) {
  double maximum = scores(row, 0);
  for (std::size_t k = 1; k < scores.cols(); ++k)
    maximum = std::max(maximum, scores(row, k));
  long double sum = 0;
  for (std::size_t k = 0; k < scores.cols(); ++k)
    sum += std::exp(static_cast<long double>(scores(row, k)) - maximum);
  return finite_gbm(static_cast<long double>(maximum) - scores(row, truth) + std::log(sum));
}
} // namespace

GBMClassifier::GBMClassifier(
    const dstruct::DataFrame& data,
    const std::string&        formula,
    GBMClassifierOptions      options)
    : formula_(formula) {
  options_ = options;
  fit(data);
}

void GBMClassifier::fit(const dstruct::DataFrame& data) {
  if (options_.n_trees == 0)
    throw std::invalid_argument("GBMClassifier: n_trees must be at least 1");
  if (!std::isfinite(options_.learning_rate) || options_.learning_rate <= 0 ||
      options_.min_samples_split < 2 || options_.min_samples_leaf == 0 || options_.max_depth == 0)
    throw std::invalid_argument(
        "GBMClassifier: positive finite learning_rate, min_samples_split>=2 and "
        "min_samples_leaf>=1 and max_depth>=1 required");
  if (!std::isfinite(options_.subsample) || options_.subsample <= 0.0 || options_.subsample > 1.0)
    throw std::invalid_argument("GBMClassifier: subsample must be in (0, 1]");

  design_              = formula_.resolve(data, ResponseKind::Categorical);
  const auto        cd = design_.build_classification_matrix(data);
  const std::size_t n  = cd.X.rows();
  const std::size_t p  = cd.X.cols();

  if (p == 0)
    throw std::invalid_argument("GBMClassifier: at least one encoded predictor required");
  check_gbm_matrix(cd.X);
  kept_row_indices_ = cd.used_row_indices;
  predictor_names_  = design_.coefficient_names;
  observations_     = n;
  training_labels_  = cd.labels;

  classes_.clear();
  for (const auto& label : cd.labels)
    if (std::find(classes_.begin(), classes_.end(), label) == classes_.end())
      classes_.push_back(label);
  std::sort(classes_.begin(), classes_.end());
  if (classes_.size() < 2)
    throw std::invalid_argument(
        "GBMClassifier: response must have at least 2 distinct classes, found " +
        std::to_string(classes_.size()));
  const std::size_t K = classes_.size();

  feature_min_.assign(p, std::numeric_limits<double>::infinity());
  feature_max_.assign(p, -std::numeric_limits<double>::infinity());
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < p; ++j) {
      feature_min_[j] = std::min(feature_min_[j], cd.X(i, j));
      feature_max_[j] = std::max(feature_max_[j], cd.X(i, j));
    }

  std::vector<std::size_t> class_index(n);
  std::vector<double>      class_count(K, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    class_index[i] = static_cast<std::size_t>(
        std::lower_bound(classes_.begin(), classes_.end(), cd.labels[i]) - classes_.begin());
    class_count[class_index[i]] += 1.0;
  }
  initial_scores_.assign(K, 0.0);
  for (std::size_t k = 0; k < K; ++k)
    initial_scores_[k] = std::log(class_count[k] / static_cast<double>(n));

  const auto predictor_frame = build_predictor_frame(cd.X);
  training_predictors_       = predictor_frame;
  const auto tree_formula    = predictor_formula(p);

  DecisionTreeRegressorOptions tree_options;
  tree_options.max_depth         = options_.max_depth;
  tree_options.min_samples_split = options_.min_samples_split;
  tree_options.min_samples_leaf  = options_.min_samples_leaf;

  linalg::DenseMatrix<double> scores(n, K, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < K; ++k)
      scores(i, k) = initial_scores_[k];

  random::SplitMix64 sampler(options_.seed);
  const auto         sample_size = static_cast<std::size_t>(
      std::max(1.0, std::round(options_.subsample * static_cast<double>(n))));
  const bool use_subsample = options_.subsample < 1.0;

  trees_.clear();
  trees_.reserve(options_.n_trees);
  training_deviance_.clear();
  training_deviance_.reserve(options_.n_trees);
  sampled_indices_.clear();
  sampled_indices_.reserve(options_.n_trees);

  for (std::size_t m = 0; m < options_.n_trees; ++m) {
    std::vector<std::vector<double>> proba(n);
    for (std::size_t i = 0; i < n; ++i)
      proba[i] = softmax_row(scores, i, K);

    std::vector<std::size_t> rows;
    if (use_subsample) {
      std::vector<std::size_t> all(n);
      std::iota(all.begin(), all.end(), 0);
      for (std::size_t i = 0; i < sample_size; ++i) {
        const std::size_t j = i + static_cast<std::size_t>(sampler.next_u64() % (n - i));
        std::swap(all[i], all[j]);
      }
      rows.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(sample_size));
      std::sort(rows.begin(), rows.end());
    }

    std::vector<DecisionTreeRegressor> round_trees;
    round_trees.reserve(K);
    for (std::size_t k = 0; k < K; ++k) {
      std::vector<double> residual(n);
      for (std::size_t i = 0; i < n; ++i)
        residual[i] = (class_index[i] == k ? 1.0 : 0.0) - proba[i][k];

      dstruct::DataFrame round_frame;
      if (use_subsample) {
        round_frame = predictor_frame.take_rows(rows);
        std::vector<double> sub_residual(rows.size());
        for (std::size_t r = 0; r < rows.size(); ++r)
          sub_residual[r] = residual[rows[r]];
        round_frame.add_column("y", std::move(sub_residual));
      } else {
        round_frame = predictor_frame;
        round_frame.add_column("y", residual);
      }

      round_trees.emplace_back(round_frame, tree_formula, tree_options);
      const auto contribution = round_trees.back().predict(predictor_frame);
      for (std::size_t i = 0; i < n; ++i)
        scores(i, k) = finite_gbm(
            static_cast<long double>(scores(i, k)) +
            static_cast<long double>(options_.learning_rate) * contribution[i]);
    }
    trees_.push_back(std::move(round_trees));
    sampled_indices_.push_back(rows);

    long double deviance = 0;
    for (std::size_t i = 0; i < n; ++i)
      deviance += gbm_log_loss(scores, i, class_index[i]);
    training_deviance_.push_back(finite_gbm(deviance / n));
  }

  fitted_classes_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t best = 0;
    for (std::size_t k = 1; k < K; ++k)
      if (scores(i, k) > scores(i, best))
        best = k;
    fitted_classes_[i] = classes_[best];
  }
}

linalg::DenseMatrix<double> GBMClassifier::score_frame(
    const dstruct::DataFrame& predictor_frame,
    std::size_t               n_rows,
    std::size_t               rounds) const {
  const std::size_t           K = classes_.size();
  linalg::DenseMatrix<double> scores(n_rows, K, 0.0);
  for (std::size_t i = 0; i < n_rows; ++i)
    for (std::size_t k = 0; k < K; ++k)
      scores(i, k) = initial_scores_[k];

  for (std::size_t m = 0; m < rounds; ++m) {
    const auto& round = trees_[m];
    for (std::size_t k = 0; k < K; ++k) {
      const auto contribution = round[k].predict(predictor_frame);
      for (std::size_t i = 0; i < n_rows; ++i)
        scores(i, k) = finite_gbm(
            static_cast<long double>(scores(i, k)) +
            static_cast<long double>(options_.learning_rate) * contribution[i]);
    }
  }
  return scores;
}

std::vector<double> GBMClassifier::feature_importance() const {
  const std::size_t   p = predictor_names_.size();
  std::vector<double> importance(p, 0.0);
  std::size_t         count = 0;
  for (const auto& round : trees_) {
    for (const auto& tree : round) {
      const auto imp = tree.feature_importance();
      for (std::size_t j = 0; j < p; ++j)
        importance[j] += imp[j];
      ++count;
    }
  }
  if (count > 0)
    for (double& v : importance)
      v /= static_cast<double>(count);
  return importance;
}

double GBMClassifier::training_accuracy() const {
  std::size_t correct = 0;
  for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
    if (fitted_classes_[i] == training_labels_[i])
      ++correct;
  return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> GBMClassifier::confusion_matrix() const {
  const std::size_t           k = classes_.size();
  linalg::DenseMatrix<double> matrix(k, k, 0.0);
  for (std::size_t i = 0; i < training_labels_.size(); ++i) {
    const auto actual_it = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
    const auto predicted_it =
        std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
    matrix(
        static_cast<std::size_t>(actual_it - classes_.begin()),
        static_cast<std::size_t>(predicted_it - classes_.begin())) += 1.0;
  }
  return matrix;
}

std::string GBMClassifier::summary() const {
  std::ostringstream out;
  out << "Call:\ngbm(formula = " << formula_.text() << ", n_trees = " << options_.n_trees
      << ", learning_rate = " << format_stat(options_.learning_rate, 3)
      << ", max_depth = " << options_.max_depth << ")\n\n";

  out << "Number of Classes: " << classes_.size() << "\n";
  out << "Levels: ";
  for (const auto& c : classes_)
    out << c << " ";
  out << "\n\n";

  out << "Feature importance:\n";
  const auto importance = feature_importance();
  for (std::size_t j = 0; j < predictor_names_.size(); ++j)
    out << "  " << predictor_names_[j] << ": " << format_stat(importance[j], 4) << "\n";

  out << "\nTraining accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  ("
      << observations_ << " observations)\n";
  if (!training_deviance_.empty())
    out << "Training deviance: " << format_stat(training_deviance_.front(), 4) << " (round 1) -> "
        << format_stat(training_deviance_.back(), 4) << " (round " << training_deviance_.size()
        << ")\n";

  return out.str();
}

void GBMClassifier::print_summary(std::ostream& os) const {
  os << summary();
}

void GBMClassifier::print_summary() const {
  print_summary(std::cout);
}

std::vector<std::string> GBMClassifier::predict(const dstruct::DataFrame& newdata) const {
  return predict_detail(newdata).class_label;
}

linalg::DenseMatrix<double> GBMClassifier::raw_scores(
    const dstruct::DataFrame& newdata,
    std::size_t               rounds) const {
  if (rounds > n_trees())
    throw std::out_of_range("GBMClassifier: prefix exceeds fitted rounds");
  const auto K = classes_.size();
  if (newdata.nrows() == 0)
    return linalg::DenseMatrix<double>(0, K);
  design_.validate_categorical_levels(newdata, "GBMClassifier::raw_scores");
  const auto matrix = design_.build_matrix(newdata, false);
  check_gbm_matrix(matrix.X);
  const auto compact = score_frame(build_predictor_frame(matrix.X), matrix.X.rows(), rounds);
  linalg::DenseMatrix<double> out(newdata.nrows(), K, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < matrix.X.rows(); ++i)
    for (std::size_t k = 0; k < K; ++k)
      out(matrix.used_row_indices[i], k) = compact(i, k);
  return out;
}
GBMClassifierPrediction GBMClassifier::predict_detail(const dstruct::DataFrame& newdata) const {
  return predict_detail(newdata, n_trees());
}
GBMClassifierPrediction GBMClassifier::predict_detail(
    const dstruct::DataFrame& newdata,
    std::size_t               rounds) const {
  const auto              scores = raw_scores(newdata, rounds);
  const auto              K      = classes_.size();
  GBMClassifierPrediction out;
  out.class_label.assign(newdata.nrows(), "");
  out.probability.resize(newdata.nrows());
  for (std::size_t i = 0; i < newdata.nrows(); ++i) {
    if (std::isnan(scores(i, 0)))
      continue;
    out.probability[i] = softmax_row(scores, i, K);
    const auto best    = static_cast<std::size_t>(
        std::max_element(out.probability[i].begin(), out.probability[i].end()) -
        out.probability[i].begin());
    out.class_label[i] = classes_[best];
  }
  return out;
}

graphics::RPlot GBMClassifier::plot_classification(
    const dstruct::DataFrame& data,
    const std::string&        x_feature,
    const std::string&        y_feature) const {
  const auto  predictions = predict(data);
  const auto& response    = design_.response_name;

  auto plot = graphics::RPlot::create();
  for (std::size_t c = 0; c < classes_.size(); ++c) {
    std::vector<double> xs, ys;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
      const auto actual = data.optional_string_at(response, i);
      if (!actual || *actual != classes_[c])
        continue;
      const auto x = data.optional_double_at(x_feature, i);
      const auto y = data.optional_double_at(y_feature, i);
      if (!x || !y)
        continue;
      xs.push_back(*x);
      ys.push_back(*y);
    }
    plot.points(
        xs, ys, classes_[c], kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
  }

  std::vector<double> mis_x, mis_y;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    const auto actual = data.optional_string_at(response, i);
    if (!actual || i >= predictions.size() || predictions[i].empty() || predictions[i] == *actual)
      continue;
    const auto x = data.optional_double_at(x_feature, i);
    const auto y = data.optional_double_at(y_feature, i);
    if (!x || !y)
      continue;
    mis_x.push_back(*x);
    mis_y.push_back(*y);
  }
  if (!mis_x.empty())
    plot.points(mis_x, mis_y, "misclassified", {15, 15, 15}, 9.0);

  plot.title("GBM Classification").x_label(x_feature).y_label(y_feature);
  return plot;
}

graphics::RPlot GBMClassifier::plot_decision_regions(
    const std::string& x_feature,
    const std::string& y_feature,
    std::size_t        grid_resolution) const {
  return plot_decision_regions(x_feature, y_feature, grid_resolution, n_trees());
}
graphics::RPlot GBMClassifier::plot_decision_regions(
    const std::string& x_feature,
    const std::string& y_feature,
    std::size_t        grid_resolution,
    std::size_t        rounds) const {
  if (x_feature == y_feature || grid_resolution < 2 || rounds > n_trees())
    throw std::invalid_argument("GBMClassifier: distinct axes, grid>=2 and valid prefix required");
  if (grid_resolution > std::numeric_limits<std::size_t>::max() / grid_resolution)
    throw std::length_error("GBMClassifier: grid size overflow");
  if (predictor_names_.size() != 2)
    throw std::invalid_argument(
        "GBMClassifier::plot_decision_regions: model must have exactly 2 predictors");
  std::size_t xi = 2, yi = 2;
  for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
    if (predictor_names_[j] == x_feature)
      xi = j;
    if (predictor_names_[j] == y_feature)
      yi = j;
  }
  if (xi == 2 || yi == 2)
    throw std::invalid_argument(
        "GBMClassifier::plot_decision_regions: x_feature/y_feature must match the "
        "model's two predictors (" +
        predictor_names_[0] + ", " + predictor_names_[1] + ")");

  double       x_min = feature_min_[xi], x_max = feature_max_[xi];
  double       y_min = feature_min_[yi], y_max = feature_max_[yi];
  const double x_pad = std::max(1e-6, (x_max - x_min) * 0.1);
  const double y_pad = std::max(1e-6, (y_max - y_min) * 0.1);
  x_min -= x_pad;
  x_max += x_pad;
  y_min -= y_pad;
  y_max += y_pad;

  std::vector<double> col0, col1;
  col0.reserve(grid_resolution * grid_resolution);
  col1.reserve(grid_resolution * grid_resolution);
  for (std::size_t gx = 0; gx < grid_resolution; ++gx) {
    for (std::size_t gy = 0; gy < grid_resolution; ++gy) {
      const double x = x_min + (x_max - x_min) * static_cast<double>(gx) /
                                   static_cast<double>(grid_resolution - 1);
      const double y = y_min + (y_max - y_min) * static_cast<double>(gy) /
                                   static_cast<double>(grid_resolution - 1);
      col0.push_back(predictor_names_[0] == x_feature ? x : y);
      col1.push_back(predictor_names_[1] == x_feature ? x : y);
    }
  }
  dstruct::DataFrame grid;
  grid.add_column(predictor_names_[0], col0);
  grid.add_column(predictor_names_[1], col1);

  const auto detail = predict_detail(grid, rounds);

  std::vector<std::vector<double>> grid_x(classes_.size()), grid_y(classes_.size());
  for (std::size_t i = 0; i < detail.class_label.size(); ++i) {
    if (detail.class_label[i].empty())
      continue;
    const auto it = std::lower_bound(classes_.begin(), classes_.end(), detail.class_label[i]);
    const auto c  = static_cast<std::size_t>(it - classes_.begin());
    grid_x[c].push_back(predictor_names_[0] == x_feature ? col0[i] : col1[i]);
    grid_y[c].push_back(predictor_names_[0] == y_feature ? col0[i] : col1[i]);
  }

  auto plot = graphics::RPlot::create();
  for (std::size_t c = 0; c < classes_.size(); ++c) {
    auto color = kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
    color.r    = static_cast<std::uint8_t>(std::min(255, color.r + (255 - color.r) * 3 / 4));
    color.g    = static_cast<std::uint8_t>(std::min(255, color.g + (255 - color.g) * 3 / 4));
    color.b    = static_cast<std::uint8_t>(std::min(255, color.b + (255 - color.b) * 3 / 4));
    plot.points(grid_x[c], grid_y[c], classes_[c] + " region", color, 3.0);
  }

  plot.title("GBM Decision Regions").x_label(x_feature).y_label(y_feature);
  return plot;
}

graphics::RPlot GBMClassifier::plot_training_deviance() const {
  std::vector<double> iteration(training_deviance_.size());
  for (std::size_t i = 0; i < training_deviance_.size(); ++i)
    iteration[i] = static_cast<double>(i + 1);

  auto plot = graphics::RPlot::create();
  plot.line(iteration, training_deviance_, "training log loss", {37, 99, 235});
  plot.title("Training Deviance").x_label("Boosting iteration").y_label("Multinomial deviance");
  return plot;
}

std::vector<std::size_t> GBMClassifier::round_sample_rows(std::size_t round) const {
  if (round == 0 || round > n_trees())
    throw std::out_of_range("GBMClassifier: training round out of range");
  if (sampled_indices_[round - 1].empty())
    return kept_row_indices_;
  std::vector<std::size_t> rows;
  for (auto i : sampled_indices_[round - 1])
    rows.push_back(kept_row_indices_[i]);
  return rows;
}
dstruct::DataFrame GBMClassifier::training_step(std::size_t round) const {
  if (round == 0 || round > n_trees())
    throw std::out_of_range("GBMClassifier: training round out of range");
  const auto                  n = observations_, K = classes_.size();
  const auto                  before = score_frame(training_predictors_, n, round - 1);
  auto                        after  = before;
  linalg::DenseMatrix<double> tree_values(n, K);
  for (std::size_t k = 0; k < K; ++k) {
    const auto contribution = trees_[round - 1][k].predict(training_predictors_);
    for (std::size_t i = 0; i < n; ++i) {
      tree_values(i, k) = contribution[i];
      after(i, k)       = finite_gbm(
          static_cast<long double>(before(i, k)) +
          static_cast<long double>(options_.learning_rate) * contribution[i]);
    }
  }
  std::vector<double> rounds, rows, indicator, old_score, old_probability, residual, value,
      new_score, new_probability, in_bag;
  std::vector<std::string> labels;
  const auto               sample = round_sample_rows(round);
  for (std::size_t i = 0; i < n; ++i) {
    const auto p0 = softmax_row(before, i, K), p1 = softmax_row(after, i, K);
    for (std::size_t k = 0; k < K; ++k) {
      rounds.push_back(round);
      rows.push_back(kept_row_indices_[i]);
      labels.push_back(classes_[k]);
      const double target = training_labels_[i] == classes_[k] ? 1 : 0;
      indicator.push_back(target);
      old_score.push_back(before(i, k));
      old_probability.push_back(p0[k]);
      residual.push_back(target - p0[k]);
      value.push_back(tree_values(i, k));
      new_score.push_back(after(i, k));
      new_probability.push_back(p1[k]);
      in_bag.push_back(
          std::binary_search(sample.begin(), sample.end(), kept_row_indices_[i]) ? 1 : 0);
    }
  }
  dstruct::DataFrame out;
  out.add_column("round", rounds);
  out.add_column("original_row", rows);
  out.add_column("class", labels);
  out.add_column("one_hot", indicator);
  out.add_column("before_score", old_score);
  out.add_column("before_probability", old_probability);
  out.add_column("pseudo_residual", residual);
  out.add_column("tree_value", value);
  out.add_column("after_score", new_score);
  out.add_column("after_probability", new_probability);
  out.add_column("in_bag", in_bag);
  return out;
}
dstruct::DataFrame GBMClassifier::step_diagnostics(std::size_t round) const {
  const auto  step  = training_step(round);
  long double inner = 0, norm = 0;
  for (std::size_t i = 0; i < step.nrows(); ++i) {
    const long double h = step.double_at("tree_value", i);
    inner += step.double_at("pseudo_residual", i) * h;
    norm += h * h;
  }
  long double before = 0;
  if (round > 1)
    before = training_deviance_[round - 2];
  else {
    const auto truth = gbm_truth(training_labels_, classes_);
    for (auto k : truth)
      before -= initial_scores_[k];
    before /= observations_;
  }
  const long double  eta   = options_.learning_rate;
  const auto         count = round_sample_rows(round).size();
  dstruct::DataFrame out;
  out.add_column("round", std::vector<double>{double(round)});
  out.add_column("sample_size", std::vector<double>{double(count)});
  out.add_column("full_sample", std::vector<double>{count == observations_ ? 1. : 0.});
  out.add_column("loss_before", std::vector<double>{finite_gbm(before)});
  out.add_column("loss_after", std::vector<double>{training_deviance_[round - 1]});
  out.add_column("gradient_tree_inner", std::vector<double>{finite_gbm(inner)});
  out.add_column("tree_squared_norm", std::vector<double>{finite_gbm(norm)});
  out.add_column(
      "quadratic_bound",
      std::vector<double>{finite_gbm(before - eta * (1 - eta / 4) * norm / observations_)});
  return out;
}
dstruct::DataFrame GBMClassifier::evaluate_frame(
    const dstruct::DataFrame&       predictors,
    const std::vector<std::string>& labels,
    std::size_t                     input_rows,
    std::size_t                     rounds) const {
  const auto                  truth = gbm_truth(labels, classes_);
  const auto                  n = labels.size(), K = classes_.size();
  linalg::DenseMatrix<double> scores(n, K);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < K; ++k)
      scores(i, k) = initial_scores_[k];
  std::vector<double> stage, accuracy, loss, brier;
  // Record one fixed-model stage; all stages use the same complete evaluation rows.
  const auto record = [&](std::size_t m) {
    long double correct = 0, total_loss = 0, total_brier = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto probability = softmax_row(scores, i, K);
      const auto best        = static_cast<std::size_t>(
          std::max_element(probability.begin(), probability.end()) - probability.begin());
      correct += best == truth[i];
      total_loss += gbm_log_loss(scores, i, truth[i]);
      for (std::size_t k = 0; k < K; ++k) {
        const long double error = probability[k] - (k == truth[i] ? 1 : 0);
        total_brier += error * error;
      }
    }
    stage.push_back(m);
    accuracy.push_back(finite_gbm(correct / n));
    loss.push_back(finite_gbm(total_loss / n));
    brier.push_back(finite_gbm(total_brier / n));
  };
  record(0);
  for (std::size_t m = 0; m < rounds; ++m) {
    for (std::size_t k = 0; k < K; ++k) {
      const auto contribution = trees_[m][k].predict(predictors);
      for (std::size_t i = 0; i < n; ++i)
        scores(i, k) = finite_gbm(
            static_cast<long double>(scores(i, k)) +
            static_cast<long double>(options_.learning_rate) * contribution[i]);
    }
    record(m + 1);
  }
  dstruct::DataFrame out;
  out.add_column("round", stage);
  out.add_column("input_rows", std::vector<double>(stage.size(), input_rows));
  out.add_column("evaluated_rows", std::vector<double>(stage.size(), n));
  out.add_column("skipped_rows", std::vector<double>(stage.size(), input_rows - n));
  out.add_column("accuracy", accuracy);
  out.add_column("mean_log_loss", loss);
  out.add_column("brier", brier);
  return out;
}
dstruct::DataFrame GBMClassifier::staged_evaluation(const dstruct::DataFrame& data) const {
  design_.validate_categorical_levels(data, "GBMClassifier::staged_evaluation");
  const auto cd = design_.build_classification_matrix(data);
  check_gbm_matrix(cd.X);
  return evaluate_frame(build_predictor_frame(cd.X), cd.labels, data.nrows(), n_trees());
}
dstruct::DataFrame GBMClassifier::evaluation(const dstruct::DataFrame& data) const {
  return evaluation(data, n_trees());
}
dstruct::DataFrame GBMClassifier::evaluation(const dstruct::DataFrame& data, std::size_t rounds)
    const {
  if (rounds > n_trees())
    throw std::out_of_range("GBMClassifier: prefix exceeds fitted rounds");
  design_.validate_categorical_levels(data, "GBMClassifier::evaluation");
  const auto cd = design_.build_classification_matrix(data);
  check_gbm_matrix(cd.X);
  return evaluate_frame(build_predictor_frame(cd.X), cd.labels, data.nrows(), rounds).tail(1);
}
dstruct::DataFrame GBMClassifier::confusion(const dstruct::DataFrame& data) const {
  return confusion(data, n_trees());
}
dstruct::DataFrame GBMClassifier::confusion(const dstruct::DataFrame& data, std::size_t rounds)
    const {
  if (rounds > n_trees())
    throw std::out_of_range("GBMClassifier: prefix exceeds fitted rounds");
  design_.validate_categorical_levels(data, "GBMClassifier::confusion");
  const auto cd = design_.build_classification_matrix(data);
  check_gbm_matrix(cd.X);
  const auto truth  = gbm_truth(cd.labels, classes_);
  const auto scores = score_frame(build_predictor_frame(cd.X), cd.X.rows(), rounds);
  const auto K      = classes_.size();
  linalg::DenseMatrix<double> counts(K, K, 0.0);
  for (std::size_t i = 0; i < truth.size(); ++i) {
    std::size_t best = 0;
    for (std::size_t k = 1; k < K; ++k)
      if (scores(i, k) > scores(i, best))
        best = k;
    ++counts(truth[i], best);
  }
  dstruct::DataFrame out;
  out.add_column("actual", classes_);
  for (std::size_t k = 0; k < K; ++k)
    out.add_column("pred_" + classes_[k], counts.col(k));
  return out;
}
dstruct::DataFrame GBMClassifier::importance_table() const {
  const auto   mean     = feature_importance();
  auto         relative = mean;
  const double total    = std::accumulate(mean.begin(), mean.end(), 0.0);
  if (total > 0)
    for (auto& v : relative)
      v /= total;
  dstruct::DataFrame out;
  out.add_column("predictor", predictor_names_);
  out.add_column("mean_normalized_gain", mean);
  out.add_column("relative_share", relative);
  return out;
}
GBMValidationResult gbm_validation_search(
    const dstruct::DataFrame&  train,
    const dstruct::DataFrame&  validation,
    const std::string&         formula,
    GBMClassifierOptions       options,
    const std::vector<double>& learning_rates) {
  if (learning_rates.empty())
    throw std::invalid_argument("gbm_validation_search: nonempty rate grid required");
  for (auto rate : learning_rates)
    if (!std::isfinite(rate) || rate <= 0)
      throw std::invalid_argument("gbm_validation_search: finite positive rates required");
  GBMValidationResult out;
  double              best = std::numeric_limits<double>::infinity();
  for (auto rate : learning_rates) {
    options.learning_rate = rate;
    const GBMClassifier model(train, formula, options);
    auto                curve = model.staged_evaluation(validation);
    curve.add_column("learning_rate", std::vector<double>(curve.nrows(), rate));
    const auto training = model.staged_evaluation(train);
    curve.add_column("training_mean_log_loss", training.numeric_column("mean_log_loss").values());
    curve.add_column("training_accuracy", training.numeric_column("accuracy").values());
    for (std::size_t m = 0; m < curve.nrows(); ++m) {
      const double loss = curve.double_at("mean_log_loss", m);
      if (loss < best) {
        best                         = loss;
        out.selected_options         = options;
        out.selected_rounds          = m;
        out.selected_options.n_trees = std::max(std::size_t(1), m);
      }
    }
    out.candidates = out.candidates.ncols() ? out.candidates.concat_rows(curve) : curve;
  }
  return out;
}

} // namespace datamunge::stats
