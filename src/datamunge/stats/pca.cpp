/// @file pca.cpp
/// @brief Checked preprocessing, normalized Gram PCA and direct projection-error diagnostics.
#include <algorithm>
#include <cmath>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/pca.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Private plotting helpers and checked extended-precision conversions.
namespace {
/// @brief Converts an intermediate to finite double, permitting underflow to zero.
/// @param value Extended arithmetic result.
/// @return Finite double result.
/// @throws std::invalid_argument Nonfinite or unrepresentable result.
double finite_value(long double value) {
  const double result = static_cast<double>(value);
  if (!std::isfinite(result))
    throw std::invalid_argument("PCA: nonfinite or unrepresentable arithmetic");
  return result;
}

/// @brief Six plotting colours, recycled in external-group lexical order.
const graphics::RGB kSeriesColors[] = {
    {37, 99, 235},
    {220, 38, 38},
    {22, 163, 74},
    {217, 119, 6},
    {124, 58, 237},
    {8, 145, 178},
};

/// @brief Copies distinct external group strings in lexical order.
/// @param labels Borrowed fitted-row labels.
/// @return Owned sorted unique strings.
std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
  std::vector<std::string> result(labels.begin(), labels.end());
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

} // namespace

PCA::PCA(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& feature_columns,
    PCAOptions                      options)
    : feature_columns_(feature_columns), options_(options) {
  fit(data);
}

void PCA::fit(const dstruct::DataFrame& data) {
  if (feature_columns_.empty())
    throw std::invalid_argument("PCA: nonempty features required");
  std::set<std::string> seen;
  for (const auto& col : feature_columns_)
    if (!seen.insert(col).second || !data.has_column(col) ||
        data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("PCA: distinct existing numeric features required");
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    bool complete = true;
    for (const auto& col : feature_columns_)
      if (data.is_null(col, i)) {
        complete = false;
        break;
      }
    if (complete)
      kept_row_indices_.push_back(i);
  }
  const auto n = kept_row_indices_.size(), p = feature_columns_.size();
  if (n < 2)
    throw std::invalid_argument("PCA: fewer than two complete rows");
  linalg::DenseMatrix<double> x(n, p);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < p; ++j)
      x(i, j) = finite_value(data.double_at(feature_columns_[j], kept_row_indices_[i]));
  mean_.assign(p, 0);
  scale_.assign(p, 1);
  for (std::size_t j = 0; j < p; ++j) {
    if (options_.center) {
      long double sum = 0;
      for (std::size_t i = 0; i < n; ++i)
        sum += x(i, j);
      mean_[j] = finite_value(sum / n);
    }
    long double square = 0;
    for (std::size_t i = 0; i < n; ++i) {
      x(i, j) = finite_value(static_cast<long double>(x(i, j)) - mean_[j]);
      square += static_cast<long double>(x(i, j)) * x(i, j);
    }
    if (options_.scale) {
      const auto sd = finite_value(std::sqrt(square / (n - 1)));
      scale_[j]     = sd > 0 ? sd : 1;
      for (std::size_t i = 0; i < n; ++i)
        x(i, j) = finite_value(static_cast<long double>(x(i, j)) / scale_[j]);
    }
  }
  covariance_ = linalg::DenseMatrix<double>(p, p);
  double norm = 0;
  for (std::size_t j = 0; j < p; ++j)
    for (std::size_t k = j; k < p; ++k) {
      long double sum = 0;
      for (std::size_t i = 0; i < n; ++i)
        sum += static_cast<long double>(x(i, j)) * x(i, k);
      const double value = finite_value(sum / (n - 1));
      covariance_(j, k) = covariance_(k, j) = value;
      norm                                  = std::max(norm, std::abs(value));
    }
  auto normalized = covariance_;
  if (norm > 0)
    for (std::size_t j = 0; j < p; ++j)
      for (std::size_t k = 0; k < p; ++k)
        normalized(j, k) /= norm;
  const auto eig = linalg::jacobi_eigen(normalized);
  loadings_      = eig.eigenvectors;
  for (double value : eig.eigenvalues)
    explained_variance_.push_back(
        std::max(0., finite_value(static_cast<long double>(value) * norm)));
  const long double threshold =
      64.L * std::max(n, p) * std::numeric_limits<double>::epsilon() * explained_variance_[0];
  for (double value : explained_variance_)
    if (value > threshold)
      ++rank_;
  scores_       = project(x);
  observations_ = n;
}

std::vector<double> PCA::explained_variance_ratio() const {
  long double total = 0;
  for (double value : explained_variance_)
    total += value;
  std::vector<double> ratios(explained_variance_.size(), 0.0);
  if (total <= 0.0)
    return ratios;
  for (std::size_t i = 0; i < explained_variance_.size(); ++i)
    ratios[i] = static_cast<double>(explained_variance_[i] / total);
  return ratios;
}

std::vector<double> PCA::cumulative_explained_variance_ratio() const {
  auto   ratios  = explained_variance_ratio();
  double running = 0.0;
  for (auto& v : ratios) {
    running += v;
    v = running;
  }
  return ratios;
}

std::vector<double> PCA::component_loadings(const std::size_t component_index) const {
  if (component_index >= loadings_.cols())
    throw std::out_of_range("PCA::component_loadings index out of range");
  return loadings_.col(component_index);
}

std::vector<double> PCA::component_scores(const std::size_t component_index) const {
  if (component_index >= scores_.cols())
    throw std::out_of_range("PCA::component_scores index out of range");
  return scores_.col(component_index);
}

linalg::DenseMatrix<double> PCA::preprocess(const dstruct::DataFrame& data) const {
  for (const auto& col : feature_columns_)
    if (!data.has_column(col) || data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("PCA: numeric query features required");
  linalg::DenseMatrix<double> z(data.nrows(), feature_columns_.size());
  for (std::size_t i = 0; i < z.rows(); ++i)
    for (std::size_t j = 0; j < z.cols(); ++j) {
      if (data.is_null(feature_columns_[j], i))
        throw std::invalid_argument("PCA: null query feature");
      const auto value = finite_value(data.double_at(feature_columns_[j], i));
      z(i, j)          = finite_value((static_cast<long double>(value) - mean_[j]) / scale_[j]);
    }
  return z;
}
linalg::DenseMatrix<double> PCA::project(const linalg::DenseMatrix<double>& z) const {
  linalg::DenseMatrix<double> out(z.rows(), num_components());
  for (std::size_t i = 0; i < z.rows(); ++i)
    for (std::size_t c = 0; c < num_components(); ++c) {
      long double sum = 0;
      for (std::size_t j = 0; j < z.cols(); ++j)
        sum += static_cast<long double>(z(i, j)) * loadings_(j, c);
      out(i, c) = finite_value(sum);
    }
  return out;
}
linalg::DenseMatrix<double> PCA::transform(const dstruct::DataFrame& newdata) const {
  return project(preprocess(newdata));
}
linalg::DenseMatrix<double> PCA::reconstruct(
    const dstruct::DataFrame& newdata,
    std::size_t               components) const {
  if (components > num_components())
    throw std::invalid_argument("PCA: invalid component prefix");
  const auto                  t = transform(newdata);
  linalg::DenseMatrix<double> out(t.rows(), num_components());
  for (std::size_t i = 0; i < t.rows(); ++i)
    for (std::size_t j = 0; j < num_components(); ++j) {
      long double value = 0;
      for (std::size_t c = 0; c < components; ++c)
        value += static_cast<long double>(t(i, c)) * loadings_(j, c);
      out(i, j) = finite_value(mean_[j] + value * scale_[j]);
    }
  return out;
}
dstruct::DataFrame PCA::reconstruction_curve(const dstruct::DataFrame& newdata) const {
  const auto z = preprocess(newdata);
  if (z.rows() == 0)
    throw std::invalid_argument("PCA: reconstruction curve requires query rows");
  const auto                       t = project(z);
  linalg::DenseMatrix<long double> approximate(z.rows(), z.cols(), 0);
  std::vector<double>              prefix, sse, rmse, original;
  for (std::size_t k = 0; k <= num_components(); ++k) {
    long double error = 0, raw_error = 0;
    for (std::size_t i = 0; i < z.rows(); ++i)
      for (std::size_t j = 0; j < z.cols(); ++j) {
        if (k)
          approximate(i, j) += static_cast<long double>(t(i, k - 1)) * loadings_(j, k - 1);
        const long double d = static_cast<long double>(z(i, j)) - approximate(i, j);
        error += d * d;
        raw_error += d * d * scale_[j] * scale_[j];
      }
    const long double entries = static_cast<long double>(z.rows()) * z.cols();
    prefix.push_back(k);
    sse.push_back(finite_value(error));
    rmse.push_back(finite_value(std::sqrt(error / entries)));
    original.push_back(finite_value(std::sqrt(raw_error / entries)));
  }
  dstruct::DataFrame out;
  out.add_column("components", prefix);
  out.add_column("preprocessed_sse", sse);
  out.add_column("preprocessed_rmse", rmse);
  out.add_column("original_rmse", original);
  return out;
}
dstruct::DataFrame PCA::spectrum() const {
  const auto share = explained_variance_ratio(), cumulative = cumulative_explained_variance_ratio();
  std::vector<double> component, tail;
  long double         remaining = 0;
  tail.resize(num_components());
  for (std::size_t c = num_components(); c-- > 0;) {
    tail[c] = finite_value(remaining * (observations_ - 1));
    remaining += explained_variance_[c];
  }
  for (std::size_t c = 0; c < num_components(); ++c)
    component.push_back(c + 1);
  dstruct::DataFrame out;
  out.add_column("component", component);
  out.add_column("variance", explained_variance_);
  out.add_column("share", share);
  out.add_column("cumulative_share", cumulative);
  out.add_column("tail_sse", tail);
  return out;
}
dstruct::DataFrame PCA::loadings_frame() const {
  dstruct::DataFrame out;
  out.add_column("feature", feature_columns_);
  for (std::size_t c = 0; c < num_components(); ++c)
    out.add_column("PC" + std::to_string(c + 1), component_loadings(c));
  return out;
}

std::string PCA::summary() const {
  std::ostringstream oss;
  print_summary(oss);
  return oss.str();
}

void PCA::print_summary() const {
  print_summary(std::cout);
}

void PCA::print_summary(std::ostream& os) const {
  os << "Principal Component Analysis\n";
  os << "Features: ";
  for (std::size_t j = 0; j < feature_columns_.size(); ++j)
    os << (j ? ", " : "") << feature_columns_[j];
  os << "\n";
  os << "Number of obs: " << observations_ << ", components: " << num_components() << "\n";
  os << (options_.center ? "Centered" : "Uncentered second moment")
     << (options_.scale ? ", scaled\n" : ", unscaled\n");

  const auto ratio      = explained_variance_ratio();
  const auto cumulative = cumulative_explained_variance_ratio();
  os << std::fixed << std::setprecision(4);
  os << "\n"
     << std::left << std::setw(12) << "Component" << std::setw(16) << "Std.Dev." << std::setw(20)
     << "Prop.Variance" << "Cum.Variance\n";
  for (std::size_t c = 0; c < num_components(); ++c) {
    os << std::left << std::setw(12) << ("PC" + std::to_string(c + 1)) << std::setw(16)
       << std::sqrt(explained_variance_[c]) << std::setw(20) << ratio[c] << cumulative[c] << "\n";
  }
}

graphics::RPlot PCA::plot_scores(const std::size_t component_x, const std::size_t component_y)
    const {
  if (component_x >= num_components() || component_y >= num_components())
    throw std::out_of_range("PCA::plot_scores component index out of range");

  auto plot = graphics::RPlot::create();
  plot.points(scores_.col(component_x), scores_.col(component_y), "", kSeriesColors[0]);
  const auto         ratio = explained_variance_ratio();
  std::ostringstream x_label, y_label;
  x_label << "PC" << (component_x + 1) << " (" << std::fixed << std::setprecision(1)
          << ratio[component_x] * 100.0 << "%)";
  y_label << "PC" << (component_y + 1) << " (" << std::fixed << std::setprecision(1)
          << ratio[component_y] * 100.0 << "%)";
  plot.title("Principal Component Analysis").x_label(x_label.str()).y_label(y_label.str());
  return plot;
}

graphics::RPlot PCA::plot_scores(
    const std::vector<std::string>& group_labels,
    const std::size_t               component_x,
    const std::size_t               component_y) const {
  if (component_x >= num_components() || component_y >= num_components())
    throw std::out_of_range("PCA::plot_scores component index out of range");
  if (group_labels.size() != observations_)
    throw std::invalid_argument(
        "PCA::plot_scores: group_labels must have one entry per fitted observation");

  auto       plot   = graphics::RPlot::create();
  const auto groups = unique_sorted(group_labels);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    std::vector<double> xs, ys;
    for (std::size_t i = 0; i < observations_; ++i) {
      if (group_labels[i] != groups[g])
        continue;
      xs.push_back(scores_(i, component_x));
      ys.push_back(scores_(i, component_y));
    }
    plot.points(
        xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
  }

  const auto         ratio = explained_variance_ratio();
  std::ostringstream x_label, y_label;
  x_label << "PC" << (component_x + 1) << " (" << std::fixed << std::setprecision(1)
          << ratio[component_x] * 100.0 << "%)";
  y_label << "PC" << (component_y + 1) << " (" << std::fixed << std::setprecision(1)
          << ratio[component_y] * 100.0 << "%)";
  plot.title("Principal Component Analysis").x_label(x_label.str()).y_label(y_label.str());
  return plot;
}

graphics::RPlot PCA::plot_scree() const {
  const auto               ratio = explained_variance_ratio();
  std::vector<double>      percentages;
  std::vector<std::string> names;
  for (std::size_t c = 0; c < ratio.size(); ++c) {
    percentages.push_back(ratio[c] * 100.0);
    names.push_back("PC" + std::to_string(c + 1));
  }
  auto plot = graphics::RPlot::barplot(percentages, names, "", kSeriesColors[0]);
  plot.title("Scree Plot").x_label("Component").y_label("% Variance Explained");
  return plot;
}

} // namespace datamunge::stats
