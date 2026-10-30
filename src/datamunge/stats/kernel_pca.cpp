/// @file kernel_pca.cpp
/// @brief Stable kernel centering, rank-aware spectral coordinates and query projection.
#include <algorithm>
#include <cmath>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/kernel_pca.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Private plotting colours and checked numeric conversions.
namespace {
/// @brief Six recycled external-label colours, in lexical group order.
constexpr graphics::RGB series_colors[] =
    {{37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178}};
/// @brief Missing query coordinates/norms.
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
/// @brief Converts extended arithmetic while rejecting overflow or invalid geometry.
/// @param value Extended intermediate value.
/// @return Finite double (underflow to zero allowed).
/// @throws std::invalid_argument Value is not finite/representable.
double finite_double(long double value) {
  const double result = static_cast<double>(value);
  if (!std::isfinite(result))
    throw std::invalid_argument("KernelPCA: nonfinite or unrepresentable kernel geometry");
  return result;
}
} // namespace
KernelPCA::KernelPCA(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& columns,
    KernelPCAOptions                options)
    : feature_columns_(columns), options_(std::move(options)) {
  fit(data);
}
long double KernelPCA::kernel_value(const std::vector<double>& a, const std::vector<double>& b)
    const {
  long double dot = 0, distance = 0;
  for (std::size_t j = 0; j < a.size(); ++j) {
    if (options_.kernel == "rbf") {
      const long double delta = static_cast<long double>(a[j]) - b[j];
      distance += delta * delta;
    } else
      dot += static_cast<long double>(a[j]) * b[j];
  }
  long double value;
  if (options_.kernel == "linear")
    value = dot;
  else if (options_.kernel == "rbf")
    value = std::expm1(-static_cast<long double>(options_.gamma) * distance);
  else
    value =
        std::pow(static_cast<long double>(options_.gamma) * dot + options_.coef0, options_.degree);
  (void)finite_double(value);
  return value;
}
void KernelPCA::fit(const dstruct::DataFrame& data) {
  if (feature_columns_.empty() || options_.n_components == 0)
    throw std::invalid_argument("KernelPCA: nonempty features and positive n_components required");
  if (options_.kernel != "linear" && options_.kernel != "rbf" && options_.kernel != "polynomial")
    throw std::invalid_argument("KernelPCA: kernel must be linear, rbf or polynomial");
  if (options_.kernel != "linear" && (!std::isfinite(options_.gamma) || options_.gamma <= 0))
    throw std::invalid_argument("KernelPCA: gamma must be finite and positive");
  if (options_.kernel == "polynomial" && (!std::isfinite(options_.degree) || options_.degree < 1 ||
                                          std::floor(options_.degree) != options_.degree ||
                                          !std::isfinite(options_.coef0) || options_.coef0 < 0))
    throw std::invalid_argument(
        "KernelPCA: polynomial degree must be a positive integer and coef0 finite/nonnegative");
  std::set<std::string> names;
  for (const auto& name : feature_columns_) {
    if (!names.insert(name).second || !data.has_column(name) ||
        data.column_type(name) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("KernelPCA: distinct numeric feature columns required: " + name);
  }
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    bool complete = true;
    for (const auto& name : feature_columns_)
      if (data.is_null(name, i)) {
        complete = false;
        break;
      }
    if (!complete)
      continue;
    std::vector<double> row;
    for (const auto& name : feature_columns_) {
      const double value = data.double_at(name, i);
      if (!std::isfinite(value))
        throw std::invalid_argument("KernelPCA: retained predictors must be finite");
      row.push_back(value);
    }
    training_.push_back(std::move(row));
    kept_row_indices_.push_back(i);
  }
  const auto n = training_.size(), p = feature_columns_.size();
  observations_ = n;
  if (n < 2 || options_.n_components >= n)
    throw std::invalid_argument("KernelPCA: n_components must be less than the complete row count");
  feature_mean_.assign(p, 0);
  for (const auto& row : training_)
    for (std::size_t j = 0; j < p; ++j)
      feature_mean_[j] += static_cast<long double>(row[j]) / n;
  linalg::DenseMatrix<long double> working(n, n);
  kernel_   = linalg::DenseMatrix<double>(n, n);
  centered_ = linalg::DenseMatrix<double>(n, n);
  kernel_mean_.assign(n, 0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i; j < n; ++j) {
      const auto value = kernel_value(training_[i], training_[j]);
      working(i, j) = working(j, i) = value;
      kernel_(i, j) = kernel_(j, i) = finite_double(value + (options_.kernel == "rbf" ? 1.L : 0.L));
    }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j)
      kernel_mean_[i] += working(i, j) / n;
    kernel_grand_mean_ += kernel_mean_[i] / n;
  }
  double scale = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i; j < n; ++j) {
      long double value = working(i, j) - kernel_mean_[i] - kernel_mean_[j] + kernel_grand_mean_;
      if (options_.kernel == "linear") {
        value = 0;
        for (std::size_t k = 0; k < p; ++k)
          value += (training_[i][k] - feature_mean_[k]) * (training_[j][k] - feature_mean_[k]);
      }
      centered_(i, j) = centered_(j, i) = finite_double(value);
      scale                             = std::max(scale, std::abs(centered_(i, j)));
    }
  if (!(scale > 0))
    throw std::invalid_argument("KernelPCA: centred kernel has zero rank");
  linalg::DenseMatrix<double> normalized(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      normalized(i, j) = centered_(i, j) / scale;
  auto eig = linalg::symmetric_eigen(normalized);
  for (double value : eig.eigenvalues)
    all_eigenvalues_.push_back(finite_double(static_cast<long double>(value) * scale));
  tolerance_ =
      64 * std::numeric_limits<double>::epsilon() * n * std::max(0., all_eigenvalues_.front());
  if (all_eigenvalues_.back() < -tolerance_)
    throw std::invalid_argument("KernelPCA: materially negative centred-kernel eigenvalue");
  for (double value : all_eigenvalues_)
    if (value > tolerance_)
      ++rank_;
  rank_ = std::min(rank_, n - 1);
  if (options_.n_components > rank_)
    throw std::invalid_argument("KernelPCA: requested components exceed numerical kernel rank");
  embedding_ = linalg::DenseMatrix<double>(n, options_.n_components);
  dual_      = linalg::DenseMatrix<double>(n, options_.n_components);
  for (std::size_t c = 0; c < options_.n_components; ++c) {
    const double lambda = all_eigenvalues_[c], root = std::sqrt(lambda);
    eigenvalues_.push_back(lambda);
    std::size_t pivot = 0;
    for (std::size_t i = 1; i < n; ++i)
      if (std::abs(eig.eigenvectors(i, c)) > std::abs(eig.eigenvectors(pivot, c)))
        pivot = i;
    const double sign = eig.eigenvectors(pivot, c) < 0 ? -1 : 1;
    for (std::size_t i = 0; i < n; ++i) {
      embedding_(i, c) = eig.eigenvectors(i, c) * sign * root;
      dual_(i, c) = finite_double(static_cast<long double>(eig.eigenvectors(i, c)) * sign / root);
    }
  }
}
std::vector<double> KernelPCA::dimension(std::size_t index) const {
  if (index >= n_components())
    throw std::out_of_range("KernelPCA::dimension: component unavailable");
  return embedding_.col(index);
}
dstruct::DataFrame KernelPCA::spectrum() const {
  std::vector<double> component, variance, share, cumulative, residual, retained, resolved;
  long double         total = 0, used = 0;
  for (std::size_t c = 0; c < rank_; ++c)
    total += all_eigenvalues_[c];
  for (std::size_t c = 0; c < observations_; ++c) {
    const double positive = c < rank_ ? all_eigenvalues_[c] : 0;
    used += positive;
    component.push_back(c + 1);
    variance.push_back(all_eigenvalues_[c] / (observations_ - 1));
    share.push_back(positive / total);
    cumulative.push_back(used / total);
    residual.push_back(std::max(0.L, total - used));
    retained.push_back(c < n_components());
    resolved.push_back(c < rank_);
  }
  dstruct::DataFrame out;
  out.add_column("component", component);
  out.add_column("eigenvalue", all_eigenvalues_);
  out.add_column("sample_variance", variance);
  out.add_column("positive_share", share);
  out.add_column("cumulative_share", cumulative);
  out.add_column("residual_energy", residual);
  out.add_column("retained", retained);
  out.add_column("resolved", resolved);
  return out;
}
dstruct::DataFrame KernelPCA::diagnostics() const {
  long double total = 0, used = 0, row_max = 0, error = 0, norm = 0;
  for (std::size_t c = 0; c < rank_; ++c)
    total += all_eigenvalues_[c];
  for (double e : eigenvalues_)
    used += e;
  for (std::size_t i = 0; i < observations_; ++i) {
    long double mean = 0;
    for (std::size_t j = 0; j < observations_; ++j)
      mean += static_cast<long double>(centered_(i, j)) / observations_;
    row_max = std::max(row_max, std::abs(mean));
    for (std::size_t c = 0; c < n_components(); ++c) {
      long double value = 0;
      for (std::size_t j = 0; j < observations_; ++j)
        value += static_cast<long double>(centered_(i, j)) * dual_(j, c);
      const long double target = embedding_(i, c), delta = value - target;
      error += delta * delta;
      norm += target * target;
    }
  }
  dstruct::DataFrame out;
  out.add_column("n", std::vector<double>{double(observations_)});
  out.add_column("p", std::vector<double>{double(feature_columns_.size())});
  out.add_column("d", std::vector<double>{double(n_components())});
  out.add_column("rank", std::vector<double>{double(rank_)});
  out.add_column("trace", std::vector<double>{finite_double(total)});
  out.add_column("retained_share", std::vector<double>{double(used / total)});
  out.add_column(
      "mean_residual_energy",
      std::vector<double>{double(std::max(0.L, total - used) / observations_)});
  out.add_column("max_centered_row_mean", std::vector<double>{double(row_max)});
  out.add_column("relative_eigen_residual", std::vector<double>{double(std::sqrt(error / norm))});
  return out;
}
void KernelPCA::validate_query(const dstruct::DataFrame& data) const {
  if (!data.nrows())
    return;
  for (const auto& name : feature_columns_)
    if (!data.has_column(name) || data.column_type(name) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("KernelPCA::transform: required numeric column missing: " + name);
}
KernelPCA::Projection KernelPCA::project_row(const dstruct::DataFrame& data, std::size_t row)
    const {
  Projection          out{std::vector<double>(n_components(), missing), missing, missing, "ok"};
  std::vector<double> query;
  for (const auto& name : feature_columns_) {
    const auto value = data.optional_double_at(name, row);
    if (!value) {
      out.status = "missing_predictor";
      return out;
    }
    if (!std::isfinite(*value)) {
      out.status = "nonfinite_predictor";
      return out;
    }
    query.push_back(*value);
  }
  std::vector<long double> cross(observations_);
  long double              mean = 0;
  for (std::size_t i = 0; i < observations_; ++i) {
    cross[i] = kernel_value(query, training_[i]);
    mean += cross[i] / observations_;
  }
  long double self = kernel_value(query, query) - 2 * mean + kernel_grand_mean_;
  for (std::size_t i = 0; i < observations_; ++i) {
    cross[i] = cross[i] - mean - kernel_mean_[i] + kernel_grand_mean_;
    if (options_.kernel == "linear") {
      cross[i] = 0;
      for (std::size_t j = 0; j < query.size(); ++j)
        cross[i] += (query[j] - feature_mean_[j]) * (training_[i][j] - feature_mean_[j]);
    }
  }
  if (options_.kernel == "linear") {
    self = 0;
    for (std::size_t j = 0; j < query.size(); ++j) {
      const auto delta = query[j] - feature_mean_[j];
      self += delta * delta;
    }
  }
  long double projected = 0;
  for (std::size_t c = 0; c < n_components(); ++c) {
    long double score = 0;
    for (std::size_t i = 0; i < observations_; ++i)
      score += cross[i] * dual_(i, c);
    out.scores[c] = finite_double(score);
    projected += score * score;
  }
  out.norm_squared      = finite_double(std::max(0.L, self));
  out.projected_squared = finite_double(projected);
  return out;
}
linalg::DenseMatrix<double> KernelPCA::transform(const dstruct::DataFrame& data) const {
  validate_query(data);
  linalg::DenseMatrix<double> out(data.nrows(), n_components(), missing);
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    const auto p = project_row(data, i);
    for (std::size_t c = 0; c < n_components(); ++c)
      out(i, c) = p.scores[c];
  }
  return out;
}
dstruct::DataFrame KernelPCA::projection_details(const dstruct::DataFrame& data) const {
  validate_query(data);
  std::vector<double>      row, norm, projected, residual;
  std::vector<std::string> status;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    const auto p = project_row(data, i);
    row.push_back(i);
    norm.push_back(p.norm_squared);
    projected.push_back(p.projected_squared);
    residual.push_back(
        p.status == "ok" ? std::max(0., p.norm_squared - p.projected_squared) : missing);
    status.push_back(p.status);
  }
  dstruct::DataFrame out;
  out.add_column("query_row", row);
  out.add_column("centered_norm_squared", norm);
  out.add_column("projected_norm_squared", projected);
  out.add_column("residual_energy", residual);
  out.add_column("status", status);
  return out;
}
std::string KernelPCA::summary() const {
  const auto         d = diagnostics();
  std::ostringstream os;
  os << "Kernel Principal Component Analysis\nKernel: " << options_.kernel
     << "\nNumber of obs: " << observations_ << ", components: " << n_components()
     << ", numerical rank: " << rank_ << "\n";
  os << std::setprecision(6)
     << "Retained feature-space energy: " << d.double_at("retained_share", 0)
     << "\nEigenvalues used (Gram scale):\n";
  for (std::size_t c = 0; c < n_components(); ++c)
    os << "  KPC" << c + 1 << ": " << eigenvalues_[c] << "\n";
  return os.str();
}
void KernelPCA::print_summary(std::ostream& os) const {
  os << summary();
}
void KernelPCA::print_summary() const {
  print_summary(std::cout);
}
graphics::RPlot KernelPCA::plot_embedding(std::size_t x, std::size_t y) const {
  if (x >= n_components() || y >= n_components())
    throw std::out_of_range("KernelPCA::plot_embedding: component unavailable");
  auto plot = graphics::RPlot::create();
  plot.points(dimension(x), dimension(y), "observations", series_colors[0]);
  return plot.title("Kernel Principal Component Analysis")
      .x_label("KPC" + std::to_string(x + 1))
      .y_label("KPC" + std::to_string(y + 1));
}
graphics::RPlot KernelPCA::plot_embedding(
    const std::vector<std::string>& labels,
    std::size_t                     x,
    std::size_t                     y) const {
  if (x >= n_components() || y >= n_components())
    throw std::out_of_range("KernelPCA::plot_embedding: component unavailable");
  if (labels.size() != observations_)
    throw std::invalid_argument(
        "KernelPCA::plot_embedding: one external label per retained row required");
  const std::set<std::string> groups(labels.begin(), labels.end());
  auto                        plot = graphics::RPlot::create();
  std::size_t                 c    = 0;
  for (const auto& group : groups) {
    std::vector<double> xx, yy;
    for (std::size_t i = 0; i < observations_; ++i)
      if (labels[i] == group) {
        xx.push_back(embedding_(i, x));
        yy.push_back(embedding_(i, y));
      }
    plot.points(xx, yy, group, series_colors[c++ % 6]);
  }
  return plot.title("Kernel Principal Component Analysis")
      .x_label("KPC" + std::to_string(x + 1))
      .y_label("KPC" + std::to_string(y + 1));
}
} // namespace datamunge::stats
