/// @file isomap.cpp
/// @brief Deterministic union-kNN graphs, dense geodesics and classical Isomap diagnostics.
#include <algorithm>
#include <cmath>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/isomap.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Internal numeric distance, correlation and plotting helpers.
namespace {
/// @brief Group/component palette, recycled in label order.
const graphics::RGB kSeriesColors[] =
    {{37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178}};

/// @brief Sorted unique categorical labels.
/// @param labels Input labels.
/// @return Owned unique labels in lexical order.
std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
  auto out = labels;
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

/// @brief Stable finite distance between two selected-feature rows.
/// @param X Retained feature matrix.
/// @param i First fitted row.
/// @param j Second fitted row.
/// @param metric Validated Euclidean or Manhattan metric.
/// @return Finite input distance; overflow is rejected.
double row_distance(
    const linalg::DenseMatrix<double>& X,
    std::size_t                        i,
    std::size_t                        j,
    DistanceMetric                     metric) {
  long double sum = 0;
  if (metric == DistanceMetric::Euclidean) {
    for (std::size_t k = 0; k < X.cols(); ++k)
      sum = std::hypot(sum, static_cast<long double>(X(i, k)) - X(j, k));
  } else
    for (std::size_t k = 0; k < X.cols(); ++k)
      sum += std::fabs(static_cast<long double>(X(i, k)) - X(j, k));
  const double distance = static_cast<double>(sum);
  if (!std::isfinite(distance))
    throw std::invalid_argument("Isomap: feature distance exceeds finite range");
  return distance;
}

/// @brief Stable centred Pearson correlation of equally sized distance vectors.
/// @param x First nonempty vector.
/// @param y Second vector, matching x.
/// @return Correlation in [-1,1], or NaN for a constant vector.
double distance_correlation(const std::vector<double>& x, const std::vector<double>& y) {
  long double mx = 0, my = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    mx += static_cast<long double>(x[i]) / x.size();
    my += static_cast<long double>(y[i]) / y.size();
  }
  long double xx = 0, yy = 0, xy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const long double a = x[i] - mx, b = y[i] - my;
    xx += a * a;
    yy += b * b;
    xy += a * b;
  }
  if (xx == 0 || yy == 0)
    return std::numeric_limits<double>::quiet_NaN();
  return std::clamp(static_cast<double>(xy / std::sqrt(xx) / std::sqrt(yy)), -1.0, 1.0);
}

/// @brief Relative pair-distance norm error, including a zero-reference convention.
/// @param estimate Estimated distances.
/// @param reference Reference distances, matching estimate.
/// @return Norm ratio; zero/zero is zero, positive/zero is infinity.
double relative_distance_error(
    const std::vector<double>& estimate,
    const std::vector<double>& reference) {
  long double error = 0, norm = 0;
  for (std::size_t i = 0; i < estimate.size(); ++i) {
    const long double d = static_cast<long double>(estimate[i]) - reference[i];
    error += d * d;
    norm += static_cast<long double>(reference[i]) * reference[i];
  }
  return norm > 0     ? static_cast<double>(std::sqrt(error / norm))
         : error == 0 ? 0.0
                      : std::numeric_limits<double>::infinity();
}
} // namespace

IsomapGraph isomap_graph(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& columns,
    std::size_t                     n_neighbors,
    DistanceMetric                  metric) {
  if (columns.empty())
    throw std::invalid_argument("Isomap: feature_columns must not be empty");
  if (n_neighbors < 2)
    throw std::invalid_argument("Isomap: n_neighbors must be at least two");
  if (metric != DistanceMetric::Euclidean && metric != DistanceMetric::Manhattan)
    throw std::invalid_argument("Isomap: unsupported distance metric");
  for (const auto& name : columns) {
    if (!data.has_column(name))
      throw std::invalid_argument("Isomap: column not found: " + name);
    if (data.column_type(name) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("Isomap: feature column must be numeric: " + name);
  }
  IsomapGraph out;
  out.feature_names = columns;
  out.n_neighbors   = n_neighbors;
  for (std::size_t row = 0; row < data.nrows(); ++row) {
    bool complete = true;
    for (const auto& name : columns)
      if (data.is_null(name, row)) {
        complete = false;
        break;
      }
    if (complete)
      out.kept_rows.push_back(row);
  }
  const auto n = out.kept_rows.size(), p = columns.size();
  if (n_neighbors >= n)
    throw std::invalid_argument("Isomap: n_neighbors must be less than the complete row count");
  out.feature_values = linalg::DenseMatrix<double>(n, p);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < p; ++j) {
      const double value = data.double_at(columns[j], out.kept_rows[i]);
      if (!std::isfinite(value))
        throw std::invalid_argument("Isomap: retained features must be finite");
      out.feature_values(i, j) = value;
    }
  out.distances = linalg::DenseMatrix<double>(n, n, 0.0);
  out.adjacency = linalg::DenseMatrix<double>(n, n, std::numeric_limits<double>::infinity());
  for (std::size_t i = 0; i < n; ++i) {
    out.adjacency(i, i) = 0;
    for (std::size_t j = i + 1; j < n; ++j)
      out.distances(i, j) = out.distances(j, i) = row_distance(out.feature_values, i, j, metric);
  }
  out.neighbors.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::size_t> others;
    others.reserve(n - 1);
    for (std::size_t j = 0; j < n; ++j)
      if (i != j)
        others.push_back(j);
    /// @brief Strict deterministic order by distance, then increasing fitted index.
    auto closer = [&](std::size_t a, std::size_t b) {
      return out.distances(i, a) < out.distances(i, b) ||
             (out.distances(i, a) == out.distances(i, b) && a < b);
    };
    std::partial_sort(
        others.begin(),
        others.begin() + static_cast<std::ptrdiff_t>(n_neighbors),
        others.end(),
        closer);
    out.neighbors[i].assign(
        others.begin(), others.begin() + static_cast<std::ptrdiff_t>(n_neighbors));
    for (auto j : out.neighbors[i])
      out.adjacency(i, j) = out.adjacency(j, i) = out.distances(i, j);
  }
  out.component_labels.assign(n, n);
  for (std::size_t start = 0; start < n; ++start) {
    if (out.component_labels[start] != n)
      continue;
    const auto               component = out.component_sizes.size();
    std::vector<std::size_t> queue{start};
    out.component_labels[start] = component;
    for (std::size_t head = 0; head < queue.size(); ++head)
      for (std::size_t j = 0; j < n; ++j)
        if (out.component_labels[j] == n && std::isfinite(out.adjacency(queue[head], j))) {
          out.component_labels[j] = component;
          queue.push_back(j);
        }
    out.component_sizes.push_back(queue.size());
  }
  return out;
}

dstruct::DataFrame IsomapGraph::edge_table() const {
  std::vector<double> row_i, row_j, distance, mutual;
  for (std::size_t i = 0; i < kept_rows.size(); ++i)
    for (std::size_t j = i + 1; j < kept_rows.size(); ++j) {
      if (!std::isfinite(adjacency(i, j)))
        continue;
      row_i.push_back(static_cast<double>(kept_rows[i]));
      row_j.push_back(static_cast<double>(kept_rows[j]));
      distance.push_back(adjacency(i, j));
      mutual.push_back(
          std::find(neighbors[i].begin(), neighbors[i].end(), j) != neighbors[i].end() &&
                  std::find(neighbors[j].begin(), neighbors[j].end(), i) != neighbors[j].end()
              ? 1.0
              : 0.0);
    }
  dstruct::DataFrame out;
  out.add_column("row_i", row_i);
  out.add_column("row_j", row_j);
  out.add_column("distance", distance);
  out.add_column("mutual", mutual);
  return out;
}

dstruct::DataFrame IsomapGraph::diagnostics() const {
  const auto  n          = kept_rows.size();
  std::size_t min_degree = n, max_degree = 0, degree_sum = 0;
  double      longest = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t degree = 0;
    for (std::size_t j = 0; j < n; ++j)
      if (i != j && std::isfinite(adjacency(i, j))) {
        ++degree;
        longest = std::max(longest, adjacency(i, j));
      }
    min_degree = std::min(min_degree, degree);
    max_degree = std::max(max_degree, degree);
    degree_sum += degree;
  }
  dstruct::DataFrame out;
  out.add_column("n", std::vector<double>{static_cast<double>(n)});
  out.add_column("k", std::vector<double>{static_cast<double>(n_neighbors)});
  out.add_column("components", std::vector<double>{static_cast<double>(component_sizes.size())});
  out.add_column("edges", std::vector<double>{static_cast<double>(degree_sum / 2)});
  out.add_column("min_degree", std::vector<double>{static_cast<double>(min_degree)});
  out.add_column("max_degree", std::vector<double>{static_cast<double>(max_degree)});
  out.add_column("mean_degree", std::vector<double>{static_cast<double>(degree_sum) / n});
  out.add_column("longest_edge", std::vector<double>{longest});
  return out;
}

graphics::RPlot IsomapGraph::plot(std::size_t feature_x, std::size_t feature_y) const {
  if (feature_x >= feature_values.cols() || feature_y >= feature_values.cols())
    throw std::out_of_range("IsomapGraph::plot: feature index out of range");
  auto                plot = graphics::RPlot::create();
  std::vector<double> x0, y0, x1, y1;
  for (std::size_t i = 0; i < kept_rows.size(); ++i)
    for (std::size_t j = i + 1; j < kept_rows.size(); ++j)
      if (std::isfinite(adjacency(i, j))) {
        x0.push_back(feature_values(i, feature_x));
        y0.push_back(feature_values(i, feature_y));
        x1.push_back(feature_values(j, feature_x));
        y1.push_back(feature_values(j, feature_y));
      }
  plot.segments(x0, y0, x1, y1, {148, 163, 184}, 1.0);
  for (std::size_t c = 0; c < component_sizes.size(); ++c) {
    std::vector<double> x, y;
    for (std::size_t i = 0; i < kept_rows.size(); ++i)
      if (component_labels[i] == c) {
        x.push_back(feature_values(i, feature_x));
        y.push_back(feature_values(i, feature_y));
      }
    plot.points(x, y, "component " + std::to_string(c + 1), kSeriesColors[c % 6]);
  }
  return plot.title("Union-kNN graph")
      .x_label(feature_names[feature_x])
      .y_label(feature_names[feature_y]);
}

Isomap::Isomap(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& columns,
    IsomapOptions                   options)
    : feature_columns_(columns), options_(options) {
  fit(data);
}

void Isomap::fit(const dstruct::DataFrame& data) {
  if (options_.n_components == 0)
    throw std::invalid_argument("Isomap: n_components must be positive");
  source_rows_  = data.nrows();
  graph_        = isomap_graph(data, feature_columns_, options_.n_neighbors, options_.metric);
  observations_ = graph_.kept_rows.size();
  const auto n  = observations_;
  if (options_.n_components >= n)
    throw std::invalid_argument("Isomap: n_components must be less than the complete row count");
  if (graph_.component_sizes.size() != 1)
    throw std::invalid_argument(
        "Isomap: neighbor graph is disconnected (" + std::to_string(graph_.component_sizes.size()) +
        " components); inspect isomap_graph() before choosing a neighborhood");
  geodesic_ = graph_.adjacency;
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = 0; i < n; ++i) {
      const double ik = geodesic_(i, k);
      if (!std::isfinite(ik))
        continue;
      for (std::size_t j = 0; j < n; ++j) {
        const double via = ik + geodesic_(k, j);
        if (via < geodesic_(i, j))
          geodesic_(i, j) = via;
      }
    }
  // Mirroring one triangle removes summation-order asymmetry, never an actual edge.
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      const double d = std::min(geodesic_(i, j), geodesic_(j, i));
      if (!std::isfinite(d) || !std::isfinite(d * d) || (d > 0 && d * d == 0))
        throw std::invalid_argument(
            "Isomap: squared shortest distances outside representable range");
      geodesic_(i, j) = geodesic_(j, i) = d;
    }
  std::vector<long double> row_mean(n, 0);
  long double              grand_mean = 0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j)
      row_mean[i] += static_cast<long double>(geodesic_(i, j)) * geodesic_(i, j) / n;
    grand_mean += row_mean[i] / n;
  }
  gram_        = linalg::DenseMatrix<double>(n, n);
  double scale = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i; j < n; ++j) {
      const double value = static_cast<double>(
          -.5L * (static_cast<long double>(geodesic_(i, j)) * geodesic_(i, j) - row_mean[i] -
                  row_mean[j] + grand_mean));
      if (!std::isfinite(value))
        throw std::invalid_argument("Isomap: centred Gram entry exceeds finite range");
      gram_(i, j) = gram_(j, i) = value;
      scale                     = std::max(scale, std::fabs(value));
    }
  auto normalized = gram_;
  if (scale > 0)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        normalized(i, j) /= scale;
  const auto eig = linalg::symmetric_eigen(normalized);
  eigenvalues_   = eig.eigenvalues;
  for (double& value : eigenvalues_) {
    value *= scale;
    if (!std::isfinite(value))
      throw std::invalid_argument("Isomap: eigenvalue exceeds finite range");
  }
  long double positive = 0, used = 0;
  for (double value : eigenvalues_)
    positive += std::max(0.0, value);
  embedding_ = linalg::DenseMatrix<double>(n, options_.n_components, 0.0);
  for (std::size_t c = 0; c < options_.n_components; ++c) {
    const double value = std::max(0.0, eigenvalues_[c]);
    used += value;
    for (std::size_t i = 0; i < n; ++i)
      embedding_(i, c) = eig.eigenvectors(i, c) * std::sqrt(value);
  }
  goodness_of_fit_ = positive > 0 ? static_cast<double>(used / positive) : 0;
}

std::vector<double> Isomap::dimension(std::size_t index) const {
  if (index >= n_components())
    throw std::out_of_range("Isomap::dimension: index out of range");
  return embedding_.col(index);
}

std::string Isomap::summary() const {
  std::ostringstream out;
  print_summary(out);
  return out.str();
}
void Isomap::print_summary() const {
  print_summary(std::cout);
}
void Isomap::print_summary(std::ostream& os) const {
  os << "Isomap\nFeatures: ";
  for (std::size_t j = 0; j < feature_columns_.size(); ++j)
    os << (j ? ", " : "") << feature_columns_[j];
  os << "\nDistance metric: "
     << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan")
     << "\nn_neighbors: " << options_.n_neighbors << "\nNumber of obs: " << observations_
     << ", dimensions: " << n_components()
     << "\nConnected components: " << graph_.component_sizes.size() << "\n"
     << std::fixed << std::setprecision(4)
     << "Goodness of fit (positive inertia only): " << goodness_of_fit_
     << "\n\nEigenvalues used:\n";
  for (std::size_t c = 0; c < n_components(); ++c)
    os << "  Dim" << c + 1 << ": " << eigenvalues_[c] << "\n";
}

graphics::RPlot Isomap::plot_embedding(std::size_t x, std::size_t y) const {
  if (x >= n_components() || y >= n_components())
    throw std::out_of_range("Isomap::plot_embedding: axis out of range");
  auto plot = graphics::RPlot::create();
  plot.points(dimension(x), dimension(y), "", kSeriesColors[0]);
  return plot.title("Isomap")
      .x_label("Dim" + std::to_string(x + 1))
      .y_label("Dim" + std::to_string(y + 1));
}

graphics::RPlot
Isomap::plot_embedding(const std::vector<std::string>& labels, std::size_t x, std::size_t y) const {
  if (x >= n_components() || y >= n_components())
    throw std::out_of_range("Isomap::plot_embedding: axis out of range");
  if (labels.size() != observations_)
    throw std::invalid_argument("Isomap::plot_embedding: labels must align to retained rows");
  auto       plot   = graphics::RPlot::create();
  const auto groups = unique_sorted(labels);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    std::vector<double> xs, ys;
    for (std::size_t i = 0; i < observations_; ++i)
      if (labels[i] == groups[g]) {
        xs.push_back(embedding_(i, x));
        ys.push_back(embedding_(i, y));
      }
    plot.points(xs, ys, groups[g], kSeriesColors[g % 6]);
  }
  return plot.title("Isomap")
      .x_label("Dim" + std::to_string(x + 1))
      .y_label("Dim" + std::to_string(y + 1));
}

dstruct::DataFrame Isomap::spectrum() const {
  long double positive = 0;
  double      largest  = 0;
  for (double value : eigenvalues_) {
    positive += std::max(0.0, value);
    largest = std::max(largest, std::fabs(value));
  }
  std::vector<double> dim, share, cumulative, negative;
  long double         used = 0;
  for (std::size_t c = 0; c < observations_; ++c) {
    const double value = eigenvalues_[c];
    used += std::max(0.0, value);
    dim.push_back(c + 1);
    share.push_back(positive > 0 ? static_cast<double>(std::max(0.0, value) / positive) : 0);
    cumulative.push_back(positive > 0 ? static_cast<double>(used / positive) : 0);
    negative.push_back(value < -1e-10 * largest ? 1.0 : 0.0);
  }
  dstruct::DataFrame out;
  out.add_column("dimension", dim);
  out.add_column("eigenvalue", eigenvalues_);
  out.add_column("positive_share", share);
  out.add_column("cumulative_positive_share", cumulative);
  out.add_column("significant_negative", negative);
  return out;
}

dstruct::DataFrame Isomap::dimension_diagnostics() const {
  const auto  n        = observations_;
  long double positive = 0, absolute = 0, negative = 0, norm2 = 0;
  for (double v : eigenvalues_) {
    positive += std::max(0.0, v);
    absolute += std::fabs(v);
    negative += std::max(0.0, -v);
    norm2 += static_cast<long double>(v) * v;
  }
  std::vector<double> dims, gof, neg, strain, rmse, relative, residual, reference;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j)
      reference.push_back(geodesic_(i, j));
  std::vector<double> embedded_distances(reference.size(), 0.0);
  long double         used = 0;
  for (std::size_t c = 0; c < n_components(); ++c) {
    used += std::max(0.0, eigenvalues_[c]);
    std::size_t         pair = 0;
    std::vector<double> distances;
    distances.reserve(reference.size());
    long double error = 0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        const double d           = embedding_(i, c) - embedding_(j, c);
        embedded_distances[pair] = std::hypot(embedded_distances[pair], d);
        distances.push_back(embedded_distances[pair++]);
        const long double e = distances.back() - geodesic_(i, j);
        error += e * e;
      }
    long double spectral_error = 0;
    for (std::size_t j = 0; j < n; ++j)
      if (j > c || eigenvalues_[j] < 0)
        spectral_error += static_cast<long double>(eigenvalues_[j]) * eigenvalues_[j];
    const double correlation = distance_correlation(reference, distances);
    dims.push_back(c + 1);
    gof.push_back(positive > 0 ? static_cast<double>(used / positive) : 0);
    neg.push_back(absolute > 0 ? static_cast<double>(negative / absolute) : 0);
    strain.push_back(norm2 > 0 ? static_cast<double>(std::sqrt(spectral_error / norm2)) : 0);
    rmse.push_back(static_cast<double>(std::sqrt(error / reference.size())));
    relative.push_back(relative_distance_error(distances, reference));
    residual.push_back(1 - correlation * correlation);
  }
  dstruct::DataFrame out;
  out.add_column("dimensions", dims);
  out.add_column("gof_positive", gof);
  out.add_column("negative_mass", neg);
  out.add_column("relative_strain", strain);
  out.add_column("geodesic_rmse", rmse);
  out.add_column("relative_geodesic_error", relative);
  out.add_column("residual_variance", residual);
  return out;
}

dstruct::DataFrame Isomap::pair_distances(std::size_t dimensions) const {
  if (!dimensions)
    dimensions = n_components();
  if (dimensions > n_components())
    throw std::out_of_range("Isomap::pair_distances: prefix out of range");
  std::vector<double> row_i, row_j, ambient, geodesic, embedded;
  for (std::size_t i = 0; i < observations_; ++i)
    for (std::size_t j = i + 1; j < observations_; ++j) {
      double d = 0;
      for (std::size_t c = 0; c < dimensions; ++c)
        d = std::hypot(d, embedding_(i, c) - embedding_(j, c));
      row_i.push_back(graph_.kept_rows[i]);
      row_j.push_back(graph_.kept_rows[j]);
      ambient.push_back(graph_.distances(i, j));
      geodesic.push_back(geodesic_(i, j));
      embedded.push_back(d);
    }
  dstruct::DataFrame out;
  out.add_column("row_i", row_i);
  out.add_column("row_j", row_j);
  out.add_column("ambient_distance", ambient);
  out.add_column("geodesic_distance", geodesic);
  out.add_column("embedded_distance", embedded);
  return out;
}

dstruct::DataFrame Isomap::reference_diagnostics(
    const dstruct::DataFrame&       reference,
    const std::vector<std::string>& columns) const {
  if (reference.nrows() != source_rows_ || columns.empty())
    throw std::invalid_argument(
        "Isomap: reference rows must match original input and columns must be nonempty");
  for (const auto& name : columns)
    if (!reference.has_column(name) ||
        reference.column_type(name) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("Isomap: reference columns must be numeric and present");
  linalg::DenseMatrix<double> truth(observations_, columns.size());
  for (std::size_t i = 0; i < observations_; ++i)
    for (std::size_t j = 0; j < columns.size(); ++j) {
      const auto value = reference.optional_double_at(columns[j], graph_.kept_rows[i]);
      if (!value || !std::isfinite(*value))
        throw std::invalid_argument(
            "Isomap: retained reference coordinates must be finite and complete");
      truth(i, j) = *value;
    }
  const auto pairs    = pair_distances();
  const auto geo      = pairs.numeric_column("geodesic_distance").values(),
             embedded = pairs.numeric_column("embedded_distance").values();
  std::vector<double> distances;
  distances.reserve(geo.size());
  for (std::size_t i = 0; i < observations_; ++i)
    for (std::size_t j = i + 1; j < observations_; ++j)
      distances.push_back(row_distance(truth, i, j, DistanceMetric::Euclidean));
  double max_edge_stretch = 0;
  for (std::size_t i = 0; i < observations_; ++i)
    for (std::size_t j = i + 1; j < observations_; ++j) {
      if (!std::isfinite(graph_.adjacency(i, j)))
        continue;
      const double target = row_distance(truth, i, j, DistanceMetric::Euclidean),
                   edge   = graph_.adjacency(i, j);
      const double ratio  = edge > 0      ? target / edge
                            : target == 0 ? 0.0
                                          : std::numeric_limits<double>::infinity();
      max_edge_stretch    = std::max(max_edge_stretch, ratio);
    }
  dstruct::DataFrame out;
  out.add_column("n", std::vector<double>{static_cast<double>(observations_)});
  out.add_column("pairs", std::vector<double>{static_cast<double>(geo.size())});
  out.add_column(
      "geodesic_relative_error", std::vector<double>{relative_distance_error(geo, distances)});
  out.add_column(
      "embedding_relative_error",
      std::vector<double>{relative_distance_error(embedded, distances)});
  out.add_column("geodesic_correlation", std::vector<double>{distance_correlation(geo, distances)});
  out.add_column(
      "embedding_correlation", std::vector<double>{distance_correlation(embedded, distances)});
  out.add_column("max_edge_stretch", std::vector<double>{max_edge_stretch});
  return out;
}

graphics::RPlot Isomap::plot_shepard(std::size_t dimensions) const {
  const auto pairs = pair_distances(dimensions);
  const auto x     = pairs.numeric_column("geodesic_distance").values(),
             y     = pairs.numeric_column("embedded_distance").values();
  const double largest =
      std::max(*std::max_element(x.begin(), x.end()), *std::max_element(y.begin(), y.end()));
  const double upper = largest > 0 ? largest * 1.05 : 1.0;
  auto         plot  = graphics::RPlot::create();
  plot.points(x, y, "", kSeriesColors[0], 2.5)
      .segments({0}, {0}, {upper}, {upper}, {90, 90, 90}, 1.0)
      .x_limits(0, upper)
      .y_limits(0, upper)
      .x_label("Graph-geodesic distance")
      .y_label("Embedded distance")
      .title("Isomap Shepard plot");
  return plot;
}
} // namespace datamunge::stats
