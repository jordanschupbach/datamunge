/// @file clustering_diagnostics.cpp
/// @brief Contingency pair counts and projection/tree graphics for fitted clusterings.
#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/stats/clustering_diagnostics.hpp>
#include <limits>
#include <map>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Implementation details shared by clustering agreement/composition summaries.
namespace {
/// @brief Ordered cluster-by-class counts; -1 retains noise separately.
using Counts = std::map<int, std::map<std::string, std::size_t>>;
/// @brief Validates aligned nonempty labels/classes and owns their ordered contingency counts.
/// @param labels Cluster ids >=0 or noise -1.
/// @param truth Equally sized class strings.
/// @return Owned nonzero cells grouped by cluster id then class string.
/// @throws std::invalid_argument Invalid sizes or label domain.
Counts contingency(const std::vector<int>& labels, const std::vector<std::string>& truth) {
  if (labels.empty() || labels.size() != truth.size())
    throw std::invalid_argument("clustering diagnostics: nonempty aligned labels/classes required");
  Counts out;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] < -1)
      throw std::invalid_argument("clustering diagnostics: label below -1");
    ++out[labels[i]][truth[i]];
  }
  return out;
}
/// @brief Number of unordered distinct pairs, computed without integer multiplication overflow.
/// @param n Cell or marginal count.
/// @return n*(n-1)/2 in extended floating precision, or zero below two.
long double pairs(std::size_t n) {
  return n < 2 ? 0 : static_cast<long double>(n) * (n - 1) / 2;
}
} // namespace
ClusterAgreement cluster_agreement(
    const std::vector<int>&         labels,
    const std::vector<std::string>& truth) {
  const auto                         counts = contingency(labels, truth);
  std::map<std::string, std::size_t> columns;
  long double                        cells = 0, rows = 0;
  std::size_t                        noise = 0, correct = 0, clusters = 0;
  for (const auto& [cluster, votes] : counts) {
    std::size_t size = 0, best = 0;
    for (const auto& [name, count] : votes) {
      columns[name] += count;
      size += count;
      best = std::max(best, count);
      cells += pairs(count);
    }
    rows += pairs(size);
    if (cluster == -1)
      noise = size;
    else {
      ++clusters;
      correct += best;
    }
  }
  long double cols = 0;
  for (const auto& [name, count] : columns)
    cols += pairs(count);
  const auto  n = labels.size(), assigned = n - noise;
  long double ari = 1;
  if (n >= 2) {
    const long double expected = rows * cols / pairs(n), denominator = (rows + cols) / 2 - expected;
    ari = denominator == 0 ? 1 : (cells - expected) / denominator;
  }
  return {
      n,
      clusters,
      noise,
      double(assigned) / n,
      assigned ? double(correct) / assigned : std::numeric_limits<double>::quiet_NaN(),
      double(ari)};
}
dstruct::DataFrame cluster_composition(
    const std::vector<int>&         labels,
    const std::vector<std::string>& truth) {
  const auto               counts = contingency(labels, truth);
  std::vector<double>      ids, sizes, majorities, purities;
  std::vector<std::string> names;
  for (const auto& [cluster, votes] : counts) {
    std::size_t size = 0, best = 0;
    std::string majority;
    for (const auto& [name, count] : votes) {
      size += count;
      if (count > best) {
        best     = count;
        majority = name;
      }
    }
    ids.push_back(cluster);
    sizes.push_back(size);
    majorities.push_back(best);
    names.push_back(majority);
    purities.push_back(double(best) / size);
  }
  dstruct::DataFrame out;
  out.add_column("cluster", ids);
  out.add_column("size", sizes);
  out.add_column("majority_class", names);
  out.add_column("majority_count", majorities);
  out.add_column("purity", purities);
  return out;
}
graphics::RPlot plot_clusters(
    const dstruct::DataFrame& data,
    const std::string&        x_feature,
    const std::string&        y_feature,
    const std::vector<int>&   labels,
    std::string               title) {
  if (labels.empty() || labels.size() != data.nrows())
    throw std::invalid_argument("plot_clusters: nonempty aligned rows/labels required");
  for (const auto& col : {x_feature, y_feature})
    if (!data.has_column(col) || data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("plot_clusters: numeric coordinates required");
  std::map<int, std::pair<std::vector<double>, std::vector<double>>> groups;
  double xmin = std::numeric_limits<double>::infinity(), xmax = -xmin, ymin = xmin, ymax = -xmin;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] < -1)
      throw std::invalid_argument("plot_clusters: label below -1");
    if (data.is_null(x_feature, i) || data.is_null(y_feature, i))
      throw std::invalid_argument("plot_clusters: null coordinate");
    const double x = data.double_at(x_feature, i), y = data.double_at(y_feature, i);
    if (!std::isfinite(x) || !std::isfinite(y))
      throw std::invalid_argument("plot_clusters: nonfinite coordinate");
    groups[labels[i]].first.push_back(x);
    groups[labels[i]].second.push_back(y);
    xmin = std::min(xmin, x);
    xmax = std::max(xmax, x);
    ymin = std::min(ymin, y);
    ymax = std::max(ymax, y);
  }
  const std::array<graphics::RGB, 8> palette = {
      graphics::RGB{37, 99, 235},
      {220, 38, 38},
      {22, 163, 74},
      {147, 51, 234},
      {234, 88, 12},
      {8, 145, 178},
      {190, 24, 93},
      {87, 83, 78}};
  auto        p = graphics::RPlot::create();
  std::size_t c = 0;
  for (const auto& [label, xy] : groups) {
    const auto color = label == -1 ? graphics::RGB{107, 114, 128} : palette[c++ % palette.size()];
    p.points(
        xy.first, xy.second, label == -1 ? "noise" : "cluster " + std::to_string(label), color, 4);
  }
  const double dx   = xmax > xmin ? xmax - xmin : .2 * std::max(1., std::abs(xmin));
  const double dy   = ymax > ymin ? ymax - ymin : .2 * std::max(1., std::abs(ymin));
  const double left = xmin - .08 * dx, right = xmax + .08 * dx;
  const double bottom = ymin - .1 * dy, top = ymax + .35 * dy;
  if (!std::isfinite(left) || !std::isfinite(right) || !std::isfinite(bottom) ||
      !std::isfinite(top))
    throw std::invalid_argument("plot_clusters: unrepresentable padded coordinate range");
  p.title(std::move(title))
      .x_label(x_feature)
      .y_label(y_feature)
      .x_limits(left, right)
      .y_limits(bottom, top);
  return p;
}
graphics::RPlot plot_dendrogram(const AgglomerativeClustering& model, std::string title) {
  const auto  n      = model.observations();
  const auto& merges = model.merge_history();
  if (n < 2 || merges.size() != n - 1)
    throw std::invalid_argument("plot_dendrogram: incomplete fitted tree");
  for (std::size_t j = 0; j < merges.size(); ++j) {
    const auto& m = merges[j];
    if (m.cluster_a >= n + j || m.cluster_b >= n + j || m.cluster_a == m.cluster_b ||
        !std::isfinite(m.distance) || m.distance < 0 ||
        m.distance > std::numeric_limits<double>::max() / 1.12)
      throw std::invalid_argument("plot_dendrogram: invalid tree geometry");
  }
  std::vector<std::size_t> stack{2 * n - 2};
  std::vector<double>      x(2 * n - 1), y(2 * n - 1, 0);
  std::size_t              cursor = 0;
  while (!stack.empty()) {
    const auto node = stack.back();
    stack.pop_back();
    if (node < n)
      x[node] = cursor++;
    else {
      const auto& m = merges[node - n];
      stack.push_back(m.cluster_b);
      stack.push_back(m.cluster_a);
    }
  }
  auto   p   = graphics::RPlot::create();
  double top = 0;
  for (std::size_t j = 0; j < merges.size(); ++j) {
    const auto& m = merges[j];
    x[n + j]      = (x[m.cluster_a] + x[m.cluster_b]) / 2;
    y[n + j]      = m.distance;
    top           = std::max(top, m.distance);
    p.line(
        {x[m.cluster_a], x[m.cluster_a], x[m.cluster_b], x[m.cluster_b]},
        {y[m.cluster_a], m.distance, m.distance, y[m.cluster_b]},
        "",
        {37, 99, 235});
  }
  top = std::max(1e-6, top);
  p.title(std::move(title))
      .x_label("leaf order (tree traversal)")
      .y_label("linkage height")
      .x_limits(-.5, n - .5)
      .y_limits(-.04 * top, 1.12 * top);
  return p;
}
} // namespace datamunge::stats
