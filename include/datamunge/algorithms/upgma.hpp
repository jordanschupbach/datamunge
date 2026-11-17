#pragma once
/// @file upgma.hpp
/// @brief Checked native upgma and actual producing diagnostics.
#include <cmath>
#include <cstddef>
#include <datamunge/plot/plot.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief One actual size-weighted cluster merge.
/// @par Example
/// @code
/// dm::algorithms::UPGMAMerge m{};
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct UPGMAMerge {
  int    left{0};       ///< Actual first child ID, leaf IDs below n.
  int    right{0};      ///< Actual second child ID.
  double height{0};     ///< Half the actual selected distance.
  int    id{0};         ///< New cluster ID, n+merge index.
  int    left_size{0};  ///< Actual leaf count in left cluster.
  int    right_size{0}; ///< Actual leaf count in right cluster.
};
/// @brief Actual UPGMA merge sequence and selected work.
/// @par Example
/// @code
/// auto r=dm::algorithms::upgma({{0,2},{2,0}});
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct UPGMAResult {
  std::vector<UPGMAMerge> merges;              ///< Actual producer merge order, root last.
  std::size_t             pair_checks{0};      ///< Actual active-cluster distance inspections.
  std::size_t             distance_updates{0}; ///< Actual new-cluster weighted averages.
};
/// @brief Build a size-weighted average-linkage rooted tree.
/// Active-row first encounter breaks equal-distance ties. Weighted means use long double intermediate arithmetic. O(n^3) pair scanning and O(n^2) storage. Leaf heights0; ultrametric output does not establish a molecular clock in the data.
/// @param dist Owned finite nonnegative exactly symmetric square matrix, zero diagonal, <=500 leaves.
/// @return Actual merges, cluster sizes and counters;0/1 leaf yields no merges.
/// @throws std::invalid_argument Invalid matrix shape, symmetry or distances.
/// @throws std::length_error More than500 leaves.
/// @par Example
/// @code
/// auto r=dm::algorithms::upgma({{0,2},{2,0}}); //height1
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline UPGMAResult upgma(std::vector<std::vector<double>> dist) {
  if (dist.size() > 500)
    throw std::length_error("UPGMA vertex cap");
  for (std::size_t i = 0; i < dist.size(); ++i) {
    if (dist[i].size() != dist.size())
      throw std::invalid_argument("UPGMA square matrix");
    for (std::size_t j = 0; j < dist.size(); ++j)
      if (!std::isfinite(dist[i][j]) || dist[i][j] < 0 || (i == j && dist[i][j] != 0))
        throw std::invalid_argument("UPGMA distances");
  }
  for (std::size_t i = 0; i < dist.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (dist[i][j] != dist[j][i])
        throw std::invalid_argument("UPGMA symmetry");
  const int   n = static_cast<int>(dist.size());
  UPGMAResult result;

  std::vector<int> id(n), size(n, 1); // active clusters, parallel arrays over the matrix rows
  for (int i = 0; i < n; ++i)
    id[i] = i;
  int next_id = n;

  while (static_cast<int>(id.size()) > 1) {
    // Find the closest pair of active clusters.
    int    bi = 0, bj = 1;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < id.size(); ++i)
      for (std::size_t j = i + 1; j < id.size(); ++j) {
        ++result.pair_checks;
        if (dist[i][j] < best) {
          best = dist[i][j];
          bi   = (int)i;
          bj   = (int)j;
        }
      }

    int si = size[bi], sj = size[bj];
    result.merges.push_back({id[bi], id[bj], best / 2.0, next_id, si, sj});

    // Weighted-average distances from the new cluster to every other active cluster.
    std::vector<double> newrow(id.size(), 0.0);
    for (std::size_t k = 0; k < id.size(); ++k)
      if ((int)k != bi && (int)k != bj) {
        newrow[k] = static_cast<double>(
            (static_cast<long double>(si) * dist[bi][k] +
             static_cast<long double>(sj) * dist[bj][k]) /
            (si + sj));
        ++result.distance_updates;
      }

    // Rebuild the active set: drop bi, bj; append the merged cluster.
    std::vector<int>    nid;
    std::vector<int>    nsize;
    std::vector<double> kept; // distances of survivors to the new cluster
    for (std::size_t k = 0; k < id.size(); ++k)
      if ((int)k != bi && (int)k != bj) {
        nid.push_back(id[k]);
        nsize.push_back(size[k]);
        kept.push_back(newrow[k]);
      }

    const int                        m = static_cast<int>(nid.size());
    std::vector<std::vector<double>> nd(m + 1, std::vector<double>(m + 1, 0.0));
    // Copy surviving-vs-surviving distances.
    std::vector<int> survivors;
    for (std::size_t k = 0; k < id.size(); ++k)
      if ((int)k != bi && (int)k != bj)
        survivors.push_back((int)k);
    for (int a = 0; a < m; ++a)
      for (int b = 0; b < m; ++b)
        nd[a][b] = dist[survivors[a]][survivors[b]];
    for (int a = 0; a < m; ++a) {
      nd[a][m] = kept[a];
      nd[m][a] = kept[a];
    }

    nid.push_back(next_id++);
    nsize.push_back(si + sj);
    id   = std::move(nid);
    size = std::move(nsize);
    dist = std::move(nd);
  }
  return result;
}

/// @brief Draw a canonical UPGMA merge result with actual branch heights.
/// Iterative traversal lays each subtree over a contiguous leaf interval. O(n) layout; caller owns saving the figure.
/// @param result Unmodified output of upgma.
/// @param n Original leaf count.
/// @param labels One label per leaf, or empty for numeric labels.
/// @return Owned native dendrogram plot.
/// @throws std::invalid_argument Result/label shape mismatch.
/// @par Example
/// @code
/// auto r=dm::algorithms::upgma({{0,2},{2,0}});auto p=dm::algorithms::upgma_plot(r,2);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline graphics::RPlot
upgma_plot(const UPGMAResult& result, std::size_t n, const std::vector<std::string>& labels = {}) {
  if ((n > 1 && result.merges.size() != n - 1) || (!labels.empty() && labels.size() != n))
    throw std::invalid_argument("UPGMA plot shape");
  auto p = graphics::RPlot::create();
  if (!n)
    return p;
  std::vector<double>              x(2 * n - 1), h(2 * n - 1);
  std::vector<std::pair<int, int>> ch(2 * n - 1, {-1, -1});
  for (auto m : result.merges) {
    ch[m.id] = {m.left, m.right};
    h[m.id]  = m.height;
  }
  std::vector<int> order, pending{int(2 * n - 2)};
  while (!pending.empty()) {
    int u = pending.back();
    pending.pop_back();
    order.push_back(u);
    if (u >= int(n)) {
      pending.push_back(ch[u].second);
      pending.push_back(ch[u].first);
    }
  }
  double leaf = 0;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    int u = *it;
    if (u < int(n))
      x[u] = leaf++;
    else
      x[u] = (x[ch[u].first] + x[ch[u].second]) / 2;
  }
  for (auto m : result.merges) {
    p.line(
        {x[m.left], x[m.left], x[m.right], x[m.right]},
        {h[m.left], m.height, m.height, h[m.right]},
        "",
        {37, 99, 235},
        2);
  }
  for (std::size_t i = 0; i < n; ++i)
    p.text(x[i], 0, labels.empty() ? std::to_string(i) : labels[i]);
  p.title("Actual UPGMA dendrogram").x_label("leaf layout").y_label("height = distance /2");
  return p;
}

} // namespace datamunge::algorithms
