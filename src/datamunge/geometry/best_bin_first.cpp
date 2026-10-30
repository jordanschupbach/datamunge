/// @file best_bin_first.cpp
/// @brief Bucketed k-d tree construction and Best Bin First k-nearest-neighbour search.

#include <datamunge/geometry/best_bin_first.hpp>

#include <numeric>
#include <stdexcept>

namespace datamunge::geometry {

/// @brief Helpers for best_bin_first.cpp.
namespace {

/// Squared Euclidean distance between two points of the same dimension.
double squared_distance(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double t = a[i] - b[i];
    d += t * t;
  }
  return d;
}

/// A queued branch: its lower bound, its node, and (for cell keys) the per-axis offsets of the
/// query from the branch's cell.
struct QueueEntry {
  double key;                  ///< Lower bound on the squared distance to any point in the branch.
  int node;                    ///< The branch's root node.
  std::vector<double> offset;  ///< Per-axis offset from the query to the cell (cell keys only).
};

}  // namespace

BestBinFirstKd::BestBinFirstKd(std::vector<std::vector<double>> points, int leaf_size, BinPriority priority)
    : points_(std::move(points)), leaf_size_(leaf_size), priority_(priority) {
  if (leaf_size < 1) throw std::invalid_argument("BestBinFirstKd: leaf_size must be at least 1");
  if (points_.empty()) return;
  dim_ = points_.front().size();
  for (const auto& p : points_)
    if (p.size() != dim_) throw std::invalid_argument("BestBinFirstKd: all points must have the same dimension");
  order_.resize(points_.size());
  std::iota(order_.begin(), order_.end(), 0);
  nodes_.reserve(2 * points_.size() / static_cast<std::size_t>(leaf_size) + 2);
  build(0, static_cast<int>(points_.size()));
}

int BestBinFirstKd::build(int begin, int end) {
  const int id = static_cast<int>(nodes_.size());
  nodes_.push_back(Node{});
  int axis = -1;
  double widest = 0.0;
  if (end - begin > leaf_size_) {
    for (std::size_t a = 0; a < dim_; ++a) {
      double lo = points_[order_[begin]][a], hi = lo;
      for (int i = begin + 1; i < end; ++i) {
        lo = std::min(lo, points_[order_[i]][a]);
        hi = std::max(hi, points_[order_[i]][a]);
      }
      if (hi - lo > widest) widest = hi - lo, axis = static_cast<int>(a);
    }
  }
  if (axis < 0) {  // small enough, or every point identical
    nodes_[id].begin = begin;
    nodes_[id].end = end;
    return id;
  }
  const int mid = begin + (end - begin) / 2;
  std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                   [&](int a, int b) { return points_[a][axis] < points_[b][axis]; });
  // Left gets order_[begin, mid), all with coordinate <= split; right gets the rest.
  double split = points_[order_[begin]][axis];
  for (int i = begin + 1; i < mid; ++i) split = std::max(split, points_[order_[i]][axis]);
  nodes_[id].axis = axis;
  nodes_[id].split = split;
  const int left = build(begin, mid);
  const int right = build(mid, end);
  nodes_[id].left = left;
  nodes_[id].right = right;
  return id;
}

std::size_t BestBinFirstKd::leaf_count() const noexcept {
  std::size_t leaves = 0;
  for (const auto& n : nodes_) leaves += n.axis < 0;
  return leaves;
}

BestBinFirstResult BestBinFirstKd::knn(const std::vector<double>& query, int k, int max_examined) const {
  if (k < 1) throw std::invalid_argument("BestBinFirstKd::knn: k must be at least 1");
  if (query.size() != dim_ && !points_.empty())
    throw std::invalid_argument("BestBinFirstKd::knn: the query has the wrong dimension");
  BestBinFirstResult result;
  if (points_.empty()) {
    result.exact = true;
    return result;
  }
  const bool cell = priority_ == BinPriority::Cell;
  // Max-heap of the k best (squared distance, index) so far.
  std::priority_queue<std::pair<double, int>> best;
  const auto kth = [&] {
    return static_cast<int>(best.size()) < k ? std::numeric_limits<double>::infinity() : best.top().first;
  };
  const auto larger_key = [](const QueueEntry& a, const QueueEntry& b) { return a.key > b.key; };
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, decltype(larger_key)> queue(larger_key);
  queue.push(QueueEntry{0.0, 0, cell ? std::vector<double>(dim_, 0.0) : std::vector<double>{}});
  bool budget_hit = false;
  while (!queue.empty() && !budget_hit) {
    QueueEntry entry = queue.top();
    queue.pop();
    if (entry.key > kth()) break;  // every remaining branch is at least this far
    int node = entry.node;
    double key = entry.key;
    std::vector<double> offset = std::move(entry.offset);
    while (nodes_[node].axis >= 0) {
      const Node& n = nodes_[node];
      const double diff = query[n.axis] - n.split;
      const int near = diff <= 0 ? n.left : n.right;
      const int far = diff <= 0 ? n.right : n.left;
      double far_key;
      if (cell) {
        // Moving into the far cell changes only this axis's offset (Arya and Mount).
        far_key = key - offset[n.axis] * offset[n.axis] + diff * diff;
        if (far_key <= kth()) {
          std::vector<double> far_offset = offset;
          far_offset[n.axis] = diff;
          queue.push(QueueEntry{far_key, far, std::move(far_offset)});
        }
      } else {
        far_key = diff * diff;
        if (far_key <= kth()) queue.push(QueueEntry{far_key, far, {}});
      }
      node = near;
    }
    ++result.leaves;
    for (int i = nodes_[node].begin; i < nodes_[node].end; ++i) {
      const int p = order_[i];
      const double d = squared_distance(query, points_[p]);
      ++result.examined;
      if (static_cast<int>(best.size()) < k) {
        best.emplace(d, p);
      } else if (d < best.top().first) {
        best.pop();
        best.emplace(d, p);
      }
      if (max_examined > 0 && result.examined >= max_examined) {
        budget_hit = true;
        break;
      }
    }
  }
  result.exact = !budget_hit;
  const std::size_t found = best.size();
  result.indices.resize(found);
  result.distances.resize(found);
  for (std::size_t i = found; i-- > 0;) {
    result.indices[i] = best.top().second;
    result.distances[i] = std::sqrt(best.top().first);
    best.pop();
  }
  return result;
}

std::vector<int> knn_brute_force(const std::vector<std::vector<double>>& points, const std::vector<double>& query, int k) {
  std::vector<std::pair<double, int>> all(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) all[i] = {squared_distance(query, points[i]), static_cast<int>(i)};
  const auto kk = std::min<std::size_t>(static_cast<std::size_t>(std::max(k, 0)), all.size());
  std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(kk), all.end());
  std::vector<int> out(kk);
  for (std::size_t i = 0; i < kk; ++i) out[i] = all[i].second;
  return out;
}

std::vector<std::vector<int>> knn_lists(const std::vector<std::vector<double>>& points, int k) {
  if (k < 1 || static_cast<std::size_t>(k) >= points.size())
    throw std::invalid_argument("knn_lists: k must be at least 1 and below the number of points");
  std::vector<std::vector<int>> lists(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (const int j : knn_brute_force(points, points[i], k + 1)) {
      if (static_cast<std::size_t>(j) != i && static_cast<int>(lists[i].size()) < k) lists[i].push_back(j);
    }
  }
  return lists;
}

}  // namespace datamunge::geometry
