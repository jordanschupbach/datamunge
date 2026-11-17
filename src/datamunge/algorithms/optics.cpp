/// @file optics.cpp
/// @brief Canonical checked serial scan-seed OPTICS and actual ordering/reachability histories.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/optics.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete admission, checked distances and one original scan-seed ordering path.
namespace {
/// @brief Add logical/operation counts without overflow.
/// @param a Original count.
/// @param b Added count.
/// @return Checked sum.
/// @throws std::length_error If addition overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("OPTICS: count geometry");
  return a + b;
}
/// @brief Multiply complete geometry without overflow.
/// @param a First dimension.
/// @param b Second dimension.
/// @return Checked product.
/// @throws std::length_error If multiplication overflows.
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("OPTICS: complete geometry");
  return a * b;
}
/// @brief Compute original safe distance order, widening complete unsafe intermediates.
/// @param a Validated finite coordinates.
/// @param b Matching validated finite coordinates.
/// @param work Optional actual selected-distance counter sink.
/// @return Finite rounded Euclidean distance.
/// @throws std::length_error If an actual selected counter overflows.
/// @throws std::overflow_error If distance cannot be represented as finite double.
double distance(const std::vector<double>& a, const std::vector<double>& b, OpticsWork* work) {
  double sum     = 0;
  bool   widened = false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double delta = a[i] - b[i];
    sum += delta * delta;
    widened |= !std::isfinite(delta) || !std::isfinite(sum);
    if (work)
      work->coordinate_differences = add(work->coordinate_differences, 1);
  }
  if (work)
    work->distance_calls = add(work->distance_calls, 1);
  if (!widened)
    return std::sqrt(sum);
  long double norm = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    norm = std::hypot(norm, static_cast<long double>(a[i]) - b[i]);
  double result = static_cast<double>(norm);
  if (!std::isfinite(norm) || !std::isfinite(result))
    throw std::overflow_error("OPTICS: finite double distance required");
  if (work)
    work->widened_distances = add(work->widened_distances, 1);
  return result;
}
/// @brief Validate every stored ordering ID and distance before extraction/profile indexing.
/// @param r Complete actual result.
/// @throws std::invalid_argument If dimensions/permutation/nonnegative-distance checks fail.
void validate_result(const OpticsResult& r) {
  auto n = r.reachability.size();
  if (r.ordering.size() != n || r.core_distance.size() != n)
    throw std::invalid_argument("OPTICS result: complete dimensions required");
  std::vector<char> seen(n);
  for (auto id : r.ordering) {
    if (id >= n || seen[id])
      throw std::invalid_argument("OPTICS result: complete permutation required");
    seen[id] = 1;
  }
  for (auto* values : {&r.reachability, &r.core_distance})
    for (double value : *values)
      if (std::isnan(value) || value < 0)
        throw std::invalid_argument("OPTICS result: nonnegative finite/+inf distances required");
}
/// @brief Execute one canonical ordering with optional complete owned observations.
/// @param data Complete original coordinate rows.
/// @param eps Finite nonnegative generating radius.
/// @param min_pts Positive neighbors including self.
/// @param history Optional owned actual-history sink.
/// @param cap Actual logical history quota,0 disables.
/// @return Actual ordering, per-ID distances, original controls and selected work.
/// @throws std::invalid_argument If complete finite/shape/control checks fail.
/// @throws std::length_error If geometry/count or actual history quota fails.
/// @throws std::overflow_error If an actual distance cannot be finite double.
OpticsRun run(
    const std::vector<std::vector<double>>& data,
    double                                  eps,
    std::size_t                             min_pts,
    OpticsTrace*                            history,
    std::size_t                             cap) {
  if (!std::isfinite(eps) || eps < 0 || !min_pts)
    throw std::invalid_argument("OPTICS: finite nonnegative radius and positive min_pts required");
  auto n = data.size(), d = n ? data[0].size() : 0;
  for (auto& row : data) {
    if (row.size() != d)
      throw std::invalid_argument("OPTICS: complete rectangular coordinates required");
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument("OPTICS: complete finite coordinates required");
  }
  auto nd = mul(n, d);
  (void)mul(mul(n, n), d);
  /// @brief Admit actual complete next owned elements/record before copying or appending.
  /// @param units Actual logical contribution.
  /// @throws std::length_error If geometry or selected history quota fails.
  auto charge = [&](std::size_t units) {
    if (!history)
      return;
    auto next = add(history->captured_entries, units);
    if (cap && next > cap)
      throw std::length_error("OPTICS: actual capture quota");
    history->captured_entries = next;
  };
  if (history) {
    charge(add(2, add(nd, mul(3, n))));
    history->data = data;
  }
  constexpr double inf = std::numeric_limits<double>::infinity();
  OpticsRun        actual;
  actual.eps     = eps;
  actual.min_pts = min_pts;
  auto& r        = actual.result;
  auto& w        = actual.work;
  r.reachability.assign(n, inf);
  r.core_distance.assign(n, inf);
  r.ordering.reserve(n);
  std::vector<char>        processed(n);
  std::vector<std::size_t> seeds;
  /// @brief Query actual inclusive neighbors and select min_pts-th sorted distance.
  /// @param p Original point ID.
  /// @return Actual original-ID neighbor list, with core distance committed in result.
  auto neighbors = [&](std::size_t p) {
    w.neighborhood_queries = add(w.neighborhood_queries, 1);
    std::vector<std::pair<std::size_t, double>> out;
    for (std::size_t q = 0; q < n; ++q) {
      auto value = distance(data[p], data[q], &w);
      if (value <= eps)
        out.push_back({q, value});
    }
    double core = inf;
    if (out.size() >= min_pts) {
      std::vector<double> distances;
      distances.reserve(out.size());
      for (auto [id, value] : out)
        distances.push_back(value);
      std::sort(distances.begin(), distances.end());
      core = distances[min_pts - 1];
    }
    r.core_distance[p] = core;
    if (history) {
      charge(add(1, mul(2, out.size())));
      history->neighborhoods.push_back({p, out, core});
    }
    return out;
  };
  /// @brief Inspect every actual core-neighbor occurrence and update seed priorities.
  /// @param nbrs Actual native queried neighbors in original-ID order.
  /// @param p Actual already-processed finite-core source.
  auto update = [&](const std::vector<std::pair<std::size_t, double>>& nbrs, std::size_t p) {
    for (auto [o, value] : nbrs) {
      w.update_candidates = add(w.update_candidates, 1);
      OpticsReachUpdate u;
      u.from      = p;
      u.to        = o;
      u.distance  = value;
      u.core      = r.core_distance[p];
      u.before    = r.reachability[o];
      u.after     = u.before;
      u.processed = processed[o];
      if (!u.processed) {
        u.proposed = std::max(u.core, value);
        if (u.before == inf) {
          u.after    = u.proposed;
          u.inserted = true;
        } else if (u.proposed < u.before) {
          u.after     = u.proposed;
          u.decreased = true;
        }
      }
      if (history) {
        charge(1);
        history->updates.push_back(u);
      }
      if (u.inserted) {
        r.reachability[o] = u.after;
        seeds.push_back(o);
        w.seed_insertions = add(w.seed_insertions, 1);
      } else if (u.decreased) {
        r.reachability[o] = u.after;
        w.decreases       = add(w.decreases, 1);
      }
    }
  };
  /// @brief Extract the first strict minimum from actual seed order, removing by swap-back.
  /// @return Actual selected original point ID.
  auto extract = [&]() {
    OpticsSelection selection;
    if (history) {
      charge(add(1, mul(2, seeds.size())));
      selection.seeds = seeds;
      selection.reachabilities.reserve(seeds.size());
      for (auto id : seeds)
        selection.reachabilities.push_back(r.reachability[id]);
    }
    std::size_t best  = 0;
    double      value = inf;
    for (std::size_t i = 0; i < seeds.size(); ++i) {
      w.seed_scan_entries = add(w.seed_scan_entries, 1);
      if (r.reachability[seeds[i]] < value) {
        value = r.reachability[seeds[i]];
        best  = i;
      }
    }
    auto id = seeds[best];
    if (history) {
      selection.winning_position = best;
      selection.point            = id;
      history->selections.push_back(std::move(selection));
    }
    seeds[best] = seeds.back();
    seeds.pop_back();
    w.seed_extractions = add(w.seed_extractions, 1);
    return id;
  };
  /// @brief Commit the actual processed flag/order and optional owned visit.
  /// @param p Actual selected original point ID.
  /// @param start True only for an outer-loop start.
  auto visit = [&](std::size_t p, bool start) {
    if (history) {
      charge(1);
      history->visits.push_back({p, r.core_distance[p], r.reachability[p], start});
    }
    processed[p] = 1;
    r.ordering.push_back(p);
  };
  for (std::size_t p = 0; p < n; ++p) {
    if (processed[p])
      continue;
    auto nbrs = neighbors(p);
    visit(p, true);
    if (r.core_distance[p] == inf)
      continue;
    seeds.clear();
    update(nbrs, p);
    while (!seeds.empty()) {
      auto q = extract();
      if (processed[q])
        continue;
      auto qn = neighbors(q);
      visit(q, false);
      if (r.core_distance[q] != inf)
        update(qn, q);
    }
  }
  return actual;
}
} // namespace
OpticsResult optics(const std::vector<std::vector<double>>& data, double eps, std::size_t min_pts) {
  return run(data, eps, min_pts, nullptr, 0).result;
}
OpticsRun
optics_run(const std::vector<std::vector<double>>& data, double eps, std::size_t min_pts) {
  return run(data, eps, min_pts, nullptr, 0);
}
OpticsTrace optics_trace(
    const std::vector<std::vector<double>>& data,
    double                                  eps,
    std::size_t                             min_pts,
    std::size_t                             max_entries) {
  OpticsTrace t;
  t.run = run(data, eps, min_pts, &t, max_entries);
  return t;
}
std::vector<int> optics_extract_dbscan(const OpticsResult& r, double eps_cluster) {
  validate_result(r);
  if (!std::isfinite(eps_cluster) || eps_cluster < 0)
    throw std::invalid_argument("OPTICS extraction: finite nonnegative cut required");
  std::vector<int> labels(r.ordering.size(), -1);
  int              cluster = -1;
  for (auto p : r.ordering)
    if (r.reachability[p] > eps_cluster) {
      if (r.core_distance[p] <= eps_cluster) {
        if (cluster == std::numeric_limits<int>::max())
          throw std::length_error("OPTICS extraction: cluster ID geometry");
        labels[p] = ++cluster;
      }
    } else
      labels[p] = cluster;
  return labels;
}
std::vector<int> optics_extract_dbscan(const OpticsRun& r, double eps_cluster) {
  if (!std::isfinite(r.eps) || r.eps < 0 || !r.min_pts || !std::isfinite(eps_cluster) ||
      eps_cluster < 0 || eps_cluster > r.eps)
    throw std::invalid_argument("OPTICS extraction: cut must be within generating radius");
  return optics_extract_dbscan(r.result, eps_cluster);
}
OpticsClusterSummary optics_cluster_summary(const std::vector<int>& labels) {
  int maximum = -1;
  for (int label : labels) {
    if (label < -1 || (label >= 0 && std::size_t(label) >= labels.size()))
      throw std::invalid_argument("OPTICS summary: labels must be noise or contiguous valid IDs");
    maximum = std::max(maximum, label);
  }
  OpticsClusterSummary summary;
  summary.labels = labels;
  summary.sizes.assign(maximum < 0 ? 0 : std::size_t(maximum) + 1, 0);
  for (int label : labels)
    if (label < 0)
      ++summary.noise;
    else
      ++summary.sizes[label];
  for (auto count : summary.sizes)
    if (!count)
      throw std::invalid_argument("OPTICS summary: no missing cluster IDs allowed");
  return summary;
}
OpticsReachabilityProfile optics_reachability_profile(
    const OpticsResult& r,
    double              undefined_height) {
  validate_result(r);
  if (!std::isfinite(undefined_height) || undefined_height < 0)
    throw std::invalid_argument("OPTICS profile: finite nonnegative display height required");
  OpticsReachabilityProfile p;
  if (undefined_height == 0) {
    double maximum = 0;
    for (double value : r.reachability)
      if (std::isfinite(value))
        maximum = std::max(maximum, value);
    undefined_height = maximum > 0 ? maximum * 1.15 : 1.;
    if (!std::isfinite(undefined_height))
      throw std::overflow_error("OPTICS profile: finite automatic ceiling required");
  }
  p.undefined_height = undefined_height;
  p.position.reserve(r.ordering.size());
  p.displayed.reserve(r.ordering.size());
  p.undefined.reserve(r.ordering.size());
  for (std::size_t i = 0; i < r.ordering.size(); ++i) {
    auto value   = r.reachability[r.ordering[i]];
    bool missing = std::isinf(value);
    p.position.push_back(double(i));
    p.undefined.push_back(missing);
    p.displayed.push_back(missing ? undefined_height : value);
  }
  return p;
}
std::vector<std::vector<double>> optics_demo_data() {
  std::mt19937_64                  rng(30);
  std::vector<std::vector<double>> data;
  /// @brief Append one original Gaussian blob using a fresh distribution object.
  /// @param cx Center x.
  /// @param cy Center y.
  /// @param sd Positive original standard deviation.
  auto blob = [&](double cx, double cy, double sd) {
    std::normal_distribution<double> normal(0, sd);
    for (int i = 0; i < 40; ++i)
      data.push_back({cx + normal(rng), cy + normal(rng)});
  };
  blob(0, 0, .20);
  blob(8, 0, .70);
  blob(4, 7, .35);
  std::uniform_real_distribution<double> uniform(-2, 10);
  for (int i = 0; i < 12; ++i)
    data.push_back({uniform(rng), uniform(rng)});
  return data;
}
/// @brief Original checked distance-helper compatibility namespace.
namespace detail {
double optics_euclidean(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    throw std::invalid_argument("OPTICS distance: matching dimensions required");
  for (auto* values : {&a, &b})
    for (double x : *values)
      if (!std::isfinite(x))
        throw std::invalid_argument("OPTICS distance: complete finite coordinates required");
  return distance(a, b, nullptr);
}
} // namespace detail
} // namespace datamunge::algorithms
