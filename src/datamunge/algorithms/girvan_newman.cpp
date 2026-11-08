/// @file girvan_newman.cpp
/// @brief Shared checked ordered-double Brandes kernel, legacy batch cuts and actual capture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/girvan_newman.hpp>
#include <limits>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private canonicalization, actual capture and shared graph kernels.
namespace {
/// @brief An owned undirected endpoint pair after canonicalization.
using Edge = std::pair<std::size_t, std::size_t>;
/// @brief Validate every endpoint before other domains and guard legacy int distances.
/// @param n Vertex count.
/// @param edges Raw ordered input edges.
/// @throws std::invalid_argument An endpoint is outside the graph.
/// @throws std::length_error Vertex count cannot fit legacy int distances.
void validate_graph(std::size_t n, const std::vector<Edge>& edges) {
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("network analysis: endpoint out of range");
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("girvan_newman: distance id capacity");
}
/// @brief Validate the legacy target, including the empty-graph target1 exception.
/// @param n Vertex count.
/// @param target Desired minimum number of components.
/// @throws std::invalid_argument Target is outside the valid domain.
void validate_target(std::size_t n, std::size_t target) {
  if (target == 0 || target > std::max<std::size_t>(1, n))
    throw std::invalid_argument("girvan_newman: invalid target community count");
}
/// @brief Add selected actual work without wrapping size_t.
/// @param value Owned current counter.
/// @param amount Actual operations/slots to add, default1.
/// @throws std::length_error Counter cannot represent the sum.
void add(std::size_t& value, std::size_t amount = 1) {
  if (amount > std::numeric_limits<std::size_t>::max() - value)
    throw std::length_error("girvan_newman: work capacity");
  value += amount;
}
/// @brief Reject nonfinite actual arithmetic before propagation or deletion.
/// @param value Required finite actual result.
/// @throws std::overflow_error A path/dependency/score is nonfinite.
void finite(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("girvan_newman: nonfinite path/dependency/score");
}
/// @brief Append one whole copied record from the executing kernel.
/// @param events Null for ordinary calls, otherwise owned current history.
/// @param cap Positive optional record quota, or0 to disable it.
/// @param e Actual copied event.
/// @throws std::length_error Positive quota or history capacity exhausted.
void emit(std::vector<GirvanNewmanEvent>* events, std::size_t cap, const GirvanNewmanEvent& e) {
  if (!events)
    return;
  if ((cap && events->size() >= cap) || events->size() == events->max_size())
    throw std::length_error("girvan_newman: complete event quota/capacity");
  events->push_back(e);
}
/// @brief Canonicalize validated input and capture the actual prepared edge list.
/// @param edges Validated raw input edges.
/// @param events Optional current history.
/// @param cap Positive optional record quota, or0.
/// @return Sorted unique non-loop ascending endpoint pairs.
/// @throws std::length_error Container/history capacity exhausted.
std::vector<Edge>
prepare(const std::vector<Edge>& edges, std::vector<GirvanNewmanEvent>* events, std::size_t cap) {
  std::vector<Edge> active;
  for (auto [u, v] : edges)
    if (u != v)
      active.push_back(std::minmax(u, v));
  std::sort(active.begin(), active.end());
  active.erase(std::unique(active.begin(), active.end()), active.end());
  GirvanNewmanEvent e;
  e.kind  = GirvanNewmanEventKind::canonical;
  e.count = active.size();
  emit(events, cap, e);
  for (std::size_t i = 0; i < active.size(); ++i) {
    e       = {};
    e.kind  = GirvanNewmanEventKind::edge;
    e.index = i;
    e.u     = active[i].first;
    e.v     = active[i].second;
    emit(events, cap, e);
  }
  return active;
}
/// @brief Build adjacency in the same sorted-active iteration order as the legacy kernel.
/// @param n Vertex count.
/// @param active Canonical active edges.
/// @return Owned adjacency vectors.
/// @throws std::length_error Container capacity exceeded.
std::vector<std::vector<std::size_t>> adjacency(std::size_t n, const std::vector<Edge>& active) {
  std::vector<std::vector<std::size_t>> out(n);
  for (auto [u, v] : active) {
    out[u].push_back(v);
    out[v].push_back(u);
  }
  return out;
}
/// @brief Compute actual legacy BFS components and count selected work.
/// @param adj Canonical-order adjacency.
/// @param work Current owned counters.
/// @return Components in ascending-root and actual BFS member order.
/// @throws std::length_error Counter/container capacity exceeded.
std::vector<std::vector<std::size_t>> components(
    const std::vector<std::vector<std::size_t>>& adj,
    GirvanNewmanWork&                            work) {
  add(work.component_passes);
  std::vector<bool>                     seen(adj.size());
  std::vector<std::vector<std::size_t>> out;
  for (std::size_t root = 0; root < adj.size(); ++root)
    if (!seen[root]) {
      out.emplace_back();
      std::queue<std::size_t> q;
      q.push(root);
      seen[root] = true;
      while (!q.empty()) {
        auto u = q.front();
        q.pop();
        out.back().push_back(u);
        add(work.component_vertices);
        for (auto v : adj[u]) {
          add(work.component_neighbor_checks);
          if (!seen[v]) {
            seen[v] = true;
            q.push(v);
          }
        }
      }
    }
  return out;
}
/// @brief Copy actual component metadata as bounded fixed primitive records.
/// @param groups Actual current BFS result, never recomputed for capture.
/// @param round Zero-based current check round.
/// @param events Optional owned history.
/// @param cap Positive optional record quota, or0.
/// @throws std::length_error History capacity/quota exhausted.
void component_records(
    const std::vector<std::vector<std::size_t>>& groups,
    std::size_t                                  round,
    std::vector<GirvanNewmanEvent>*              events,
    std::size_t                                  cap) {
  GirvanNewmanEvent e;
  e.kind  = GirvanNewmanEventKind::components;
  e.round = round;
  e.count = groups.size();
  emit(events, cap, e);
  for (std::size_t i = 0; i < groups.size(); ++i) {
    e       = {};
    e.kind  = GirvanNewmanEventKind::component;
    e.round = round;
    e.index = i;
    e.count = groups[i].size();
    emit(events, cap, e);
    for (auto u : groups[i]) {
      e.kind = GirvanNewmanEventKind::vertex;
      e.u    = u;
      emit(events, cap, e);
    }
  }
}
/// @brief Execute one actual ordered-double all-source Brandes edge accumulation.
/// @param n Vertex count with representable int distances.
/// @param active Sorted canonical edges.
/// @param adj Matching adjacency in legacy order.
/// @param work Owned current actual counters.
/// @param max_paths Largest actual initialized/updated path count, updated.
/// @param integer_guarantee Cleared before a certified integer addition can exceed2^53.
/// @param round Zero-based actual GN check round, or0 for standalone scoring.
/// @param events Optional actual primitive history.
/// @param cap Positive optional history quota, or0.
/// @return Actual raw source-summed edge scores in active order.
/// @throws std::overflow_error A required actual numeric result is nonfinite.
/// @throws std::length_error Counter/container/history capacity exceeded.
std::vector<double> brandes(
    std::size_t                                  n,
    const std::vector<Edge>&                     active,
    const std::vector<std::vector<std::size_t>>& adj,
    GirvanNewmanWork&                            work,
    double&                                      max_paths,
    bool&                                        integer_guarantee,
    std::size_t                                  round,
    std::vector<GirvanNewmanEvent>*              events,
    std::size_t                                  cap) {
  std::vector<double> score(active.size());
  if (active.empty())
    return score;
  for (std::size_t source = 0; source < n; ++source) {
    std::vector<std::vector<std::size_t>> pred(n);
    std::vector<int>                      dist(n, -1);
    std::vector<double>                   paths(n), dependency(n);
    std::vector<std::size_t>              order;
    std::queue<std::size_t>               q;
    dist[source]  = 0;
    paths[source] = 1;
    q.push(source);
    add(work.source_initializations);
    add(work.source_slots, n);
    max_paths = std::max(max_paths, 1.);
    GirvanNewmanEvent e;
    e.kind   = GirvanNewmanEventKind::source;
    e.round  = round;
    e.source = source;
    emit(events, cap, e);
    while (!q.empty()) {
      auto u = q.front();
      q.pop();
      order.push_back(u);
      add(work.bfs_pops);
      e          = {};
      e.kind     = GirvanNewmanEventKind::pop;
      e.round    = round;
      e.source   = source;
      e.u        = u;
      e.distance = dist[u];
      e.paths_u  = paths[u];
      emit(events, cap, e);
      for (auto v : adj[u]) {
        add(work.neighbor_checks);
        e          = {};
        e.kind     = GirvanNewmanEventKind::scan;
        e.round    = round;
        e.source   = source;
        e.u        = u;
        e.v        = v;
        e.distance = dist[v];
        e.paths_v  = paths[v];
        emit(events, cap, e);
        if (dist[v] < 0) {
          dist[v] = dist[u] + 1;
          q.push(v);
          add(work.discoveries);
          e.kind     = GirvanNewmanEventKind::discover;
          e.distance = dist[v];
          emit(events, cap, e);
        }
        if (dist[v] == dist[u] + 1) {
          // While certified, operands are exact nonnegative integers <=2^53.
          // Subtraction stays exact in that integer range. Check before rounding:
          // the mathematical2^53+1 can otherwise round back to2^53 unnoticed.
          if (integer_guarantee && paths[u] > 9007199254740992.0 - paths[v])
            integer_guarantee = false;
          paths[v] += paths[u];
          finite(paths[v]);
          max_paths = std::max(max_paths, paths[v]);
          pred[v].push_back(u);
          add(work.path_updates);
          e.kind    = GirvanNewmanEventKind::path;
          e.paths_u = paths[u];
          e.paths_v = paths[v];
          emit(events, cap, e);
        }
      }
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      auto v = *it;
      for (auto u : pred[v]) {
        double contribution = (paths[u] / paths[v]) * (1 + dependency[v]);
        finite(contribution);
        dependency[u] += contribution;
        finite(dependency[u]);
        Edge edge = std::minmax(u, v);
        auto pos  = std::lower_bound(
            active.begin(), active.end(), edge, [&work](const Edge& a, const Edge& b) {
              add(work.edge_search_comparisons);
              return a < b;
            });
        const auto index = static_cast<std::size_t>(pos - active.begin());
        score[index] += contribution;
        finite(score[index]);
        add(work.dependency_updates);
        e              = {};
        e.kind         = GirvanNewmanEventKind::dependency;
        e.round        = round;
        e.source       = source;
        e.u            = u;
        e.v            = v;
        e.paths_u      = paths[u];
        e.paths_v      = paths[v];
        e.dependency_v = dependency[v];
        e.contribution = contribution;
        e.dependency_u = dependency[u];
        e.index        = index;
        e.score        = score[index];
        emit(events, cap, e);
      }
    }
  }
  for (std::size_t i = 0; i < score.size(); ++i) {
    add(work.edge_scores);
    GirvanNewmanEvent e;
    e.kind  = GirvanNewmanEventKind::edge_score;
    e.round = round;
    e.index = i;
    e.score = score[i];
    emit(events, cap, e);
  }
  return score;
}
/// @brief Execute actual legacy batch cuts with optional primitive capture.
/// @param n Validated vertex count.
/// @param edges Fully validated raw edges.
/// @param target Validated minimum component target.
/// @param events Optional owned current history.
/// @param cap Positive optional complete-record quota, or0.
/// @return Owned final groups, actual work and numeric diagnostics.
/// @throws std::overflow_error Nonfinite required numeric update.
/// @throws std::length_error Counter/container/history capacity exceeded.
GirvanNewmanRun run(
    std::size_t                     n,
    const std::vector<Edge>&        edges,
    std::size_t                     target,
    std::vector<GirvanNewmanEvent>* events,
    std::size_t                     cap) {
  GirvanNewmanRun out;
  out.work.input_edges = edges.size();
  auto active          = prepare(edges, events, cap);
  while (true) {
    const auto        round = out.work.rounds;
    GirvanNewmanEvent e;
    e.kind  = GirvanNewmanEventKind::round;
    e.round = round;
    e.count = active.size();
    emit(events, cap, e);
    auto adj    = adjacency(n, active);
    auto groups = components(adj, out.work);
    component_records(groups, round, events, cap);
    if (groups.size() >= target || active.empty()) {
      e       = {};
      e.kind  = GirvanNewmanEventKind::stop;
      e.round = round;
      e.count = groups.size();
      emit(events, cap, e);
      out.groups = std::move(groups);
      return out;
    }
    auto score = brandes(
        n,
        active,
        adj,
        out.work,
        out.max_paths,
        out.path_counts_integer_guaranteed,
        round,
        events,
        cap);
    add(out.work.rounds);
    const double maximum = *std::max_element(score.begin(), score.end());
    finite(maximum);
    e         = {};
    e.kind    = GirvanNewmanEventKind::maximum;
    e.round   = round;
    e.maximum = maximum;
    emit(events, cap, e);
    std::vector<Edge> kept;
    for (std::size_t i = 0; i < active.size(); ++i) {
      const bool keep = score[i] < maximum - 1e-12;
      add(out.work.edge_decisions);
      e          = {};
      e.kind     = GirvanNewmanEventKind::keep;
      e.round    = round;
      e.index    = i;
      e.accepted = keep;
      emit(events, cap, e);
      if (keep)
        kept.push_back(active[i]);
      else
        add(out.work.removed_edges);
    }
    active.swap(kept);
  }
}
/// @brief Run the shared standalone edge-score computation after validation.
/// @param n Validated vertex count.
/// @param edges Validated raw edges.
/// @param events Optional owned actual events.
/// @param cap Positive optional complete-record quota, or0.
/// @return Owned actual raw/conventional scores, canonical edges and diagnostics.
/// @throws std::overflow_error Nonfinite actual arithmetic.
/// @throws std::length_error Counter/container/history capacity exceeded.
EdgeBetweenness betweenness(
    std::size_t                     n,
    const std::vector<Edge>&        edges,
    std::vector<GirvanNewmanEvent>* events,
    std::size_t                     cap) {
  EdgeBetweenness out;
  out.work.input_edges = edges.size();
  out.edges            = prepare(edges, events, cap);
  const auto adj       = adjacency(n, out.edges);
  out.raw_scores       = brandes(
      n,
      out.edges,
      adj,
      out.work,
      out.max_paths,
      out.path_counts_integer_guaranteed,
      0,
      events,
      cap);
  out.scores.reserve(out.raw_scores.size());
  for (double value : out.raw_scores)
    out.scores.push_back(value / 2);
  return out;
}
/// @brief Multiply whole generated graph dimensions before allocation.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Checked product.
/// @throws std::length_error size_t multiplication overflow.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("network graph: count overflow");
  return a * b;
}
/// @brief Check a generated graph's whole vertex admission and output capacity.
/// @param n Whole vertex count.
/// @param e Whole edge count.
/// @param cap Positive optional whole-vertex quota, or0.
/// @throws std::length_error Int ids, optional quota or edge-vector capacity exceeded.
void graph_capacity(std::size_t n, std::size_t e, std::size_t cap) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) || (cap && n > cap) ||
      e > std::vector<Edge>{}.max_size())
    throw std::length_error("network graph: whole count capacity/quota");
}
} // namespace

std::vector<std::vector<std::size_t>> girvan_newman(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             target) {
  validate_graph(n, edges);
  validate_target(n, target);
  return run(n, edges, target, nullptr, 0).groups;
}
GirvanNewmanRun girvan_newman_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             target) {
  validate_graph(n, edges);
  validate_target(n, target);
  return run(n, edges, target, nullptr, 0);
}
GirvanNewmanTrace girvan_newman_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             target,
    std::size_t                                             cap) {
  validate_graph(n, edges);
  validate_target(n, target);
  GirvanNewmanTrace out;
  out.run = run(n, edges, target, &out.events, cap);
  return out;
}
EdgeBetweenness edge_betweenness(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
  validate_graph(n, edges);
  return betweenness(n, edges, nullptr, 0);
}
EdgeBetweennessTrace edge_betweenness_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             cap) {
  validate_graph(n, edges);
  EdgeBetweennessTrace out;
  out.result = betweenness(n, edges, &out.events, cap);
  return out;
}
double graph_modularity(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::vector<std::size_t>>&            groups) {
  validate_graph(n, edges);
  std::vector<std::size_t> labels(n, groups.size());
  std::size_t              assigned = 0;
  for (std::size_t c = 0; c < groups.size(); ++c) {
    if (groups[c].empty())
      throw std::invalid_argument("graph modularity: empty group");
    for (auto u : groups[c]) {
      if (u >= n || labels[u] != groups.size())
        throw std::invalid_argument("graph modularity: invalid/duplicate vertex");
      labels[u] = c;
      ++assigned;
    }
  }
  if (assigned != n)
    throw std::invalid_argument("graph modularity: incomplete partition");
  const auto active = prepare(edges, nullptr, 0);
  if (active.empty())
    return 0;
  std::vector<std::size_t> internal(groups.size()), degree(groups.size());
  for (auto [u, v] : active) {
    add(degree[labels[u]]);
    add(degree[labels[v]]);
    if (labels[u] == labels[v])
      add(internal[labels[u]]);
  }
  const double m = static_cast<double>(active.size());
  double       q = 0;
  for (std::size_t c = 0; c < groups.size(); ++c) {
    const double fraction = static_cast<double>(degree[c]) / (2 * m);
    q += static_cast<double>(internal[c]) / m - fraction * fraction;
  }
  return q;
}
NetworkGraph
clique_bridge_graph(std::size_t left, std::size_t right, std::size_t bridges, std::size_t cap) {
  if (!left || !right || bridges > std::min(left, right))
    throw std::invalid_argument("clique bridge graph: invalid dimensions/bridges");
  std::size_t n = left;
  add(n, right);
  const auto  a = multiply(left, left - 1) / 2, b = multiply(right, right - 1) / 2;
  std::size_t count = a;
  add(count, b);
  add(count, bridges);
  graph_capacity(n, count, cap);
  NetworkGraph out;
  out.vertex_count = n;
  out.edges.reserve(count);
  for (std::size_t i = 0; i < left; ++i)
    for (std::size_t j = i + 1; j < left; ++j)
      out.edges.emplace_back(i, j);
  for (std::size_t i = 0; i < right; ++i)
    for (std::size_t j = i + 1; j < right; ++j)
      out.edges.emplace_back(left + i, left + j);
  for (std::size_t k = 0; k < bridges; ++k)
    out.edges.emplace_back(left - 1 - k, left + k);
  return out;
}
NetworkGraph layered_graph(std::size_t layers, std::size_t width, std::size_t cap) {
  if (!layers || !width)
    throw std::invalid_argument("layered graph: positive dimensions required");
  const auto n = multiply(layers, width), count = multiply(layers - 1, multiply(width, width));
  graph_capacity(n, count, cap);
  NetworkGraph out;
  out.vertex_count = n;
  out.edges.reserve(count);
  for (std::size_t l = 0; l + 1 < layers; ++l)
    for (std::size_t a = 0; a < width; ++a)
      for (std::size_t b = 0; b < width; ++b)
        out.edges.emplace_back(l * width + a, (l + 1) * width + b);
  return out;
}
} // namespace datamunge::algorithms
