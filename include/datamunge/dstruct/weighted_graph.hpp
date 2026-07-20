#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::dstruct {

/// @brief A graph with weighted edges, usable as either directed or undirected (chosen once
///        at construction). Separate from DirectedGraph/UndirectedGraph (which are simpler,
///        unweighted, and don't carry this class's algorithmic machinery) rather than adding
///        weights to those -- keeps both simpler classes' tested behavior untouched.
template <typename Vertex, typename Weight = double>
class WeightedGraph {
 public:
  using vertex_type = Vertex;
  using weight_type = Weight;
  using size_type = std::size_t;

  explicit WeightedGraph(bool directed = true) : directed_(directed) {}

  WeightedGraph(std::initializer_list<Vertex> vertices, bool directed = true) : directed_(directed) {
    for (const auto& vertex : vertices) {
      add_vertex(vertex);
    }
  }

  WeightedGraph(const std::vector<Vertex>& vertices, bool directed = true) : directed_(directed) {
    for (const auto& vertex : vertices) {
      add_vertex(vertex);
    }
  }

  [[nodiscard]] bool directed() const { return directed_; }
  [[nodiscard]] bool empty() const { return vertices_.empty(); }
  [[nodiscard]] size_type vertex_count() const { return vertices_.size(); }
  [[nodiscard]] size_type edge_count() const { return edge_count_; }

  void clear() {
    vertices_.clear();
    vertex_index_.clear();
    adjacency_.clear();
    edge_count_ = 0;
  }

  bool add_vertex(const Vertex& vertex) {
    if (has_vertex(vertex)) {
      return false;
    }
    vertex_index_.emplace(vertex, vertices_.size());
    vertices_.push_back(vertex);
    adjacency_.emplace(vertex, std::vector<std::pair<Vertex, Weight>>{});
    return true;
  }

  bool remove_vertex(const Vertex& vertex) {
    if (!has_vertex(vertex)) {
      return false;
    }

    if (directed_) {
      edge_count_ -= adjacency_.at(vertex).size(); // outgoing edges from vertex (including a self-loop, if any)
      for (const auto& src : vertices_) {
        if (src == vertex) {
          continue;
        }
        auto& list = adjacency_.at(src);
        const auto before = list.size();
        erase_to(list, vertex);
        edge_count_ -= (before - list.size()); // incoming edges to vertex
      }
    } else {
      for (const auto& [neighbor, weight] : adjacency_.at(vertex)) {
        (void)weight;
        if (neighbor == vertex) {
          continue;
        }
        erase_to(adjacency_.at(neighbor), vertex);
      }
      edge_count_ -= adjacency_.at(vertex).size();
    }

    adjacency_.erase(vertex);
    const auto removed_index = vertex_index_.at(vertex);
    vertex_index_.erase(vertex);
    vertices_.erase(vertices_.begin() + static_cast<std::ptrdiff_t>(removed_index));
    for (size_type i = removed_index; i < vertices_.size(); ++i) vertex_index_.at(vertices_[i]) = i;
    return true;
  }

  [[nodiscard]] bool has_vertex(const Vertex& vertex) const { return vertex_index_.find(vertex) != vertex_index_.end(); }

  /// @brief Replaces the weight if the edge already exists (rather than rejecting the call,
  ///        unlike DirectedGraph/UndirectedGraph's add_edge -- weights are the whole point
  ///        of this class, so re-adding with a new weight is the natural way to update one).
  bool add_edge(const Vertex& src, const Vertex& dst, const Weight& weight) {
    add_vertex(src);
    add_vertex(dst);

    auto& forward = adjacency_.at(src);
    const auto it = std::find_if(forward.begin(), forward.end(), [&](const auto& p) { return p.first == dst; });
    const bool is_new = (it == forward.end());
    if (is_new) {
      forward.emplace_back(dst, weight);
    } else {
      it->second = weight;
    }

    if (!directed_ && src != dst) {
      auto& backward = adjacency_.at(dst);
      const auto bit = std::find_if(backward.begin(), backward.end(), [&](const auto& p) { return p.first == src; });
      if (bit == backward.end()) {
        backward.emplace_back(src, weight);
      } else {
        bit->second = weight;
      }
    }

    if (is_new) ++edge_count_;
    return is_new;
  }

  bool remove_edge(const Vertex& src, const Vertex& dst) {
    if (!has_edge(src, dst)) {
      return false;
    }
    erase_to(adjacency_.at(src), dst);
    if (!directed_ && src != dst) {
      erase_to(adjacency_.at(dst), src);
    }
    --edge_count_;
    return true;
  }

  [[nodiscard]] bool has_edge(const Vertex& src, const Vertex& dst) const {
    const auto it = adjacency_.find(src);
    if (it == adjacency_.end()) {
      return false;
    }
    const auto& list = it->second;
    return std::find_if(list.begin(), list.end(), [&](const auto& p) { return p.first == dst; }) != list.end();
  }

  [[nodiscard]] Weight edge_weight(const Vertex& src, const Vertex& dst) const {
    const auto& list = adjacency_for(src);
    const auto it = std::find_if(list.begin(), list.end(), [&](const auto& p) { return p.first == dst; });
    if (it == list.end()) {
      throw std::out_of_range("WeightedGraph::edge_weight: no such edge");
    }
    return it->second;
  }

  [[nodiscard]] std::vector<Vertex> vertices() const { return vertices_; }

  /// @brief Outgoing neighbors if directed(); all incident neighbors if !directed().
  [[nodiscard]] std::vector<Vertex> neighbors(const Vertex& vertex) const {
    std::vector<Vertex> result;
    for (const auto& [neighbor, weight] : adjacency_for(vertex)) {
      (void)weight;
      result.push_back(neighbor);
    }
    return result;
  }

  struct ShortestPaths {
    /// @brief Only contains entries for vertices reachable from the source (the source
    ///        itself maps to 0).
    std::unordered_map<Vertex, Weight> distance;
    /// @brief The predecessor of every reachable, non-source vertex on its shortest path.
    std::unordered_map<Vertex, Vertex> predecessor;

    /// @brief Reconstructs the shortest path to @p target as a vertex sequence starting at
    ///        the source, or an empty vector if @p target isn't reachable.
    [[nodiscard]] std::vector<Vertex> path_to(const Vertex& target) const {
      if (distance.find(target) == distance.end()) {
        return {};
      }
      std::vector<Vertex> path{target};
      Vertex current = target;
      while (predecessor.find(current) != predecessor.end()) {
        current = predecessor.at(current);
        path.push_back(current);
      }
      std::reverse(path.begin(), path.end());
      return path;
    }
  };

  /// @brief Dijkstra's algorithm. Throws std::invalid_argument if any edge has a negative
  ///        weight (use bellman_ford() instead).
  [[nodiscard]] ShortestPaths dijkstra(const Vertex& source) const {
    require_vertex(source);
    for (const auto& v : vertices_) {
      for (const auto& [neighbor, weight] : adjacency_.at(v)) {
        (void)neighbor;
        if (weight < Weight{}) {
          throw std::invalid_argument("WeightedGraph::dijkstra: negative edge weight (use bellman_ford instead)");
        }
      }
    }

    using QueueEntry = std::pair<Weight, Vertex>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> frontier;

    ShortestPaths result;
    result.distance.emplace(source, Weight{});
    frontier.emplace(Weight{}, source);

    std::unordered_map<Vertex, bool> finalized;
    while (!frontier.empty()) {
      const auto [dist, current] = frontier.top();
      frontier.pop();
      if (finalized.find(current) != finalized.end()) {
        continue;
      }
      finalized.emplace(current, true);

      for (const auto& [neighbor, weight] : adjacency_.at(current)) {
        const Weight candidate = dist + weight;
        const auto it = result.distance.find(neighbor);
        if (it == result.distance.end() || candidate < it->second) {
          result.distance[neighbor] = candidate;
          result.predecessor[neighbor] = current;
          frontier.emplace(candidate, neighbor);
        }
      }
    }

    return result;
  }

  /// @brief Same computation as dijkstra(), but returned as (vertex, distance) pairs in
  ///        vertices() order rather than an unordered_map -- a binding-friendly shape (plain
  ///        vector<pair<>>, using the same std_pair/std_vector machinery already relied on
  ///        elsewhere) for language bindings where std::unordered_map isn't cleanly exposed.
  [[nodiscard]] std::vector<std::pair<Vertex, Weight>> dijkstra_distances(const Vertex& source) const {
    const auto result = dijkstra(source);
    std::vector<std::pair<Vertex, Weight>> out;
    out.reserve(result.distance.size());
    for (const auto& v : vertices_) {
      const auto it = result.distance.find(v);
      if (it != result.distance.end()) out.emplace_back(v, it->second);
    }
    return out;
  }

  struct BellmanFordResult : ShortestPaths {
    /// @brief True if a negative-weight cycle reachable from the source exists, in which
    ///        case distance/predecessor are not meaningful (shortest paths are undefined).
    bool has_negative_cycle{false};
  };

  [[nodiscard]] BellmanFordResult bellman_ford(const Vertex& source) const {
    require_vertex(source);

    BellmanFordResult result;
    result.distance.emplace(source, Weight{});

    const auto relax_once = [&]() {
      bool changed = false;
      for (const auto& src : vertices_) {
        const auto src_it = result.distance.find(src);
        if (src_it == result.distance.end()) {
          continue;
        }
        for (const auto& [dst, weight] : adjacency_.at(src)) {
          const Weight candidate = src_it->second + weight;
          const auto dst_it = result.distance.find(dst);
          if (dst_it == result.distance.end() || candidate < dst_it->second) {
            result.distance[dst] = candidate;
            result.predecessor[dst] = src;
            changed = true;
          }
        }
      }
      return changed;
    };

    for (size_type i = 0; i + 1 < vertices_.size(); ++i) {
      if (!relax_once()) break;
    }
    // One more full pass: if anything still improves, a negative cycle is reachable.
    result.has_negative_cycle = relax_once();

    return result;
  }

  /// @brief Same computation as bellman_ford(), but returned as (vertex, distance) pairs in
  ///        vertices() order rather than an unordered_map -- see dijkstra_distances() for why.
  ///        Empty (with has_negative_cycle() separately reporting the reason) when a negative
  ///        cycle makes the distances themselves meaningless.
  [[nodiscard]] std::vector<std::pair<Vertex, Weight>> bellman_ford_distances(const Vertex& source) const {
    const auto result = bellman_ford(source);
    std::vector<std::pair<Vertex, Weight>> out;
    if (result.has_negative_cycle) return out;
    out.reserve(result.distance.size());
    for (const auto& v : vertices_) {
      const auto it = result.distance.find(v);
      if (it != result.distance.end()) out.emplace_back(v, it->second);
    }
    return out;
  }

  [[nodiscard]] bool bellman_ford_has_negative_cycle(const Vertex& source) const {
    return bellman_ford(source).has_negative_cycle;
  }

  /// @brief All-pairs shortest distances (Floyd-Warshall). Unreachable pairs are simply
  ///        absent from the returned map's inner maps.
  [[nodiscard]] std::unordered_map<Vertex, std::unordered_map<Vertex, Weight>> floyd_warshall() const {
    std::unordered_map<Vertex, std::unordered_map<Vertex, Weight>> dist;
    for (const auto& v : vertices_) dist[v][v] = Weight{};
    for (const auto& src : vertices_) {
      for (const auto& [dst, weight] : adjacency_.at(src)) {
        // Look up via find() (never operator[]) before deciding to insert/update: operator[]
        // would default-construct a 0-valued entry as a side effect of merely checking
        // whether one exists, making the "existing" check below vacuously true forever.
        const auto existing = dist[src].find(dst);
        if (existing == dist[src].end() || weight < existing->second) dist[src][dst] = weight;
      }
    }

    for (const auto& k : vertices_) {
      for (const auto& i : vertices_) {
        const auto ik_it = dist[i].find(k);
        if (ik_it == dist[i].end()) continue;
        // Copy the value out rather than keeping the iterator: the j-loop below mutates
        // dist[i] (possibly rehashing it), which would invalidate an iterator held across
        // those mutations for the rest of this i-iteration.
        const Weight ik = ik_it->second;
        for (const auto& j : vertices_) {
          const auto kj_it = dist[k].find(j);
          if (kj_it == dist[k].end()) continue;
          const Weight candidate = ik + kj_it->second;
          auto& row = dist[i];
          const auto ij_it = row.find(j);
          if (ij_it == row.end() || candidate < ij_it->second) row[j] = candidate;
        }
      }
    }

    return dist;
  }

  /// @brief Kruskal's minimum spanning tree/forest, as a new undirected WeightedGraph
  ///        containing every vertex and the selected edges. Throws std::logic_error if
  ///        directed() (MST is only defined for undirected graphs).
  [[nodiscard]] WeightedGraph minimum_spanning_tree() const {
    if (directed_) {
      throw std::logic_error("WeightedGraph::minimum_spanning_tree: only defined for undirected graphs");
    }

    struct Edge {
      Vertex a, b;
      Weight weight;
    };
    std::vector<Edge> edges;
    // vertex_index_ gives a stable arbitrary total order, used purely to avoid double-adding
    // each undirected edge once from each endpoint (Vertex itself need not be orderable, so
    // this can't just compare a < b directly).
    for (const auto& src : vertices_) {
      for (const auto& [dst, weight] : adjacency_.at(src)) {
        if (vertex_index_.at(src) <= vertex_index_.at(dst)) edges.push_back({src, dst, weight});
      }
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) { return x.weight < y.weight; });

    std::unordered_map<Vertex, Vertex> parent;
    for (const auto& v : vertices_) parent[v] = v;
    const auto find_root = [&](Vertex v) {
      while (parent[v] != v) v = parent[v];
      return v;
    };

    WeightedGraph mst(vertices_, /*directed=*/false);
    for (const auto& e : edges) {
      const Vertex ra = find_root(e.a);
      const Vertex rb = find_root(e.b);
      if (ra == rb) continue;
      parent[ra] = rb;
      mst.add_edge(e.a, e.b, e.weight);
    }
    return mst;
  }

  /// @brief Connected components (ignoring weights). Throws std::logic_error if directed()
  ///        (use strongly_connected_components() instead).
  [[nodiscard]] std::vector<std::vector<Vertex>> connected_components() const {
    if (directed_) {
      throw std::logic_error("WeightedGraph::connected_components: only defined for undirected graphs; "
                              "use strongly_connected_components() for a directed graph");
    }

    std::unordered_map<Vertex, bool> visited;
    std::vector<std::vector<Vertex>> components;
    for (const auto& start : vertices_) {
      if (visited.find(start) != visited.end()) continue;
      std::vector<Vertex> component;
      std::vector<Vertex> stack{start};
      visited.emplace(start, true);
      while (!stack.empty()) {
        Vertex current = std::move(stack.back());
        stack.pop_back();
        component.push_back(current);
        for (const auto& [neighbor, weight] : adjacency_.at(current)) {
          (void)weight;
          if (visited.find(neighbor) == visited.end()) {
            visited.emplace(neighbor, true);
            stack.push_back(neighbor);
          }
        }
      }
      components.push_back(std::move(component));
    }
    return components;
  }

  /// @brief Strongly connected components (Tarjan's algorithm), each as a vector of
  ///        vertices; components are returned in reverse-topological order (a component with
  ///        no edges leaving it to another component comes first). Throws std::logic_error
  ///        if !directed() (use connected_components() instead).
  [[nodiscard]] std::vector<std::vector<Vertex>> strongly_connected_components() const {
    if (!directed_) {
      throw std::logic_error("WeightedGraph::strongly_connected_components: only defined for directed graphs; "
                              "use connected_components() for an undirected graph");
    }

    TarjanState state;
    std::vector<std::vector<Vertex>> result;
    for (const auto& v : vertices_) {
      if (state.index.find(v) == state.index.end()) {
        tarjan_strongconnect(v, state, result);
      }
    }
    return result;
  }

  /// @brief True if the graph contains a cycle: a self-loop (either kind), or -- for a
  ///        directed graph, a directed cycle (found via 3-color DFS); for an undirected
  ///        graph, two distinct paths between some pair of vertices.
  [[nodiscard]] bool has_cycle() const {
    for (const auto& v : vertices_) {
      if (has_edge(v, v)) return true;
    }

    if (directed_) {
      std::unordered_map<Vertex, int> color; // 0 = white (unvisited), 1 = gray (on stack), 2 = black (done)
      for (const auto& start : vertices_) {
        if (color.find(start) == color.end() && has_directed_cycle_from(start, color)) return true;
      }
      return false;
    }

    std::unordered_map<Vertex, bool> visited;
    for (const auto& start : vertices_) {
      if (visited.find(start) != visited.end()) continue;
      if (has_undirected_cycle_from(start, visited)) return true;
    }
    return false;
  }

 private:
  bool directed_;
  std::vector<Vertex> vertices_;
  std::unordered_map<Vertex, size_type> vertex_index_;
  std::unordered_map<Vertex, std::vector<std::pair<Vertex, Weight>>> adjacency_;
  size_type edge_count_ = 0;

  static void erase_to(std::vector<std::pair<Vertex, Weight>>& list, const Vertex& target) {
    const auto it = std::find_if(list.begin(), list.end(), [&](const auto& p) { return p.first == target; });
    if (it != list.end()) list.erase(it);
  }

  void require_vertex(const Vertex& vertex) const {
    if (!has_vertex(vertex)) throw std::out_of_range("WeightedGraph vertex not found");
  }

  [[nodiscard]] const std::vector<std::pair<Vertex, Weight>>& adjacency_for(const Vertex& vertex) const {
    const auto it = adjacency_.find(vertex);
    if (it == adjacency_.end()) throw std::out_of_range("WeightedGraph vertex not found");
    return it->second;
  }

  // Mark-at-push-time iterative DFS, same reasoning as UndirectedGraph::has_cycle_from.
  [[nodiscard]] bool has_undirected_cycle_from(const Vertex& start, std::unordered_map<Vertex, bool>& visited) const {
    std::vector<std::pair<Vertex, Vertex>> stack{{start, start}};
    visited.emplace(start, true);
    while (!stack.empty()) {
      auto [current, parent] = stack.back();
      stack.pop_back();
      bool skipped_parent_edge = false;
      for (const auto& [neighbor, weight] : adjacency_.at(current)) {
        (void)weight;
        if (!skipped_parent_edge && neighbor == parent && current != start) {
          skipped_parent_edge = true;
          continue;
        }
        if (visited.find(neighbor) != visited.end()) return true;
        visited.emplace(neighbor, true);
        stack.emplace_back(neighbor, current);
      }
    }
    return false;
  }

  // Recursive 3-color DFS for directed-cycle detection: a gray (on-stack) neighbor means a
  // back edge, i.e. a cycle. Recursive (not iterative) since the natural iterative
  // formulation needs an explicit "child iterator position" per stack frame to correctly
  // pop back to gray->black -- the recursive version is far less error-prone to get right,
  // and graphs deep enough to matter for stack size aren't this class's target use case.
  [[nodiscard]] bool has_directed_cycle_from(const Vertex& v, std::unordered_map<Vertex, int>& color) const {
    color[v] = 1;
    for (const auto& [neighbor, weight] : adjacency_.at(v)) {
      (void)weight;
      const auto it = color.find(neighbor);
      if (it == color.end()) {
        if (has_directed_cycle_from(neighbor, color)) return true;
      } else if (it->second == 1) {
        return true;
      }
    }
    color[v] = 2;
    return false;
  }

  struct TarjanState {
    std::unordered_map<Vertex, size_type> index;
    std::unordered_map<Vertex, size_type> low_link;
    std::unordered_map<Vertex, bool> on_stack;
    std::vector<Vertex> stack;
    size_type next_index{0};
  };

  void tarjan_strongconnect(const Vertex& v, TarjanState& state, std::vector<std::vector<Vertex>>& result) const {
    state.index[v] = state.next_index;
    state.low_link[v] = state.next_index;
    ++state.next_index;
    state.stack.push_back(v);
    state.on_stack[v] = true;

    for (const auto& [neighbor, weight] : adjacency_.at(v)) {
      (void)weight;
      if (state.index.find(neighbor) == state.index.end()) {
        tarjan_strongconnect(neighbor, state, result);
        state.low_link[v] = std::min(state.low_link[v], state.low_link[neighbor]);
      } else if (state.on_stack[neighbor]) {
        state.low_link[v] = std::min(state.low_link[v], state.index[neighbor]);
      }
    }

    if (state.low_link[v] == state.index[v]) {
      std::vector<Vertex> component;
      Vertex w;
      do {
        w = state.stack.back();
        state.stack.pop_back();
        state.on_stack[w] = false;
        component.push_back(w);
      } while (!(w == v));
      result.push_back(std::move(component));
    }
  }
};

} // namespace datamunge::dstruct
