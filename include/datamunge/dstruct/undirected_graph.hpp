#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::dstruct {

template <typename Vertex>
class UndirectedGraph {
 public:
  using vertex_type = Vertex;
  using size_type = std::size_t;

  UndirectedGraph() = default;

  UndirectedGraph(std::initializer_list<Vertex> vertices) {
    for (const auto& vertex : vertices) {
      add_vertex(vertex);
    }
  }

  explicit UndirectedGraph(const std::vector<Vertex>& vertices) {
    for (const auto& vertex : vertices) {
      add_vertex(vertex);
    }
  }

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
    adjacency_.emplace(vertex, std::vector<Vertex>{});
    return true;
  }

  bool add_vertex(Vertex&& vertex) {
    if (has_vertex(vertex)) {
      return false;
    }

    const auto index = vertices_.size();
    vertices_.push_back(std::move(vertex));
    const Vertex& stored = vertices_.back();
    vertex_index_.emplace(stored, index);
    adjacency_.emplace(stored, std::vector<Vertex>{});
    return true;
  }

  bool remove_vertex(const Vertex& vertex) {
    if (!has_vertex(vertex)) {
      return false;
    }

    const auto neighbors_copy = adjacency_.at(vertex);
    for (const auto& neighbor : neighbors_copy) {
      if (neighbor == vertex) {
        continue;
      }
      erase_one(adjacency_.at(neighbor), vertex);
      --edge_count_;
    }
    if (has_edge(vertex, vertex)) {
      --edge_count_;
    }

    adjacency_.erase(vertex);

    const auto removed_index = vertex_index_.at(vertex);
    vertex_index_.erase(vertex);
    vertices_.erase(vertices_.begin() + static_cast<std::ptrdiff_t>(removed_index));
    reindex_vertices(removed_index);
    return true;
  }

  [[nodiscard]] bool has_vertex(const Vertex& vertex) const {
    return vertex_index_.find(vertex) != vertex_index_.end();
  }

  bool add_edge(const Vertex& a, const Vertex& b) {
    add_vertex(a);
    add_vertex(b);
    if (has_edge(a, b)) {
      return false;
    }

    adjacency_.at(a).push_back(b);
    if (a != b) {
      adjacency_.at(b).push_back(a);
    }
    ++edge_count_;
    return true;
  }

  bool remove_edge(const Vertex& a, const Vertex& b) {
    if (!has_edge(a, b)) {
      return false;
    }

    erase_one(adjacency_.at(a), b);
    if (a != b) {
      erase_one(adjacency_.at(b), a);
    }
    --edge_count_;
    return true;
  }

  [[nodiscard]] bool has_edge(const Vertex& a, const Vertex& b) const {
    const auto it = adjacency_.find(a);
    if (it == adjacency_.end()) {
      return false;
    }

    const auto& neighbors = it->second;
    return std::find(neighbors.begin(), neighbors.end(), b) != neighbors.end();
  }

  [[nodiscard]] std::vector<Vertex> vertices() const { return vertices_; }

  [[nodiscard]] std::vector<Vertex> neighbors(const Vertex& vertex) const {
    const auto it = adjacency_.find(vertex);
    if (it == adjacency_.end()) {
      throw std::out_of_range("UndirectedGraph vertex not found");
    }
    return it->second;
  }

  [[nodiscard]] size_type degree(const Vertex& vertex) const { return neighbors(vertex).size(); }

  [[nodiscard]] std::vector<Vertex> bfs(const Vertex& start) const {
    require_vertex(start);

    std::unordered_map<Vertex, bool> visited;
    std::queue<Vertex> frontier;
    std::vector<Vertex> order;

    frontier.push(start);
    visited.emplace(start, true);

    while (!frontier.empty()) {
      Vertex current = frontier.front();
      frontier.pop();
      order.push_back(current);

      for (const auto& neighbor : adjacency_.at(current)) {
        if (visited.find(neighbor) != visited.end()) {
          continue;
        }
        visited.emplace(neighbor, true);
        frontier.push(neighbor);
      }
    }

    return order;
  }

  [[nodiscard]] std::vector<Vertex> dfs(const Vertex& start) const {
    require_vertex(start);

    std::unordered_map<Vertex, bool> visited;
    std::vector<Vertex> stack{start};
    std::vector<Vertex> order;

    while (!stack.empty()) {
      Vertex current = std::move(stack.back());
      stack.pop_back();
      if (visited.find(current) != visited.end()) {
        continue;
      }

      visited.emplace(current, true);
      order.push_back(current);

      const auto& neighbors_of_current = adjacency_.at(current);
      for (auto it = neighbors_of_current.rbegin(); it != neighbors_of_current.rend(); ++it) {
        if (visited.find(*it) == visited.end()) {
          stack.push_back(*it);
        }
      }
    }

    return order;
  }

  [[nodiscard]] bool contains_path(const Vertex& src, const Vertex& dst) const {
    require_vertex(src);
    require_vertex(dst);

    if (src == dst) {
      return true;
    }

    const auto reachable = bfs(src);
    return std::find(reachable.begin(), reachable.end(), dst) != reachable.end();
  }

  /// @brief The graph's connected components, each as a vector of vertices in BFS-visit
  ///        order; components themselves appear in vertices() order.
  [[nodiscard]] std::vector<std::vector<Vertex>> connected_components() const {
    std::unordered_map<Vertex, bool> visited;
    std::vector<std::vector<Vertex>> components;

    for (const auto& vertex : vertices_) {
      if (visited.find(vertex) != visited.end()) {
        continue;
      }
      auto component = bfs(vertex);
      for (const auto& v : component) {
        visited.emplace(v, true);
      }
      components.push_back(std::move(component));
    }

    return components;
  }

  /// @brief True if the graph contains a cycle (a self-loop, or two distinct paths between
  ///        some pair of vertices) in any component.
  [[nodiscard]] bool has_cycle() const {
    for (const auto& vertex : vertices_) {
      if (has_edge(vertex, vertex)) {
        return true;
      }
    }

    std::unordered_map<Vertex, bool> visited;
    for (const auto& start : vertices_) {
      if (visited.find(start) != visited.end()) {
        continue;
      }
      if (has_cycle_from(start, visited)) {
        return true;
      }
    }

    return false;
  }

 private:
  std::vector<Vertex> vertices_;
  std::unordered_map<Vertex, size_type> vertex_index_;
  std::unordered_map<Vertex, std::vector<Vertex>> adjacency_;
  size_type edge_count_ = 0;

  static void erase_one(std::vector<Vertex>& values, const Vertex& target) {
    const auto it = std::find(values.begin(), values.end(), target);
    if (it != values.end()) {
      values.erase(it);
    }
  }

  void reindex_vertices(size_type start) {
    for (size_type index = start; index < vertices_.size(); ++index) {
      vertex_index_.at(vertices_[index]) = index;
    }
  }

  void require_vertex(const Vertex& vertex) const {
    if (!has_vertex(vertex)) {
      throw std::out_of_range("UndirectedGraph vertex not found");
    }
  }

  // Iterative DFS with an explicit (vertex, parent) stack. Marks each vertex visited at the
  // moment it's PUSHED (not when popped) -- pushing every neighbor exactly once this way
  // means any neighbor found already-visited (other than the parent we just came from) is
  // necessarily reachable via a second, different path, i.e. a genuine cycle; marking at pop
  // time instead can push the same vertex from two different frontier branches before either
  // is processed and silently miss that cycle.
  [[nodiscard]] bool has_cycle_from(const Vertex& start, std::unordered_map<Vertex, bool>& visited) const {
    std::vector<std::pair<Vertex, Vertex>> stack{{start, start}};
    visited.emplace(start, true);

    while (!stack.empty()) {
      auto [current, parent] = stack.back();
      stack.pop_back();

      bool skipped_parent_edge = false;
      for (const auto& neighbor : adjacency_.at(current)) {
        if (!skipped_parent_edge && neighbor == parent && current != start) {
          skipped_parent_edge = true;
          continue;
        }
        if (visited.find(neighbor) != visited.end()) {
          return true;
        }
        visited.emplace(neighbor, true);
        stack.emplace_back(neighbor, current);
      }
    }

    return false;
  }
};

} // namespace datamunge::dstruct
