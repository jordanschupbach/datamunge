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
class DirectedGraph {
 public:
  using vertex_type = Vertex;
  using size_type = std::size_t;

  DirectedGraph() = default;

  DirectedGraph(std::initializer_list<Vertex> vertices) {
    for (const auto& vertex : vertices) {
      add_vertex(vertex);
    }
  }

  explicit DirectedGraph(const std::vector<Vertex>& vertices) {
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
    outgoing_.clear();
    incoming_.clear();
    edge_count_ = 0;
  }

  bool add_vertex(const Vertex& vertex) {
    if (has_vertex(vertex)) {
      return false;
    }

    vertex_index_.emplace(vertex, vertices_.size());
    vertices_.push_back(vertex);
    outgoing_.emplace(vertex, std::vector<Vertex>{});
    incoming_.emplace(vertex, std::vector<Vertex>{});
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
    outgoing_.emplace(stored, std::vector<Vertex>{});
    incoming_.emplace(stored, std::vector<Vertex>{});
    return true;
  }

  bool remove_vertex(const Vertex& vertex) {
    if (!has_vertex(vertex)) {
      return false;
    }

    const auto outgoing_copy = outgoing_.at(vertex);
    const auto incoming_copy = incoming_.at(vertex);

    for (const auto& dst : outgoing_copy) {
      erase_one(incoming_.at(dst), vertex);
      --edge_count_;
    }

    for (const auto& src : incoming_copy) {
      if (src == vertex) {
        continue;
      }
      erase_one(outgoing_.at(src), vertex);
      --edge_count_;
    }

    outgoing_.erase(vertex);
    incoming_.erase(vertex);

    const auto removed_index = vertex_index_.at(vertex);
    vertex_index_.erase(vertex);
    vertices_.erase(vertices_.begin() + static_cast<std::ptrdiff_t>(removed_index));
    reindex_vertices(removed_index);
    return true;
  }

  [[nodiscard]] bool has_vertex(const Vertex& vertex) const {
    return vertex_index_.find(vertex) != vertex_index_.end();
  }

  bool add_edge(const Vertex& src, const Vertex& dst) {
    add_vertex(src);
    add_vertex(dst);
    if (has_edge(src, dst)) {
      return false;
    }

    outgoing_.at(src).push_back(dst);
    incoming_.at(dst).push_back(src);
    ++edge_count_;
    return true;
  }

  bool remove_edge(const Vertex& src, const Vertex& dst) {
    if (!has_edge(src, dst)) {
      return false;
    }

    erase_one(outgoing_.at(src), dst);
    erase_one(incoming_.at(dst), src);
    --edge_count_;
    return true;
  }

  [[nodiscard]] bool has_edge(const Vertex& src, const Vertex& dst) const {
    const auto it = outgoing_.find(src);
    if (it == outgoing_.end()) {
      return false;
    }

    const auto& neighbors = it->second;
    return std::find(neighbors.begin(), neighbors.end(), dst) != neighbors.end();
  }

  [[nodiscard]] std::vector<Vertex> vertices() const { return vertices_; }

  [[nodiscard]] std::vector<Vertex> outgoing_neighbors(const Vertex& vertex) const {
    return adjacency_for(outgoing_, vertex);
  }

  [[nodiscard]] std::vector<Vertex> incoming_neighbors(const Vertex& vertex) const {
    return adjacency_for(incoming_, vertex);
  }

  [[nodiscard]] size_type out_degree(const Vertex& vertex) const {
    return outgoing_neighbors(vertex).size();
  }

  [[nodiscard]] size_type in_degree(const Vertex& vertex) const {
    return incoming_neighbors(vertex).size();
  }

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

      for (const auto& neighbor : outgoing_.at(current)) {
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

      const auto& neighbors = outgoing_.at(current);
      for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
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

  [[nodiscard]] DirectedGraph transpose() const {
    DirectedGraph result(vertices_);
    for (const auto& src : vertices_) {
      for (const auto& dst : outgoing_.at(src)) {
        result.add_edge(dst, src);
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<Vertex> topological_sort() const {
    std::unordered_map<Vertex, size_type> indegree;
    std::queue<Vertex> ready;
    std::vector<Vertex> order;

    for (const auto& vertex : vertices_) {
      indegree.emplace(vertex, incoming_.at(vertex).size());
    }

    for (const auto& vertex : vertices_) {
      if (indegree.at(vertex) == 0) {
        ready.push(vertex);
      }
    }

    while (!ready.empty()) {
      Vertex current = ready.front();
      ready.pop();
      order.push_back(current);

      for (const auto& neighbor : outgoing_.at(current)) {
        auto& degree = indegree.at(neighbor);
        --degree;
        if (degree == 0) {
          ready.push(neighbor);
        }
      }
    }

    if (order.size() != vertices_.size()) {
      throw std::invalid_argument("DirectedGraph::topological_sort requires an acyclic graph");
    }

    return order;
  }

 private:
  std::vector<Vertex> vertices_;
  std::unordered_map<Vertex, size_type> vertex_index_;
  std::unordered_map<Vertex, std::vector<Vertex>> outgoing_;
  std::unordered_map<Vertex, std::vector<Vertex>> incoming_;
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
      throw std::out_of_range("DirectedGraph vertex not found");
    }
  }

  [[nodiscard]] static std::vector<Vertex> adjacency_for(
      const std::unordered_map<Vertex, std::vector<Vertex>>& adjacency,
      const Vertex& vertex) {
    const auto it = adjacency.find(vertex);
    if (it == adjacency.end()) {
      throw std::out_of_range("DirectedGraph vertex not found");
    }
    return it->second;
  }
};

} // namespace datamunge::dstruct
