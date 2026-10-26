#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <datamunge/dstruct/directed_graph.hpp>

namespace datamunge::dstruct {

template <typename Vertex>
class DirectedAcyclicGraph {
 public:
  bool add_vertex(const Vertex& vertex) { return graph_.add_vertex(vertex); }
  bool add_edge(const Vertex& source, const Vertex& destination) {
    if (source == destination || (graph_.has_vertex(destination) && graph_.has_vertex(source) && graph_.contains_path(destination, source))) {
      throw std::invalid_argument("edge would create a cycle");
    }
    return graph_.add_edge(source, destination);
  }
  bool contains_path(const Vertex& source, const Vertex& destination) const { return graph_.contains_path(source, destination); }
  std::vector<Vertex> topological_order() const { return graph_.topological_sort(); }
  std::size_t vertex_count() const { return graph_.vertex_count(); }
  std::size_t edge_count() const { return graph_.edge_count(); }
 private: DirectedGraph<Vertex> graph_;
};

}  // namespace datamunge::dstruct
