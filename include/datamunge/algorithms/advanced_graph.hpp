#pragma once

#include <datamunge/algorithms/edmonds_karp.hpp>
#include <datamunge/algorithms/euclidean_mst.hpp>
#include <datamunge/algorithms/dinic.hpp>
#include <datamunge/algorithms/graph_layout.hpp>
#include <datamunge/algorithms/mst_result.hpp>
#include <datamunge/algorithms/chu_liu_edmonds.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

MaxFlowResult push_relabel_max_flow(std::size_t vertex_count,
                                    const std::vector<CapacityEdge>& edges,
                                    std::size_t source, std::size_t sink);

struct MinCutResult {
  std::size_t              cut_size{0};
  std::vector<std::size_t> side_a;
  std::vector<std::size_t> side_b;
};

MinCutResult karger_min_cut(std::size_t vertex_count,
                            const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                            std::size_t trials = 0, std::uint64_t seed = 42);



} // namespace datamunge::algorithms
