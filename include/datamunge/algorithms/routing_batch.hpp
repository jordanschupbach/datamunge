#pragma once
#include <datamunge/algorithms/mst_result.hpp>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
using WeightedEdge = std::tuple<std::size_t,std::size_t,double>;
struct LongestPathResult { std::vector<double> distance; std::vector<std::size_t> predecessor; };
LongestPathResult dag_longest_paths(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t source);
MinimumSpanningTree boruvka(std::size_t n,const std::vector<WeightedEdge>& edges);
MinimumSpanningTree reverse_delete(std::size_t n,const std::vector<WeightedEdge>& edges);
std::vector<std::size_t> nonblocking_switch_routes(
 std::size_t input_groups,std::size_t output_groups,
 const std::vector<std::pair<std::size_t,std::size_t>>& connections);
struct JohnsonAllPairsResult { std::vector<std::vector<double>> distance; bool has_negative_cycle{false}; };
JohnsonAllPairsResult johnson_all_pairs_shortest_paths(std::size_t n,const std::vector<WeightedEdge>& edges);
std::vector<std::vector<bool>> transitive_closure(
 std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,bool reflexive=true);
}
