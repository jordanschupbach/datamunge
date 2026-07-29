#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

struct HITSResult {
  std::vector<double> hubs;
  std::vector<double> authorities;
  std::size_t         iterations{0};
};

std::vector<double> page_rank(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& directed_edges,
    double damping = 0.85, double tolerance = 1e-12, std::size_t max_iterations = 1000);

std::vector<double> trust_rank(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& directed_edges,
    const std::vector<std::size_t>& trusted_seeds,
    double damping = 0.85, double tolerance = 1e-12, std::size_t max_iterations = 1000);

HITSResult hits(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& directed_edges,
    double tolerance = 1e-12, std::size_t max_iterations = 1000);

std::vector<std::vector<std::size_t>> girvan_newman(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& undirected_edges,
    std::size_t target_communities = 2);

} // namespace datamunge::algorithms
