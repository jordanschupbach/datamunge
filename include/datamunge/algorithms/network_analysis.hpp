#include <datamunge/algorithms/trust_rank.hpp>
#pragma once

/// @file network_analysis.hpp
/// @brief Link analysis on directed graphs (PageRank, TrustRank, HITS) and Girvan–Newman communities.

#include <cstddef>
#include <datamunge/algorithms/girvan_newman.hpp>
#include <datamunge/algorithms/hits.hpp>
#include <datamunge/algorithms/page_rank.hpp>
#include <datamunge/linalg/csr.hpp>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief The PageRank linear system \f$(I-\alpha P^\top)x=\frac{1-\alpha}{n}\mathbf 1\f$ in sparse form.
struct PageRankSystem {
  linalg::SparseCSR<double>
      matrix; ///< \f$A=I-\alpha P^\top\f$, with \f$P_{uv}=1/\deg^+(u)\f$ for each edge u -> v.
  linalg::SparseCSR<double> transpose; ///< \f$A^\top=I-\alpha P\f$.
  std::vector<double>       rhs;       ///< \f$b=\frac{1-\alpha}{n}\mathbf 1\f$.
};

/// @brief Builds the PageRank system whose solution is the page_rank() vector.
///
/// \f$A\f$ is non-symmetric, and \f$\mathbf 1^\top A=(1-\alpha)\mathbf 1^\top\f$ because P is
/// row-stochastic, so the solution sums to one. Repeated edges count with multiplicity.
///
/// @param vertex_count Number of vertices n.
/// @param directed_edges Edges u -> v.
/// @param damping \f$\alpha\in[0,1)\f$.
/// @return The matrix, its transpose and the right-hand side.
/// @throws std::invalid_argument for an out-of-range vertex, a damping outside [0, 1), or a
///         vertex with no out-edges (dangling vertices would need a dense correction).
///
/// @par Example
/// @code
/// const auto sys = dm::algorithms::page_rank_system(n, edges, 0.85);
/// const auto x = dm::linalg::gmres(sys.matrix, sys.rhs).x;   // equals page_rank(n, edges, 0.85)
/// @endcode
/// @see page_rank()
PageRankSystem page_rank_system(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& directed_edges,
    double                                                  damping = 0.85);

/// @brief TrustRank: PageRank whose random jumps (and dangling mass) go only to trusted seeds.
///
/// @param vertex_count Number of vertices n.
/// @param directed_edges Edges u -> v.
/// @param trusted_seeds Seed vertices; jumps land on each with equal probability.
/// @param damping \f$\alpha\in[0,1)\f$.
/// @param tolerance Stop when the L1 change between iterates is below this.
/// @param max_iterations Iteration cap.
/// @return Trust scores that sum to one.
/// @throws std::invalid_argument for no seeds, an out-of-range vertex or seed, or bad parameters.
std::vector<double> trust_rank(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& directed_edges,
    const std::vector<std::size_t>&                         trusted_seeds,
    double                                                  damping        = 0.85,
    double                                                  tolerance      = 1e-12,
    std::size_t                                             max_iterations = 1000);

} // namespace datamunge::algorithms
