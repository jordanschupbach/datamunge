#pragma once

/// @file gauss_hermite.hpp
/// @brief Gauss-Hermite rules for expectations under a normal distribution.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Nodes and weights of a Gauss-Hermite rule for standard normal expectations.
/// @par Example
/// @code
/// dm::stats::GaussHermiteRule r;
/// r.nodes = {0.0};
/// r.weights = {1.0};
/// std::println("{}", r.nodes.size()); // 1
/// @endcode
struct GaussHermiteRule {
    std::vector<double> nodes;   ///< Points z_j, symmetric about 0 and increasing.
    std::vector<double> weights; ///< Positive weights summing to 1.
};

/// @brief Rule with \f$E[f(Z)]\approx\sum_j w_jf(z_j)\f$ for \f$Z\sim N(0,1)\f$.
///
/// Built by the Golub-Welsch method from the Jacobi matrix of the probabilists' Hermite
/// polynomials (off-diagonal entries \f$\sqrt{j}\f$): the nodes are its eigenvalues and the weights
/// the squared first components of its normalised eigenvectors. The rule is exact for polynomials of
/// degree below 2 * points. For \f$X\sim N(\mu,\sigma^2)\f$ use the nodes \f$\mu+\sigma z_j\f$.
/// @param points Number of nodes (at least 1).
/// @return The rule.
/// @throws std::invalid_argument When points is 0.
/// @par Example
/// @code
/// const auto r = dm::stats::normal_gauss_hermite(5);
/// double m4 = 0;
/// for (std::size_t j = 0; j < 5; ++j)
///     m4 += r.weights[j] * std::pow(r.nodes[j], 4);
/// std::println("{:.6f}", m4); // 3.000000
/// @endcode
GaussHermiteRule normal_gauss_hermite(std::size_t points);

} // namespace datamunge::stats
