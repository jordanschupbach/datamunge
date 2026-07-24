#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::fem {

/// @brief A 1D finite-element mesh: a strictly increasing sequence of node coordinates.
///        Element i spans [nodes[i], nodes[i+1]] for i in [0, num_elements()) -- no separate
///        connectivity array is needed since consecutive nodes are always the two endpoints
///        of one linear (P1) element. `nodes` is a public field for direct C++ construction,
///        but (matching datamunge::ode::ODESolution's t/y fields -- a swig-jse R-backend
///        codegen bug hits plain vector<double> struct *field* getters) is not exposed to
///        language bindings; node_at()/num_nodes() are the cross-language-safe accessors.
struct Mesh1D {
    std::vector<double> nodes;

    [[nodiscard]] std::size_t num_nodes() const { return nodes.size(); }
    [[nodiscard]] std::size_t num_elements() const { return nodes.empty() ? 0 : nodes.size() - 1; }
    [[nodiscard]] double node_at(std::size_t index) const { return nodes.at(index); }
    [[nodiscard]] double element_length(std::size_t element_index) const {
        return nodes.at(element_index + 1) - nodes.at(element_index);
    }
};

/// @brief Builds a uniform mesh of @p num_elements equal-length elements on [a, b].
///        Throws std::invalid_argument if num_elements is 0 or b <= a.
[[nodiscard]] Mesh1D make_uniform_mesh1d(double a, double b, std::size_t num_elements);

} // namespace datamunge::fem
