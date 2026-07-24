#pragma once

#include <datamunge/fem/mesh2d.hpp>
#include <datamunge/fem/source_function.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::fem {

/// @brief The nodal solution of a 2D FEM Poisson solve.
struct FEM2DResult {
    std::vector<datamunge::geometry::Point2D> nodes;
    std::vector<double> u;

    [[nodiscard]] std::size_t size() const { return u.size(); }
    [[nodiscard]] datamunge::geometry::Point2D node_at(std::size_t index) const { return nodes.at(index); }
    [[nodiscard]] double value_at(std::size_t index) const { return u.at(index); }
};

/// @brief Galerkin finite-element solver for the 2D Poisson equation
///        -k * (u_xx + u_yy) = f(x, y) on a triangular mesh, using piecewise-linear (P1)
///        Lagrange elements and Dirichlet boundary conditions only (this "basic" 2D solver
///        does not offer FEM1D's Neumann/Robin options). Internally assembles a sparse
///        stiffness system and solves it with the conjugate-gradient solver
///        (datamunge::linalg::cg) -- the assembled matrices themselves are never exposed,
///        matching this codebase's rule that linalg's sparse/dense matrix types are never
///        SWIG-bound.
class FEM2D {
public:
    /// @brief Solves -k * Laplacian(u) = f with Dirichlet data u = dirichlet_values[i] at node
    ///        dirichlet_nodes[i] (every other node is a free unknown); k is the constant,
    ///        isotropic diffusion coefficient. Throws std::invalid_argument if
    ///        dirichlet_nodes.size() != dirichlet_values.size(), if any index in
    ///        dirichlet_nodes is out of range, or if every node is constrained (nothing to
    ///        solve for).
    [[nodiscard]] static FEM2DResult solve(const Mesh2D& mesh, double k, ScalarField2D& f,
                                            const std::vector<int>& dirichlet_nodes,
                                            const std::vector<double>& dirichlet_values);

    /// @brief Same as solve(), but f is one of a fixed set of named source terms instead of a
    ///        live user-supplied callback. source_params is interpreted positionally, with the
    ///        listed defaults used for any parameter past the end of source_params:
    ///          - "zero":     f(x, y) = 0.                                source_params unused.
    ///          - "constant": f(x, y) = c.                                {c=1}.
    ///          - "sine":     f(x, y) = amplitude*sin(fx*x)*sin(fy*y).    {amplitude=1, fx=1, fy=1}.
    ///        Throws std::invalid_argument for an unknown source name.
    [[nodiscard]] static FEM2DResult solve_builtin(const Mesh2D& mesh, double k, const std::string& source,
                                                    const std::vector<double>& source_params,
                                                    const std::vector<int>& dirichlet_nodes,
                                                    const std::vector<double>& dirichlet_values);
};

} // namespace datamunge::fem
