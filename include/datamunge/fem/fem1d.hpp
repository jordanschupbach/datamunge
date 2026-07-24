#pragma once

#include <datamunge/fem/mesh1d.hpp>
#include <datamunge/fem/source_function.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::fem {

enum class BCType { Dirichlet, Neumann, Robin };

/// @brief A boundary condition at one end of a 1D domain, expressed in terms of the outward
///        normal derivative du/dn (i.e. +u'(x) at the right end, -u'(x) at the left end --
///        this sign convention is what makes the same struct/formula apply symmetrically at
///        both ends):
///          Dirichlet: u = value.
///          Neumann:   p * du/dn = value (value is the prescribed outward flux; positive
///                     means flux leaving the domain through this boundary).
///          Robin:     p * du/dn + robin_coefficient * (u - value) = 0 (a linear/convective
///                     mix of the two -- value is the ambient/reference value, robin_coefficient
///                     the transfer coefficient; robin_coefficient = 0 reduces to Neumann with
///                     flux 0).
struct BoundaryCondition1D {
    BCType type{BCType::Dirichlet};
    double value{0.0};
    double robin_coefficient{0.0};
};

/// @brief The nodal solution of a steady-state 1D FEM solve, plus lightweight
///        cross-language-safe accessors (see datamunge::ode::ODESolution for the same
///        "public fields for C++, accessor methods for bindings" split rationale).
struct FEM1DResult {
    std::vector<double> nodes;
    std::vector<double> u;

    [[nodiscard]] std::size_t size() const { return u.size(); }
    [[nodiscard]] double node_at(std::size_t index) const { return nodes.at(index); }
    [[nodiscard]] double value_at(std::size_t index) const { return u.at(index); }
};

/// @brief One time slice of a transient (time-dependent) 1D FEM solve.
struct FEM1DTimeSeries {
    std::vector<double> nodes;
    std::vector<double> times;
    std::vector<std::vector<double>> u; // u[k] is the nodal solution at times[k]

    [[nodiscard]] std::size_t num_steps() const { return times.size(); }
    [[nodiscard]] std::size_t num_nodes() const { return nodes.size(); }
    [[nodiscard]] double node_at(std::size_t index) const { return nodes.at(index); }
    [[nodiscard]] double time_at(std::size_t step) const { return times.at(step); }
    [[nodiscard]] double value_at(std::size_t step, std::size_t node_index) const { return u.at(step).at(node_index); }
};

/// @brief Galerkin finite-element solver for 1D linear two-point boundary value problems
///        -(p u')' + q u = f(x) on [a, b] (steady state), or its parabolic (time-dependent)
///        counterpart u_t - (p u')' + q u = f(x) (transient), using piecewise-linear (P1)
///        Lagrange basis functions on the supplied mesh. p and q are taken as piecewise
///        constant per element (the common case: known per-element material/diffusion/
///        reaction coefficients) while the source term f may vary continuously in space.
class FEM1D {
public:
    /// @brief Solves the steady-state BVP -(p u')' + q u = f with the given boundary
    ///        conditions. p and q must each be either a single value (constant across the
    ///        whole domain) or exactly one value per element (mesh.num_elements()).
    ///        Throws std::invalid_argument for a mismatched size.
    [[nodiscard]] static FEM1DResult solve(const Mesh1D& mesh, const std::vector<double>& p,
                                            const std::vector<double>& q, ScalarField1D& f,
                                            const BoundaryCondition1D& left, const BoundaryCondition1D& right);

    /// @brief Same as solve(), but f is one of a fixed set of named source terms instead of a
    ///        live user-supplied callback -- the only option for bindings without director
    ///        support. source_params is interpreted positionally per source name, with the
    ///        listed defaults used for any parameter past the end of source_params:
    ///          - "zero":       f(x) = 0.                            source_params unused.
    ///          - "constant":   f(x) = c.                            {c=1}.
    ///          - "linear":     f(x) = m*x + c.                       {m=1, c=0}.
    ///          - "polynomial": f(x) = sum_i source_params[i] * x^i. an empty list means f=0.
    ///          - "sine":       f(x) = amplitude*sin(freq*x+phase).  {amplitude=1, freq=1, phase=0}.
    ///        Throws std::invalid_argument for an unknown source name.
    [[nodiscard]] static FEM1DResult solve_builtin(const Mesh1D& mesh, const std::vector<double>& p,
                                                    const std::vector<double>& q, const std::string& source,
                                                    const std::vector<double>& source_params,
                                                    const BoundaryCondition1D& left,
                                                    const BoundaryCondition1D& right);

    /// @brief Solves the transient (parabolic) problem u_t - (p u')' + q u = f(x) [f and the
    ///        boundary conditions are held fixed in time -- this "basic" transient solver does
    ///        not support a time-varying source or BCs] from t=0 to t=t_end with initial
    ///        condition u0 (one value per mesh node), via unconditionally stable backward
    ///        Euler time-stepping with fixed step dt. Throws std::invalid_argument if
    ///        u0.size() != mesh.num_nodes(), or if dt <= 0 or t_end <= 0.
    [[nodiscard]] static FEM1DTimeSeries
    solve_transient(const Mesh1D& mesh, const std::vector<double>& p, const std::vector<double>& q,
                    ScalarField1D& f, const BoundaryCondition1D& left, const BoundaryCondition1D& right,
                    const std::vector<double>& u0, double t_end, double dt);
};

} // namespace datamunge::fem
