#pragma once
/// @file fem1d.hpp
/// @brief Owned P1 finite-element solutions for steady and backward-Euler one-dimensional problems.
#include <datamunge/fem/mesh1d.hpp>
#include <datamunge/fem/source_function.hpp>
#include <cstddef>
#include <string>
#include <vector>

/// @brief Native mesh-based P1 finite-element models and owned scalar-field solutions.
namespace datamunge::fem {
/// @brief Select an endpoint value or an outward-conormal natural boundary condition.
/// The conormal p*du/dn uses -u' at the left and +u' at the right. Physical
/// diffusive outward flux is its negative; these are different sign conventions.
/// @par Example
/// @code
/// std::print("{}\n",dm::fem::BCType::Neumann); // Neumann
/// @endcode
/// @see BoundaryCondition1D
enum class BCType {
    Dirichlet, ///< Prescribe u=value at the endpoint.
    Neumann, ///< Prescribe outward conormal p*du/dn=value.
    Robin ///< Prescribe p*du/dn+robin_coefficient*(u-value)=0.
};

/// @brief Owned endpoint data for Dirichlet, outward-conormal Neumann or Robin conditions.
/// Dirichlet uses value as u; Neumann uses it as p*du/dn; Robin uses it as the
/// reference value in p*du/dn+alpha*(u-value)=0. Alpha zero gives zero conormal.
/// For coercive diffusion/heat models use nonnegative Robin alpha and sufficient
/// constraints/reaction to remove the constant nullspace. Unsupported enum values
/// and nonfinite data are outside the solver's preconditions, not fully validated.
/// @par Example
/// @code
/// const dm::fem::BoundaryCondition1D right{dm::fem::BCType::Robin,4,3};
/// std::print("{} {}\n",right.value,right.robin_coefficient); // 4 3
/// @endcode
/// @see FEM1D, BCType
struct BoundaryCondition1D {
    BCType type{BCType::Dirichlet}; ///< Supported condition kind; defaults to Dirichlet.
    double value{0.0}; ///< Prescribed value, conormal or Robin reference, as selected by type.
    double robin_coefficient{0.0}; ///< Robin alpha; ignored for other kinds.
};

/// @brief Owned nodal P1 steady solution, with checked scalar accessors for language bindings.
/// The continuous finite-element field is the piecewise-linear interpolant of
/// these values; nodal agreement alone does not imply agreement between nodes.
/// Copies own their arrays, with no mesh/callback alias. Public edits can break
/// array consistency; accessors check the actual requested vector bounds.
/// @par Example
/// @code
/// const auto r=dm::fem::FEM1D::solve_builtin(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},"constant",{1},{},{});
/// std::print("{} {}\n",r.size(),r.value_at(1)); // 3 0.125
/// @endcode
/// @see FEM1D::solve(), algorithms::linear_interpolate()
struct FEM1DResult {
    std::vector<double> nodes; ///< Owned spatial coordinates in original mesh order.
    std::vector<double> u; ///< Owned nodal values, aligned with nodes when returned by a solver.

    /// @brief Count stored solution values without changing either array.
    /// @return Actual u.size(); no allocation or shared mutation.
    [[nodiscard]] std::size_t size() const { return u.size(); }
    /// @brief Read an owned spatial coordinate with vector bounds checking.
    /// @param index Zero-based coordinate index.
    /// @return Stored coordinate by value.
    /// @throws std::out_of_range If index is outside nodes.
    [[nodiscard]] double node_at(std::size_t index) const { return nodes.at(index); }
    /// @brief Read an owned nodal value with vector bounds checking.
    /// @param index Zero-based value index.
    /// @return Stored solution value by value.
    /// @throws std::out_of_range If index is outside u.
    [[nodiscard]] double value_at(std::size_t index) const { return u.at(index); }
};

/// @brief Owned initial state and actual strictly advancing backward-Euler time slices.
/// Solver output has one u row per time and one value per spatial node. Initial
/// time zero/u0 are retained as supplied; later rows satisfy imposed boundary data.
/// Times retain rounded nominal-step products with a shortened final interval;
/// an already reached endpoint is not appended twice. Copies retain no mesh/source
/// alias. Public edits may break consistency; accessors check actual vector bounds.
/// @par Example
/// @code
/// const auto t=dm::fem::FEM1D::solve_transient_builtin(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},"constant",{1},{},{},{0,0,0},.5,.5);
/// std::print("{} {:.6f}\n",t.num_steps(),t.value_at(1,1)); // 2 0.107143
/// @endcode
/// @see FEM1D::solve_transient(), FEM1D::solve_transient_builtin()
struct FEM1DTimeSeries {
    std::vector<double> nodes; ///< Owned spatial coordinates.
    std::vector<double> times; ///< Owned actual time levels, including initial zero.
    std::vector<std::vector<double>> u; ///< Owned nodal rows; u[k] corresponds to times[k].

    /// @brief Count stored time levels, including the initial state.
    /// @return Actual times.size(), rather than the number of time advances.
    [[nodiscard]] std::size_t num_steps() const { return times.size(); }
    /// @brief Count stored spatial coordinates.
    /// @return Actual nodes.size(); no allocation or shared mutation.
    [[nodiscard]] std::size_t num_nodes() const { return nodes.size(); }
    /// @brief Read a spatial coordinate with vector bounds checking.
    /// @param index Zero-based coordinate index.
    /// @return Owned coordinate by value.
    /// @throws std::out_of_range If index is outside nodes.
    [[nodiscard]] double node_at(std::size_t index) const { return nodes.at(index); }
    /// @brief Read an actual stored time level with bounds checking.
    /// @param step Zero-based frame index, including initial frame zero.
    /// @return Actual stored time by value.
    /// @throws std::out_of_range If step is outside times.
    [[nodiscard]] double time_at(std::size_t step) const { return times.at(step); }
    /// @brief Read a stored nodal time-slice value with both vector bounds checked.
    /// @param step Zero-based time-frame index in u.
    /// @param node_index Zero-based spatial index within that row.
    /// @return Stored nodal value by value.
    /// @throws std::out_of_range If either index is outside the actual stored vectors.
    [[nodiscard]] double value_at(std::size_t step,std::size_t node_index) const { return u.at(step).at(node_index); }
};

/// @brief Solve steady or transient one-dimensional divergence-form problems with P1 Galerkin FEM.
/// Models -(p*u')'+q*u=f and u_t-(p*u')'+q*u=f on a caller-supplied mesh.
/// Requires at least two finite strictly increasing nodes, finite coefficient/
/// boundary/source data and finite assembled arithmetic with nonzero Thomas pivots;
/// those mathematical/geometry preconditions are not fully validated. Positive p,
/// nonnegative q/Robin data and adequate constraints give the usual coercive model.
/// Piecewise-constant p/q can be scalar or one per element. Results own their
/// values; input mesh/data are unchanged. Calls share no mutable global state,
/// while callback side effects/concurrency belong to the caller.
/// @par Example
/// @code
/// const auto r=dm::fem::FEM1D::solve_builtin(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},"constant",{1},{},{});
/// std::print("{}\n",r.value_at(1)); // 0.125
/// @endcode
/// @par References
/// - MFEM, Weak Formulations: https://mfem.org/fem_weak_form/ . This native implementation
///   uses P1 intervals, three-point Gauss-Legendre loads and a tridiagonal direct solve.
/// @see Mesh1D, FEM1DResult, FEM1DTimeSeries
class FEM1D {
public:
    /// @brief Solve the steady P1 system for -(p*u')'+q*u=f with endpoint boundary data.
    /// Follows FEM1D's mesh/coefficient/finite-arithmetic/nonsingular-pivot preconditions.
    /// Uses three-point Gauss-Legendre quadrature of f times each local basis, natural
    /// boundary terms and symmetric Dirichlet elimination, then Thomas elimination.
    /// O(N) work/storage for N nodes; invokes the source three times per element.
    /// No callback pointer is retained. Independent calls share no mutable state.
    /// @param mesh Borrowed mesh with at least two finite strictly increasing nodes.
    /// @param p Borrowed diffusion coefficient: length1 or one per element.
    /// @param q Borrowed reaction coefficient: length1 or one per element.
    /// @param f Borrowed spatial source; evaluate() may have caller-owned side effects.
    /// @param left Left endpoint condition, with outward normal -1.
    /// @param right Right endpoint condition, with outward normal +1.
    /// @return Owned original nodes and computed nodal values.
    /// @throws std::invalid_argument If p/q are empty or have the wrong lengths.
    /// @throws std::bad_alloc If assembly/result storage cannot be allocated; callback exceptions propagate.
    /// @par Example
    /// @code
    /// struct Unit final:dm::fem::ScalarField1D{double evaluate(double) override{return 1;}} source;
    /// const auto r=dm::fem::FEM1D::solve(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},source,{},{});
    /// std::print("{}\n",r.value_at(1)); // 0.125
    /// @endcode
    /// @par References
    /// - https://mfem.org/fem_weak_form/ ; native interval P1 and quadrature choices as FEM1D.
    /// @see solve_builtin(), solve_transient()
    [[nodiscard]] static FEM1DResult solve(const Mesh1D& mesh,const std::vector<double>& p,
        const std::vector<double>& q,ScalarField1D& f,const BoundaryCondition1D& left,const BoundaryCondition1D& right);

    /// @brief Solve the steady system using an owned named spatial source, without a director callback.
    /// Shares the live-source solver and its mathematical/precondition/storage contracts.
    /// Sources: zero=0 (params unused); constant=c ({1}); linear=m*x+c ({1,0});
    /// polynomial=sum(params[i]*x^i) (empty gives0); sine=A*sin(w*x+phase) ({1,1,0}).
    /// Missing trailing parameters use those defaults; extra parameters are ignored
    /// except polynomial, which uses all entries. Name matching is exact/case-sensitive.
    /// @param mesh Borrowed valid finite interval mesh; unchanged.
    /// @param p Borrowed scalar/per-element diffusion coefficient.
    /// @param q Borrowed scalar/per-element reaction coefficient.
    /// @param source Exact name zero, constant, linear, polynomial or sine.
    /// @param source_params Borrowed positional source parameters, with defaults described above.
    /// @param left Left endpoint data using outward normal -1.
    /// @param right Right endpoint data using outward normal +1.
    /// @return Owned computed nodal result, retaining no source or input reference.
    /// @throws std::invalid_argument For an unknown source or malformed p/q lengths.
    /// @throws std::bad_alloc If owned parameter/assembly/result storage cannot be allocated.
    /// @par Example
    /// @code
    /// const auto r=dm::fem::FEM1D::solve_builtin(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},"constant",{}, {},{});
    /// std::print("{}\n",r.value_at(1)); // 0.125
    /// @endcode
    /// @par References
    /// - https://mfem.org/fem_weak_form/ ; source definitions are native convenience choices.
    /// @see solve(), solve_transient_builtin()
    [[nodiscard]] static FEM1DResult solve_builtin(const Mesh1D& mesh,const std::vector<double>& p,
        const std::vector<double>& q,const std::string& source,const std::vector<double>& source_params,
        const BoundaryCondition1D& left,const BoundaryCondition1D& right);

    /// @brief Integrate u_t-(p*u')'+q*u=f with P1 space and actual backward-Euler time intervals.
    /// Shares FEM1D's spatial/finite-arithmetic/nonzero-pivot preconditions. Source
    /// and boundary data represent a time-independent model; the live callback is
    /// actually reevaluated three times per element at every time advance, not cached.
    /// Stores initial u0 as supplied, then imposes boundary data at later solves.
    /// Nominal step dt is shortened at t_end; actual positive time differences enter
    /// the mass term. Stops once t_end is reached, avoiding a duplicate rounded last
    /// time even when ceil(t_end/dt) overestimates the number of advances.
    /// O(S*N) work/output storage for S advances/N nodes, O(N) assembly temporaries;
    /// no optional trace or quota. Backward Euler is unconditionally stable for the
    /// coercive semidiscrete model, not a guarantee of arbitrary-step accuracy or
    /// finite arithmetic for arbitrary parameters. Input arrays remain unchanged;
    /// callback exceptions/side effects/concurrent safety belong to caller.
    /// @param mesh Borrowed valid mesh with at least two finite increasing nodes.
    /// @param p Borrowed scalar/per-element diffusion coefficient.
    /// @param q Borrowed scalar/per-element reaction coefficient.
    /// @param f Borrowed spatial callback, reevaluated at every advance.
    /// @param left Fixed left condition, with outward normal -1.
    /// @param right Fixed right condition, with outward normal +1.
    /// @param u0 Borrowed finite initial nodal values, exactly one per mesh node.
    /// @param t_end Positive finite final time, in time units.
    /// @param dt Positive finite nominal time step in the same units; may exceed t_end.
    /// @return Owned nodes, initial state and strictly advancing slices ending at t_end.
    /// @throws std::invalid_argument For fewer than two nodes, wrong coefficient/u0 lengths or nonpositive/nonfinite time controls.
    /// @throws std::length_error If frame counts, vector capacities or total scalar count cannot be represented.
    /// @throws std::overflow_error If rounded time products cannot advance before t_end.
    /// @throws std::bad_alloc If output/assembly storage cannot be allocated; callback exceptions propagate.
    /// @par Example
    /// @code
    /// struct Zero final:dm::fem::ScalarField1D{double evaluate(double) override{return 0;}} source;
    /// const auto t=dm::fem::FEM1D::solve_transient(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},source,{},{},{0,1,0},.5,.5);
    /// std::print("{} {:.6f}\n",t.num_steps(),t.value_at(1,1)); // 2 0.142857
    /// @endcode
    /// @par References
    /// - Arnold (2012), Numerical Analysis of Partial Differential Equations, chapter5 section2.
    ///   https://www-users.cse.umn.edu/~arnold/8445.f12/notes.pdf . Native load quadrature,
    ///   fixed data and rounded/shortened time grid are implementation choices.
    /// @see solve_transient_builtin(), FEM1DTimeSeries
    [[nodiscard]] static FEM1DTimeSeries solve_transient(const Mesh1D& mesh,const std::vector<double>& p,
        const std::vector<double>& q,ScalarField1D& f,const BoundaryCondition1D& left,
        const BoundaryCondition1D& right,const std::vector<double>& u0,double t_end,double dt);

    /// @brief Integrate the same backward-Euler model with a named source and no callback subclass.
    /// Uses the same private source construction as solve_builtin() and the same
    /// actual time loop as solve_transient(); no second solver or per-language glue.
    /// Names/defaults: zero=0; constant=c ({1}); linear=m*x+c ({1,0});
    /// polynomial=sum(params[i]*x^i) (empty0); sine=A*sin(w*x+phase) ({1,1,0}).
    /// Extra entries ignored except polynomial; exact case-sensitive names. Source
    /// objects are local and synchronous; results own values, not callback aliases.
    /// All spatial/time/initial-state, complexity and stability contracts follow
    /// solve_transient(); ordinary output has no optional history/quota.
    /// @param mesh Borrowed valid finite increasing mesh, with at least two nodes.
    /// @param p Borrowed scalar/per-element diffusion coefficient.
    /// @param q Borrowed scalar/per-element reaction coefficient.
    /// @param source Exact name zero, constant, linear, polynomial or sine.
    /// @param source_params Borrowed positional parameters; omitted trailing values use the defaults above.
    /// @param left Fixed left boundary condition using outward normal -1.
    /// @param right Fixed right boundary condition using outward normal +1.
    /// @param u0 Borrowed finite initial values, exactly one per spatial node.
    /// @param t_end Positive finite final time.
    /// @param dt Positive finite nominal time step; final interval is shortened as needed.
    /// @return Owned initial state and actual strictly advancing backward-Euler slices.
    /// @throws std::invalid_argument For an unknown source, malformed coefficient/u0 lengths, fewer than two nodes or nonpositive/nonfinite time controls.
    /// @throws std::length_error If frame/container/total scalar counts cannot be represented.
    /// @throws std::overflow_error If rounded time products cannot advance.
    /// @throws std::bad_alloc If parameters, output or assembly storage cannot be allocated.
    /// @par Example
    /// @code
    /// const auto t=dm::fem::FEM1D::solve_transient_builtin(dm::fem::make_uniform_mesh1d(0,1,2),{1},{0},"constant",{1},{},{},{0,0,0},.5,.5);
    /// std::print("{} {:.6f}\n",t.num_steps(),t.value_at(1,1)); // 2 0.107143
    /// @endcode
    /// @par References
    /// - https://www-users.cse.umn.edu/~arnold/8445.f12/notes.pdf ; same native discretization as solve_transient().
    /// @see solve_builtin(), solve_transient()
    [[nodiscard]] static FEM1DTimeSeries solve_transient_builtin(const Mesh1D& mesh,const std::vector<double>& p,
        const std::vector<double>& q,const std::string& source,const std::vector<double>& source_params,
        const BoundaryCondition1D& left,const BoundaryCondition1D& right,const std::vector<double>& u0,double t_end,double dt);
};
} // namespace datamunge::fem
