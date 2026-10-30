#pragma once

/// @file hierarchical_bspline.hpp
/// @brief Coefficient-carrying tensor B-splines, knot refinement, tensor P-spline fits,
///        and adaptive hierarchical (HB) and truncated hierarchical (THB) B-splines.
///
/// Everything here uses a runtime dimension d: points are `std::vector<double>` of length
/// d, multi-indices are `std::vector<int>` of length d, and coefficient nets are stored in
/// row-major order (axis 0 slowest, last axis fastest), matching fda::BSpline's design
/// matrix columns. Hierarchy cells are addressed by integer ids; the root is cell 0 and a
/// child slot is a bit mask whose bit k selects the upper half of axis k.

#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <datamunge/fda/bspline.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/sparse_coo.hpp>

namespace datamunge::fda {

/// @brief Linear-system method used by the spline least-squares fits.
///
/// Every fit forms symmetric normal equations (B^T B + P) c = B^T y. Cholesky is exact but
/// costs O(K^3) for K unknowns; conjugate gradient on the sparse system costs roughly
/// O(nnz) per iteration and suits large, well-conditioned (usually penalized) systems.
/// @par Example
/// @code
/// dm::fda::SplineSolveOptions options{dm::fda::SplineSolver::ConjugateGradient};
/// @endcode
/// @see SplineSolveOptions
enum class SplineSolver {
    Automatic,        ///< Cholesky up to SplineSolveOptions::direct_limit unknowns, else CG.
    Cholesky,         ///< Dense Cholesky factorization of the normal equations.
    ConjugateGradient ///< Jacobi-preconditioned CG on the sparse normal equations.
};

/// @brief Solver settings shared by fit_tensor_bspline(), fit_tensor_pspline() and the
///        hierarchical fitting methods.
///
/// @par Example
/// @code
/// dm::fda::SplineSolveOptions options;
/// options.method = dm::fda::SplineSolver::Cholesky;
/// @endcode
/// @see SplineSolver
struct SplineSolveOptions {
    SplineSolver method{SplineSolver::Automatic}; ///< Requested solver.
    std::size_t  direct_limit{2000}; ///< Automatic switches from Cholesky to CG above this many unknowns.
    double       tolerance{1e-12}; ///< CG relative residual tolerance.
    std::size_t  max_iterations{0}; ///< CG iteration cap; 0 means 10 times the unknown count.
};

/// @brief A refined B-spline basis together with the operator that maps coarse
///        coefficients onto it.
///
/// For any coefficient vector c on the coarse basis, `matrix * c` (equivalently apply())
/// gives coefficients on the refined basis that represent exactly the same spline.
/// @par Example
/// @code
/// auto coarse = dm::fda::BSplineBasis::open_uniform(2, 4);
/// auto refined = dm::fda::dyadic_refine(coarse);
/// std::print("{}\n", refined.basis.basis_size()); // 6
/// @endcode
/// @see refine_basis(), dyadic_refine()
struct BSplineRefinement {
    BSplineBasis              basis;  ///< Refined basis (same degree and domain).
    linalg::SparseCOO<double> matrix; ///< Refined-by-coarse knot-insertion operator, compressed.

    /// @brief Maps coarse coefficients to the refined basis.
    /// @param coefficients One coefficient per coarse basis function.
    /// @return One coefficient per refined basis function, representing the same spline.
    /// @throws std::invalid_argument If @p coefficients does not match the coarse basis size.
    /// @par Example
    /// @code
    /// auto r = dm::fda::dyadic_refine(dm::fda::BSplineBasis::open_uniform(1, 2));
    /// std::print("{}\n", r.apply({0.0, 2.0})); // [0, 1, 2]
    /// @endcode
    std::vector<double> apply(const std::vector<double>& coefficients) const;
};

/// @brief Inserts knots into a clamped B-spline basis (Oslo algorithm).
///
/// Builds the refined knot vector (old knots merged with @p inserted_knots) and the sparse
/// discrete-B-spline operator that re-expresses every coarse spline on it exactly. Each
/// row has at most degree + 1 nonzeros and the work is O(m p^2) for m refined functions.
///
/// @param basis Basis to refine; its knot vector must be clamped (first and last knots
///              repeated degree + 1 times, as produced by BSplineBasis::open_uniform()).
/// @param inserted_knots Knots to add, strictly inside the domain, in any order. Repeats
///                       are allowed while every knot multiplicity stays <= degree + 1.
/// @return The refined basis and its refinement operator.
/// @throws std::invalid_argument If the basis is not clamped, a knot is outside the open
///                               domain or non-finite, or a multiplicity would exceed degree + 1.
/// @par Example
/// @code
/// auto coarse = dm::fda::BSplineBasis::open_uniform(1, 2);   // knots 0 0 1 1
/// auto r = dm::fda::refine_basis(coarse, {0.25});
/// std::print("{}\n", r.apply({0.0, 4.0}));                   // [0, 1, 4]
/// @endcode
/// @par References
/// - E. Cohen, T. Lyche and R. Riesenfeld (1980). Discrete B-splines and subdivision
///   techniques in computer-aided geometric design and computer graphics. Computer Graphics
///   and Image Processing 14(2), 87-111. https://doi.org/10.1016/0146-664X(80)90040-4
/// - W. Boehm (1980). Inserting new knots into B-spline curves. Computer-Aided Design
///   12(4), 199-201. https://doi.org/10.1016/0010-4485(80)90154-2
/// @see dyadic_refine()
BSplineRefinement refine_basis(const BSplineBasis& basis, const std::vector<double>& inserted_knots);

/// @brief Halves every knot span of a clamped basis (inserts each positive span's midpoint).
///
/// This is the nested-space step between consecutive levels of HB/THB splines: the refined
/// space contains the coarse one, and the operator maps coefficients exactly.
/// @param basis Clamped basis to refine.
/// @return The refined basis and its refinement operator.
/// @throws std::invalid_argument If the basis is not clamped.
/// @par Example
/// @code
/// auto r = dm::fda::dyadic_refine(dm::fda::BSplineBasis::open_uniform(3, 5));
/// std::print("{} -> {}\n", 5, r.basis.basis_size()); // 5 -> 7
/// @endcode
/// @par References
/// - E. Cohen, T. Lyche and R. Riesenfeld (1980). Discrete B-splines and subdivision
///   techniques in computer-aided geometric design and computer graphics.
///   https://doi.org/10.1016/0146-664X(80)90040-4
/// @see refine_basis(), TensorBSpline::dyadic_refine()
BSplineRefinement dyadic_refine(const BSplineBasis& basis);

/// @brief A scalar tensor-product B-spline f(x) = sum_j c_j B_j(x) with owned coefficients.
///
/// With one basis this is an ordinary spline curve; with d bases a surface or volume. The
/// coefficient net is row-major over the per-axis basis sizes (last axis fastest), the same
/// column order as fda::BSpline::evaluate(). Evaluation clamps each coordinate into the
/// basis domain, so points outside the domain take the boundary value. Const methods are
/// safe to call concurrently while the object is not mutated.
/// @par Example
/// @code
/// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);     // linear on [0, 1]
/// dm::fda::TensorBSpline f({b, b}, {0, 1, 1, 2});         // f(x, y) = x + y
/// std::print("{:.2f}\n", f.evaluate({0.25, 0.5}));        // 0.75
/// @endcode
/// @par References
/// - C. de Boor (1972). On calculating with B-splines. Journal of Approximation Theory
///   6(1), 50-62. https://doi.org/10.1016/0021-9045(72)90080-9
/// @see fit_tensor_bspline(), fit_tensor_pspline(), BSpline
class TensorBSpline {
public:
    /// @brief One-dimensional spline curve on @p basis.
    /// @param basis Basis for the single axis.
    /// @param coefficients One coefficient per basis function.
    /// @throws std::invalid_argument If the coefficient count differs from basis.basis_size().
    /// @par Example
    /// @code
    /// dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(1, 2), {1.0, 3.0});
    /// std::print("{}\n", f.evaluate(0.5)); // 2
    /// @endcode
    TensorBSpline(BSplineBasis basis, std::vector<double> coefficients);

    /// @brief Tensor-product spline over one basis per axis.
    /// @param bases One basis per dimension (at least one).
    /// @param coefficients Row-major net whose size is the product of the basis sizes.
    /// @throws std::invalid_argument If @p bases is empty or the net size does not match.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(0, 2);  // piecewise constant
    /// dm::fda::TensorBSpline f({b, b}, {1, 2, 3, 4});
    /// std::print("{}\n", f.evaluate({0.75, 0.25}));        // 3
    /// @endcode
    TensorBSpline(std::vector<BSplineBasis> bases, std::vector<double> coefficients);

    /// @brief Number of parameter dimensions d.
    /// @return bases().size().
    std::size_t dimensions() const noexcept { return bases_.size(); }

    /// @brief The per-axis bases.
    /// @return Reference valid for the lifetime of this object.
    const std::vector<BSplineBasis>& bases() const noexcept { return bases_; }

    /// @brief Per-axis basis sizes, i.e. the shape of the coefficient net.
    /// @return d sizes in axis order.
    std::vector<int> shape() const;

    /// @brief Total number of coefficients (product of shape()).
    /// @return Coefficient count.
    std::size_t coefficient_count() const noexcept { return coefficients_.size(); }

    /// @brief The row-major coefficient net.
    /// @return Reference valid until the next mutation of this object.
    const std::vector<double>& coefficients() const noexcept { return coefficients_; }

    /// @brief Lower corner of the parameter domain (each basis's first clamped knot).
    /// @return d coordinates.
    std::vector<double> domain_min() const;

    /// @brief Upper corner of the parameter domain.
    /// @return d coordinates.
    std::vector<double> domain_max() const;

    /// @brief Reads one coefficient by per-axis basis index.
    /// @param index d indices, each in [0, shape()[k]).
    /// @return The coefficient.
    /// @throws std::invalid_argument If index.size() != dimensions().
    /// @throws std::out_of_range If a component is out of range.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TensorBSpline f({b, b}, {0, 1, 2, 3});
    /// std::print("{}\n", f.coefficient({1, 0})); // 2
    /// @endcode
    double coefficient(const std::vector<int>& index) const;

    /// @brief Overwrites one coefficient by per-axis basis index.
    /// @param index d indices, each in [0, shape()[k]).
    /// @param value New coefficient.
    /// @throws std::invalid_argument If index.size() != dimensions().
    /// @throws std::out_of_range If a component is out of range.
    /// @par Example
    /// @code
    /// dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(1, 2), {0, 0});
    /// f.set_coefficient({1}, 5.0);
    /// std::print("{}\n", f.evaluate(1.0)); // 5
    /// @endcode
    void set_coefficient(const std::vector<int>& index, double value);

    /// @brief Evaluates a one-dimensional spline at @p x (clamped into the domain).
    /// @param x Finite coordinate.
    /// @return f(x).
    /// @throws std::invalid_argument If dimensions() != 1 or @p x is not finite.
    /// @par Example
    /// @code
    /// dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(1, 2), {1.0, 3.0});
    /// std::print("{}\n", f.evaluate(2.0)); // 3  (clamped to x = 1)
    /// @endcode
    double evaluate(double x) const;

    /// @brief Evaluates at one d-dimensional point (each coordinate clamped into the domain).
    /// @param point d finite coordinates.
    /// @return f(point).
    /// @throws std::invalid_argument If point.size() != dimensions() or a coordinate is not finite.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TensorBSpline f({b, b}, {0, 1, 1, 2});
    /// std::print("{}\n", f.evaluate({1.0, 1.0})); // 2
    /// @endcode
    double evaluate(const std::vector<double>& point) const;

    /// @brief Evaluates at many points.
    /// @param points Points, each of length d.
    /// @return One value per point, in input order.
    /// @throws std::invalid_argument As for evaluate(const std::vector<double>&).
    /// @par Example
    /// @code
    /// dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(1, 2), {0.0, 1.0});
    /// std::print("{}\n", f.evaluate_points({{0.0}, {0.5}})); // [0, 0.5]
    /// @endcode
    std::vector<double> evaluate_points(const std::vector<std::vector<double>>& points) const;

    /// @brief Evaluates on the Cartesian product of per-axis sample values.
    /// @param axes d non-empty coordinate lists.
    /// @return Row-major grid of values (axis 0 slowest, last axis fastest).
    /// @throws std::invalid_argument If axes.size() != dimensions() or an axis is empty.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TensorBSpline f({b, b}, {0, 1, 1, 2});
    /// std::print("{}\n", f.evaluate_grid({{0, 1}, {0, 1}})); // [0, 1, 1, 2]
    /// @endcode
    std::vector<double> evaluate_grid(const std::vector<std::vector<double>>& axes) const;

    /// @brief Dyadically refines every axis and maps the coefficients, giving an identical
    ///        function on the finer, nested tensor basis.
    /// @return The refined spline.
    /// @throws std::invalid_argument If some axis basis is not clamped.
    /// @par Example
    /// @code
    /// dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(2, 4), {0, 1, 3, 2});
    /// auto g = f.dyadic_refine();
    /// std::print("{} {:.6f}\n", g.coefficient_count(), g.evaluate(0.3) - f.evaluate(0.3)); // 6 0.000000
    /// @endcode
    /// @see dyadic_refine(const BSplineBasis&)
    TensorBSpline dyadic_refine() const;

private:
    /// @brief THB levels edit their coefficient nets in place through flat indices.
    friend class TruncatedHierarchicalBSpline;

    std::vector<BSplineBasis> bases_;        ///< One basis per axis.
    std::vector<double>       coefficients_; ///< Row-major coefficient net.
    std::vector<std::size_t>  strides_;      ///< Row-major strides (last axis stride 1).

    /// @brief Validates @p index and converts it to a flat offset into coefficients_.
    /// @param index Per-axis basis indices.
    /// @return Flat row-major offset.
    /// @throws std::invalid_argument On a dimension mismatch.
    /// @throws std::out_of_range On an out-of-range component.
    std::size_t flat_index(const std::vector<int>& index) const;
};

/// @brief Sparse anisotropic tensor difference penalty P = sum_k lambda_k (D_k^T D_k)
///        acting on a row-major coefficient net.
///
/// D_k takes order-q differences of adjacent coefficients along axis k (q = 0 gives the
/// identity, i.e. ridge). Returned compressed; it is symmetric positive semi-definite.
/// @param shape Per-axis coefficient counts (each > @p order).
/// @param lambdas Nonnegative per-axis weights, same length as @p shape.
/// @param order Difference order q.
/// @return The K by K penalty, K = product of @p shape.
/// @throws std::invalid_argument On empty/mismatched inputs, a negative or non-finite
///                               lambda, or an axis with size <= @p order.
/// @par Example
/// @code
/// auto p = dm::fda::tensor_difference_penalty({3}, {1.0}, 1);
/// std::print("{}\n", p.nnz()); // 7
/// @endcode
/// @par References
/// - P. H. C. Eilers, I. D. Currie and M. Durban (2006). Fast and compact smoothing on large
///   multidimensional grids. Computational Statistics & Data Analysis 50(1), 61-76.
///   https://doi.org/10.1016/j.csda.2004.07.008
/// @see fit_tensor_pspline(), pspline_penalty_matrix()
linalg::SparseCOO<double> tensor_difference_penalty(const std::vector<int>& shape,
                                                    const std::vector<double>& lambdas,
                                                    int order);

/// @brief Fits a tensor-product B-spline to scattered data by ordinary least squares.
///
/// Builds the sparse design B (each row has at most prod(degree_k + 1) nonzeros), then
/// solves (B^T B) c = B^T y. Coordinates outside a basis domain are clamped to it.
/// @param bases One basis per dimension.
/// @param points Sample locations, each of length bases.size().
/// @param values One response per point.
/// @param options Solver settings (default: automatic).
/// @return The fitted spline.
/// @throws std::invalid_argument On empty or mismatched inputs, or non-finite data.
/// @throws std::runtime_error If the normal equations are singular (some basis function has
///                            too little data; use a penalized fit or fewer functions).
/// @par Example
/// @code
/// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
/// auto f = dm::fda::fit_tensor_bspline({b}, {{0.0}, {0.5}, {1.0}}, {1.0, 2.0, 3.0});
/// std::print("{:.3f}\n", f.evaluate(0.25)); // 1.500
/// @endcode
/// @par References
/// - C. de Boor (2001). A Practical Guide to Splines, revised edition. Springer.
/// @see fit_tensor_pspline()
TensorBSpline fit_tensor_bspline(std::vector<BSplineBasis> bases,
                                 const std::vector<std::vector<double>>& points,
                                 const std::vector<double>& values,
                                 const SplineSolveOptions& options = SplineSolveOptions());

/// @brief Fits a tensor-product P-spline: least squares plus an anisotropic difference
///        penalty, solving (B^T B + P) c = B^T y with P from tensor_difference_penalty().
///
/// All-zero lambdas reproduce fit_tensor_bspline(). Positive lambdas keep the system
/// well-posed even with fewer points than coefficients (for order 0, or once the data
/// determine the order-q polynomial null space).
/// @param bases One basis per dimension.
/// @param points Sample locations, each of length bases.size().
/// @param values One response per point.
/// @param lambdas Nonnegative per-axis smoothing weights, one per basis.
/// @param order Difference order (default 2, the conventional curvature penalty).
/// @param options Solver settings (default: automatic).
/// @return The fitted spline.
/// @throws std::invalid_argument On empty/mismatched inputs, invalid lambdas, or an axis
///                               with no more functions than @p order.
/// @throws std::runtime_error If the penalized normal equations are singular.
/// @par Example
/// @code
/// auto b = dm::fda::BSplineBasis::open_uniform(3, 8);
/// auto f = dm::fda::fit_tensor_pspline({b}, {{0.0}, {1.0}}, {0.0, 1.0}, {1e3});
/// std::print("{:.3f}\n", f.evaluate(0.5)); // 0.500
/// @endcode
/// @par References
/// - P. H. C. Eilers and B. D. Marx (1996). Flexible smoothing with B-splines and penalties.
///   https://doi.org/10.1214/ss/1038425655
/// - P. H. C. Eilers, I. D. Currie and M. Durban (2006). Fast and compact smoothing on large
///   multidimensional grids. https://doi.org/10.1016/j.csda.2004.07.008
/// @see fit_tensor_bspline(), fit_pspline()
TensorBSpline fit_tensor_pspline(std::vector<BSplineBasis> bases,
                                 const std::vector<std::vector<double>>& points,
                                 const std::vector<double>& values,
                                 const std::vector<double>& lambdas,
                                 int order = 2,
                                 const SplineSolveOptions& options = SplineSolveOptions());

#ifndef SWIG
/// @brief Implementation helpers shared by the hierarchical splines (not a public API).
namespace detail {

/// @brief Dyadic 2^d-ary tree of axis-aligned boxes shared by the hierarchical splines.
///
/// Cells live in a flat vector and are addressed by id (root 0). Children are created on
/// demand by halving every axis; slot bit k selects the upper half of axis k.
class OrthantCellTree {
public:
    /// @brief Creates the root cell [lo, hi].
    /// @param lo Lower corner.
    /// @param hi Upper corner, with lo[k] < hi[k].
    /// @throws std::invalid_argument On empty/mismatched/non-finite bounds, lo >= hi, or
    ///                               more than 20 dimensions.
    OrthantCellTree(std::vector<double> lo, std::vector<double> hi);

    /// @brief Number of dimensions d. @return d.
    std::size_t dimensions() const noexcept { return root_lo_.size(); }
    /// @brief Children per cell, 2^d. @return 2^d.
    int arity() const noexcept { return 1 << static_cast<int>(dimensions()); }
    /// @brief Number of cells created so far. @return Cell count.
    int size() const noexcept { return static_cast<int>(depth_.size()); }
    /// @brief Greatest depth of any cell. @return Maximum depth (root is 0).
    int max_depth() const noexcept { return max_depth_; }

    /// @brief Throws std::out_of_range unless @p cell is a valid id. @param cell Cell id.
    void check(int cell) const;
    /// @brief Throws std::out_of_range unless 0 <= @p slot < arity(). @param slot Child slot.
    void check_slot(int slot) const;

    /// @brief Cell lower corner. @param cell Cell id. @return d coordinates.
    const std::vector<double>& lo(int cell) const { return lo_[static_cast<std::size_t>(cell)]; }
    /// @brief Cell upper corner. @param cell Cell id. @return d coordinates.
    const std::vector<double>& hi(int cell) const { return hi_[static_cast<std::size_t>(cell)]; }
    /// @brief Cell depth. @param cell Cell id. @return Depth (root 0).
    int depth(int cell) const { return depth_[static_cast<std::size_t>(cell)]; }
    /// @brief Parent id. @param cell Cell id. @return Parent id, or -1 for the root.
    int parent(int cell) const { return parent_[static_cast<std::size_t>(cell)]; }
    /// @brief Child id. @param cell Cell id. @param slot Child slot. @return Id or -1 if absent.
    int child(int cell, int slot) const;
    /// @brief True when no child of @p cell exists. @param cell Cell id. @return Leaf flag.
    bool is_leaf(int cell) const { return children_[static_cast<std::size_t>(cell)].empty(); }

    /// @brief Returns the child at @p slot, creating it if needed.
    /// @param cell Parent id. @param slot Child slot. @return Child id.
    int ensure_child(int cell, int slot);

    /// @brief Box of the (possibly not yet created) child at @p slot.
    /// @param cell Parent id. @param slot Child slot.
    /// @return Pair (lo, hi) of the child box.
    std::pair<std::vector<double>, std::vector<double>> child_box(int cell, int slot) const;

    /// @brief Slot of @p cell's child that contains @p point (upper half on ties at the max).
    /// @param cell Cell id. @param point d coordinates. @return Child slot.
    int slot_for(int cell, const std::vector<double>& point) const;

    /// @brief Whether @p point lies in the closed box of @p cell.
    /// @param cell Cell id. @param point d coordinates. @return Containment flag.
    bool contains(int cell, const std::vector<double>& point) const;

    /// @brief Deepest existing cell containing @p point.
    /// @param point d coordinates. @return Cell id, or -1 outside the root box.
    int locate(const std::vector<double>& point) const;

    /// @brief Ids of every cell at @p depth. @param depth Depth. @return Ids in creation order.
    const std::vector<int>& cells_at_depth(int depth) const;

    /// @brief Whether the open box (lo, hi) intersects the box [blo, bhi].
    /// @param lo Box A lower corner. @param hi Box A upper corner.
    /// @param blo Box B lower corner. @param bhi Box B upper corner.
    /// @return True when the interiors overlap.
    static bool overlaps(const std::vector<double>& lo, const std::vector<double>& hi,
                         const std::vector<double>& blo, const std::vector<double>& bhi);

    /// @brief Whether the part of [blo, bhi] inside the root is covered by cells of depth
    ///        >= @p target_depth (i.e. the box lies in the refined domain Omega^target_depth).
    /// @param target_depth Depth to test. @param blo Box lower corner. @param bhi Box upper corner.
    /// @return Coverage flag.
    bool covered_to_depth(int target_depth, const std::vector<double>& blo,
                          const std::vector<double>& bhi) const;

private:
    std::vector<double>              root_lo_;   ///< Root lower corner.
    std::vector<double>              root_hi_;   ///< Root upper corner.
    std::vector<std::vector<double>> lo_;        ///< Per-cell lower corners.
    std::vector<std::vector<double>> hi_;        ///< Per-cell upper corners.
    std::vector<int>                 depth_;     ///< Per-cell depth.
    std::vector<int>                 parent_;    ///< Per-cell parent id (-1 for root).
    std::vector<std::vector<int>>    children_;  ///< Per-cell child ids (empty for leaves, else arity()).
    std::vector<std::vector<int>>    by_depth_;  ///< Cell ids grouped by depth.
    int                              max_depth_{0}; ///< Deepest cell depth.

    /// @brief Recursive helper for covered_to_depth().
    /// @param cell Current cell. @param target_depth Depth to test.
    /// @param blo Box lower corner. @param bhi Box upper corner. @return Coverage flag.
    bool covered_impl(int cell, int target_depth, const std::vector<double>& blo,
                      const std::vector<double>& bhi) const;
};

/// @brief Sparse per-axis refinement operator: row i lists (coarse index, weight).
using RefinementRows = std::vector<std::vector<std::pair<int, double>>>;

/// @brief One axis's level-to-level refinement operator stored both ways.
struct AxisRefinement {
    RefinementRows rows; ///< rows[i]: (coarse j, R(i, j)) for refined index i.
    RefinementRows cols; ///< cols[j]: (refined i, R(i, j)) for coarse index j.
};

} // namespace detail
#endif

/// @brief Classical (overlay) hierarchical B-spline in the style of Forsey and Bartels:
///        every cell of a dyadic tree may carry its own local "detail" TensorBSpline.
///
/// evaluate(x) sums the patch of every cell on the root-to-leaf path containing x. Each
/// patch is evaluated in its cell's local coordinates, (x - cell_min) / (cell_max -
/// cell_min) in [0, 1]^d, so patch bases should span [0, 1]. Patches are independent
/// (no shared nested basis and no truncation); see TruncatedHierarchicalBSpline for the
/// THB-spline construction. Cells are identified by integer ids (root 0).
/// @par Example
/// @code
/// dm::fda::HierarchicalBSpline h({0.0}, {1.0});
/// auto b = dm::fda::BSplineBasis::open_uniform(0, 1);              // constant patch
/// h.set_spline(h.root(), dm::fda::TensorBSpline(b, {1.0}));
/// h.refine(h.root(), 1, dm::fda::TensorBSpline(b, {2.0}));          // upper half
/// std::print("{} {}\n", h.evaluate({0.25}), h.evaluate({0.75}));    // 1 3
/// @endcode
/// @par References
/// - D. R. Forsey and R. H. Bartels (1988). Hierarchical B-spline refinement. ACM SIGGRAPH
///   Computer Graphics 22(4), 205-212. https://doi.org/10.1145/378456.378512
/// @see TruncatedHierarchicalBSpline
class HierarchicalBSpline {
public:
#ifndef SWIG
    /// @brief Callback used by refine_region(): given a cell's bounds, returns the patch to
    ///        install there, or std::nullopt to leave the cell's patch unchanged.
    using SplineFactory = std::function<std::optional<TensorBSpline>(const std::vector<double>& cell_min,
                                                                     const std::vector<double>& cell_max)>;
#endif

    /// @brief Creates a hierarchy whose root cell is [domain_min, domain_max], with no patches.
    /// @param domain_min Lower corner (length d >= 1).
    /// @param domain_max Upper corner, with domain_min[k] < domain_max[k].
    /// @throws std::invalid_argument On empty, mismatched or non-finite bounds, min >= max,
    ///                               or more than 20 dimensions.
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0, 0.0}, {1.0, 2.0});
    /// std::print("{} {}\n", h.dimensions(), h.cell_count()); // 2 1
    /// @endcode
    HierarchicalBSpline(std::vector<double> domain_min, std::vector<double> domain_max);

    /// @brief Number of dimensions d. @return d.
    std::size_t dimensions() const noexcept { return tree_.dimensions(); }
    /// @brief Number of cells created so far. @return Cell count.
    int cell_count() const noexcept { return tree_.size(); }
    /// @brief Root cell id. @return Always 0.
    int root() const noexcept { return 0; }
    /// @brief Parent id. @param cell Cell id. @return Parent id, -1 for the root.
    /// @throws std::out_of_range For an invalid id.
    int parent(int cell) const;
    /// @brief Child id at @p slot. @param cell Cell id. @param slot Child slot in [0, 2^d).
    /// @return Child id, or -1 when not created. @throws std::out_of_range For invalid ids/slots.
    int child(int cell, int slot) const;
    /// @brief Depth of @p cell (root 0). @param cell Cell id. @return Depth.
    /// @throws std::out_of_range For an invalid id.
    int depth(int cell) const;
    /// @brief Lower corner of @p cell. @param cell Cell id. @return d coordinates.
    /// @throws std::out_of_range For an invalid id.
    std::vector<double> cell_min(int cell) const;
    /// @brief Upper corner of @p cell. @param cell Cell id. @return d coordinates.
    /// @throws std::out_of_range For an invalid id.
    std::vector<double> cell_max(int cell) const;

    /// @brief Deepest existing cell containing @p point.
    /// @param point d coordinates.
    /// @return Cell id, or -1 if @p point is outside the domain.
    /// @throws std::invalid_argument If point.size() != dimensions().
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0}, {1.0});
    /// int c = h.refine(h.root(), 1);
    /// std::print("{} {}\n", h.locate({0.9}) == c, h.locate({2.0})); // true -1
    /// @endcode
    int locate(const std::vector<double>& point) const;

    /// @brief Whether @p cell carries a patch. @param cell Cell id. @return Flag.
    /// @throws std::out_of_range For an invalid id.
    bool has_spline(int cell) const;
    /// @brief The patch of @p cell. @param cell Cell id.
    /// @return Reference valid until the patch is replaced or cleared.
    /// @throws std::out_of_range For an invalid id. @throws std::invalid_argument If no patch.
    const TensorBSpline& spline(int cell) const;
    /// @brief Installs (or replaces) the patch of @p cell.
    /// @param cell Cell id. @param spline Patch in cell-local [0, 1]^d coordinates.
    /// @throws std::out_of_range For an invalid id.
    /// @throws std::invalid_argument If the patch dimension differs from dimensions().
    void set_spline(int cell, TensorBSpline spline);
    /// @brief Removes the patch of @p cell (no-op if none). @param cell Cell id.
    /// @throws std::out_of_range For an invalid id.
    void clear_spline(int cell);

    /// @brief Creates all 2^d children of @p cell (existing ones are kept), without patches.
    /// @param cell Cell id.
    /// @return The 2^d child ids in slot order.
    /// @throws std::out_of_range For an invalid id.
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
    /// std::print("{}\n", h.subdivide(h.root()).size()); // 4
    /// @endcode
    std::vector<int> subdivide(int cell);

    /// @brief Creates (if needed) only the child of @p cell at @p slot.
    /// @param cell Cell id. @param slot Child slot; bit k selects the upper half of axis k.
    /// @return The child id.
    /// @throws std::out_of_range For an invalid id or slot.
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0}, {1.0});
    /// int c = h.refine(h.root(), 0);
    /// std::print("{}\n", h.cell_max(c)); // [0.5]
    /// @endcode
    int refine(int cell, int slot);

    /// @brief Creates (if needed) the child at @p slot and installs @p spline on it.
    /// @param cell Cell id. @param slot Child slot. @param spline Patch for the child.
    /// @return The child id.
    /// @throws std::out_of_range For an invalid id or slot.
    /// @throws std::invalid_argument If the patch dimension differs from dimensions().
    /// @par Example
    /// See the class example.
    int refine(int cell, int slot, TensorBSpline spline);

    /// @brief Subdivides, down to depth @p depth, every cell overlapping the open box
    ///        (region_min, region_max).
    /// @param region_min Lower corner. @param region_max Upper corner (min < max per axis).
    /// @param depth Deepest depth to create (cells already deeper are untouched).
    /// @throws std::invalid_argument On mismatched sizes or min >= max.
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0}, {1.0});
    /// h.refine_region({0.0}, {0.3}, 2);
    /// std::print("{}\n", h.cell_count()); // 4
    /// @endcode
    void refine_region(const std::vector<double>& region_min, const std::vector<double>& region_max, int depth);

#ifndef SWIG
    /// @brief Like refine_region(region_min, region_max, depth) but calls @p factory on every
    ///        touched cell (new or existing) and installs any patch it returns.
    /// @param region_min Lower corner. @param region_max Upper corner. @param depth Deepest depth.
    /// @param factory Patch callback; std::nullopt leaves a cell unchanged.
    /// @throws std::invalid_argument On invalid bounds or a patch of the wrong dimension.
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0}, {1.0});
    /// auto b = dm::fda::BSplineBasis::open_uniform(0, 1);
    /// h.refine_region({0.0}, {0.3}, 1, [&](auto&, auto&) { return dm::fda::TensorBSpline(b, {1.0}); });
    /// std::print("{}\n", h.evaluate({0.1})); // 1
    /// @endcode
    void refine_region(const std::vector<double>& region_min, const std::vector<double>& region_max, int depth,
                       const SplineFactory& factory);
#endif

    /// @brief Least-squares fits a new patch on @p cell so the whole hierarchy matches
    ///        @p values at @p points.
    ///
    /// The patch explains only what is left after every other patch on the path (the
    /// residual values - evaluate(points)), in the spirit of Forsey-Bartels editing, so pass
    /// raw data values rather than residuals. Points are global coordinates inside the cell.
    /// @param cell Cell id; must not already have a patch (clear_spline() first to refit).
    /// @param bases Patch bases on [0, 1], one per dimension.
    /// @param points Sample locations inside the cell.
    /// @param values One raw data value per point.
    /// @param options Solver settings.
    /// @throws std::out_of_range For an invalid id.
    /// @throws std::invalid_argument If the cell has a patch, inputs are empty/mismatched, or a
    ///                               point lies outside the cell.
    /// @throws std::runtime_error If the local normal equations are singular.
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0}, {1.0});
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// h.fit_cell(h.root(), {b}, {{0.0}, {1.0}}, {1.0, 3.0});
    /// std::print("{}\n", h.evaluate({0.5})); // 2
    /// @endcode
    void fit_cell(int cell, std::vector<BSplineBasis> bases, const std::vector<std::vector<double>>& points,
                  const std::vector<double>& values, const SplineSolveOptions& options = SplineSolveOptions());

    /// @brief Penalized (P-spline) variant of fit_cell() with per-axis @p lambdas and
    ///        difference @p order; all-zero lambdas give fit_cell().
    /// @param cell Cell id without a patch. @param bases Patch bases on [0, 1].
    /// @param points Sample locations inside the cell. @param values Raw data values.
    /// @param lambdas Nonnegative per-axis weights. @param order Difference order (default 2).
    /// @param options Solver settings.
    /// @throws std::out_of_range For an invalid id.
    /// @throws std::invalid_argument As for fit_cell() and fit_tensor_pspline().
    /// @throws std::runtime_error If the penalized normal equations are singular.
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0}, {1.0});
    /// auto b = dm::fda::BSplineBasis::open_uniform(3, 10);
    /// h.fit_cell_penalized(h.root(), {b}, {{0.0}, {1.0}}, {0.0, 2.0}, {1e3});
    /// std::print("{:.3f}\n", h.evaluate({0.5})); // 1.000
    /// @endcode
    void fit_cell_penalized(int cell, std::vector<BSplineBasis> bases, const std::vector<std::vector<double>>& points,
                            const std::vector<double>& values, const std::vector<double>& lambdas, int order = 2,
                            const SplineSolveOptions& options = SplineSolveOptions());

    /// @brief Sum of the patches along the path of cells containing @p point.
    /// @param point d coordinates inside the domain.
    /// @return Hierarchy value (0 where no patch applies).
    /// @throws std::invalid_argument On a size mismatch or a point outside the domain.
    /// @par Example
    /// See the class example.
    double evaluate(const std::vector<double>& point) const;

    /// @brief Evaluates at many points. @param points Points of length d.
    /// @return One value per point. @throws std::invalid_argument As for evaluate().
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0}, {1.0});
    /// std::print("{}\n", h.evaluate_points({{0.1}, {0.9}})); // [0, 0]
    /// @endcode
    std::vector<double> evaluate_points(const std::vector<std::vector<double>>& points) const;

    /// @brief Evaluates on a Cartesian grid. @param axes d non-empty coordinate lists.
    /// @return Row-major values (last axis fastest).
    /// @throws std::invalid_argument On a dimension mismatch, empty axis, or outside point.
    /// @par Example
    /// @code
    /// dm::fda::HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
    /// std::print("{}\n", h.evaluate_grid({{0.0, 1.0}, {0.5}}).size()); // 2
    /// @endcode
    std::vector<double> evaluate_grid(const std::vector<std::vector<double>>& axes) const;

private:
    detail::OrthantCellTree                   tree_;    ///< Cell geometry and topology.
    std::vector<std::optional<TensorBSpline>> patches_; ///< Per-cell optional patch.

    /// @brief Throws std::invalid_argument unless @p spline has dimensions() axes.
    /// @param spline Patch to check.
    void check_patch(const TensorBSpline& spline) const;
    /// @brief Creates the child (if needed) and grows patches_. @param cell Parent. @param slot Slot.
    /// @return Child id.
    int make_child(int cell, int slot);
    /// @brief Maps a global point into @p cell's local [0, 1]^d coordinates.
    /// @param cell Cell id. @param point Global coordinates. @return Local coordinates.
    std::vector<double> to_local(int cell, const std::vector<double>& point) const;
    /// @brief Shared input validation and residual computation for fit_cell*.
    /// @param cell Cell id. @param points Global points. @param values Raw values.
    /// @param who Caller name for messages. @return (local points, residuals).
    std::pair<std::vector<std::vector<double>>, std::vector<double>> prepare_fit(
        int cell, const std::vector<std::vector<double>>& points, const std::vector<double>& values,
        const char* who) const;
    /// @brief Recursive refine_region() worker.
    /// @param cell Current cell. @param lo Region lower corner. @param hi Region upper corner.
    /// @param depth Target depth. @param factory Optional patch callback (may be null).
    void refine_region_impl(int cell, const std::vector<double>& lo, const std::vector<double>& hi, int depth,
                            const std::function<std::optional<TensorBSpline>(const std::vector<double>&,
                                                                             const std::vector<double>&)>* factory);
};

/// @brief Truncated hierarchical B-spline (THB-spline) of Giannelli, Juttler and Speleers.
///
/// Level 0 is the caller's TensorBSpline; level l + 1 is the dyadic refinement of level l.
/// A dyadic cell tree defines the nested domains Omega^0 (whole domain) contains Omega^1
/// contains ..., where Omega^l is the union of cells of depth >= l. A level-l basis
/// function is active when its support lies in Omega^l but not in Omega^(l+1). The THB
/// basis consists of the active functions, each truncated: its representation on finer
/// levels drops the finer functions whose support lies in the finer refined domain. The
/// resulting basis is a partition of unity, preserves the continuity of the underlying
/// splines, and spans the hierarchical spline space.
///
/// Internally each level stores a full coefficient net whose entries outside the next
/// level's refined domain always equal the refinement of the coarser level; coefficient
/// edits are pushed down through all finer levels with truncation, so evaluation needs
/// only the finest level covering a point. Refinement never changes the represented
/// function. Bases must be clamped. Cells are integer ids (root 0).
/// @par Example
/// @code
/// auto b = dm::fda::BSplineBasis::open_uniform(2, 4);
/// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline({b, b}, std::vector<double>(16, 1.0)));
/// thb.refine_region({0.0, 0.0}, {0.5, 0.5}, 1);
/// std::print("{} {:.3f}\n", thb.level_count(), thb.evaluate({0.2, 0.2})); // 2 1.000
/// @endcode
/// @par References
/// - C. Giannelli, B. Juttler and H. Speleers (2012). THB-splines: The truncated basis for
///   hierarchical splines. Computer Aided Geometric Design 29(7), 485-498.
///   https://doi.org/10.1016/j.cagd.2012.03.025
/// - D. R. Forsey and R. H. Bartels (1988). Hierarchical B-spline refinement.
///   https://doi.org/10.1145/378456.378512
/// @see HierarchicalBSpline, dyadic_refine()
class TruncatedHierarchicalBSpline {
public:
    /// @brief Starts a hierarchy whose level 0 (and root cell / domain) is @p base.
    /// @param base Level-0 spline; every axis basis must be clamped.
    /// @throws std::invalid_argument If an axis basis is not clamped or d > 20.
    /// @par Example
    /// See the class example.
    explicit TruncatedHierarchicalBSpline(TensorBSpline base);

    /// @brief Number of dimensions d. @return d.
    std::size_t dimensions() const noexcept { return tree_.dimensions(); }
    /// @brief Number of materialized levels (1 + deepest cell depth). @return Level count.
    int level_count() const noexcept { return static_cast<int>(levels_.size()); }
    /// @brief The level-@p l spline on the whole domain; it equals the hierarchy wherever
    ///        Omega^l but not Omega^(l+1) applies.
    /// @param level Level in [0, level_count()).
    /// @return Reference valid until the next refinement or coefficient edit.
    /// @throws std::out_of_range For an invalid level.
    const TensorBSpline& level(int level) const;

    /// @brief Number of cells. @return Cell count.
    int cell_count() const noexcept { return tree_.size(); }
    /// @brief Root cell id. @return Always 0.
    int root() const noexcept { return 0; }
    /// @brief Parent id. @param cell Cell id. @return Parent id, -1 for the root.
    /// @throws std::out_of_range For an invalid id.
    int parent(int cell) const;
    /// @brief Child id. @param cell Cell id. @param slot Child slot.
    /// @return Child id or -1. @throws std::out_of_range For invalid ids/slots.
    int child(int cell, int slot) const;
    /// @brief Depth (= level used there if it is a leaf). @param cell Cell id. @return Depth.
    /// @throws std::out_of_range For an invalid id.
    int depth(int cell) const;
    /// @brief Lower corner. @param cell Cell id. @return d coordinates.
    /// @throws std::out_of_range For an invalid id.
    std::vector<double> cell_min(int cell) const;
    /// @brief Upper corner. @param cell Cell id. @return d coordinates.
    /// @throws std::out_of_range For an invalid id.
    std::vector<double> cell_max(int cell) const;
    /// @brief Deepest cell containing @p point (its depth is the finest level there).
    /// @param point d coordinates. @return Cell id, or -1 outside the domain.
    /// @throws std::invalid_argument If point.size() != dimensions().
    int locate(const std::vector<double>& point) const;

    /// @brief Creates (if needed) the child of @p cell at @p slot, materializing the next
    ///        level when first reached. The represented function is unchanged.
    /// @param cell Cell id. @param slot Child slot. @return Child id.
    /// @throws std::out_of_range For an invalid id or slot.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
    /// thb.refine(thb.root(), 0);
    /// std::print("{} {}\n", thb.level_count(), thb.evaluate({0.25})); // 2 0.25
    /// @endcode
    int refine(int cell, int slot);

    /// @brief Creates all 2^d children of @p cell. @param cell Cell id.
    /// @return Child ids in slot order. @throws std::out_of_range For an invalid id.
    /// @par Example
    /// See refine().
    std::vector<int> subdivide(int cell);

    /// @brief Subdivides, down to depth @p depth, every cell overlapping the open box
    ///        (region_min, region_max). The represented function is unchanged.
    /// @param region_min Lower corner. @param region_max Upper corner (min < max per axis).
    /// @param depth Deepest level to create.
    /// @throws std::invalid_argument On mismatched sizes or min >= max.
    /// @par Example
    /// See the class example.
    void refine_region(const std::vector<double>& region_min, const std::vector<double>& region_max, int depth);

    /// @brief Lower corner of the support box of basis function @p index at @p level.
    /// @param level Level. @param index Multi-index into that level's net.
    /// @return d coordinates. @throws std::out_of_range For invalid levels/indices.
    /// @throws std::invalid_argument On a dimension mismatch.
    std::vector<double> support_min(int level, const std::vector<int>& index) const;
    /// @brief Upper corner of the support box. @param level Level. @param index Multi-index.
    /// @return d coordinates. @throws std::out_of_range For invalid levels/indices.
    /// @throws std::invalid_argument On a dimension mismatch.
    std::vector<double> support_max(int level, const std::vector<int>& index) const;

    /// @brief Whether level-@p level function @p index is in the THB basis (support in
    ///        Omega^level but not in Omega^(level+1)).
    /// @param level Level. @param index Multi-index.
    /// @return Activity flag.
    /// @throws std::out_of_range For invalid levels/indices.
    /// @throws std::invalid_argument On a dimension mismatch.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
    /// thb.subdivide(thb.root());
    /// std::print("{} {}\n", thb.is_active(0, {0}), thb.is_active(1, {0})); // false true
    /// @endcode
    bool is_active(int level, const std::vector<int>& index) const;

    /// @brief Every active multi-index at @p level, in row-major order.
    /// @param level Level. @return Multi-indices. @throws std::out_of_range For an invalid level.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
    /// std::print("{}\n", thb.active_indices(0).size()); // 2
    /// @endcode
    std::vector<std::vector<int>> active_indices(int level) const;

    /// @brief Total number of THB basis functions over all levels. @return Count.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
    /// thb.refine(thb.root(), 0);
    /// std::print("{}\n", thb.basis_size()); // 3
    /// @endcode
    std::size_t basis_size() const;

    /// @brief THB coefficient of a basis function (for active functions; inactive entries
    ///        are internal bookkeeping).
    /// @param level Level. @param index Multi-index. @return Coefficient.
    /// @throws std::out_of_range For invalid levels/indices.
    /// @throws std::invalid_argument On a dimension mismatch.
    double coefficient(int level, const std::vector<int>& index) const;

    /// @brief Sets the coefficient of an active function and propagates the change, with
    ///        truncation, to every finer level. The edit is local to the function's support.
    /// @param level Level. @param index Active multi-index. @param value New coefficient.
    /// @throws std::out_of_range For invalid levels/indices.
    /// @throws std::invalid_argument On a dimension mismatch or an inactive index.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 0.0}));
    /// thb.set_coefficient(0, {1}, 2.0);
    /// std::print("{}\n", thb.evaluate({0.5})); // 1
    /// @endcode
    void set_coefficient(int level, const std::vector<int>& index, double value);

    /// @brief Value of the truncated basis function (level, index) at @p point.
    /// @param level Level. @param index Multi-index (inactive functions give 0).
    /// @param point d coordinates inside the domain.
    /// @return Truncated basis value.
    /// @throws std::out_of_range For invalid levels/indices.
    /// @throws std::invalid_argument On a size mismatch or a point outside the domain.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
    /// std::print("{}\n", thb.truncated_basis(0, {1}, {0.25})); // 0.25
    /// @endcode
    double truncated_basis(int level, const std::vector<int>& index, const std::vector<double>& point) const;

    /// @brief Least-squares update of the active coefficients at @p level only.
    ///
    /// Solves for the change of the level's active coefficients that best explains the
    /// residual values - evaluate(points), using the exact truncated basis, so pass raw
    /// data values. Points shallower than @p level are unaffected by these functions and
    /// are ignored. Typical use is coarse-to-fine: fit level 0, refine where residuals are
    /// large, then fit the new level.
    /// @param level Level with at least one active function.
    /// @param points Sample locations inside the domain. @param values One raw value per point.
    /// @param options Solver settings.
    /// @throws std::out_of_range For an invalid level.
    /// @throws std::invalid_argument On empty/mismatched inputs, points outside the domain, or
    ///                               a level with no active functions.
    /// @throws std::runtime_error If the normal equations are singular.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 0.0}));
    /// thb.fit_level(0, {{0.0}, {1.0}}, {1.0, 3.0});
    /// std::print("{}\n", thb.evaluate({0.5})); // 2
    /// @endcode
    void fit_level(int level, const std::vector<std::vector<double>>& points, const std::vector<double>& values,
                   const SplineSolveOptions& options = SplineSolveOptions());

    /// @brief Penalized variant of fit_level(): adds the level's full-grid tensor difference
    ///        penalty restricted to its active functions, applied to the coefficient change.
    /// @param level Level. @param points Sample locations. @param values Raw values.
    /// @param lambdas Nonnegative per-axis weights. @param order Difference order (default 2).
    /// @param options Solver settings.
    /// @throws std::out_of_range For an invalid level.
    /// @throws std::invalid_argument As for fit_level() and tensor_difference_penalty().
    /// @throws std::runtime_error If the normal equations are singular.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(3, 8);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, std::vector<double>(8, 0.0)));
    /// thb.fit_level_penalized(0, {{0.0}, {1.0}}, {0.0, 1.0}, {1e3});
    /// std::print("{:.3f}\n", thb.evaluate({0.5})); // 0.500
    /// @endcode
    void fit_level_penalized(int level, const std::vector<std::vector<double>>& points,
                             const std::vector<double>& values, const std::vector<double>& lambdas, int order = 2,
                             const SplineSolveOptions& options = SplineSolveOptions());

    /// @brief Least-squares fit of every active coefficient on every level at once
    ///        (replaces the current function).
    /// @param points Sample locations inside the domain. @param values One value per point.
    /// @param options Solver settings.
    /// @throws std::invalid_argument On empty/mismatched inputs or points outside the domain.
    /// @throws std::runtime_error If some active function has too little data (singular).
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 0.0}));
    /// thb.refine(thb.root(), 1);                      // adds a knot at 0.5
    /// thb.fit({{0.0}, {0.25}, {0.5}, {0.75}, {1.0}}, {1.0, 0.5, 0.0, 0.5, 1.0}); // |2x - 1|
    /// std::print("{:.3f}\n", thb.evaluate({0.75})); // 0.500
    /// @endcode
    void fit(const std::vector<std::vector<double>>& points, const std::vector<double>& values,
             const SplineSolveOptions& options = SplineSolveOptions());

    /// @brief Penalized all-level fit: adds lambda_k times the squared order-q differences
    ///        along axis k of every run of q + 1 neighbouring active coefficients of a level.
    ///
    /// Differences that would involve an inactive coefficient are omitted, so for q >= 1
    /// constant coefficients (and, generally, polynomials of degree < q in the coefficient
    /// index on each level) are unpenalized; q = 0 is ridge regression on all coefficients.
    /// @param points Sample locations. @param values One value per point.
    /// @param lambdas Nonnegative per-axis weights. @param order Difference order (default 2).
    /// @param options Solver settings.
    /// @throws std::invalid_argument As for fit() and tensor_difference_penalty().
    /// @throws std::runtime_error If the normal equations are singular.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(2, 6);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, std::vector<double>(6, 0.0)));
    /// thb.refine_region({0.0}, {0.5}, 1);
    /// thb.fit_penalized({{0.0}, {1.0}}, {2.0, 2.0}, {1.0}, 1);
    /// std::print("{:.3f}\n", thb.evaluate({0.3})); // 2.000
    /// @endcode
    void fit_penalized(const std::vector<std::vector<double>>& points, const std::vector<double>& values,
                       const std::vector<double>& lambdas, int order = 2, const SplineSolveOptions& options = SplineSolveOptions());

    /// @brief Evaluates the hierarchical spline (finest level covering @p point).
    /// @param point d coordinates inside the domain. @return Value.
    /// @throws std::invalid_argument On a size mismatch or a point outside the domain.
    /// @par Example
    /// See the class example.
    double evaluate(const std::vector<double>& point) const;

    /// @brief Evaluates at many points. @param points Points of length d.
    /// @return One value per point. @throws std::invalid_argument As for evaluate().
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
    /// std::print("{}\n", thb.evaluate_points({{0.0}, {0.5}})); // [0, 0.5]
    /// @endcode
    std::vector<double> evaluate_points(const std::vector<std::vector<double>>& points) const;

    /// @brief Evaluates on a Cartesian grid. @param axes d non-empty coordinate lists.
    /// @return Row-major values (last axis fastest).
    /// @throws std::invalid_argument On a dimension mismatch, empty axis, or outside point.
    /// @par Example
    /// @code
    /// auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
    /// dm::fda::TruncatedHierarchicalBSpline thb(dm::fda::TensorBSpline(b, {0.0, 1.0}));
    /// std::print("{}\n", thb.evaluate_grid({{0.0, 1.0}})); // [0, 1]
    /// @endcode
    std::vector<double> evaluate_grid(const std::vector<std::vector<double>>& axes) const;

private:
    /// @brief Sparse level-indexed vector: flat row-major index to value.
    using SparseLevelVector = std::unordered_map<std::size_t, double>;

    detail::OrthantCellTree                     tree_;       ///< Nested-domain cell tree.
    std::vector<TensorBSpline>                  levels_;     ///< Per-level coefficient nets.
    std::vector<std::vector<detail::AxisRefinement>> refine_ops_; ///< refine_ops_[l][k]: axis-k op l -> l+1.
    mutable std::vector<std::unordered_map<std::size_t, bool>> inside_cache_; ///< Memoized Omega^l support tests.

    /// @brief Throws std::out_of_range for an invalid level. @param level Level.
    void check_level(int level) const;
    /// @brief Validated flat index of @p index at @p level.
    /// @param level Level. @param index Multi-index. @return Flat offset.
    std::size_t flat(int level, const std::vector<int>& index) const;
    /// @brief Multi-index of flat offset @p f at @p level. @param level Level. @param f Offset.
    /// @return Multi-index.
    std::vector<int> unflat(int level, std::size_t f) const;
    /// @brief Whether the support of (level, flat index) lies in Omega^level (memoized).
    /// @param level Level. @param f Flat index. @return Flag.
    bool inside(int level, std::size_t f) const;
    /// @brief Active test on a flat index. @param level Level. @param f Flat index. @return Flag.
    bool active_flat(int level, std::size_t f) const;
    /// @brief Active flat indices at @p level, ascending. @param level Level. @return Indices.
    std::vector<std::size_t> active_flat_indices(int level) const;
    /// @brief Materializes levels up to @p level (inclusive) by dyadic refinement.
    /// @param level Required level.
    void ensure_level(int level);
    /// @brief Clears memoized activity tests (after the tree changes).
    void invalidate();
    /// @brief Pushes a coefficient change at @p level down to every finer level with truncation.
    /// @param level Level of the change. @param delta Sparse change of that level's coefficients.
    void propagate(int level, const SparseLevelVector& delta);
    /// @brief Truncated basis values at @p point grouped by level: entry l maps level-l flat
    ///        indices (active or not) to their truncated weight.
    /// @param point Point inside the domain. @param min_level Coarsest level to report.
    /// @return One sparse vector per level from @p min_level to the point's depth.
    std::vector<SparseLevelVector> truncated_rows(const std::vector<double>& point, int min_level) const;
    /// @brief Shared worker for fit_level() and fit_level_penalized().
    /// @param level Level. @param points Points. @param values Raw values.
    /// @param lambdas Penalty weights (empty for none). @param order Difference order.
    /// @param options Solver settings.
    void fit_level_impl(int level, const std::vector<std::vector<double>>& points, const std::vector<double>& values,
                        const std::vector<double>* lambdas, int order, const SplineSolveOptions& options);
    /// @brief Shared worker for fit() and fit_penalized().
    /// @param points Points. @param values Values. @param lambdas Penalty weights or null.
    /// @param order Difference order. @param options Solver settings.
    void fit_all_impl(const std::vector<std::vector<double>>& points, const std::vector<double>& values,
                      const std::vector<double>* lambdas, int order, const SplineSolveOptions& options);
    /// @brief Recomputes every level's non-own coefficients from the coarser level (after a
    ///        bulk coefficient change at all levels).
    void repropagate_all();
};

} // namespace datamunge::fda
