#pragma once

/// @file linear_solvers.hpp
/// @brief Dense and matrix-free linear solvers: Jacobi, Gauss-Seidel, CG, BiCG, Thomas, Gauss-Jordan.

#include <functional>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned legacy direct-solve result: computed cutoff success and solution.
/// For gaussian_elimination(), solved=false means an absolute pivot fell below
/// 1e-12, not a mathematical singularity certificate. No unique-solution or
/// forward-accuracy guarantee follows from solved=true. Empty valid input succeeds.
/// Field order and aggregate layout are unchanged; no caller storage is borrowed.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gaussian_elimination({{2}}, {4});
/// std::cout << s.solved << ' ' << s.x[0]; // 1 2
/// @endcode
/// @see gaussian_elimination_solve()
struct LinearSolution {
    bool                solved{false}; ///< True when the selected direct method completed.
    std::vector<double> x;             ///< Owned solution, empty on Gaussian cutoff failure.
};

/// @brief Checked legacy Gaussian elimination with earliest-tie partial pivoting.
/// Full square shapes, matching RHS and finite entries are validated before
/// indexing or arithmetic (by-value argument copies can precede validation).
/// Preserves original ordered double operations and absolute pivot test <1e-12;
/// equality is accepted. A cutoff failure returns solved=false/empty x and is not
/// a rank certificate: uniform scaling can change acceptance. Empty valid input
/// succeeds. Nonfinite stored arithmetic throws; roundoff/underflow and element
/// growth remain possible. O(n³) time/O(n²) owned storage, no caller mutation.
/// @param a Finite n-by-n coefficient rows, taken by value.
/// @param b Finite length-n RHS, taken by value.
/// @return Owned legacy-layout result, approximate x on success.
/// @throws std::invalid_argument Non-square, size mismatch or nonfinite input.
/// @throws std::length_error Container or work-counter capacities exceeded.
/// @throws std::overflow_error A stored assignment or widened growth becomes nonfinite.
/// @throws std::bad_alloc Argument copies, working state or output allocation fail.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gaussian_elimination({{0,1},{2,1}}, {1,3});
/// std::cout << s.x[0] << ' ' << s.x[1]; // 1 1
/// @endcode
/// @par References
/// - LAPACK DGETRF, partial-pivot row interchanges (blocked variant):
///   https://netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f_source.html .
/// - J. H. Wilkinson (1961), Error Analysis of Direct Methods of Matrix Inversion,
///   Journal of the ACM8(3),281–330, doi:10.1145/321075.321076 .
/// @see gaussian_elimination_solve(), gaussian_elimination_trace(), LinearSolution
LinearSolution gaussian_elimination(std::vector<std::vector<double>> a, std::vector<double> b);

/// @brief The result of an iterative linear solve: whether it @ref converged to the tolerance, the
///        approximate solution @ref x, the number of @ref iterations, and the final @ref residual
///        @c ||A x - b||.
struct IterativeSolution {
    bool                converged{false}; ///< Whether the tolerance was reached.
    std::vector<double> x;                ///< The approximate solution.
    int                 iterations{0};    ///< Iterations performed.
    double              residual{0.0};    ///< Final residual norm ||A x - b||.
};

/// @brief Checked forward in-place Gauss-Seidel from a zero initial iterate.
/// Validates full finite square shape, matching RHS, nonzero diagonal and controls
/// before any sweep. Uses the newest coordinates with widened sums and checked
/// double assignments. Inclusive absolute recomputed residual stopping; empty/zero
/// RHS can converge without a sweep. General matrices need not converge. O(kn²)
/// time and O(n) scratch; no input mutation. Layout/signature remain unchanged.
/// @param a Finite square coefficient matrix with nonzero diagonal.
/// @param b Finite matching right-hand side.
/// @param max_iter Nonnegative sweep budget, default 1000.
/// @param tol Finite nonnegative absolute residual threshold, default 1e-10.
/// @return Owned legacy-layout solution; zero-budget residual is ||b||.
/// @throws std::invalid_argument Invalid shape, entries, diagonal or controls.
/// @throws std::overflow_error Computed arithmetic cannot fit its format.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_seidel({{2}}, {4});
/// std::cout << s.x[0] << ' ' << s.converged; // 2 1
/// @endcode
/// @par References
/// - Barrett et al., Templates for the Solution of Linear Systems, 2nd ed.,
///   https://netlib.org/templates/templates.html , forward Gauss-Seidel.
/// @see gauss_seidel_trace(), IterativeSolution
IterativeSolution gauss_seidel(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                               int max_iter = 1000, double tol = 1e-10);

/// @brief Checked dense conjugate gradient for a symmetric positive-definite system.
/// Full finite/square/size and exact stored symmetry validation precedes solving.
/// Positive definiteness is a precondition, not certified: encountered nonpositive
/// curvature returns unconverged. Starts at zero and uses inclusive absolute
/// recomputed residual stopping, including initially. Norms/inner products widen
/// to long double; vector assignments round to checked double. Two products per
/// completed step include a true-residual check. Recursive convergence without
/// true convergence returns unconverged; conjugate_gradient_solve() exposes why.
/// Exact-arithmetic n-step termination is not a floating guarantee. O(kn²) work,
/// O(n) scratch, no input mutation; original result fields/signature are preserved.
/// @param a Finite exactly symmetric square matrix, SPD by contract.
/// @param b Finite matching right-hand side.
/// @param max_iter Nonnegative iteration budget, default 1000.
/// @param tol Finite nonnegative absolute true-residual cutoff, default 1e-10.
/// @return Owned iterate with recomputed final residual, including zero-budget runs.
/// @throws std::invalid_argument Invalid shape, entries, symmetry or controls.
/// @throws std::overflow_error Computed arithmetic cannot fit its format.
/// @par Example
/// @code
/// const auto s = dm::algorithms::conjugate_gradient({{2}}, {4});
/// std::cout << s.x[0] << ' ' << s.iterations; // 2 1
/// @endcode
/// @par References
/// - Hestenes and Stiefel (1952), Methods of Conjugate Gradients for Solving Linear
///   Systems, https://doi.org/10.6028/jres.049.044 .
/// @see conjugate_gradient_solve(), conjugate_gradient_trace(), IterativeSolution
IterativeSolution conjugate_gradient(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                                     int max_iter = 1000, double tol = 1e-10);

/// @brief The *Thomas algorithm* (tridiagonal matrix algorithm): solves a tridiagonal system in
///        *linear time* @c O(n) -- a specialized Gaussian elimination that exploits the banded
///        structure, needing no pivoting when the matrix is diagonally dominant. The system is given
///        by its three diagonals: @p sub (below), @p diag (main), @p super (above), and @p rhs.
///        Tridiagonal systems arise everywhere -- cubic splines, 1-D finite differences, implicit
///        PDE steps.
///
/// @param sub   the sub-diagonal (length @c n; @c sub[0] is unused).
/// @param diag  the main diagonal (length @c n).
/// @param super the super-diagonal (length @c n; @c super[n-1] is unused).
/// @param rhs   the right-hand side (length @c n).
/// @return the solution vector (length @c n).
std::vector<double> thomas_solve(std::vector<double> sub, std::vector<double> diag,
                                 std::vector<double> super, std::vector<double> rhs);

/// @brief Owned legacy Gauss–Jordan result: computed cutoff success, solution and inverse.
/// Both arrays are empty on cutoff failure. Successful arrays contain floating
/// approximations, not exact arithmetic certificates. Field order and aggregate
/// layout are preserved. Empty valid square input succeeds with empty arrays.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_jordan({{2}}, {4});
/// std::cout << s.x[0] << ' ' << s.inverse[0][0]; // 2 0.5
/// @endcode
/// @see gauss_jordan_solve()
struct GaussJordanResult {
    bool                             solved{false}; ///< True when all selected pivots passed the cutoff.
    std::vector<double>              x;             ///< Owned approximate solution; empty on failure.
    std::vector<std::vector<double>> inverse;        ///< Owned approximate inverse; empty on failure.
};

/// @brief Checked legacy full augmented Gauss–Jordan reduction with partial pivoting.
/// Validates all finite square rows and matching finite RHS before indexing or
/// inverse allocation; by-value argument copies can precede validation. Preserves
/// strict earliest pivot ties, absolute pivot test <1e-12 (equality accepted),
/// and ordered full-block double divisions/updates. Failure is not a rank or
/// conditioning certificate: uniform scaling can change acceptance. Empty valid
/// input succeeds. Stored nonfinite arithmetic throws; rounding/underflow and
/// inaccurate inverses remain possible. O(n³) time/O(n²) storage. The explicit
/// identity block adds arithmetic cost; no caller mutation or shared state.
/// @param a Finite square coefficients, taken by value.
/// @param b Finite matching RHS, taken by value.
/// @return Owned legacy-layout solution and inverse, both arrays empty on cutoff failure.
/// @throws std::invalid_argument Invalid complete shape, size or nonfinite entry.
/// @throws std::length_error Count/container/work-counter capacity exceeded.
/// @throws std::overflow_error A stored computed assignment is nonfinite.
/// @throws std::bad_alloc Argument, inverse, working or result allocation fails.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_jordan({{2}}, {4});
/// std::cout << s.solved << ' ' << s.inverse[0][0]; // 1 0.5
/// @endcode
/// @par References
/// - Tsao (1989), On the Equivalence of Gaussian Elimination and Gauss-Jordan
///   Reduction in Solving Linear Equations, NASA TM101466, Section3 (related
///   diagonal-reduction variant), https://ntrs.nasa.gov/citations/19890011339 .
/// - Peters and Wilkinson (1975), On the Stability of Gauss-Jordan Elimination
///   with Pivoting, Communications of the ACM18(1),20–24, doi:10.1145/360569.360653 .
/// @see gauss_jordan_solve(), gauss_jordan_trace(), GaussJordanResult
GaussJordanResult gauss_jordan(std::vector<std::vector<double>> a, std::vector<double> b);

/// @brief *Successive over-relaxation* (SOR): accelerates Gauss-Seidel by *over-shooting* each update
///        with a relaxation factor @p omega. Each new value is a blend
///        @c x_i <- (1-omega) x_i + omega * (Gauss-Seidel update), and for @c omega in @c (1,2) the
///        extra push can cut the number of sweeps by an order of magnitude; @c omega = 1 is exactly
///        Gauss-Seidel. The optimal @p omega depends on the spectral radius of the iteration matrix,
///        and @p omega must lie in @c (0,2) for convergence (Kahan's theorem).
///
/// @param a the coefficient matrix (@c n x @c n).
/// @param b the right-hand side.
/// @param omega the relaxation factor in @c (0,2).
/// @param max_iter the iteration cap.
/// @param tol the convergence tolerance on the residual norm.
/// @return the @ref IterativeSolution.
IterativeSolution sor(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                      double omega, int max_iter = 1000, double tol = 1e-10);

/// @brief The *biconjugate gradient* (BiCG) method: extends conjugate gradient to *non-symmetric*
///        systems by running two coupled recurrences -- one with @c A and a shadow one with @c A^T --
///        that keep the residuals bi-orthogonal. It needs no symmetry or definiteness, at the cost of
///        two matrix-vector products per step (one by @c A, one by @c A^T) and the risk of breakdown.
///        Like CG it terminates in at most @c n steps in exact arithmetic.
///
/// @param a the (square, possibly non-symmetric) matrix.
/// @param b the right-hand side.
/// @param max_iter the iteration cap.
/// @param tol the convergence tolerance on the residual norm.
/// @return the @ref IterativeSolution.
IterativeSolution biconjugate_gradient(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                                       int max_iter = 1000, double tol = 1e-10);

/// @brief A linear map \f$v\mapsto Mv\f$ supplied as a function, for matrix-free solvers.
using LinearOperator = std::function<std::vector<double>(const std::vector<double>&)>;

/// @brief Result of the matrix-free biconjugate_gradient(): the iterate, its history and why it stopped.
struct BiCGResult {
    std::vector<double> x;                  ///< The final iterate.
    bool                converged{false};   ///< Whether the relative residual reached the tolerance.
    bool                breakdown{false};   ///< Whether a vanishing \f$\tilde r^\top r\f$ or \f$\tilde p^\top Ap\f$ stopped it.
    int                 iterations{0};      ///< Iterations taken.
    int                 products{0};        ///< Operator applications (two per iteration: \f$A\f$ and \f$A^\top\f$).
    std::vector<double> residual_history;   ///< \f$\|r_k\|/\|b\|\f$ for k = 0 .. iterations (recursively updated residual).
};

/// @brief Matrix-free BiCG for \f$Ax=b\f$ from \f$x_0=0\f$, with shadow residual \f$\tilde r_0\f$ = @p shadow
///        (default \f$b\f$).
///
/// Each iteration applies @p apply to the direction \f$p_k\f$ and @p apply_transpose to the
/// shadow direction \f$\tilde p_k\f$, then updates
/// \f$\alpha_k=\tilde r_k^\top r_k/\tilde p_k^\top Ap_k\f$, \f$x_{k+1}=x_k+\alpha_kp_k\f$,
/// \f$r_{k+1}=r_k-\alpha_kAp_k\f$, \f$\tilde r_{k+1}=\tilde r_k-\alpha_kA^\top\tilde p_k\f$,
/// \f$\beta_k=\tilde r_{k+1}^\top r_{k+1}/\tilde r_k^\top r_k\f$ and the two directions. It stops when
/// \f$\|r_k\|\le\text{tol}\,\|b\|\f$, or with `breakdown` set when either inner product falls
/// below \f$10^{-15}\f$ of the product of the norms involved, or when the shadow residual has
/// collapsed far faster than the residual, \f$\|\tilde r_k\|/\|\tilde r_0\|\le10^{-10}\|r_k\|/\|b\|\f$ (in
/// exact arithmetic \f$\tilde r_k\f$ would be zero: a serious breakdown).
///
/// @param apply \f$v\mapsto Av\f$.
/// @param apply_transpose \f$v\mapsto A^\top v\f$.
/// @param b The right-hand side.
/// @param max_iter The iteration cap.
/// @param tol The relative residual tolerance.
/// @param shadow The shadow residual \f$\tilde r_0\f$; empty means \f$b\f$. It must not be orthogonal to
///        \f$b\f$, and a random vector avoids accidental structure (for example a left eigenvector of A).
/// @return The iterate, the residual history and the stopping reason.
/// @throws std::invalid_argument if @p shadow or an operator result has the wrong size.
///
/// @par Example
/// @code
/// const auto A = dm::linalg::to_csr(coo), At = dm::linalg::to_csr(coo_transposed);
/// const auto run = dm::algorithms::biconjugate_gradient(
///     [&](const auto& v) { return A.spmv(v); }, [&](const auto& v) { return At.spmv(v); }, b, 500, 1e-10);
/// @endcode
/// @see biconjugate_gradient(const std::vector<std::vector<double>>&, const std::vector<double>&, int, double)
BiCGResult biconjugate_gradient(const LinearOperator& apply, const LinearOperator& apply_transpose,
                                const std::vector<double>& b, int max_iter = 1000, double tol = 1e-10,
                                const std::vector<double>& shadow = {});

/// @brief *Levinson recursion*: solves a *symmetric Toeplitz* system @c T x = b in @c O(n^2) time
///        (versus @c O(n^3) for a general solver), by exploiting the constant-diagonal structure. A
///        Toeplitz matrix is defined entirely by its first row; the algorithm grows the solution one
///        dimension at a time, reusing a forward/backward vector pair. Toeplitz systems arise in
///        signal processing (linear prediction, autoregressive fitting) and time series.
///
/// @param first_row the first row of the symmetric Toeplitz matrix (so @c T(i,j)=first_row[|i-j|]);
///        @c first_row[0] must be nonzero.
/// @param rhs the right-hand side (same length as @p first_row).
/// @return the solution vector.
std::vector<double> levinson_solve(const std::vector<double>& first_row, const std::vector<double>& rhs);

} // namespace datamunge::algorithms
