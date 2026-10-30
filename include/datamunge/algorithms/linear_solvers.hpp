#pragma once

/// @file linear_solvers.hpp
/// @brief Dense and matrix-free linear solvers: Jacobi, Gauss-Seidel, CG, BiCG, Thomas, Gauss-Jordan.

#include <functional>
#include <vector>

namespace datamunge::algorithms {

/// @brief The result of a direct linear solve: whether the system was @ref solved (a unique solution
///        exists) and the solution vector @ref x.
struct LinearSolution {
    bool                solved{false}; ///< false if the matrix is singular / no unique solution.
    std::vector<double> x;             ///< the solution (empty when @c solved is false).
};

/// @brief *Gaussian elimination* with *partial pivoting*: the standard direct method for @c A x = b.
///        It reduces the augmented matrix to upper-triangular form by row operations -- at each step
///        swapping in the row with the largest pivot magnitude for numerical stability -- then solves
///        by back substitution. Runs in @c O(n^3). Reports failure when the matrix is singular (a
///        vanishing pivot).
///
/// @param a the coefficient matrix (@c n x @c n; taken by value and overwritten internally).
/// @param b the right-hand side (length @c n).
/// @return the @ref LinearSolution.
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

/// @brief The *Gauss-Seidel method*: a stationary iterative solver for @c A x = b. It sweeps the
///        equations, solving the @c i-th for @c x_i using the *most recently updated* values of the
///        other unknowns (in-place, unlike Jacobi's simultaneous update), which typically converges
///        about twice as fast. It is guaranteed to converge when @c A is strictly diagonally
///        dominant or symmetric positive-definite. Each sweep costs @c O(n^2).
///
/// @param a the coefficient matrix (@c n x @c n).
/// @param b the right-hand side.
/// @param max_iter the iteration cap.
/// @param tol the convergence tolerance on the residual norm.
/// @return the @ref IterativeSolution.
IterativeSolution gauss_seidel(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                               int max_iter = 1000, double tol = 1e-10);

/// @brief The *conjugate gradient* method: the classic Krylov-subspace iterative solver for a
///        *symmetric positive-definite* @c A. It minimizes the quadratic @c 1/2 x^T A x - b^T x along
///        a sequence of mutually @c A-conjugate search directions, so that in exact arithmetic it
///        reaches the exact solution in at most @c n steps -- and far fewer when the eigenvalues are
///        clustered. Each iteration costs one matrix-vector product, @c O(n^2) for a dense matrix
///        (much less for a sparse one, which is CG's real domain).
///
/// @param a a symmetric positive-definite matrix (@c n x @c n).
/// @param b the right-hand side.
/// @param max_iter the iteration cap.
/// @param tol the convergence tolerance on the residual norm.
/// @return the @ref IterativeSolution.
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

/// @brief The result of @ref gauss_jordan: whether the system was @ref solved, the solution @ref x,
///        and the full matrix @ref inverse (a by-product of reducing @c A to the identity).
struct GaussJordanResult {
    bool                             solved{false}; ///< false when A is singular.
    std::vector<double>              x;       ///< the solution to @c A x = b.
    std::vector<std::vector<double>> inverse; ///< @c A^{-1} (empty when singular).
};

/// @brief *Gauss-Jordan elimination*: reduces the augmented matrix @c [A | I | b] all the way to
///        *reduced* row-echelon form @c [I | A^{-1} | x] using full forward and backward
///        elimination with partial pivoting. Unlike plain Gaussian elimination (which stops at
///        upper-triangular and back-substitutes), Gauss-Jordan clears above the pivots too, so it
///        yields the *matrix inverse* alongside the solution. Runs in @c O(n^3) (about 50% more
///        arithmetic than Gaussian elimination, which is why it is preferred only when the inverse
///        itself is wanted).
///
/// @param a the coefficient matrix (@c n x @c n; taken by value).
/// @param b the right-hand side.
/// @return the @ref GaussJordanResult with the solution and the inverse.
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
