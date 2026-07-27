#pragma once

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
    bool                converged{false};
    std::vector<double> x;
    int                 iterations{0};
    double              residual{0.0};
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

} // namespace datamunge::algorithms
