#pragma once
/// @file inverse_iteration.hpp
/// @brief Fixed-shift inverse iteration with one factorization and actual iterate records.
#include <cmath>
#include <cstddef>
#include <datamunge/algorithms/linear_algebra.hpp>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
namespace detail {

/// @brief Dense double matrix retained for the Rayleigh-iteration compatibility helpers.
using EigMatrix = std::vector<std::vector<double>>;

/// @brief Legacy Rayleigh-iteration elimination; keeps its existing tiny-pivot policy.
/// @param M Owned square matrix.
/// @param b Owned matching right-hand side.
/// @return Legacy solution, not used by the new fixed-shift inverse kernel.
inline std::vector<double> eig_solve(EigMatrix M, std::vector<double> b) {
  const int n = static_cast<int>(b.size());
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::fabs(M[r][col]) > std::fabs(M[piv][col]))
        piv = r;
    std::swap(M[col], M[piv]);
    std::swap(b[col], b[piv]);
    double d = M[col][col];
    if (std::fabs(d) < 1e-300)
      d = 1e-300; // near-singular: regularize
    for (int r = 0; r < n; ++r) {
      if (r == col)
        continue;
      const double f = M[r][col] / d;
      for (int c = col; c < n; ++c)
        M[r][c] -= f * M[col][c];
      b[r] -= f * b[col];
    }
  }
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = M[i][i] != 0 ? b[i] / M[i][i] : 0.0;
  return x;
}

/// @brief Legacy square matrix-vector product for Rayleigh iteration.
/// @param A Borrowed square matrix.
/// @param x Borrowed matching vector.
/// @return Product.
inline std::vector<double> eig_matvec(const EigMatrix& A, const std::vector<double>& x) {
  const int           n = static_cast<int>(A.size());
  std::vector<double> y(n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      y[i] += A[i][j] * x[j];
  return y;
}

/// @brief Legacy in-place Euclidean normalization for Rayleigh iteration.
/// @param v Mutable vector.
/// @return Pre-normalization norm.
inline double eig_normalize(std::vector<double>& v) {
  double nrm = 0;
  for (double x : v)
    nrm += x * x;
  nrm = std::sqrt(nrm);
  if (nrm > 1e-300)
    for (double& x : v)
      x /= nrm;
  return nrm;
}

} // namespace detail

/// @brief Fixed solve count and optional complete-input/retained-vector ceilings.
/// No residual-based early stopping: iterations means exactly that many solves.
/// Zero disables either optional size cap, but zero iterations requests only the
/// normalized start and its Rayleigh diagnostics, without factorization. Complete
/// n*(iterations+1) retained-vector admission precedes numerical work for traces.
/// Caps exclude borrowed input, working LU, scalar metadata, vector capacity and
/// byte/peak RAM. Ordinary runs ignore max_history_values.
/// @par Example
/// @code
/// const dm::algorithms::InverseIterationOptions o;
/// std::print("{} {}\n",o.iterations,o.max_order); // 50 1024
/// @endcode
/// @see inverse_iteration_trace()
struct InverseIterationOptions {
  std::size_t iterations{50};  ///< Exact number of normalized inverse solves.
  std::size_t max_order{1024}; ///< Complete matrix order cap; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Retained vector entries including step zero; zero disables.
};
/// @brief One actual normalized iterate and its original-matrix diagnostics.
/// All values and vector entries are owned. Residual uses the stored double
/// vector and Rayleigh value; relative_residual divides by
/// (Frobenius(A)+abs(value))*norm(vector), or is zero for a zero denominator.
/// These are numerical backward-error diagnostics, not a convergence guarantee.
/// @par Example
/// @code
/// const auto t=dm::algorithms::inverse_iteration_trace({{2,0},{0,5}},2.1,{1});
/// std::print("{} {}\n",t.steps[1].iteration,t.steps[1].pair.vector.size()); // 1 2
/// @endcode
/// @see inverse_iteration_trace()
struct InverseIterationStep {
  std::size_t iteration{0};         ///< Zero is the normalized start; k is after solve k.
  EigenPair   pair;                 ///< Actual Rayleigh value and owned normalized iterate.
  double      residual{0};          ///< Euclidean norm of A*x-value*x in original matrix units.
  double      relative_residual{0}; ///< Frobenius-scaled residual, dimensionless.
};
/// @brief Final pair and actual selected work from one execution.
/// Does not retain input or any intermediate vectors. No shared mutable state.
/// Exactly one LU factorization when iterations>0, otherwise zero; each solve
/// uses that same fixed-shift LU. Diagnostics do not count as inverse solves.
/// @par Example
/// @code
/// const auto r=dm::algorithms::inverse_iteration_run({{2,0},{0,5}},2.1,{3});
/// std::print("{} {}\n",r.factorizations,r.solves); // 1 3
/// @endcode
/// @see inverse_iteration_run()
struct InverseIterationRun {
  EigenPair   pair;                 ///< Final owned normalized iterate and Rayleigh estimate.
  double      residual{0};          ///< Actual final absolute residual.
  double      relative_residual{0}; ///< Actual final Frobenius-scaled residual.
  std::size_t factorizations{0};    ///< Actual complete fixed-shift LU factorizations.
  std::size_t solves{0};            ///< Actual inverse solves followed by normalization.
};
/// @brief Actual chronological normalized iterates from the producing loop.
/// Includes step zero and every completed solve; no repeated-prefix replay.
/// All vectors are owned, and final run diagnostics match the last step.
/// @par Example
/// @code
/// const auto t=dm::algorithms::inverse_iteration_trace({{2,0},{0,5}},2.1,{3});
/// std::print("{}\n",t.steps.size()); // 4
/// @endcode
/// @see inverse_iteration_trace()
struct InverseIterationTrace {
  InverseIterationRun               run;   ///< Actual final pair, diagnostics and selected work.
  std::vector<InverseIterationStep> steps; ///< Owned initial and actual solve boundaries.
};
/// @brief Estimate an eigenpair by a fixed shift and an equal-entry start.
/// Borrowed finite nonempty square A is unchanged. Intended for real symmetric
/// A: convergence to a uniquely nearest eigenvalue requires a nonzero starting
/// component in its eigenspace. Repeated/equidistant eigenvalues, absent start
/// components and nonsymmetric matrices have no such guarantee. Eigenvector sign
/// is arbitrary. Defaults to 50 solves. Zero iterations returns initial diagnostics;
/// negative counts are invalid. Uses one extended-precision partial-pivot LU,
/// robust norm normalization and the Rayleigh quotient divided by x^T*x.
/// Unlike the old implementation, exact zero pivots throw rather than silently
/// regularizing. Rounded near-singular matrices need not have exact zero pivots;
/// residuals must be inspected. Output may differ from the old double elimination.
/// Time O(n^3+k*n^2), workspace O(n^2), without a retained history.
/// @param A Borrowed nonempty finite square dense matrix.
/// @param mu Finite fixed shift, never updated by Rayleigh estimates.
/// @param iters Nonnegative solve count, default 50.
/// @return Owned Rayleigh value and normalized vector.
/// @throws std::invalid_argument On shape, nonfinite input or negative iters.
/// @throws std::domain_error On an exact zero LU pivot or zero solve direction.
/// @throws std::overflow_error On nonfinite arithmetic or unrepresentable diagnostics.
/// @throws std::length_error On impossible container sizes.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto p=dm::algorithms::inverse_iteration({{2,0},{0,5}},2.1,12);
/// std::print("{:.6f}\n",p.value); // 2.000000
/// @endcode
/// @par References
/// - Netlib Templates, Inverse Iteration: https://www.netlib.org/utk/people/JackDongarra/etemplates/node96.html
/// @see inverse_iteration_run(), inverse_iteration_trace()
EigenPair inverse_iteration(const std::vector<std::vector<double>>& A, double mu, int iters = 50);
/// @brief Run fixed-shift inverse iteration with optional custom start and selected work.
/// Same numerical semantics/errors as inverse_iteration(). Empty start selects
/// equal positive entries; a supplied finite matching nonzero vector is normalized.
/// Input/start are borrowed only during the call. Complete order admission precedes
/// matrix reads/copying; max_history_values is ignored. No symmetry check or
/// convergence assertion. Extended-precision LU is platform dependent. No mutation
/// or shared state. O(n^3+k*n^2) time, O(n^2) workspace and O(n) returned vector.
/// @param A Borrowed finite nonempty square matrix.
/// @param mu Finite fixed shift.
/// @param options Fixed solve count and complete order cap; defaults above.
/// @param start Borrowed optional finite matching nonzero start; empty means equal entries.
/// @return Owned final pair, residual diagnostics and actual factorization/solve counts.
/// @throws std::invalid_argument On shape, finite-input or start violations.
/// @throws std::domain_error On exact zero pivot or zero direction.
/// @throws std::overflow_error On arithmetic/diagnostic range failure.
/// @throws std::length_error On enabled order cap or container-size overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::inverse_iteration_run({{2,0},{0,5}},2.1,{4},{0,1});
/// std::print("{:.0f} {}\n",r.pair.value,r.solves); // 5 4
/// @endcode
/// @see inverse_iteration()
InverseIterationRun inverse_iteration_run(
    const std::vector<std::vector<double>>& A,
    double                                  mu,
    InverseIterationOptions                 options = {},
    const std::vector<double>&              start   = {});
/// @brief Capture each actual fixed-shift iterate including the normalized start.
/// Uses the same producing kernel as inverse_iteration_run(), with complete
/// (iterations+1)*n retained-vector admission before copying/factorization. Zero
/// cap disables this optional ceiling; size arithmetic remains checked. No partial
/// trace returns on failure. Scalars, final duplicate vector, capacity, LU/input
/// and byte/peak RAM are excluded from the vector-entry ceiling. Additional
/// O(k*n) retained entries; no reruns of shorter prefixes.
/// @param A Borrowed finite nonempty square matrix.
/// @param mu Finite fixed shift.
/// @param options Exact solve count and complete order/history policies.
/// @param start Borrowed optional finite matching nonzero start; empty means equal entries.
/// @return Owned actual steps and final run diagnostics/work.
/// @throws std::invalid_argument On shape/nonfinite input/start violations.
/// @throws std::domain_error On exact zero pivot or zero direction.
/// @throws std::overflow_error On arithmetic/diagnostic range failure.
/// @throws std::length_error On enabled cap or count/container-size overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::inverse_iteration_trace({{2,0},{0,5}},2.1,{0});
/// std::print("{} {}\n",t.steps.size(),t.run.factorizations); // 1 0
/// @endcode
/// @see inverse_iteration_run()
InverseIterationTrace inverse_iteration_trace(
    const std::vector<std::vector<double>>& A,
    double                                  mu,
    InverseIterationOptions                 options = {},
    const std::vector<double>&              start   = {});
} // namespace datamunge::algorithms
