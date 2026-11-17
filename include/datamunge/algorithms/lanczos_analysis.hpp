#pragma once
/// @file lanczos_analysis.hpp
/// @brief Original symmetric study fixture and explicit Ritz vector/residual diagnostics of actual Lanczos output.
#include <cstddef>
#include <datamunge/algorithms/lanczos_iteration.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned actual native tridiagonal eigensystem and explicit full-matrix Ritz diagnostics.
/// values are descending native Jacobi approximations; coefficients[j][i] is
/// mode i's coefficient of actual basis vector V[j]. vectors[i] is the formed
/// full Ritz vector. Residuals explicitly evaluate A*y-theta*y, rather than using
/// only terminal_beta*last_coefficient, which omits reorthogonalization/roundoff
/// defects. Orthogonality/projection errors are observed widened maxima, not
/// exact certificates or evidence the supplied result originated from A.
/// @par Example
/// @code
/// const std::vector<std::vector<double>> A={{2}};
/// const auto a=dm::algorithms::lanczos_ritz_analysis(A,dm::algorithms::lanczos_iteration(A,{1},1));
/// std::print("{} {}\n",a.values[0],a.residual_norms[0]); // 2 0
/// @endcode
/// @see lanczos_ritz_analysis()
struct LanczosRitzAnalysis {
  std::vector<std::vector<double>> tridiagonal; ///< Actual T formed from alpha and first m-1 beta.
  std::vector<double>              values; ///< Actual native Jacobi eigenvalues of T, descending.
  std::vector<std::vector<double>>
      coefficients; ///< Actual m-by-m Jacobi eigenvectors, modes as columns.
  std::vector<std::vector<double>> vectors; ///< Formed full Ritz vectors, one owned row per mode.
  std::vector<long double> vector_norms;    ///< Explicit widened Euclidean norms of formed vectors.
  std::vector<long double>
              residual_norms; ///< Explicit widened full A*y-theta*y norms, not normalized.
  long double orthogonality_error{
      0};                          ///< max absolute entry of actual V^T V-I, widened diagnostics.
  long double projection_error{0}; ///< max absolute entry of actual V^T A V-T, widened diagnostics.
  long double projected_eigen_residual{
      0}; ///< Largest actual norm of T*q-theta*q across native modes.
};
/// @brief Recreate the original30-by30 symmetric matrix using its exact32-bit LCG fixture.
/// Start uint32 state2024; update s=1103515245*s+12345 modulo2^32 for upper
/// triangle row order; value=((s>>16)%1000)/500.0-1; mirror to lower triangle.
/// Synthetic deterministic fixture, not a random-matrix distribution claim.
/// @return Owned complete original30-by30 finite exactly symmetric matrix.
/// @throws std::bad_alloc On matrix allocation failure.
/// @par Example
/// @code
/// const auto A=dm::algorithms::lanczos_demo_matrix();std::print("{} {}\n",A.size(),A[0].size()); // 30 30
/// @endcode
/// @see lanczos_trace()
std::vector<std::vector<double>> lanczos_demo_matrix();
/// @brief Form the actual T eigensystem and explicitly evaluate full Ritz vectors/residuals.
/// Complete finite exactly symmetric A, compatible finite actual basis/alpha/beta
/// shapes and mode quota checked before allocations. Does not validate provenance
/// or require exact orthogonality. Uses native Jacobi on T(tolerance1e-13,100 sweeps),
/// forms y=sum_j V[j]*q[j] with widened sums rounded to double, then computes
/// actual residual/norm/Gram/projection diagnostics in long double. No Lanczos
/// rerun or prefix replay. Requires nonzero formed vectors; no universal spectral
/// approximation/convergence guarantee. O(m*n*n+n*m*m+m^3) dense work and O(n*m+m*m)
/// owned/runtime diagnostic storage; max_modes is a count, not bytes/peak RAM.
/// @param A Borrowed complete finite exactly symmetric original matrix.
/// @param result Borrowed complete m-mode result, m<=n; extra finite terminal beta allowed.
/// @param max_modes Complete m ceiling, default1000, zero disables optional cap.
/// @return Owned descending native Ritz eigensystem and explicit actual diagnostics.
/// @throws std::invalid_argument On matrix/result shape/finite/symmetry or zero formed vector.
/// @throws std::length_error On enabled mode quota or matrix/container geometry limits.
/// @throws std::overflow_error On nonfinite/unrepresentable diagnostic arithmetic.
/// @throws std::runtime_error If projected eigensystem residual exceeds1e-9*max(1,max_abs_T)*sqrt(m).
/// @throws std::bad_alloc On eigensystem/vector/diagnostic allocation failure.
/// @par Example
/// @code
/// const std::vector<std::vector<double>> A={{2,0},{0,3}};
/// const auto r=dm::algorithms::lanczos_iteration(A,{1,0},2);
/// std::print("{}\n",dm::algorithms::lanczos_ritz_analysis(A,r).values[0]); // 2
/// @endcode
/// @par References
/// - Lanczos(1950), An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators, https://nvlpubs.nist.gov/nistpubs/jres/045/jresv45n4p255_A1b.pdf.
/// @see LanczosRitzAnalysis, lanczos_trace()
LanczosRitzAnalysis lanczos_ritz_analysis(
    const std::vector<std::vector<double>>& A,
    const LanczosResult&                    result,
    std::size_t                             max_modes = 1000);
} // namespace datamunge::algorithms
