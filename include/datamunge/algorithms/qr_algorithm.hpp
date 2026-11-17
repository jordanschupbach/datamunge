#pragma once
/// @file qr_algorithm.hpp
/// @brief Checked dense symmetric shifted QR iteration with actual deflation histories.
#include <cstddef>
#include <vector>
/// @brief Canonical algorithms and their owned execution records.
namespace datamunge::algorithms {
/// @brief Owned row-major square matrix for the symmetric QR solver.
/// @par Example
/// @code
/// const dm::algorithms::QrMatrix matrix{{2,1},{1,2}};
/// std::print("{}\n",matrix.size()); // 2
/// @endcode
/// @see qr_eigenvalues_run()
using QrMatrix = std::vector<std::vector<double>>;
/// @brief Work and history policies for dense Householder QR; zero size caps disable that cap.
/// @par Example
/// @code
/// const dm::algorithms::QrOptions x{};
/// std::print("{}\n",x.max_order); // 256
/// @endcode
/// @see qr_eigenvalues_run()
struct QrOptions {
  std::size_t max_order{256};       ///< Largest admitted matrix order; zero disables.
  std::size_t max_iterations{1000}; ///< Maximum QR steps per active block, including zero.
  double      tolerance{1e-12};     ///< Positive finite relative trailing-row deflation tolerance.
  std::size_t max_trace_entries{1000000}; ///< Stored matrix scalars plus records; zero disables.
};
/// @brief One actual QR step or pre-step deflation of a diagonal trailing row.
/// @par Example
/// @code
/// const dm::algorithms::QrStep x{};
/// std::print("{}\n",x.active); // 0
/// @endcode
/// @see qr_eigenvalues_run()
struct QrStep {
  std::size_t active{};        ///< Leading active block order before deflation.
  std::size_t iteration{};     ///< QR steps performed on this block, zero for initial deflation.
  double      shift{};         ///< Actual Wilkinson shift; zero for a pre-step deflation.
  double      trailing_norm{}; ///< Euclidean off-diagonal norm of the complete trailing row.
  bool        deflated{};      ///< Whether the complete trailing row passed the relative tolerance.
  QrMatrix    matrix;          ///< Complete matrix immediately after the recorded operation.
};
/// @brief Owned result; histories describe the actual producer, not repeated capped reruns.
/// @par Example
/// @code
/// const dm::algorithms::QrResult x{};
/// std::print("{}\n",x.iterations); // 0
/// @endcode
/// @see qr_eigenvalues_run()
struct QrResult {
  std::vector<double> eigenvalues; ///< Ascending values, including multiplicities.
  QrMatrix            matrix; ///< Complete final matrix, with deflated row/column couplings zeroed.
  std::vector<QrStep> steps;  ///< Actual step records, populated only by qr_eigenvalues_trace().
  std::size_t         iterations{}; ///< Total dense QR steps over all blocks.
  std::size_t         deflations{}; ///< Number of accepted eigenvalues, including the final scalar.
};
/// @brief Compute symmetric eigenvalues by dense shifted Householder QR.
/// Checks square finite input and symmetry (absolute difference at most64 machine eps times the largest input-entry magnitude), then symmetrizes accepted roundoff. Householder reflectors retain orthogonality at singular shifts. Deflation checks every off-diagonal element of the trailing row, not just its neighbor. Dense steps cost O(m^3), plus O(n^2) working storage; no preliminary tridiagonal reduction or cubic-rate guarantee is claimed. Empty input returns an empty result. Input is unchanged; all outputs are owned.
/// @param matrix Finite symmetric square input.
/// @param options Order, per-block iteration, relative tolerance and history policy.
/// @return Complete converged spectrum and actual total step/deflation counts.
/// @throws std::invalid_argument Malformed input, nonsymmetry, or nonpositive/nonfinite tolerance.
/// @throws std::length_error Order policy exceeded.
/// @throws std::runtime_error A block fails to converge within the step budget.
/// @throws std::overflow_error An intermediate result is nonfinite.
/// @par Example
/// @code
/// auto r = dm::algorithms::qr_eigenvalues_run({{2,1},{1,2}});
/// std::print("{:.0f} {:.0f}\n", r.eigenvalues[0], r.eigenvalues[1]); // 1 3
/// @endcode
/// @par References
/// - J. G. F. Francis (1961), The QR Transformation, Part I, https://doi.org/10.1093/comjnl/4.3.265
/// @see qr_eigenvalues_trace(), qr_eigenvalues()
QrResult qr_eigenvalues_run(QrMatrix matrix, const QrOptions& options = {});
/// @brief Capture complete matrices and actual trailing-row decisions from the same QR producer.
/// Records each real step and initial exact/relative deflations. A record costs n*n+1 history entries, admitted before copying; zero history cap disables. Runtime and working storage match qr_eigenvalues_run(), with O(s*n*n) retained history for s records.
/// @param matrix Finite symmetric square input, copied by value.
/// @param options Solver and history policies.
/// @return Converged spectrum, final matrix and actual selected history.
/// @throws std::length_error History or order policy exceeded.
/// @throws std::invalid_argument Input or tolerance invalid.
/// @throws std::runtime_error Iteration budget exhausted.
/// @throws std::overflow_error Nonfinite arithmetic.
/// @par Example
/// @code
/// auto r = dm::algorithms::qr_eigenvalues_trace({{1,0},{0,3}});
/// std::print("{}\n", r.iterations); // 0
/// @endcode
/// @see qr_eigenvalues_run()
QrResult qr_eigenvalues_trace(QrMatrix matrix, const QrOptions& options = {});
/// @brief Return the ascending symmetric spectrum using the checked QR producer.
/// The signature is preserved. A nonconvergent budget now throws rather than silently accepting an unconverged diagonal. Exact diagonal matrices require no QR steps. Uses the default order policy; use qr_eigenvalues_run() to change it.
/// @param matrix Finite symmetric square input.
/// @param max_iter Nonnegative maximum QR steps per active block.
/// @param tol Positive finite relative deflation tolerance.
/// @return Ascending eigenvalues.
/// @throws std::invalid_argument Negative iteration count, bad tolerance or malformed input.
/// @throws std::runtime_error Iteration budget exhausted.
/// @throws std::length_error Default order policy exceeded.
/// @throws std::overflow_error Nonfinite arithmetic.
/// @par Example
/// @code
/// auto e = dm::algorithms::qr_eigenvalues({{4,0},{0,-2}});
/// std::print("{:.0f} {:.0f}\n", e[0],e[1]); // -2 4
/// @endcode
/// @see qr_eigenvalues_run()
std::vector<double> qr_eigenvalues(QrMatrix matrix, int max_iter = 1000, double tol = 1e-12);
/// @brief Construct a finite symmetric tridiagonal Toeplitz matrix.
/// Returns diagonal d and immediate off-diagonal e, all other entries zero. Used as an exact structured QR control; its analytic eigenvalues are d+2e*cos(j*pi/(n+1)),j=1..n. Count0 returns an empty matrix but coefficients are still validated. O(n^2) owned storage/initialization.
/// @param n Matrix order.
/// @param diagonal Finite diagonal value, default2.
/// @param off_diagonal Finite neighboring value, default-1.
/// @param max_order Allocation policy, default256; zero disables.
/// @return Owned symmetric n-by-n row matrix.
/// @throws std::invalid_argument Nonfinite coefficient.
/// @throws std::length_error Order allocation policy exceeded.
/// @par Example
/// @code
/// auto a=dm::algorithms::qr_tridiagonal(3);
/// std::print("{} {}\n",a[1][1],a[1][2]); // 2 -1
/// @endcode
/// @see qr_eigenvalues_run()
QrMatrix qr_tridiagonal(
    std::size_t n,
    double      diagonal     = 2,
    double      off_diagonal = -1,
    std::size_t max_order    = 256);
} // namespace datamunge::algorithms
