#pragma once
/// @file power_iteration.hpp
/// @brief Seeded or explicitly initialized dense power iteration with actual products and iterates.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned real Rayleigh estimate and a normalized represented vector.
/// Shared with fixed-shift inverse iteration; no convergence certificate is implied.
/// @par Example
/// @code
/// const dm::algorithms::EigenPair p{};std::print("{}\n",p.value); // 0
/// @endcode
struct EigenPair {
  double              value{0}; ///< Actual final Rayleigh value, zero for an empty matrix.
  std::vector<double> vector;   ///< Owned final normalized iterate, empty for an empty matrix.
};
/// @brief Requested-iteration and actual retained-entry policies.
/// Zero disables the corresponding policy; limits are not byte or wall-time budgets.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::PowerIterationOptions{}.max_iterations); // 1000000
/// @endcode
struct PowerIterationOptions {
  std::size_t max_iterations{1000000}; ///< Maximum requested iterations admitted before execution.
  std::size_t max_trace_entries{
      1000000}; ///< Actual initial coordinates plus per-attempt record and three coordinate vectors.
};
/// @brief One actual matrix product and its accepted or stopped normalization.
/// The Rayleigh and residual values refer to before and its actual product, not after. Sign alignment preserves the historical negative-eigenvalue convention.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::power_iteration_trace({{2,1},{1,2}},{1,0},1).steps[0].rayleigh_before); // 2
/// @endcode
struct PowerIterationStep {
  std::size_t         iteration;    ///< Zero-based attempted normalization.
  std::vector<double> before;       ///< Actual normalized input to this product.
  std::vector<double> product;      ///< Actual row-major matrix-times-before result.
  std::vector<double> after;        ///< Actual next iterate, or unchanged before on a tiny product.
  double              product_norm; ///< Actual sqrt(sum(product_i squared)).
  double              rayleigh_before; ///< Actual dot(before,product).
  double residual_before; ///< Norm of product-rayleigh_before*before, evaluated with hypot.
  bool   sign_flipped;    ///< Actual negative consecutive-vector dot product triggered negation.
  bool   accepted;        ///< False when product_norm<1e-18 stops the producer.
};
/// @brief Owned final eigenpair and actual selected native work.
/// Finite fixed runs may be unresolved; no spectral gap, symmetry or dominant-component certificate is returned.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::power_iteration_run({{2,1},{1,2}},1).attempts); // 1
/// @endcode
struct PowerIterationRun {
  EigenPair pair; ///< Actual final normalized iterate and final computed Rayleigh value.
  double    residual_norm{
      0}; ///< Actual hypot norm of the final matrix product minus the final Rayleigh-scaled iterate.
  std::size_t attempts{0};     ///< Actual matrix products considered for normalization.
  std::size_t accepted{0};     ///< Actual successful normalization updates.
  std::size_t matvec_calls{0}; ///< Actual dense products, including the final Rayleigh product.
  std::size_t scalar_products{
      0}; ///< Actual scalar multiply-add terms in those dense products; excludes dot/residual work.
  bool tiny_product{false}; ///< Actual product norm below1e-18 stopped iteration.
};
/// @brief Owned explicit starting vector, actual steps and ordinary outcome.
/// All vectors are captured at actual producing boundaries, without restarting for each requested iteration count.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::power_iteration_trace({{2,1},{1,2}},{1,0},2).steps.size()); // 2
/// @endcode
struct PowerIterationTrace {
  PowerIterationRun               run;     ///< Actual final pair/work.
  std::vector<double>             initial; ///< Actual normalized initialization.
  std::vector<PowerIterationStep> steps; ///< Actual attempted products and normalization decisions.
};
/// @brief Run the native seeded dense power-iteration producer.
/// Complete finite square validation precedes products. Ordinary work O((iterations+1)*n*n), O(n) additional storage. Empty0x0 returns an empty pair. The native1e-18 tiny-product threshold and consecutive-vector sign alignment are preserved. A symmetric matrix with a unique largest-magnitude eigenvalue and a nonzero initial component supports the usual asymptotic direction argument; a finite run does not prove convergence. Inputs are borrowed unchanged and state is local.
/// @param matrix Borrowed complete finite square matrix.
/// @param iterations Requested nonnegative updates, default300.
/// @param seed Local SplitMix initialization seed, default0x9E3779B97F4A7C15.
/// @return Owned actual pair.
/// @throws std::invalid_argument If the matrix is ragged/nonsquare/nonfinite, iterations is negative, or an explicit start is malformed/nonfinite or has computed norm below1e-18.
/// @throws std::length_error If the requested-iteration or actual-history policy is exceeded.
/// @throws std::overflow_error If a producing product, dot, norm or selected counter is not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::power_iteration({{3}},2).value); // 3
/// @endcode
/// @par References
/// - G. H. Golub and C. F. Van Loan, Matrix Computations, fourth edition, Johns Hopkins University Press,2013, chapter7 (power iteration).
EigenPair power_iteration(
    const std::vector<std::vector<double>>& matrix,
    int                                     iterations = 300,
    std::uint64_t                           seed       = 0x9E3779B97F4A7C15ULL);
/// @brief Run the native seeded dense power-iteration producer.
/// Complete finite square validation precedes products. Ordinary work O((iterations+1)*n*n), O(n) additional storage. Empty0x0 returns an empty pair. The native1e-18 tiny-product threshold and consecutive-vector sign alignment are preserved. A symmetric matrix with a unique largest-magnitude eigenvalue and a nonzero initial component supports the usual asymptotic direction argument; a finite run does not prove convergence. Inputs are borrowed unchanged and state is local.
/// @param matrix Borrowed complete finite square matrix.
/// @param iterations Requested nonnegative updates, default300.
/// @param seed Local SplitMix initialization seed, default0x9E3779B97F4A7C15.
/// @param options Requested-iteration and history policies; history cap unused without capture.
/// @return Owned actual pair and selected work.
/// @throws std::invalid_argument If the matrix is ragged/nonsquare/nonfinite, iterations is negative, or an explicit start is malformed/nonfinite or has computed norm below1e-18.
/// @throws std::length_error If the requested-iteration or actual-history policy is exceeded.
/// @throws std::overflow_error If a producing product, dot, norm or selected counter is not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::power_iteration_run({{3}},2).matvec_calls); // 3
/// @endcode
/// @par References
/// - G. H. Golub and C. F. Van Loan, Matrix Computations, fourth edition, Johns Hopkins University Press,2013, chapter7 (power iteration).
PowerIterationRun power_iteration_run(
    const std::vector<std::vector<double>>& matrix,
    int                                     iterations = 300,
    std::uint64_t                           seed       = 0x9E3779B97F4A7C15ULL,
    PowerIterationOptions                   options    = {});
/// @brief Run the same power producer from an explicit vector.
/// Complete finite square validation precedes products. Ordinary work O((iterations+1)*n*n), O(n) additional storage. Empty0x0 returns an empty pair. The native1e-18 tiny-product threshold and consecutive-vector sign alignment are preserved. A symmetric matrix with a unique largest-magnitude eigenvalue and a nonzero initial component supports the usual asymptotic direction argument; a finite run does not prove convergence. Inputs are borrowed unchanged and state is local. No history is retained.
/// @param matrix Borrowed complete finite square matrix.
/// @param start Borrowed matching finite vector with computed norm at least1e-18, normalized once; empty iff the matrix is empty.
/// @param iterations Requested nonnegative updates, default300.
/// @param options Requested-iteration and actual-history policies.
/// @return Owned actual pair/work.
/// @throws std::invalid_argument If the matrix is ragged/nonsquare/nonfinite, iterations is negative, or an explicit start is malformed/nonfinite or has computed norm below1e-18.
/// @throws std::length_error If the requested-iteration or actual-history policy is exceeded.
/// @throws std::overflow_error If a producing product, dot, norm or selected counter is not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::power_iteration_from({{2,1},{1,2}},{1,0},0).pair.value); // 2
/// @endcode
/// @par References
/// - G. H. Golub and C. F. Van Loan, Matrix Computations, fourth edition, Johns Hopkins University Press,2013, chapter7 (power iteration).
PowerIterationRun power_iteration_from(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>&              start,
    int                                     iterations = 300,
    PowerIterationOptions                   options    = {});
/// @brief Run the same power producer from an explicit vector.
/// Complete finite square validation precedes products. Ordinary work O((iterations+1)*n*n), O(n) additional storage. Empty0x0 returns an empty pair. The native1e-18 tiny-product threshold and consecutive-vector sign alignment are preserved. A symmetric matrix with a unique largest-magnitude eigenvalue and a nonzero initial component supports the usual asymptotic direction argument; a finite run does not prove convergence. Inputs are borrowed unchanged and state is local. Captures3*n+1 logical entries per actual attempt plus n initial entries; worst history O(iterations*n).
/// @param matrix Borrowed complete finite square matrix.
/// @param start Borrowed matching finite vector with computed norm at least1e-18, normalized once; empty iff the matrix is empty.
/// @param iterations Requested nonnegative updates, default300.
/// @param options Requested-iteration and actual-history policies.
/// @return Owned actual pair/work and producing histories.
/// @throws std::invalid_argument If the matrix is ragged/nonsquare/nonfinite, iterations is negative, or an explicit start is malformed/nonfinite or has computed norm below1e-18.
/// @throws std::length_error If the requested-iteration or actual-history policy is exceeded.
/// @throws std::overflow_error If a producing product, dot, norm or selected counter is not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::power_iteration_trace({{2,1},{1,2}},{1,0},1).steps[0].product); // [2, 1]
/// @endcode
/// @par References
/// - G. H. Golub and C. F. Van Loan, Matrix Computations, fourth edition, Johns Hopkins University Press,2013, chapter7 (power iteration).
PowerIterationTrace power_iteration_trace(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>&              start,
    int                                     iterations = 300,
    PowerIterationOptions                   options    = {});
} // namespace datamunge::algorithms
