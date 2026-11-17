#pragma once
/// @file strassen.hpp
/// @brief Checked padded seven-product multiplication and actual block histories.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own a finite rectangular row-major double matrix for checked multiplication.
/// Rows own their scalars. Rectangularity and finiteness are checked when consumed by a multiplication function.
/// @par Example
/// @code
/// dm::algorithms::StrassenMatrix a{{1,2}};
/// std::print("{}\n",a[0].size()); // 2
/// @endcode
/// @see strassen_run()
using StrassenMatrix = std::vector<std::vector<double>>;
/// @brief Own the StrassenOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StrassenOptions r{};
/// (void)r;
/// @endcode
/// @see strassen_run()
struct StrassenOptions {
  std::size_t leaf_size{
      64}; ///< Positive schoolbook crossover order; default2x2 therefore executes eight products.
  std::size_t max_padded_cells{4194304}; ///< Complete one-square geometry; zero disables policy.
  std::size_t max_scalar_products{
      100000000}; ///< Complete executed leaf products; zero disables policy.
  std::size_t max_block_additions{
      1000000000};                 ///< Actual block element sums/differences; zero disables policy.
  std::size_t max_records{100000}; ///< Actual captured postorder blocks; zero disables policy.
  std::size_t max_history_values{1000000}; ///< Actual copied matrix entries; zero disables policy.
};
/// @brief Own the StrassenStep contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StrassenStep r{};
/// (void)r;
/// @endcode
/// @see strassen_run()
struct StrassenStep {
  std::string    path;              ///< Root or actual slash-separated seven-product child indices.
  std::size_t    depth{};           ///< Root depth0.
  StrassenMatrix result;            ///< Actual complete square block returned at this point.
  std::size_t    scalar_products{}; ///< Cumulative actual leaf products through this return.
  std::size_t block_additions{}; ///< Cumulative actual block sums/differences through this return.
};
/// @brief Own the StrassenRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StrassenRun r{};
/// (void)r;
/// @endcode
/// @see strassen_run()
struct StrassenRun {
  StrassenMatrix product;     ///< Owned trimmed m by n actual result.
  std::size_t padded_order{}; ///< Actual power-of-two order; zero for empty inner/output geometry.
  std::size_t scalar_products{}; ///< Actual leaf scalar multiplications.
  std::size_t
      leaf_accumulations{}; ///< Actual additions to schoolbook accumulators, including initial zeros.
  std::size_t block_additions{};   ///< Actual scalar block sums and differences, excluding copies.
  std::size_t recursive_calls{};   ///< Actual square multiplication calls including leaves.
  std::size_t leaf_calls{};        ///< Actual schoolbook leaf calls.
  std::vector<StrassenStep> steps; ///< Optional owned actual postorder return values.
};
/// @brief Multiply through the actual padded seven-product producer.
/// Finite rectangular operands with matching inner dimensions are required. Empty compatible shapes produce an owned zero-shaped result. Pads to a power-of-two square; positive leaf_size defaults64. Seven recursive products per internal call, eighteen half-block additions/subtractions. O(s^log2(7)) arithmetic for fixed crossover, with padding and substantial temporary storage; this is not a runtime speed guarantee. Default2x2 uses eight schoolbook products. Geometry and complete leaf-work checked before allocation; actual block/capture budgets checked during execution. Policies are event/value counts, not peak bytes. Floating rounding/overflow applies; no shared state or caller mutation.
/// @param a Borrowed finite rectangular left matrix m by k.
/// @param b Borrowed finite rectangular right matrix k by n.
/// @param options Positive crossover and complete geometry/work/capture policies; zero disables budgets.
/// @return Owned product, actual executed counts and optional postorder block states.
/// @throws std::invalid_argument Ragged/nonfinite operands, incompatible dimensions or zero crossover.
/// @throws std::length_error Geometry, complete/actual work, capture or counter capacity exceeded.
/// @throws std::overflow_error Actual arithmetic is not finite.
/// @par Example
/// @code
/// auto r=dm::algorithms::strassen_run({{2}},{{3}});
/// std::print("{} {}\n",r.product[0][0],r.scalar_products); // 6 1
/// @endcode
/// @par References
/// - V. Strassen, Gaussian Elimination is Not Optimal, Numerische Mathematik13 (1969),354-356, doi:10.1007/BF02165411.
/// @see strassen_multiply()
StrassenRun
strassen_run(const StrassenMatrix& a, const StrassenMatrix& b, const StrassenOptions& options = {});
/// @brief Multiply through the actual padded seven-product producer with postorder block capture.
/// Finite rectangular operands with matching inner dimensions are required. Empty compatible shapes produce an owned zero-shaped result. Pads to a power-of-two square; positive leaf_size defaults64. Seven recursive products per internal call, eighteen half-block additions/subtractions. O(s^log2(7)) arithmetic for fixed crossover, with padding and substantial temporary storage; this is not a runtime speed guarantee. Default2x2 uses eight schoolbook products. Geometry and complete leaf-work checked before allocation; actual block/capture budgets checked during execution. Policies are event/value counts, not peak bytes. Floating rounding/overflow applies; no shared state or caller mutation.
/// @param a Borrowed finite rectangular left matrix m by k.
/// @param b Borrowed finite rectangular right matrix k by n.
/// @param options Positive crossover and complete geometry/work/capture policies; zero disables budgets.
/// @return Owned product, actual executed counts and optional postorder block states.
/// @throws std::invalid_argument Ragged/nonfinite operands, incompatible dimensions or zero crossover.
/// @throws std::length_error Geometry, complete/actual work, capture or counter capacity exceeded.
/// @throws std::overflow_error Actual arithmetic is not finite.
/// @par Example
/// @code
/// auto r=dm::algorithms::strassen_trace({{2}},{{3}});
/// std::print("{} {}\n",r.product[0][0],r.scalar_products); // 6 1
/// @endcode
/// @par References
/// - V. Strassen, Gaussian Elimination is Not Optimal, Numerische Mathematik13 (1969),354-356, doi:10.1007/BF02165411.
/// @see strassen_multiply()
StrassenRun strassen_trace(
    const StrassenMatrix&  a,
    const StrassenMatrix&  b,
    const StrassenOptions& options = {});
/// @brief Multiply with the historical default crossover and empty result for incompatible dimensions.
/// Finite rectangular operands with matching inner dimensions are required. Empty compatible shapes produce an owned zero-shaped result. Pads to a power-of-two square; positive leaf_size defaults64. Seven recursive products per internal call, eighteen half-block additions/subtractions. O(s^log2(7)) arithmetic for fixed crossover, with padding and substantial temporary storage; this is not a runtime speed guarantee. Default2x2 uses eight schoolbook products. Geometry and complete leaf-work checked before allocation; actual block/capture budgets checked during execution. Policies are event/value counts, not peak bytes. Floating rounding/overflow applies; no shared state or caller mutation. After validating complete operands, incompatible dimensions return an empty matrix for compatibility.
/// @param a Borrowed finite rectangular left operand.
/// @param b Borrowed finite rectangular right operand.
/// @return Owned product, or empty on incompatible inner dimensions.
/// @throws std::invalid_argument Ragged or nonfinite operand.
/// @throws std::length_error Default geometry/work budget exceeded.
/// @throws std::overflow_error Nonfinite actual arithmetic.
/// @par Example
/// @code
/// auto r=dm::algorithms::strassen_multiply({{2}},{{3}});
/// std::print("{}\n",r[0][0]); // 6
/// @endcode
/// @see strassen_run()
StrassenMatrix strassen_multiply(const StrassenMatrix& a, const StrassenMatrix& b);
/// @brief Generate the two original100 by100 uniform matrix operands.
/// Local mt19937 seed1 with uniform[-1,1]; generates all left entries before all right entries in row-major order. Owned deterministic fixture with no shared state.
/// @return Two original finite100 by100 operands in left/right order.
/// @par Example
/// @code
/// auto p=dm::algorithms::strassen_example_matrices();
/// std::print("{} {}\n",p.size(),p[0].size()); // 2 100
/// @endcode
/// @see strassen_run()
std::vector<StrassenMatrix> strassen_example_matrices();
/// @brief Compute a checked rectangular schoolbook reference product with actual counts.
/// Uses the same finite validation and scalar multiply/add ordering as the leaf kernel, directly at unpadded dimensions. Complete m*k*n checked against max_scalar_products before execution; other Strassen-only geometry/capture policies are unused. Owned result, no shared state.
/// @param a Borrowed finite rectangular m by k operand.
/// @param b Borrowed finite rectangular k by n operand.
/// @param options Complete scalar-product budget; default100million.
/// @return Owned unpadded product and actual scalar/accumulation counts; no recursive block records.
/// @throws std::invalid_argument Ragged/nonfinite operand or incompatible dimensions.
/// @throws std::length_error Complete product work or capacity exceeded.
/// @throws std::overflow_error Actual arithmetic nonfinite.
/// @par Example
/// @code
/// auto r=dm::algorithms::strassen_schoolbook_run({{2}},{{3}});
/// std::print("{}\n",r.product[0][0]); // 6
/// @endcode
/// @see strassen_run()
StrassenRun strassen_schoolbook_run(
    const StrassenMatrix&  a,
    const StrassenMatrix&  b,
    const StrassenOptions& options = {});
} // namespace datamunge::algorithms
