#pragma once
/// @file toom3.hpp
/// @brief Checked exact Toom-3 coefficient arithmetic and actual recursive interpolation states.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own ToomOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ToomOptions r{};
/// (void)r;
/// @endcode
struct ToomOptions {
  std::size_t max_digits{
      10000};                ///< Complete supplied decimal bytes per operand; zero disables policy.
  std::size_t toom_leaf{48}; ///< Positive maximum coefficient width sent to Karatsuba.
  std::size_t karatsuba_leaf{32}; ///< Positive maximum width sent to schoolbook multiplication.
  std::size_t max_operations{
      100000000}; ///< Actual scalar coefficient arithmetic events, including products and carry steps; zero disables.
  std::size_t max_records{100000}; ///< Actual captured recursive return records; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual record plus serialized coefficient values retained; zero disables.
};
/// @brief Own ToomStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ToomStep r{};
/// (void)r;
/// @endcode
struct ToomStep {
  std::string kind;    ///< Actual schoolbook, karatsuba or toom3 branch.
  std::size_t depth{}; ///< Actual root-relative recursive depth.
  std::size_t split{}; ///< Actual coefficient chunk width, zero for schoolbook.
  std::vector<std::string>
      left; ///< Actual exact little-endian input coefficients as signed decimal strings.
  std::vector<std::string> right; ///< Actual exact little-endian second input coefficients.
  std::vector<std::vector<std::string>>
      products; ///< Actual Toom products at0,1,-1,2,infinity; empty for other branches.
  std::vector<std::vector<std::string>>
      interpolated; ///< Actual Toom c0..c4 coefficient vectors; empty for other branches.
  std::vector<std::string>
      result; ///< Actual returned exact little-endian coefficient convolution before decimal carry.
};
/// @brief Own ToomRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ToomRun r{};
/// (void)r;
/// @endcode
struct ToomRun {
  std::string value;              ///< Actual normalized nonnegative decimal product.
  std::size_t toom_calls{};       ///< Actual five-product internal Toom calls.
  std::size_t karatsuba_calls{};  ///< Actual three-product internal Karatsuba calls.
  std::size_t schoolbook_calls{}; ///< Actual scalar-convolution leaf calls.
  std::size_t scalar_products{};  ///< Actual schoolbook coefficient multiplications.
  std::size_t
      operations{}; ///< Actual selected coefficient arithmetic events, including products/carries.
  std::vector<ToomStep> steps; ///< Optional actual postorder recursive states.
};
/// @brief Multiply through actual exact checked Toom/Karatsuba/schoolbook branches.
/// Accept ASCII nonnegative decimal strings, including leading zeros and the legacy empty-string zero convention; reject signs/whitespace/other bytes. Complete byte lengths and positive crossover thresholds checked before parsing. Coefficients are private checked signed128 integers; every add/subtract/multiply is overflow-checked and interpolation divisions must be exact. Split into three coefficient blocks and evaluate at0,1,-1,2,infinity, multiply five times recursively, interpolate c0..c4 and shift/add them before decimal carry. The actual evaluation point is+2, correcting the old private comment saying-2. Default Toom threshold48 delegates to Karatsuba threshold32, so small examples do not execute five products. Fixed crossovers give O(n^log3(5)) arithmetic asymptotically, with smaller-size Karatsuba/schoolbook work and substantial temporary storage. Actual checked coefficient capacity/work budgets can reject large runs; this is not an unlimited-size integer package. Owned results/histories and no shared state.
/// @param a Borrowed complete ASCII decimal operand; empty means0.
/// @param b Borrowed complete second decimal operand.
/// @param options Complete digits, positive crossovers and actual arithmetic/capture policies.
/// @return Owned actual decimal product, selected arithmetic counts and optional postorder coefficient states.
/// @throws std::invalid_argument Malformed decimal bytes or zero crossover.
/// @throws std::length_error Complete digit geometry or actual arithmetic/history/counter capacity exceeded.
/// @throws std::overflow_error Actual checked signed128 coefficient arithmetic exceeds capacity.
/// @throws std::logic_error Interpolation division is not exact or final carry is negative.
/// @par Example
/// @code
/// auto r=dm::algorithms::toom3_run("123","456");
/// std::print("{}\n",r.value); // 56088
/// @endcode
/// @see toom3_mul()
ToomRun toom3_run(const std::string& a, const std::string& b, const ToomOptions& options = {});
/// @brief Multiply through actual exact checked Toom/Karatsuba/schoolbook branches.
/// Accept ASCII nonnegative decimal strings, including leading zeros and the legacy empty-string zero convention; reject signs/whitespace/other bytes. Complete byte lengths and positive crossover thresholds checked before parsing. Coefficients are private checked signed128 integers; every add/subtract/multiply is overflow-checked and interpolation divisions must be exact. Split into three coefficient blocks and evaluate at0,1,-1,2,infinity, multiply five times recursively, interpolate c0..c4 and shift/add them before decimal carry. The actual evaluation point is+2, correcting the old private comment saying-2. Default Toom threshold48 delegates to Karatsuba threshold32, so small examples do not execute five products. Fixed crossovers give O(n^log3(5)) arithmetic asymptotically, with smaller-size Karatsuba/schoolbook work and substantial temporary storage. Actual checked coefficient capacity/work budgets can reject large runs; this is not an unlimited-size integer package. Owned results/histories and no shared state.
/// @param a Borrowed complete ASCII decimal operand; empty means0.
/// @param b Borrowed complete second decimal operand.
/// @param options Complete digits, positive crossovers and actual arithmetic/capture policies.
/// @return Owned actual decimal product, selected arithmetic counts and optional postorder coefficient states.
/// @throws std::invalid_argument Malformed decimal bytes or zero crossover.
/// @throws std::length_error Complete digit geometry or actual arithmetic/history/counter capacity exceeded.
/// @throws std::overflow_error Actual checked signed128 coefficient arithmetic exceeds capacity.
/// @throws std::logic_error Interpolation division is not exact or final carry is negative.
/// @par Example
/// @code
/// auto r=dm::algorithms::toom3_trace("123","456");
/// std::print("{}\n",r.value); // 56088
/// @endcode
/// @see toom3_mul()
ToomRun toom3_trace(const std::string& a, const std::string& b, const ToomOptions& options = {});
/// @brief Preserve checked decimal Toom-3 multiplication with default crossovers.
/// Accept ASCII nonnegative decimal strings, including leading zeros and the legacy empty-string zero convention; reject signs/whitespace/other bytes. Complete byte lengths and positive crossover thresholds checked before parsing. Coefficients are private checked signed128 integers; every add/subtract/multiply is overflow-checked and interpolation divisions must be exact. Split into three coefficient blocks and evaluate at0,1,-1,2,infinity, multiply five times recursively, interpolate c0..c4 and shift/add them before decimal carry. The actual evaluation point is+2, correcting the old private comment saying-2. Default Toom threshold48 delegates to Karatsuba threshold32, so small examples do not execute five products. Fixed crossovers give O(n^log3(5)) arithmetic asymptotically, with smaller-size Karatsuba/schoolbook work and substantial temporary storage. Actual checked coefficient capacity/work budgets can reject large runs; this is not an unlimited-size integer package. Owned results/histories and no shared state.
/// @param a Borrowed complete ASCII decimal operand; empty means0.
/// @param b Borrowed complete second decimal operand.
/// @return Owned normalized exact decimal product.
/// @throws std::invalid_argument Malformed decimal bytes.
/// @throws std::length_error Default digits/work or capacity exceeded.
/// @throws std::overflow_error Actual coefficient capacity exceeded.
/// @throws std::logic_error Unexpected exact-interpolation/carry invariant failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::toom3_mul("123","456")); // 56088
/// @endcode
/// @see toom3_run()
std::string toom3_mul(const std::string& a, const std::string& b);
} // namespace datamunge::algorithms
