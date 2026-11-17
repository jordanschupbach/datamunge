#pragma once
/// @file lagrange.hpp
/// @brief Checked direct Lagrange evaluation with actual product/sum observations.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual selected direct-product operations excluding validation, allocation and time.
/// On complete n-node evaluation, terms=sum_additions=n and ratio_evaluations=
/// term_multiplications=n*(n-1). Empty uses the historical zero convention.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lagrange_analyze({0,1},{0,1},.5);
/// std::print("{} {}\n",a.work.terms,a.work.ratio_evaluations); // 2 2
/// @endcode
/// @see lagrange_analyze()
struct LagrangeWork {
  std::size_t terms{0};                ///< Actual final node terms produced.
  std::size_t ratio_evaluations{0};    ///< Actual non-diagonal basis ratios computed.
  std::size_t term_multiplications{0}; ///< Actual term-by-ratio multiplications.
  std::size_t sum_additions{0};        ///< Actual sequential additions of final terms.
};
/// @brief Owned direct interpolation value and actual same-execution selected work.
/// @par Example
/// @code
/// std::print("{:.2f}\n",dm::algorithms::lagrange_analyze({-1,0,1},{2,1,2},.5).value); // 1.25
/// @endcode
/// @see LagrangeTrace
struct LagrangeAnalysis {
  double       value{0}; ///< Actual sequential double polynomial evaluation.
  LagrangeWork work;     ///< Actual selected producing counters.
};
/// @brief Actual ratio and pre/post product from one non-diagonal factor application.
/// term_index identifies the original node term; excluded_node identifies its
/// k!=j factor. widened_differences marks fallback when a double subtraction
/// overflowed; ordinary finite-difference arithmetic retains historical order.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lagrange_trace({0,1},{0,1},.5);
/// std::print("{} {}\n",t.factors[0].term_index,t.factors[0].excluded_node); // 0 1
/// @endcode
/// @see LagrangeTrace
struct LagrangeFactor {
  std::size_t term_index{0};    ///< Actual original term j.
  std::size_t excluded_node{0}; ///< Actual original factor k, different from j.
  double      ratio{0};         ///< Actual finite double ratio multiplied into the running term.
  double      term_before{0};   ///< Actual product before this multiplication, initially y[j].
  double      term_after{0};    ///< Actual product after this multiplication.
  bool        widened_differences{
      false}; ///< Whether nonfinite double differences required widened fallback.
};
/// @brief Actual final node contribution and sequential sum boundary values.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lagrange_trace({0,1},{0,1},.5);
/// std::print("{} {}\n",t.terms[1].value,t.terms[1].sum_after); // 0.5 0.5
/// @endcode
/// @see LagrangeTrace
struct LagrangeTerm {
  std::size_t node{0};       ///< Actual original term identity.
  double      value{0};      ///< Actual final y[j]-weighted product.
  double      sum_before{0}; ///< Actual sequential sum before adding this term.
  double      sum_after{0};  ///< Actual sequential sum after adding this term.
};
/// @brief Owned actual original nodes, factor products, term sums and final evaluation.
/// captured_entries=2*n+n+n*(n-1)=n*n+2*n counts original x/y values plus term and
/// factor records, excluding scalar fields/capacities/runtime copies/bytes/RAM.
/// Complete input/count/optional quota admitted before copies/evaluation. No
/// second interpolation or reconstructed products; all histories own their data.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lagrange_trace({-1,0,1},{2,1,2},.5);
/// std::print("{} {}\n",t.terms.size(),t.factors.size()); // 3 6
/// @endcode
/// @see lagrange_trace()
struct LagrangeTrace {
  std::vector<double> xs;       ///< Owned original finite distinct abscissae, original order.
  std::vector<double> ys;       ///< Owned matched finite original values.
  double              query{0}; ///< Actual finite query, interpolation/extrapolation both allowed.
  double              value{0}; ///< Actual finite sequential double evaluation.
  LagrangeWork        work;     ///< Actual same-execution selected operations.
  std::vector<LagrangeFactor> factors; ///< Actual chronological j-then-k non-diagonal products.
  std::vector<LagrangeTerm> terms; ///< Actual chronological final contributions and sum boundaries.
  std::size_t captured_entries{0}; ///< Complete original-value+factor+term record count.
};
/// @brief Evaluate the direct Lagrange interpolating polynomial at a finite query.
/// Require matched finite x/y arrays and pairwise distinct x values, including
/// signed-zero equality; nodes may be unsorted. Empty returns historical0, even
/// though it does not define a unique polynomial. Query must be finite even then.
/// For each j, start term=y[j], multiply(x-x[k])/(x[j]-x[k]) in original k order,
/// then add to double sum. Preserve ordinary finite-difference arithmetic/order;
/// if either subtraction is nonfinite, compute both differences/ratio in long
/// double then round ratio once. Ratio, every product and every sum must remain
/// finite double or throw: this is not a robust solver for all finite data.
/// No exact-node shortcut/barycentric caching, conditioning guarantee or error
/// estimate. O(n*n) work including duplicate validation, O(1) kernel state beyond
/// borrowed inputs; no optional history/quota/mutation/shared state.
/// @param xs Borrowed complete finite distinct node coordinates, possibly empty.
/// @param ys Borrowed complete matched finite node values.
/// @param x Finite evaluation coordinate, may lie outside the node range.
/// @return Actual finite double evaluation, historical0 on empty inputs.
/// @throws std::invalid_argument On shape/nonfinite query/data or duplicate abscissae.
/// @throws std::length_error On actual operation count overflow.
/// @throws std::overflow_error On nonfinite ratio, product or sequential sum.
/// @par Example
/// @code
/// std::print("{:.2f}\n",dm::algorithms::lagrange_interpolate({-1,0,1},{2,1,2},.5)); // 1.25
/// @endcode
/// @par References
/// - Berrut and Trefethen(2004), Barycentric Lagrange Interpolation, SIAM Review46(3),501–517, https://doi.org/10.1137/S0036144502417715.
///   This API deliberately evaluates the direct product form, not the paper's barycentric algorithm.
/// @see lagrange_analyze(), lagrange_trace()
double lagrange_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x);
/// @brief Return the actual direct interpolation value and selected producing work without history.
/// Same validation/arithmetic/finite-intermediate and O(n*n) contracts as
/// lagrange_interpolate(); no input copies, optional quota or repeated evaluation.
/// @param xs Borrowed complete distinct finite nodes.
/// @param ys Borrowed complete matched finite values.
/// @param x Actual finite evaluation coordinate.
/// @return Owned actual value and selected operation counts.
/// @throws std::invalid_argument On shape/finite/duplicate violations.
/// @throws std::length_error On work-count overflow.
/// @throws std::overflow_error On nonfinite numeric intermediate/output.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::lagrange_analyze({}, {},0).value); // 0
/// @endcode
/// @par References
/// - Berrut and Trefethen(2004), Barycentric Lagrange Interpolation, https://doi.org/10.1137/S0036144502417715.
/// @see lagrange_interpolate(), LagrangeWork
LagrangeAnalysis
lagrange_analyze(const std::vector<double>& xs, const std::vector<double>& ys, double x);
/// @brief Capture every actual direct-product ratio/multiplication and final term addition once.
/// Same input/arithmetic contracts as lagrange_interpolate(). Complete n*n+2*n
/// record count is checked before original copies and full history allocation;
/// default million-entry optional quota, zero disables only that ceiling.
/// O(n*n) owned capture; quota counts records rather than bytes/peak memory.
/// @param xs Borrowed complete distinct finite original abscissae.
/// @param ys Borrowed complete matched finite original values.
/// @param x Actual finite evaluation coordinate.
/// @param max_entries Complete original-value/factor/term ceiling, default1000000, zero off.
/// @return Owned complete actual observations, never partial result or prefix replay.
/// @throws std::invalid_argument On complete input/query violations before copies.
/// @throws std::length_error On complete count/quota/container limits.
/// @throws std::overflow_error On nonfinite numeric intermediate/output.
/// @throws std::bad_alloc On original/history allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lagrange_trace({0,1},{0,1},.5,8);
/// std::print("{}\n",t.captured_entries); // 8
/// @endcode
/// @par References
/// - Berrut and Trefethen(2004), Barycentric Lagrange Interpolation, https://doi.org/10.1137/S0036144502417715.
/// @see LagrangeTrace, lagrange_analyze()
LagrangeTrace lagrange_trace(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     x,
    std::size_t                max_entries = 1000000);
} // namespace datamunge::algorithms
