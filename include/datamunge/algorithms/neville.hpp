#pragma once
/// @file neville.hpp
/// @brief Checked Neville interpolation with actual in-place triangular-update observations.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Optional complete node/history policies; zero disables either.
/// @par Example
/// @code
/// dm::algorithms::NevilleOptions o;std::print("{}\n",o.max_trace_entries); // 1000000
/// @endcode
/// @see neville_trace()
struct NevilleOptions {
  std::size_t max_nodes{
      1000000}; ///< Complete input node cap, ignored by ordinary convenience entry point.
  std::size_t max_trace_entries{
      1000000}; ///< Complete owned logical capture cap, ignored without tracing.
};
/// @brief Actual interpolation result and selected same-producer operation counts.
/// @par Example
/// @code
/// auto a=dm::algorithms::neville_analyze({-1,0,1},{1,0,1},.5);std::print("{} {}\n",a.value,a.updates); // 0.25 3
/// @endcode
/// @see neville_analyze()
struct NevilleAnalysis {
  double      value{0};   ///< Actual final in-place value, historical0 for empty input.
  std::size_t nodes{0};   ///< Actual complete node count.
  std::size_t updates{0}; ///< Actual recurrence updates, n(n-1)/2; excludes complete validation.
  std::size_t widened_updates{
      0}; ///< Actual updates requiring long-double fallback for nonfinite double intermediates.
};
/// @brief Actual operands and committed value of one in-place Neville update.
/// Fill degree increases1..n-1; ending index j decreases n-1..degree so both
/// operands still belong to the preceding degree. No report-side replay.
/// @par Example
/// @code
/// auto t=dm::algorithms::neville_trace({0,1},{0,2},.25);std::print("{}\n",t.steps[0].value); // 0.5
/// @endcode
/// @see NevilleTrace
struct NevilleStep {
  std::size_t degree{0};       ///< Actual degree of combined interpolant.
  std::size_t first{0};        ///< Actual original start index j-degree.
  std::size_t last{0};         ///< Actual original end index j.
  double      right_before{0}; ///< Actual old p[j], interpolant over first+1..last.
  double      left_before{0};  ///< Actual old p[j-1], interpolant over first..last-1.
  double      value{0};        ///< Actual committed p[j].
  bool        widened{false};  ///< Actual widened fallback route.
};
/// @brief Owned original nodes/query, actual update history and final working vector.
/// Capture counts query1, inputs2n, final working values n and every actual
/// update record n(n-1)/2. Complete units1+3n+n(n-1)/2 are pre-admitted before
/// copies/recurrence. Fixed record fields are one logical unit, not bytes.
/// @par Example
/// @code
/// auto t=dm::algorithms::neville_trace({0,1},{0,2},.25);std::print("{}\n",t.captured_entries); // 8
/// @endcode
/// @see neville_trace()
struct NevilleTrace {
  std::vector<double>      xs;       ///< Owned original distinct abscissae in caller order.
  std::vector<double>      ys;       ///< Owned matched original values in caller order.
  double                   query{0}; ///< Original finite evaluation point.
  NevilleAnalysis          analysis; ///< Actual result and selected work.
  std::vector<NevilleStep> steps;    ///< Actual degree/in-place update order.
  std::vector<double>
      final_working; ///< Actual final n-slot vector, including mixed-degree retained values.
  std::size_t captured_entries{0}; ///< Complete admitted logical history units.
};
/// @brief Evaluate a finite distinct-node polynomial through the original in-place recurrence.
/// Complete matched finite xs/ys, finite query and pairwise distinct nodes are
/// validated first; arbitrary node order, empty and singleton inputs allowed.
/// Returns historical0 for empty arrays. No input mutation/retained references.
/// Ordinary finite double intermediates retain the legacy arithmetic/order;
/// nonfinite double differences/products trigger a long-double update fallback.
/// Any nonfinite final double update throws. No general stability ordering versus
/// Lagrange, monotone degree convergence or certified error estimate is promised.
/// O(n squared) work including complete pairwise distinctness validation; O(n)
/// working storage. Ordinary entry point disables optional policies.
/// @param xs Original distinct finite nodes, any order.
/// @param ys Matching finite observations.
/// @param x Finite query, including extrapolation.
/// @return Actual polynomial value, or the explicit empty-input0 convention.
/// @throws std::invalid_argument If complete shape/finite/distinctness checks fail.
/// @throws std::length_error If complete recurrence/container geometry overflows.
/// @throws std::overflow_error If an update cannot be represented by finite double.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::neville_interpolate({-1,0,1},{1,0,1},.5)); // 0.25
/// @endcode
/// @par References
/// - E. H. Neville(1934), Iterative interpolation, Journal of the Indian Mathematical Society20,87–120.
/// @see neville_analyze(), neville_trace()
double neville_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x);
/// @brief Return the actual same-producer value and recurrence counters.
/// Same complete domain/fallback semantics as neville_interpolate(); optional
/// complete node cap applies before working allocation, after input validation.
/// @param xs Original finite distinct nodes.
/// @param ys Matching finite values.
/// @param x Finite query.
/// @param options Complete node cap, default1000000, zero disables; capture cap ignored.
/// @return Actual result, nodes, updates and fallback count.
/// @throws std::invalid_argument If complete input validation fails.
/// @throws std::length_error If count geometry/node policy is exceeded.
/// @throws std::overflow_error If an update cannot be represented finitely.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::neville_analyze({0,1},{0,2},.5).updates); // 1
/// @endcode
/// @see NevilleAnalysis
NevilleAnalysis neville_analyze(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     x,
    NevilleOptions             options = {});
/// @brief Capture every actual recurrence operand and committed update.
/// Same complete domain as neville_interpolate(). All history units are known
/// upfront and admitted before copying/evaluation. No partial result/truncation.
/// @param xs Original complete distinct finite nodes, owned by returned trace.
/// @param ys Original matching finite values, owned by returned trace.
/// @param x Finite original query.
/// @param options Complete node/history limits, defaults1000000, zero disables each.
/// @return Owned complete actual history, working values and same-producer result.
/// @throws std::invalid_argument If complete input validation fails before policies.
/// @throws std::length_error If geometry or either complete policy is exceeded.
/// @throws std::overflow_error If an actual update is nonfinite after fallback.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::neville_trace({}, {},.5).analysis.value); // 0
/// @endcode
/// @see NevilleTrace
NevilleTrace neville_trace(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     x,
    NevilleOptions             options = {});
/// @brief Original-node-order prefix study with a known analytic comparison.
/// Differences compare successive prefix values; first difference0 is a
/// placeholder with no predecessor. Neither differences nor actual finite
/// reference errors are an a posteriori certificate for arbitrary data.
/// @par Example
/// @code
/// auto s=dm::algorithms::neville_cosine_study();std::print("{}\n",s.degrees.size()); // 6
/// @endcode
/// @see neville_cosine_study(), neville_zero_sample_study()
struct NevilleStudy {
  std::vector<double> xs;       ///< Owned complete original nodes.
  std::vector<double> ys;       ///< Owned complete original samples.
  double              query{0}; ///< Actual common query.
  double              truth{0}; ///< Analytic comparison value of the stated generating function.
  std::vector<double> degrees;  ///< Actual prefix degrees1..n-1, exactly integer-valued.
  std::vector<double>
      estimates; ///< Actual canonical prefix evaluations, no nearest-node rearrangement.
  std::vector<double> absolute_errors; ///< Actual abs(estimate-truth).
  std::vector<double>
      successive_changes; ///< Actual absolute adjacent estimate changes, first0 placeholder.
};
/// @brief Preserve the original cosine prefixes0,.5,..,3 and query1.25.
/// First two nodes0,.5 extrapolate; they are not the nearest bracketing nodes.
/// Produces all six prefix calls, O(n cubed) total work for n=7 fixed nodes.
/// @return Owned original table, actual prefix estimates and cos(1.25) reference.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::neville_cosine_study().query); // 1.25
/// @endcode
/// @see NevilleStudy
NevilleStudy neville_cosine_study();
/// @brief Produce a zero-change/error counterexample from f(x)=x(x-1)(x-2).
/// Nodes0,1,2 all have exact sample0, yet f(.5)=.375. Degree1 and2 prefix
/// estimates both0, so their difference0 does not bound the actual error. No fit.
/// @return Owned complete fixture and actual two prefix evaluations.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::neville_zero_sample_study().truth); // 0.375
/// @endcode
/// @see neville_cosine_study()
NevilleStudy neville_zero_sample_study();
} // namespace datamunge::algorithms
