#pragma once
/// @file gram_schmidt.hpp
/// @brief Checked modified Gram–Schmidt, selected work and actual owned projection records.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief Absolute residual acceptance and complete input/optional capture limits.
/// The threshold is an order- and scale-dependent heuristic, not a numerical rank
/// certificate. Pass two repeats the modified projection sweep before one final
/// normalization; it does not change the threshold or promise exact orthogonality.
/// @par Example
/// @code
/// dm::algorithms::GramSchmidtOptions o;
/// std::print("{} {}\n",o.tolerance,o.passes); // 1e-12 1
/// @endcode
/// @see gram_schmidt_run(), gram_schmidt_trace()
struct GramSchmidtOptions {
  double tolerance{
      1e-12}; ///< Finite nonnegative absolute residual norm threshold; accept strictly above.
  unsigned    passes{1}; ///< One or two projection passes per input row; default one.
  std::size_t max_input_entries{
      1000000}; ///< Complete original rows plus coordinates; zero disables cap.
  std::size_t max_trace_entries{
      1000000}; ///< Actual retained floating values, as defined by GramSchmidtTrace.
};
/// @brief Selected operations in the producing native loop, not timings or all CPU work.
/// Dot products count one request per projection plus one residual self-dot per
/// input row. Products/additions count d terms per dot, including the initial
/// zero addition. Updates count d products and d subtractions per projection;
/// divisions count normalization coordinates of accepted rows only.
/// @par Example
/// @code
/// const auto r=dm::algorithms::gram_schmidt_run({{1,0},{1,1}});
/// std::print("{} {}\n",r.work.projections,r.work.dot_products); // 1 3
/// @endcode
/// @see GramSchmidtRun
struct GramSchmidtWork {
  std::size_t rows{0};         ///< Input row count.
  std::size_t dimension{0};    ///< Common row width; zero on empty input.
  std::size_t projections{0};  ///< Actual projection/subtract steps over all passes.
  std::size_t dot_products{0}; ///< Actual scalar dot requests, including residual norms.
  std::size_t dot_terms{0}; ///< Actual products and additions in those dot requests, d per request.
  std::size_t update_terms{0}; ///< Actual projection-product/subtraction pairs, d per projection.
  std::size_t divisions{0};    ///< Actual normalization divisions.
};
/// @brief Owned accepted row vectors, their original row identities and actual work.
/// Rows are accepted when their computed residual norm is strictly above the
/// absolute tolerance. They approximate an orthonormal set; one pass can lose
/// orthogonality or accept dependent rounding noise. No exact span/rank guarantee.
/// Empty input returns empty; rectangular zero-width rows are all skipped.
/// No references to caller input survive. Copying owns all nested storage.
/// @par Example
/// @code
/// const auto r=dm::algorithms::gram_schmidt_run({{1,0},{2,0},{0,1}});
/// std::print("{} {}\n",r.basis.size(),r.input_rows[1]); // 2 2
/// @endcode
/// @see gram_schmidt_run(), GramSchmidtTrace
struct GramSchmidtRun {
  std::vector<std::vector<double>>
                           basis; ///< Actual normalized accepted row vectors, in arrival order.
  std::vector<std::size_t> input_rows; ///< Original row indices aligned with basis rows.
  GramSchmidtWork work; ///< Selected actual work, excluding validation/capture/allocations.
};
/// @brief One actual modified projection, with owned complete before/after residuals.
/// coefficient is the dot of before with the current accepted basis row, not a
/// replay from the original input. Pass indices are zero based. All vectors own
/// their values; basis identifies a row in trace.run.basis.
/// @par Example
/// @code
/// const auto t=dm::algorithms::gram_schmidt_trace({{1,0},{1,1}});
/// std::print("{} {}\n",t.projections[0].coefficient,t.projections[0].after[1]); // 1 1
/// @endcode
/// @see GramSchmidtTrace
struct GramSchmidtProjection {
  std::size_t         input_row{0};   ///< Current original row index.
  std::size_t         basis_row{0};   ///< Producing accepted basis index.
  unsigned            pass{0};        ///< Producing pass, zero or one.
  double              coefficient{0}; ///< Actual rounded dot(before,basis row).
  std::vector<double> before;         ///< Owned actual residual read by the dot request.
  std::vector<double> after;          ///< Owned actual residual after the subtraction loop.
};
/// @brief Actual final residual and acceptance decision for one original row.
/// residual is stored before normalization. accepted_rows counts basis rows
/// immediately after the decision, including this row if accepted.
/// @par Example
/// @code
/// const auto t=dm::algorithms::gram_schmidt_trace({{3,4}});
/// std::print("{} {}\n",t.rows[0].norm,t.rows[0].accepted); // 5 true
/// @endcode
/// @see GramSchmidtTrace
struct GramSchmidtRow {
  std::size_t         input_row{0};     ///< Original row index.
  std::vector<double> residual;         ///< Owned final actual unnormalized residual.
  double              norm{0};          ///< Actual sqrt of its sequential double self-dot.
  bool                accepted{false};  ///< Actual strict threshold decision.
  std::size_t         accepted_rows{0}; ///< Actual accepted basis count after this decision.
};
/// @brief Owned actual projections, row decisions and result from one shared execution.
/// Capture performs no arithmetic replay. Floating-value quota units are
/// 1+2d per projection (coefficient, before and after) and 1+d per row decision
/// (norm and residual). Metadata, ordinary basis/output/work, capacities and
/// allocation overhead are excluded: this is not a byte or peak-RAM budget.
/// Online admission precedes each dot request and capture vector copies; zero
/// disables the policy cap. Failure returns no partial trace. With full rank,
/// O(k²d) retained history can be much larger than O(kd) ordinary working/output
/// storage. Caller inputs are never retained or mutated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::gram_schmidt_trace({{1,0},{1,1}});
/// std::print("{}\n",t.captured_entries); // 11
/// @endcode
/// @see gram_schmidt_trace(), GramSchmidtProjection, GramSchmidtRow
struct GramSchmidtTrace {
  GramSchmidtRun                     run; ///< Actual producing basis, source identities and work.
  std::vector<GramSchmidtProjection> projections; ///< Actual ordered projection records.
  std::vector<GramSchmidtRow> rows; ///< Actual ordered final row decisions, including skipped rows.
  std::size_t captured_entries{0};  ///< Retained floating-value units, excluding all metadata.
};
/// @brief Apply the preserved one-pass modified ordering and absolute 1e-12 residual test.
/// Rows are input vectors. Sequential double dot products, projection updates and
/// sqrt(sum of squares) retain ordinary original results; scaling can change
/// acceptance and rounding can lose orthogonality. No pivoting, relative rank
/// test, robust scaled norm or exact-span certificate is supplied. Validates
/// rectangular shape and finite inputs before internal copying/arithmetic;
/// nonfinite intermediate arithmetic throws instead of returning invalid basis
/// vectors. Empty/zero-width semantics follow GramSchmidtRun. Independent calls
/// share no state; concurrent mutation of borrowed input is outside the contract.
/// O(k²d) selected arithmetic, O(kd) working/output storage, no capture or caps.
/// @param vectors Borrowed rectangular finite row vectors, unmodified.
/// @return Owned accepted normalized vectors in original arrival order.
/// @throws std::invalid_argument On ragged/nonfinite input.
/// @throws std::overflow_error On nonfinite intermediate arithmetic.
/// @throws std::length_error On work/container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto q=dm::algorithms::gram_schmidt({{2,1},{1,3}});
/// std::print("{:.6f} {:.6f}\n",q[0][0],q[1][0]); // 0.894427 -0.447214
/// @endcode
/// @par References
/// - L. Giraud, J. Langou, M. Rozložník (2005), The loss of orthogonality in the
///   Gram-Schmidt orthogonalization process,
///   https://doi.org/10.1016/j.camwa.2005.08.009 . This routine implements sequential
///   modified sweeps with an absolute skip test.
/// @see gram_schmidt_run(), gram_schmidt_trace()
std::vector<std::vector<double>> gram_schmidt(const std::vector<std::vector<double>>& vectors);
/// @brief Execute the shared modified kernel with options, source rows and selected work.
/// Same arithmetic/limitations as gram_schmidt(); options allow a second full
/// projection sweep and a different absolute threshold. Complete rectangular and
/// finite validation, plus option validation, precede input quota and arithmetic.
/// The complete input cap counts k+kd entries, including zero-width row headers;
/// zero disables it. Trace quota is ignored. No histories or input references.
/// @param vectors Borrowed rectangular finite input rows.
/// @param options Finite nonnegative tolerance, passes one/two and complete input cap.
/// @return Owned accepted vectors, original row IDs and selected actual work.
/// @throws std::invalid_argument On malformed input/options, before caps.
/// @throws std::overflow_error On nonfinite intermediate arithmetic.
/// @throws std::length_error On input cap, checked counts or container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// dm::algorithms::GramSchmidtOptions o; o.passes=2;
/// const auto r=dm::algorithms::gram_schmidt_run({{1,0},{1,1}},o);
/// std::print("{}\n",r.work.projections); // 2
/// @endcode
/// @see GramSchmidtOptions, GramSchmidtRun
GramSchmidtRun gram_schmidt_run(
    const std::vector<std::vector<double>>& vectors,
    GramSchmidtOptions                      options = {});
/// @brief Capture actual before/coefficient/after projections and final row decisions online.
/// Same kernel, accepted basis, work and input rules as run; capture performs no
/// extra dot products. Actual quota units and ownership follow GramSchmidtTrace.
/// Row-decision admission occurs before the residual self-dot; projection
/// admission before its coefficient dot and before copying state. No partial
/// history is returned on failure; zero trace cap disables only that policy cap.
/// @param vectors Borrowed finite rectangular input, unmodified.
/// @param options Input/capture caps and arithmetic options, defaults as documented.
/// @return Owned actual arithmetic records, row decisions and producing result.
/// @throws std::invalid_argument On malformed input/options, before caps.
/// @throws std::overflow_error On nonfinite intermediate arithmetic.
/// @throws std::length_error On input/capture cap, work or container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::gram_schmidt_trace({{1,0},{2,0}});
/// std::print("{} {}\n",t.rows.size(),t.rows[1].accepted); // 2 false
/// @endcode
/// @see GramSchmidtTrace, gram_schmidt_run()
GramSchmidtTrace gram_schmidt_trace(
    const std::vector<std::vector<double>>& vectors,
    GramSchmidtOptions                      options = {});
} // namespace datamunge::algorithms
