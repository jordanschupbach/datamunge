#pragma once
/// @file linde_buzo_gray.hpp
/// @brief Checked multiplicative-split LBG with actual Lloyd stages and final-codebook-consistent assignment/distortion.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned actual final codebook, nearest assignments and mean squared Euclidean distortion.
/// Final assignment/distortion are re-evaluated against each actually returned
/// stage codebook, correcting the old pre-update objective/stale labels. The
/// multiplicative split can leave zero centroids unchanged and retain empty
/// codewords, so requested size does not imply that many distinct used centers.
/// @par Example
/// @code
/// const auto r=dm::algorithms::linde_buzo_gray({{1},{3}},1);
/// std::print("{} {}\n",r.codebook[0][0],r.distortion); // 2 1
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see linde_buzo_gray_analyze(), linde_buzo_gray_trace()
struct LBGResult {
  std::vector<std::vector<double>>
      codebook; ///< Actual returned centroids; rounded-up power-of-two size for nonempty requested runs.
  std::vector<std::size_t>
      assignment; ///< Actual nearest returned-codebook index per original point; earliest-index ties.
  double distortion{
      0}; ///< Actual mean squared nearest distance to returned codebook,0 on empty result.
  std::vector<double>
      distortion_history; ///< Actual final nearest distortion at codebook sizes1,2,4,..., including initial stage.
};
/// @brief Actual selected producing distance, assignment, centroid and split work.
/// Excludes complete validation, copying/allocation/capture and elapsed time.
/// Trace adds no distance recomputation; final consistency passes occur in all modes.
/// @par Example
/// @code
/// const auto a=dm::algorithms::linde_buzo_gray_analyze({{1},{3}},1);
/// std::print("{}\n",a.work.assignment_passes>=2); // true
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see linde_buzo_gray_analyze(), linde_buzo_gray_trace()
struct LBGWork {
  std::size_t assignment_passes{
      0}; ///< Actual full nearest-codebook passes, including final consistency pass per stage.
  std::size_t point_assignments{0};    ///< Actual nearest-center decisions, n per assignment pass.
  std::size_t distance_evaluations{0}; ///< Actual complete point-center squared distances.
  std::size_t coordinate_squares{0}; ///< Actual squared coordinate contributions inside distances.
  std::size_t centroid_reads{
      0}; ///< Actual coordinate reads for initial/global or Lloyd assigned-centroid sums.
  std::size_t centroid_updates{
      0}; ///< Actual assigned nonempty centroid coordinate writes, including initial mean.
  std::size_t split_writes{0}; ///< Actual multiplied low/high child-coordinate writes.
  std::size_t lloyd_iterations{
      0}; ///< Actual complete nearest-assignment/centroid-update passes across stages.
};
/// @brief Owned actual nearest-codebook predictions, mean squared distortion and selected distance work.
/// @par Example
/// @code
/// const auto q=dm::algorithms::lbg_quantize({{1},{3}},{{0},{4}});
/// std::print("{} {}\n",q.assignment[0],q.assignment[1]); // 0 1
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see linde_buzo_gray_analyze(), linde_buzo_gray_trace()
struct LBGQuantization {
  std::vector<std::size_t>
          assignment;    ///< Actual earliest-index nearest codeword for every original point.
  double  distortion{0}; ///< Actual mean squared Euclidean nearest distance,0 for empty data.
  LBGWork work;          ///< Actual assignment/distance work; centroid/split fields remain0.
};
/// @brief Owned actual LBG result, requested/rounded target and same-execution producing work.
/// @par Example
/// @code
/// const auto a=dm::algorithms::linde_buzo_gray_analyze({{1},{3}},3);
/// std::print("{}\n",a.rounded_size); // 4
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see linde_buzo_gray_analyze(), linde_buzo_gray_trace()
struct LBGAnalysis {
  LBGResult   result; ///< Actual final result with codebook-consistent assignments/objectives.
  std::size_t requested_size{0}; ///< Original requested codeword count.
  std::size_t rounded_size{0};   ///< Actual next-power-of-two target;0 on empty/zero-target no-op.
  LBGWork     work;              ///< Actual selected work from one complete run.
};
/// @brief Owned actual producing Lloyd assignment/update evidence for one real iteration.
/// No post-update distortion is invented for this record; stage final consistency
/// evaluation and the next real iteration provide separate observed objectives.
/// @par Example
/// @code
/// const auto t=dm::algorithms::linde_buzo_gray_trace({{1},{3}},1);
/// std::print("{}\n",t.iterations[0].before_distortion); // 1
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see linde_buzo_gray_analyze(), linde_buzo_gray_trace()
struct LBGIteration {
  std::size_t stage{0};     ///< Actual zero-based doubling stage, initial centroid stage0.
  std::size_t iteration{0}; ///< Actual zero-based Lloyd pass within this stage.
  std::vector<std::vector<double>>
      before; ///< Actual complete codebook used by the nearest-assignment pass.
  std::vector<std::size_t>
         assignment; ///< Actual nearest assignment to before codebook, not necessarily to after.
  double before_distortion{0};     ///< Actual mean nearest squared distortion to before codebook.
  std::vector<std::size_t> counts; ///< Actual number of points assigned to each before codeword.
  std::vector<std::vector<double>>
       after; ///< Actual centroid-updated codebook; empty codewords retain their before values.
  bool tolerance_stop{
      false}; ///< Actual relative pre-update distortion-change condition fired after this committed update.
};
/// @brief Owned actual initial/final codebooks and final nearest consistency result for one size.
/// @par Example
/// @code
/// const auto t=dm::algorithms::linde_buzo_gray_trace({{1},{3}},1);
/// std::print("{}\n",t.stages[0].size); // 1
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see linde_buzo_gray_analyze(), linde_buzo_gray_trace()
struct LBGStage {
  std::size_t size{0}; ///< Actual power-of-two number of codewords in this stage.
  std::vector<std::vector<double>>
      initial; ///< Actual centroid or multiplicatively split starting codebook.
  std::vector<std::vector<double>> final; ///< Actual codebook after its real Lloyd updates.
  std::vector<std::size_t> assignment; ///< Actual final nearest assignments to this final codebook.
  double distortion{0}; ///< Actual final nearest squared distortion, equal to result history entry.
  std::size_t iterations{0}; ///< Actual complete Lloyd updates at this size.
  bool        tolerance_stop{
      false}; ///< True only when the actual relative pre-update comparison fired, false at iteration cap.
};
/// @brief Owned original data and every actual split-stage/assignment/update record from one execution.
/// Complete trace or exception. Actual quota counts n*d input coordinates;
/// each stage record plus2*k*d codebook coordinates+n final assignments;
/// each iteration record plus2*k*d coordinates+n assignments+k counts.
/// Admitted before each record/copy; unknown convergence length is not fabricated.
/// Excludes fields/row headers/capacities/output/runtime copies and bytes/RAM.
/// Default million actual logical records;0 optional cap off.
/// @par Example
/// @code
/// const auto t=dm::algorithms::linde_buzo_gray_trace({{1},{3}},2);
/// std::print("{}\n",t.stages.size()); // 2
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see linde_buzo_gray_analyze(), linde_buzo_gray_trace()
struct LBGTrace {
  std::vector<std::vector<double>> data;     ///< Owned original complete rectangular finite points.
  LBGAnalysis                      analysis; ///< Actual final consistent result/target/work.
  std::vector<LBGStage> stages; ///< Actual completed codebook sizes and final nearest observations.
  std::vector<LBGIteration>
      iterations; ///< Actual chronological Lloyd before/assignment/after records; no prefix reruns.
  std::size_t captured_entries{
      0}; ///< Actual input-coordinate/stage/iteration/codebook-coordinate/assignment/count logical records.
};
/// @brief Fit a checked multiplicative-split, finite-budget generalized Lloyd codebook.
/// Initial center is the global mean. Double via c*(1-epsilon),c*(1+epsilon)
/// until size>=request, preserving the historical rounded-up behavior. Lloyd
/// assigns to earliest nearest center, then updates nonempty means; empty
/// centers retain their values. After at least two passes, stop if previous
/// pre-update distortion minus current pre-update distortion is strictly less
/// than tol*(previous+1e-12), otherwise cap iterations. Final nearest pass at
/// each size makes output labels/objective consistent with returned centroids.
/// Ordinary finite double order retained; centroid sums have widened fallback.
/// Zero-centroid multiplicative splitting may never separate symmetric data.
/// No global optimum, distinct/used centers, nested codebooks, across-size
/// monotonicity, or general convergence/quality guarantee. Empty data valid.
/// Typical O(I*n*K*d) producing work; no input mutation or shared state.
/// @param data Borrowed complete rectangular finite points with positive dimension when nonempty.
/// @param codebook_size Requested size, rounded up to a power of two;0 returns empty result after complete validation.
/// @param epsilon Finite nonnegative multiplicative split fraction, default0.01;0 permits identical children.
/// @param lloyd_iters Positive maximum Lloyd passes per size, default100.
/// @param tol Finite nonnegative relative pre-update distortion-change threshold, default1e-6.
/// @return Owned final codebook with actual nearest assignments/distortion and consistent per-size history.
/// @throws std::invalid_argument On complete ragged/nonfinite/zero-dimensional data or invalid numeric/iteration parameters.
/// @throws std::length_error On rounded target/container/selected-work geometry.
/// @throws std::overflow_error On nonfinite/unrepresentable distance, distortion, split or centroid arithmetic.
/// @throws std::bad_alloc On runtime/output allocations.
/// @par Example
/// @code
/// const auto r=dm::algorithms::linde_buzo_gray({{1},{3}},1);
/// std::print("{}\n",r.distortion); // 1
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see LBGResult, LBGTrace, lbg_quantize()
LBGResult linde_buzo_gray(
    const std::vector<std::vector<double>>& data,
    std::size_t                             codebook_size,
    double                                  epsilon     = .01,
    std::size_t                             lloyd_iters = 100,
    double                                  tol         = 1e-6);
/// @brief Return the actual consistent LBG fit and selected producing operations without history.
/// Same complete contracts/stopping/split/consistency passes as linde_buzo_gray().
/// Count-only retains runtime codebooks/assignments but no optional snapshots.
/// @param data Borrowed complete rectangular finite points with positive dimension when nonempty.
/// @param codebook_size Requested size, rounded up to a power of two;0 returns empty result after complete validation.
/// @param epsilon Finite nonnegative multiplicative split fraction, default0.01;0 permits identical children.
/// @param lloyd_iters Positive maximum Lloyd passes per size, default100.
/// @param tol Finite nonnegative relative pre-update distortion-change threshold, default1e-6.
/// @return Owned final result, actual requested/rounded target and real work.
/// @throws std::invalid_argument On complete ragged/nonfinite/zero-dimensional data or invalid numeric/iteration parameters.
/// @throws std::length_error On rounded target/container/selected-work geometry.
/// @throws std::overflow_error On nonfinite/unrepresentable distance, distortion, split or centroid arithmetic.
/// @throws std::bad_alloc On runtime/output allocations.
/// @par Example
/// @code
/// const auto a=dm::algorithms::linde_buzo_gray_analyze({{1},{3}},2);
/// std::print("{}\n",a.result.codebook.size()); // 2
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see LBGResult, LBGTrace, lbg_quantize()
LBGAnalysis linde_buzo_gray_analyze(
    const std::vector<std::vector<double>>& data,
    std::size_t                             codebook_size,
    double                                  epsilon     = .01,
    std::size_t                             lloyd_iters = 100,
    double                                  tol         = 1e-6);
/// @brief Capture every actual codebook-size stage and Lloyd assignment/update from one fit.
/// Same shared producer as ordinary/count-only APIs. Multiplicative zero-center
/// collapse and retained empty codewords are real states, not suppressed.
/// @param data Borrowed complete rectangular finite points with positive dimension when nonempty.
/// @param codebook_size Requested size, rounded up to a power of two;0 returns empty result after complete validation.
/// @param epsilon Finite nonnegative multiplicative split fraction, default0.01;0 permits identical children.
/// @param lloyd_iters Positive maximum Lloyd passes per size, default100.
/// @param tol Finite nonnegative relative pre-update distortion-change threshold, default1e-6.
/// @param max_entries Actual logical capture cap, default1000000;0 disables optional cap.
/// @return Owned complete original/stages/iterations/result/work or exception; no silent partial trace.
/// @throws std::invalid_argument On complete ragged/nonfinite/zero-dimensional data or invalid numeric/iteration parameters.
/// @throws std::length_error On rounded target/container/selected-work geometry.
/// @throws std::overflow_error On nonfinite/unrepresentable distance, distortion, split or centroid arithmetic.
/// @throws std::bad_alloc On runtime/output allocations.
/// @throws std::length_error On actual complete-record quota/count limits.
/// @par Example
/// @code
/// const auto t=dm::algorithms::linde_buzo_gray_trace({{-1},{1}},2);
/// std::print("{} {}\n",t.analysis.result.distortion,t.analysis.result.codebook[0][0]); // 1 0
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see LBGResult, LBGTrace, lbg_quantize()
LBGTrace linde_buzo_gray_trace(
    const std::vector<std::vector<double>>& data,
    std::size_t                             codebook_size,
    double                                  epsilon     = .01,
    std::size_t                             lloyd_iters = 100,
    double                                  tol         = 1e-6,
    std::size_t                             max_entries = 1000000);
/// @brief Assign complete finite points to the earliest nearest codeword and observe mean squared distortion.
/// No fitting/mutation; shared actual assignment kernel. O(n*K*d) work and
/// O(n) output. Caller can evaluate independent queries without data leakage.
/// @param data Borrowed complete rectangular finite points; empty valid.
/// @param codebook Borrowed complete finite nonempty positive-dimensional codebook matching point dimension.
/// @return Owned actual nearest indices, mean squared distance and selected distance work.
/// @throws std::invalid_argument On complete ragged/nonfinite/zero-dimensional data or invalid numeric/iteration parameters.
/// @throws std::length_error On rounded target/container/selected-work geometry.
/// @throws std::overflow_error On nonfinite/unrepresentable distance, distortion, split or centroid arithmetic.
/// @throws std::bad_alloc On runtime/output allocations.
/// @par Example
/// @code
/// const auto q=dm::algorithms::lbg_quantize({{1},{3}},{{0},{4}});
/// std::print("{}\n",q.distortion); // 1
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see LBGResult, LBGTrace, lbg_quantize()
LBGQuantization lbg_quantize(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::vector<double>>& codebook);
/// @brief Produce the original600-point seeded two-dimensional mixture fixture.
/// Centers(0,0),(6,1),(2,6),(7,7),(3,3), mt19937_64 seed12 and
/// normal noise sd0.6, x then y. Standard-library normal mapping is implementation
/// specific; no portable fixed sample stream or model-quality promise.
/// @return Owned original points,120 per center in original center/order sequence.
/// @throws std::bad_alloc On fixture allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::lbg_demo_mixture().size()); // 600
/// @endcode
/// @par References
/// - Linde, Buzo and Gray(1980), An Algorithm for Vector Quantizer Design, IEEE Transactions on Communications28(1),84–95, https://doi.org/10.1109/TCOM.1980.1094577.
/// @see LBGResult, LBGTrace, lbg_quantize()
std::vector<std::vector<double>> lbg_demo_mixture();
} // namespace datamunge::algorithms
