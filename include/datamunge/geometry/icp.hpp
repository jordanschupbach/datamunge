#pragma once
/// @file icp.hpp
/// @brief Checked point-to-point planar ICP with actual matching/update histories.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <vector>
namespace datamunge::geometry {
/// @brief Accumulated planar rigid pose, update count and final working-cloud RMSE.
/// Legacy aggregate field order/types are preserved. Angle is an unwrapped sum in
/// radians; translation composes incremental rotations/translations. RMSE now
/// measures final updated working points, correcting the old pre-update value.
/// Applying the accumulated pose once can differ from repeated updates by rounding.
/// Empty-cloud RMSE zero is a convention, not a registration-quality claim.
/// @par Example
/// @code
/// const auto r=dm::geometry::icp({{0,0}},{{2,3}},1);
/// std::print("{} {} {}\n",r.translation.x,r.translation.y,r.rmse); // 2 3 0
/// @endcode
/// @see icp_run(), IcpStop
struct IcpResult {
  double  angle{0};          ///< Accumulated unwrapped angle in radians.
  Point2D translation{0, 0}; ///< Accumulated source-to-target translation.
  int     iterations{0};     ///< Completed actual rigid updates.
  double  rmse{
      0}; ///< Final working-cloud directed nearest-target RMSE; zero for empty-cloud convention.
};
/// @brief Reason the producing loop stopped, without asserting global pose recovery.
/// A pre-update error plateau is a numerical stopping rule, not a global optimum,
/// unique correspondence or recovered ground truth certificate.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::icp_run({},{}).stop==dm::geometry::IcpStop::EmptyCloud); // true
/// @endcode
/// @see icp_run()
enum class IcpStop {
  EmptyCloud,      ///< At least one input empty; identity/zero-RMSE convention, no work.
  IterationLimit,  ///< Requested update budget exhausted, including zero.
  PreUpdatePlateau ///< Strict change between consecutive pre-update RMSE values is below tol.
};
/// @brief Optional complete input and actual-history admission policies.
/// Zero disables each optional cap. A step costs 1+5*N logical entries: one
/// record, two coordinates for each before/after point, and one match index per
/// source point. Excludes input copies, ordinary/transient arrays, final aligned
/// cloud/result/work, scalar metadata and vector capacity; not byte/peak RAM.
/// @par Example
/// @code
/// const dm::geometry::IcpOptions o;std::print("{}\n",o.max_points); // 1000000
/// @endcode
/// @see icp_trace()
struct IcpOptions {
  std::size_t max_points{
      1000000}; ///< Complete source-plus-target point count, before copy/working storage.
  std::size_t max_history_entries{
      1000000}; ///< Online complete step units, checked before its matching pass/copies/update.
};
/// @brief Selected producing work, separate from extra error-observation passes.
/// Counts exclude validation, centroids, covariance, trigonometry, copies,
/// allocations and comparisons. Ordinary final RMSE adds one nearest pass;
/// traced steps add fixed-match and nearest error observations after each update.
/// @par Example
/// @code
/// const auto r=dm::geometry::icp_run({{0,0}},{{1,0}},1);
/// std::print("{} {}\n",r.work.distance_evaluations,r.work.point_updates); // 1 1
/// @endcode
/// @see icp_run(), icp_trace()
struct IcpWork {
  std::size_t distance_evaluations{
      0};                       ///< Producing squared distances in pre-update nearest searches.
  std::size_t point_updates{0}; ///< Actual working-point coordinate-pair updates.
  std::size_t diagnostic_distance_evaluations{
      0}; ///< Extra squared distances for final/trace-only error observations.
};
/// @brief Actual owned match/update boundary from a single ICP iteration.
/// Before/after preserve evolving working coordinates, rather than reconstructed
/// accumulated-pose points. Correspondences allow many source points per target;
/// strict-less ascending target ties choose the first. Post-update errors are
/// additional observations and never influence the preserved legacy stop rule.
/// @par Example
/// @code
/// const auto t=dm::geometry::icp_trace({{0,0}},{{1,0}},1);
/// std::print("{} {}\n",t.steps[0].pre_rmse,t.steps[0].post_nearest_rmse); // 1 0
/// @endcode
/// @see icp_trace()
struct IcpStep {
  std::vector<Point2D>     before;         ///< Owned source working points before this update.
  std::vector<std::size_t> target_indices; ///< Actual selected target index for each before point.
  std::vector<Point2D>     after;          ///< Actual updated working points.
  double pre_rmse{0}; ///< Directed nearest RMSE before the update, used by legacy plateau stop.
  double post_fixed_rmse{0}; ///< Observed RMSE after update against this step's frozen matches.
  double post_nearest_rmse{
      0}; ///< Observed RMSE after update against newly searched nearest targets.
  double    incremental_angle{0};    ///< Actual solved incremental angle, radians.
  Point2D   incremental_translation; ///< Actual solved incremental translation.
  IcpResult pose; ///< Accumulated pose/update count at this boundary, with post-nearest RMSE.
};
/// @brief Owned final working points and explicit stop/work information.
/// The result pose is accumulated during production. Aligned points are the
/// actual repeated-update cloud; applying that pose once has ordinary rounding
/// differences. No caller mutation, retained references or shared state.
/// @par Example
/// @code
/// const auto r=dm::geometry::icp_run({{0,0}},{{1,2}},1);
/// std::print("{} {}\n",r.aligned[0].x,r.aligned[0].y); // 1 2
/// @endcode
/// @see icp_run()
struct IcpRun {
  IcpResult result; ///< Accumulated pose and final working-cloud directed RMSE.
  std::vector<Point2D>
          aligned; ///< Owned actual final working points; source copy for empty/zero budget.
  IcpStop stop{IcpStop::IterationLimit}; ///< Actual termination reason, not a global-optimum flag.
  IcpWork work;                          ///< Actual producing versus observation counts.
};
/// @brief Owned original clouds, actual steps and their final producing result.
/// No repeated-prefix solves or reconstructed intermediate states. Storage adds
/// source/target copies and O(K*N) captured step values to ordinary workspace.
/// @par Example
/// @code
/// const auto t=dm::geometry::icp_trace({{0,0}},{{1,0}},1);
/// std::print("{} {}\n",t.source[0].x,t.steps.size()); // 0 1
/// @endcode
/// @see icp_trace()
struct IcpTrace {
  IcpRun               run;    ///< Final result of this same execution.
  std::vector<Point2D> source; ///< Owned complete original source, in original order.
  std::vector<Point2D> target; ///< Owned complete original target, in original order.
  std::vector<IcpStep> steps;  ///< Actual ordered matching/update boundaries.
};
/// @brief Register finite planar source points onto finite target points.
/// Point-to-point directed nearest-target ICP starts from identity and permits
/// many-to-one matches. Empty clouds return identity/zero RMSE without a quality
/// claim; zero budget returns identity with actual initial RMSE if both nonempty.
/// Nonnegative finite tolerance uses strict absolute differences of consecutive
/// PRE-update RMSE values, tested after an update. Zero tolerance forces the
/// full budget. This preserves legacy finite-input update/stop arithmetic while
/// correcting returned RMSE to the final working cloud. It does not certify
/// true pose recovery, a unique alignment or a global optimum. Degenerate zero
/// covariance uses atan2's computed angle (ordinarily zero for all-zero sums).
/// No mutation/retained references/shared state. Brute-force nearest search
/// costs O((K+1)*N*M) for nonempty clouds including final observations;
/// ordinary workspace O(N); trace adds O(K*N) storage and
/// diagnostic observations. All arithmetic is checked double arithmetic with
/// no automatic rescaling; finite large coordinates can still overflow.
/// @param source Borrowed complete finite planar source cloud, size N.
/// @param target Borrowed complete finite planar target cloud, size M.
/// @param max_iters Nonnegative maximum updates, default 100.
/// @param tol Finite nonnegative absolute pre-update RMSE-change tolerance, default 1e-14.
/// Legacy entry point disables both optional caps.
/// @return Owned pose/update count/final working RMSE.
/// @throws std::invalid_argument On nonfinite coordinates/tolerance, negative tolerance or budget.
/// @throws std::length_error On count overflow or enabled complete input/history cap.
/// @throws std::overflow_error On nonfinite producing/diagnostic arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::geometry::icp({{0,0}},{{1,0}},1);
/// std::print("{}\n",r.rmse); // 0
/// @endcode
/// @par References
/// - P. J. Besl and N. D. McKay (1992), A method for registration of 3-D shapes,
///   IEEE TPAMI 14,239–256. doi:10.1109/34.121791. This is the planar
///   point-to-point, brute-force nearest-neighbor variant, not their 3-D solver.
/// @see IcpOptions, IcpStop, icp_run(), icp_trace()
IcpResult icp(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    int                         max_iters = 100,
    double                      tol       = 1e-14);
/// @brief Register once and retain actual final points, stop and work.
/// Point-to-point directed nearest-target ICP starts from identity and permits
/// many-to-one matches. Empty clouds return identity/zero RMSE without a quality
/// claim; zero budget returns identity with actual initial RMSE if both nonempty.
/// Nonnegative finite tolerance uses strict absolute differences of consecutive
/// PRE-update RMSE values, tested after an update. Zero tolerance forces the
/// full budget. This preserves legacy finite-input update/stop arithmetic while
/// correcting returned RMSE to the final working cloud. It does not certify
/// true pose recovery, a unique alignment or a global optimum. Degenerate zero
/// covariance uses atan2's computed angle (ordinarily zero for all-zero sums).
/// No mutation/retained references/shared state. Brute-force nearest search
/// costs O((K+1)*N*M) for nonempty clouds including final observations;
/// ordinary workspace O(N); trace adds O(K*N) storage and
/// diagnostic observations. All arithmetic is checked double arithmetic with
/// no automatic rescaling; finite large coordinates can still overflow.
/// @param source Borrowed complete finite planar source cloud, size N.
/// @param target Borrowed complete finite planar target cloud, size M.
/// @param max_iters Nonnegative maximum updates, default 100.
/// @param tol Finite nonnegative absolute pre-update RMSE-change tolerance, default 1e-14.
/// @param options Complete input/history policies, default million each; zero disables optional cap.
/// @return Owned actual final state and work.
/// @throws std::invalid_argument On nonfinite coordinates/tolerance, negative tolerance or budget.
/// @throws std::length_error On count overflow or enabled complete input/history cap.
/// @throws std::overflow_error On nonfinite producing/diagnostic arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::geometry::icp_run({{0,0}},{{1,0}},1);
/// std::print("{}\n",r.result.rmse); // 0
/// @endcode
/// @par References
/// - P. J. Besl and N. D. McKay (1992), A method for registration of 3-D shapes,
///   IEEE TPAMI 14,239–256. doi:10.1109/34.121791. This is the planar
///   point-to-point, brute-force nearest-neighbor variant, not their 3-D solver.
/// @see IcpOptions, IcpStop, icp_run(), icp_trace()
IcpRun icp_run(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    int                         max_iters = 100,
    double                      tol       = 1e-14,
    IcpOptions                  options   = {});
/// @brief Capture actual matches and evolving points from one registration.
/// Point-to-point directed nearest-target ICP starts from identity and permits
/// many-to-one matches. Empty clouds return identity/zero RMSE without a quality
/// claim; zero budget returns identity with actual initial RMSE if both nonempty.
/// Nonnegative finite tolerance uses strict absolute differences of consecutive
/// PRE-update RMSE values, tested after an update. Zero tolerance forces the
/// full budget. This preserves legacy finite-input update/stop arithmetic while
/// correcting returned RMSE to the final working cloud. It does not certify
/// true pose recovery, a unique alignment or a global optimum. Degenerate zero
/// covariance uses atan2's computed angle (ordinarily zero for all-zero sums).
/// No mutation/retained references/shared state. Brute-force nearest search
/// costs O((K+1)*N*M) for nonempty clouds including final observations;
/// ordinary workspace O(N); trace adds O(K*N) storage and
/// diagnostic observations. All arithmetic is checked double arithmetic with
/// no automatic rescaling; finite large coordinates can still overflow.
/// @param source Borrowed complete finite planar source cloud, size N.
/// @param target Borrowed complete finite planar target cloud, size M.
/// @param max_iters Nonnegative maximum updates, default 100.
/// @param tol Finite nonnegative absolute pre-update RMSE-change tolerance, default 1e-14.
/// @param options Complete input/history policies, default million each; zero disables optional cap.
/// @return Owned original inputs, actual steps and final state.
/// @throws std::invalid_argument On nonfinite coordinates/tolerance, negative tolerance or budget.
/// @throws std::length_error On count overflow or enabled complete input/history cap.
/// @throws std::overflow_error On nonfinite producing/diagnostic arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::geometry::icp_trace({{0,0}},{{1,0}},1);
/// std::print("{}\n",r.run.result.rmse); // 0
/// @endcode
/// @par References
/// - P. J. Besl and N. D. McKay (1992), A method for registration of 3-D shapes,
///   IEEE TPAMI 14,239–256. doi:10.1109/34.121791. This is the planar
///   point-to-point, brute-force nearest-neighbor variant, not their 3-D solver.
/// @see IcpOptions, IcpStop, icp_run(), icp_trace()
IcpTrace icp_trace(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    int                         max_iters = 100,
    double                      tol       = 1e-14,
    IcpOptions                  options   = {});
} // namespace datamunge::geometry
