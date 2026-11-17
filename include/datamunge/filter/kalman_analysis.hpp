#pragma once
/// @file kalman_analysis.hpp
/// @brief Original full filter-family study, actual API-stage capture and checked diagnostics.
#include <cstddef>
#include <datamunge/filter/filter.hpp>
#include <vector>
namespace datamunge::filter {
/// @brief Owned actual prior, predicted and corrected state/covariance from one native step.
/// Captures getter outputs around the producing predict/update, with no replay.
/// Snapshot storage is O(n^2); no model/input references are retained.
/// @par Example
/// @code
/// const auto s=dm::filter::kalman_tiny_step();std::print("{} {}\n",s.predicted.P[0][0],s.filtered.x[0]); // 5 1.25
/// @endcode
/// @see kalman_capture_step()
struct KalmanStage {
  KalmanState prior;     ///< Actual state/covariance before prediction.
  KalmanState predicted; ///< Actual state/covariance after prediction, before correction.
  KalmanState filtered;  ///< Actual state/covariance after correction.
};
/// @brief Capture one actual native predict/update step on a supplied mutable filter.
/// Caller provides valid model/finite data under KalmanFilter's contracts.
/// A failure does not roll back already committed prediction/update; no partial
/// snapshot is returned. Copies/getters do not run another estimator step.
/// @param filter Borrowed mutable native filter; finishes at corrected posterior on success.
/// @param measurement Borrowed compatible observation.
/// @return Owned actual prior/predicted/filtered state and covariance snapshots.
/// @throws std::invalid_argument On incompatible observation shape.
/// @throws std::runtime_error On underlying LU failure.
/// @throws std::bad_alloc On snapshot/native temporary allocation failure.
/// @par Example
/// @code
/// dm::filter::KalmanFilter k({{1}},{{1}},{{1}},{{3}},{0},{{4}});
/// std::print("{}\n",dm::filter::kalman_capture_step(k,{2}).filtered.P[0][0]); // 1.875
/// @endcode
/// @see kalman_tiny_step()
KalmanStage kalman_capture_step(KalmanFilter& filter, const std::vector<double>& measurement);
/// @brief Execute the fixed scalar prior0/P4, F=H=1,Q1,R3,measurement2 teaching control.
/// Owned output from actual native prediction/correction, no RNG or shared state.
/// @return Actual three-stage snapshots; predicted variance5, posterior mean1.25/variance1.875.
/// @throws std::bad_alloc On native/snapshot allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::filter::kalman_tiny_step().filtered.x[0]); // 1.25
/// @endcode
/// @see kalman_capture_step()
KalmanStage kalman_tiny_step();
/// @brief Owned original synthetic observations/truth for all report filter families.
/// Linear/nonlinear sequences have50steps, dt1/true velocity2, seed42 mt19937.
/// Position-only SD2 observations are drawn first, then monotonic2x+sin(x)
/// observations with SD.5 using the same engine/new distribution. Acceleration
/// control has40noiseless steps from rest with acceleration.5. Exact normal
/// mapping is standard-library dependent. No external dataset or caller views.
/// @par Example
/// @code
/// const auto d=dm::filter::kalman_demo_data();std::print("{} {}\n",d.linear_truth.size(),d.acceleration_truth.size()); // 50 40
/// @endcode
/// @see kalman_family_study()
struct KalmanStudyData {
  std::vector<std::vector<double>>
                      linear_measurements; ///< Original position-only noisy scalar observations.
  std::vector<double> linear_truth;        ///< True positions2,4,...,100.
  std::vector<std::vector<double>>
      nonlinear_measurements; ///< Original noisy2x+sin(x) scalar observations, same engine continued.
  std::vector<double> nonlinear_truth; ///< Original true positions for nonlinear observations.
  std::vector<double>
      acceleration_measurements; ///< Original noiseless constant-acceleration scalar positions.
  std::vector<double>
      acceleration_truth; ///< Complete true positions corresponding to acceleration input.
};
/// @brief Generate the complete original three synthetic sequences in their historical draw order.
/// Fixed50+50+40step owned fixture, no global RNG or mutation. Distribution mapping
/// depends on the standard library just as in the original report.
/// @return Owned complete original observations and truth.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::filter::kalman_demo_data().linear_truth.back()); // 100
/// @endcode
/// @see KalmanStudyData
KalmanStudyData kalman_demo_data();
/// @brief Native study populations/sigma parameter and complete input/population ceilings.
/// Limits count original input steps and ensemble+particle members, excluding
/// multiple method outputs, matrix capacities, byte counts and peak RAM. Zero
/// disables optional ceilings only; counts/UKF sigma-scale representability checked.
/// @par Example
/// @code
/// const dm::filter::KalmanStudyOptions o;std::print("{} {}\n",o.ensemble_size,o.particle_count); // 300 300
/// @endcode
/// @see kalman_family_study()
struct KalmanStudyOptions {
  int    ensemble_size{300};  ///< Actual native EnKF member count, at least2, original300.
  int    particle_count{300}; ///< Actual native PF particle count, at least2, original300.
  double ukf_alpha{
      1e-3}; ///< Actual native UKF sigma alpha, finite positive and representable2+lambda.
  std::size_t max_input_steps{
      10000}; ///< Complete linear+nonlinear+acceleration input-step ceiling, default10000, zero off.
  std::size_t max_population{100000}; ///< Complete EnKF+PF member ceiling, default100000, zero off.
};
/// @brief Owned actual outputs of every original model-family run on the same admitted data.
/// Linear Kalman/RTS uses one actual forward pass plus native backward pass.
/// EKF/UKF/EnKF/PF each run independently on the same nonlinear observations;
/// information form and alpha-beta use original linear observations. Alpha-beta-
/// gamma uses the separate acceleration sequence. No full-prefix replay, callbacks
/// or filter objects retained. Native exceptions propagate without partial output.
/// @par Example
/// @code
/// const auto s=dm::filter::kalman_family_study(dm::filter::kalman_demo_data());
/// std::print("{} {}\n",s.linear.filtered.size(),s.alpha_beta_gamma.size()); // 50 40
/// @endcode
/// @see kalman_family_study()
struct KalmanFamilyStudy {
  KalmanStudyData          data;   ///< Owned complete original admitted observations/truth.
  SmoothResult             linear; ///< Actual native causal and full-sequence RTS trajectories.
  std::vector<KalmanState> ekf;    ///< Actual nonlinear extended-Kalman posteriors.
  std::vector<KalmanState>
      ukf; ///< Actual nonlinear unscented-Kalman posteriors, not assumed accurate.
  std::vector<KalmanState> information; ///< Actual information-form linear posteriors.
  std::vector<KalmanState> ensemble; ///< Actual nonlinear300-default-member EnKF posteriors, seed7.
  std::vector<KalmanState> particle; ///< Actual nonlinear300-default-particle PF posteriors, seed7.
  std::vector<AlphaBetaState> alpha_beta; ///< Actual fixed(.5,.3),dt1 linear-input tracker states.
  std::vector<AlphaBetaGammaState>
      alpha_beta_gamma; ///< Actual fixed(.5,.1,.01),dt1 acceleration tracker states.
};
/// @brief Run the original complete model-family comparison through native APIs once each.
/// FixedF[[1,1],[0,1]],H[1,0],Q=.01I,R4,Rnonlinear.25,x0=0,P0=100I.
/// Finite matched nonempty scalar observations/truth are validated completely,
/// and complete step/population ceilings admitted before copies/filter construction.
/// No filtering algorithm is reimplemented. Nonlinear model approximation may
/// fail or produce poor finite estimates; there is no ranking/accuracy guarantee.
/// Work includes all actual requested population steps, not a selected subset.
/// @param data Borrowed complete original-shaped scalar data/truth, lengths may vary but must match each sequence.
/// @param options Actual populations/UKF alpha and complete record ceilings.
/// @return Owned complete data and actual full trajectories for all eight filter/tracker forms plus RTS.
/// @throws std::invalid_argument On empty/mismatched/nonfinite data, populations<2 or invalid representable sigma scale.
/// @throws std::length_error On complete input/population ceilings or count overflow.
/// @throws std::runtime_error On underlying linear-algebra/filter failure.
/// @throws std::overflow_error On nonfinite produced study state/covariance.
/// @throws std::bad_alloc On native/output allocation failure.
/// @par Example
/// @code
/// const auto s=dm::filter::kalman_family_study(dm::filter::kalman_demo_data());
/// std::print("{}\n",s.information.size()); // 50
/// @endcode
/// @see kalman_demo_ukf(), KalmanStudyOptions
KalmanFamilyStudy kalman_family_study(const KalmanStudyData& data, KalmanStudyOptions options = {});
/// @brief Run just the native nonlinear UKF on original study model/data with an explicit alpha.
/// Validates the complete nonlinear scalar sequence/truth and optional input-step
/// count before construction. Same fixed F/Q/Rnonlinear/x0/P0 as the family study.
/// An independent parameter run is not prefix replay or an accuracy certificate.
/// @param data Borrowed nonlinear observations/truth, finite matched and nonempty.
/// @param alpha Actual finite positive native sigma parameter with positive representable2+lambda.
/// @param max_steps Complete nonlinear input-step ceiling, default10000, zero off.
/// @return Owned actual complete nonlinear UKF posterior trajectory.
/// @throws std::invalid_argument On data/alpha/sigma-scale violations.
/// @throws std::length_error On enabled complete step ceiling.
/// @throws std::runtime_error On native linear-algebra/filter failure.
/// @throws std::overflow_error On nonfinite produced state/covariance.
/// @throws std::bad_alloc On native/output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::filter::kalman_demo_ukf(dm::filter::kalman_demo_data(),.1).size()); // 50
/// @endcode
/// @see kalman_family_study()
std::vector<KalmanState>
kalman_demo_ukf(const KalmanStudyData& data, double alpha = 1e-3, std::size_t max_steps = 10000);
/// @brief Copy a checked finite state-mean coordinate across a complete captured trajectory.
/// No filter replay, mutation or retained views. O(T) owned output.
/// @param states Borrowed captured state trajectory, empty allowed.
/// @param coordinate Required state coordinate at every row, default0.
/// @return Owned complete coordinate values.
/// @throws std::invalid_argument On absent/nonfinite requested coordinate.
/// @throws std::bad_alloc On output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::filter::kalman_state_column({{{2},{{1}}}},0)[0]); // 2
/// @endcode
/// @see kalman_position_mse()
std::vector<double> kalman_state_column(
    const std::vector<KalmanState>& states,
    std::size_t                     coordinate = 0);
/// @brief Measure actual realized position MSE against a matched finite truth sequence.
/// This statistic is not conditional covariance or a universal accuracy guarantee.
/// Widened accumulation, O(T) work/constant storage, no mutation or retained views.
/// @param states Borrowed nonempty captured trajectory with position coordinate0.
/// @param truth Borrowed same-length finite true positions.
/// @return Actual mean squared position error in squared position units.
/// @throws std::invalid_argument On empty/mismatched/nonfinite/absent position.
/// @throws std::overflow_error On nonfinite produced diagnostic.
/// @par Example
/// @code
/// std::print("{}\n",dm::filter::kalman_position_mse({{{2},{{1}}}},{1})); // 1
/// @endcode
/// @see kalman_state_column()
long double kalman_position_mse(
    const std::vector<KalmanState>& states,
    const std::vector<double>&      truth);
/// @brief Diagnose the smaller eigenvalue of a finite2-by2 matrix's symmetric part.
/// A rounded observation, not an exact PSD certificate; antisymmetry is averaged.
/// Constant work/storage, no mutation. Useful for filtered-smoothed covariance
/// differences under the fixed two-state model without replaying an estimator.
/// @param matrix Borrowed finite2-by2 array in nested-vector form.
/// @return Rounded smaller eigenvalue of the symmetric part.
/// @throws std::invalid_argument On wrong shape/nonfinite entries.
/// @throws std::overflow_error On nonfinite produced diagnostic.
/// @par Example
/// @code
/// std::print("{}\n",dm::filter::kalman_min_eigenvalue2({{2,0},{0,1}})); // 1
/// @endcode
/// @see KalmanState
long double kalman_min_eigenvalue2(const std::vector<std::vector<double>>& matrix);
} // namespace datamunge::filter
