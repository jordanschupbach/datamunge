#pragma once
/// @file rbf_network.hpp
/// @brief Checked native Gaussian-basis regression, actual Lloyd passes and explicit closed-form comparison APIs.
#include <cstddef>
#include <cstdint>
#include <vector>
/// @brief Canonical Gaussian basis regression and owned execution records.
namespace datamunge::algorithms {
/// @brief Owned legacy-layout Gaussian-feature scalar regression model.
/// @par Example
/// @code
/// const dm::algorithms::RBFNetwork value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RBFNetwork {
  std::vector<std::vector<double>> centers;  ///< Owned K finite input-space centers.
  double                           width{1}; ///< Actual positive finite common Gaussian sigma.
  std::vector<double>              weights;  ///< Actual K output weights.
  double                           bias{0};  ///< Actual finite intercept.
};
/// @brief Complete training input, selected-distance and actual-history policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RBFTrainingOptions value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RBFTrainingOptions {
  std::size_t max_points{1000000};  ///< Complete observation count.
  std::size_t max_centers{256};     ///< Actual center count after min(requested,n).
  std::size_t max_dimensions{1024}; ///< Complete input coordinate dimension.
  std::size_t max_iterations{1000}; ///< Requested fixed Lloyd passes.
  std::size_t max_distance_tests{
      10000000}; ///< Actual squared-distance calls in Lloyd, width and features.
  std::size_t max_matrix_entries{1000000}; ///< Complete design plus normal-system scalar count.
  std::size_t max_trace_entries{
      1000000}; ///< Actual initial/pass/design/system retained scalars and headers.
};
/// @brief One actual completed Lloyd assignment/update pass.
/// @par Example
/// @code
/// const dm::algorithms::RBFKmeansPass value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RBFKmeansPass {
  std::size_t                      iteration{}; ///< Zero-based actual fixed Lloyd pass.
  std::vector<std::vector<double>> before;      ///< Actual center positions used for assignment.
  std::vector<std::size_t>
      assignments; ///< Actual selected nearest-center id per observation, first wins ties.
  std::vector<std::size_t> counts; ///< Actual observation count assigned to each center.
  std::vector<std::vector<double>>
      after; ///< Actual updated means; empty clusters keep their previous center.
};
/// @brief Owned closed-form baseline and actual producer diagnostics.
/// @par Example
/// @code
/// const dm::algorithms::RBFTrainingRun value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RBFTrainingRun {
  RBFNetwork model; ///< Actual canonical trained scalar model.
  std::vector<std::vector<double>>
      initial_centers; ///< Actual sampled initial centers, only captured by trace.
  std::vector<RBFKmeansPass>       passes; ///< Actual optional fixed-count Lloyd histories.
  std::vector<std::vector<double>> design; ///< Actual optional n-by-(K+1) features, bias first.
  std::vector<std::vector<double>> normal_matrix; ///< Actual optional ridge normal system.
  std::vector<double>              rhs;           ///< Actual optional normal-system right side.
  std::size_t distance_tests{}; ///< Actual squared-distance evaluations across training stages.
};
/// @brief Fit a checked native Gaussian-basis closed-form regression baseline.
/// Validates finite nonempty rectangular inputs and matching finite targets, finite nonnegative ridge and finite width. K=min(requested,n); K0 is an explicit intercept-only baseline without Lloyd. Initializes centers with the historical mt19937_64 random-sample stream and executes exactly the requested Lloyd passes, retaining empty-cluster centers. A positive width is used directly; nonpositive finite width selects the historical maximum-center-distance/sqrt(2K) heuristic, falling back to1. Forms actual bias-first Gaussian design and ridge normal equations, penalizing the bias too. Uses widened arithmetic and checked partial-pivot elimination, throwing on an exact singular system instead of silently dropping its coefficients. Time O(passes*n*K*d+n*K^2+K^3), O(n*K+K^2) working storage; inputs unchanged, outputs owned. This is an independent closed-form comparison for reports whose demonstrated neural fitting uses explicit tensor optimization.
/// @param X Finite nonempty rectangular rows with positive dimension.
/// @param y Finite scalar targets, one per row.
/// @param num_centers Requested centers, capped by n; zero selects intercept-only.
/// @param ridge_lambda Finite nonnegative regularization, default1e-6.
/// @param width Positive explicit sigma; nonpositive finite selects heuristic, default0.
/// @param kmeans_iters Actual fixed Lloyd passes, default20.
/// @param seed Native initial-center seed, default0.
/// @param options Input, work, matrix and history policies.
/// @return Owned actual baseline model and selected training work.
/// @throws std::invalid_argument Input shape/value, ridge or width invalid.
/// @throws std::length_error Input/work/matrix/history policy exceeded.
/// @throws std::domain_error Actual normal system is singular.
/// @throws std::overflow_error Unrepresentable geometry or coefficient.
/// @par Example
/// @code
/// auto r=dm::algorithms::rbf_train_run({{0},{1}},{2,4},0,0);
/// std::print("{}\n",r.model.bias); // 3
/// @endcode
/// @par References
/// - Broomhead and Lowe (1988), Multivariable Functional Interpolation and Adaptive Networks, https://www.complex-systems.com/abstracts/v02_i03_a05/
/// @see rbf_train_trace(), rbf_fit_output()
RBFTrainingRun rbf_train_run(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    std::size_t                             num_centers,
    double                                  ridge_lambda = 1e-6,
    double                                  width        = 0,
    std::size_t                             kmeans_iters = 20,
    std::uint64_t                           seed         = 0,
    const RBFTrainingOptions&               options      = {});
/// @brief Capture actual sampled centers, Lloyd passes, design and normal equations.
/// Same single producer as rbf_train_run(). Retained scalar entries count initial center coordinates, real before/after centers and assignments/counts, then actual design/system/right-side values, plus one header for each capture. Admission precedes copies. Zero capture cap disables that bound; diagnostic narrowing may throw if a retained system scalar exceeds finite double range.
/// @param X Finite rectangular input rows.
/// @param y Finite matching targets.
/// @param num_centers Requested centers.
/// @param ridge_lambda Nonnegative finite ridge, default1e-6.
/// @param width Explicit positive sigma or finite automatic selector, default0.
/// @param kmeans_iters Actual fixed Lloyd passes, default20.
/// @param seed Initial-center seed, default0.
/// @param options Input, work and actual scalar-history policies.
/// @return Actual optional full producer histories and closed-form comparison model.
/// @throws std::invalid_argument Invalid complete input/control.
/// @throws std::length_error Any execution/history policy exceeded.
/// @throws std::domain_error Singular actual normal system.
/// @throws std::overflow_error Unrepresentable geometry/coefficient/diagnostic.
/// @par Example
/// @code
/// auto t=dm::algorithms::rbf_train_trace({{0},{1}},{2,4},1,1e-3,1,2,1);
/// std::print("{}\n",t.passes.size()); // 2
/// @endcode
/// @see rbf_train_run()
RBFTrainingRun rbf_train_trace(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    std::size_t                             num_centers,
    double                                  ridge_lambda = 1e-6,
    double                                  width        = 0,
    std::size_t                             kmeans_iters = 20,
    std::uint64_t                           seed         = 0,
    const RBFTrainingOptions&               options      = {});
/// @brief Return the checked canonical closed-form model through the preserved signature.
/// Delegates to rbf_train_run() with the same defaults and model field layout. Complete validation and singular-system failure replace previous unchecked shapes/zero-center indexing and silent coefficient dropping. For tensor neural fitting, use nn::rbf_features() with an explicit optimizer loop and keep this as a separate baseline.
/// @param X Finite rectangular observations.
/// @param y Finite matching targets.
/// @param num_centers Requested centers, including zero.
/// @param ridge_lambda Nonnegative finite ridge, default1e-6.
/// @param width Positive finite sigma or finite automatic selector, default0.
/// @param kmeans_iters Actual fixed Lloyd passes, default20.
/// @param seed Initial-center seed, default0.
/// @return Owned canonical Gaussian regression model.
/// @throws std::invalid_argument Invalid data/control.
/// @throws std::length_error Default policies exceeded.
/// @throws std::domain_error Singular system.
/// @throws std::overflow_error Unrepresentable arithmetic.
/// @par Example
/// @code
/// auto m=dm::algorithms::rbf_train({{0},{1}},{2,4},0,0);
/// std::print("{}\n",m.bias); // 3
/// @endcode
/// @see rbf_train_run()
RBFNetwork rbf_train(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    std::size_t                             num_centers,
    double                                  ridge_lambda = 1e-6,
    double                                  width        = 0,
    std::size_t                             kmeans_iters = 20,
    std::uint64_t                           seed         = 0);
/// @brief Fit only the checked output coefficients for caller-selected fixed Gaussian centers.
/// Uses the same native feature/normal-system solver as rbf_train_run(), with no Lloyd placement or width heuristic. Penalizes all output coefficients including bias, permitting a like-for-like independent comparison to an explicitly optimized tensor model. Center inputs are copied; no fitting iterations are hidden. Same shape/value, complexity and error semantics as the native training run after center placement.
/// @param X Finite nonempty rectangular observations.
/// @param y Finite matching targets.
/// @param centers Finite matching-dimensional fixed centers, possibly empty.
/// @param width Positive finite Gaussian sigma.
/// @param ridge_lambda Nonnegative finite ridge, default1e-6.
/// @param options Input, matrix and selected-distance policies.
/// @return Owned fixed-center closed-form comparison model.
/// @throws std::invalid_argument Invalid inputs/control.
/// @throws std::length_error Execution policy exceeded.
/// @throws std::domain_error Singular normal system.
/// @throws std::overflow_error Unrepresentable arithmetic.
/// @par Example
/// @code
/// auto m=dm::algorithms::rbf_fit_output({{0},{1}},{2,4},{},1,0);
/// std::print("{}\n",m.bias); // 3
/// @endcode
/// @see rbf_train_run(), nn::rbf_features()
RBFNetwork rbf_fit_output(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    const std::vector<std::vector<double>>& centers,
    double                                  width,
    double                                  ridge_lambda = 1e-6,
    const RBFTrainingOptions&               options      = {});
/// @brief Predict with a completely checked native closed-form comparison model.
/// Validates every center/weight and finite matching input coordinates before feature evaluation. Width must be finite/positive and weights must match centers. Empty-center model predicts its finite bias. Uses the same widened Gaussian feature formula as fitting; O(K*d) time, constant extra working storage, no model mutation.
/// @param model Owned finite valid scalar RBF model.
/// @param x Finite positive-dimensional input matching every center.
/// @return Actual finite scalar comparison prediction.
/// @throws std::invalid_argument Invalid model/input shape, width or value.
/// @throws std::overflow_error Unrepresentable feature arithmetic or prediction.
/// @par Example
/// @code
/// dm::algorithms::RBFNetwork m{{},1,{},3};
/// std::print("{}\n",dm::algorithms::rbf_predict(m,{0})); // 3
/// @endcode
/// @see rbf_fit_output()
double rbf_predict(const RBFNetwork& model, const std::vector<double>& x);
/// @brief Owned original synthetic regression observations and independent noiseless grid.
/// @par Example
/// @code
/// const dm::algorithms::RBFExampleData value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RBFExampleData {
  std::vector<std::vector<double>> input; ///< Actual61 original one-dimensional input rows.
  std::vector<double>
      target; ///< Actual original std::normal noise realization added to the analytic target.
  std::vector<double> grid;  ///< Actual201 independent noiseless evaluation coordinates.
  std::vector<double> truth; ///< Actual sin(x)+sin(3x)/3 values on that grid.
};
/// @brief Generate the original61-point seed11 noisy regression fixture and201-point truth grid.
/// Uses x=2*pi*i/60, i0..60, and one mt19937_64(seed)/normal_distribution(0,.15) stream. The noiseless201-point grid is independent evaluation against the known analytic function, not held-out observations. O(61+201) owned input storage; no model fitting or preprocessing occurs. Convert these canonical observations to tensors for neural model computation.
/// @param seed Original native noise seed, default11.
/// @return Owned original observations and analytic evaluation grid.
/// @par Example
/// @code
/// auto d=dm::algorithms::rbf_example_data();
/// std::print("{} {}\n",d.input.size(),d.grid.size()); // 61 201
/// @endcode
/// @see rbf_train_run(), nn::rbf_features()
RBFExampleData rbf_example_data(std::uint64_t seed = 11);
} // namespace datamunge::algorithms
