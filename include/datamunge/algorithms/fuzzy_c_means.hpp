#pragma once
/// @file fuzzy_c_means.hpp
/// @brief Checked Euclidean fuzzy c-means, supplied starts and actual update diagnostics.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned row-major points, centers or membership rows.
/// Empty rows can represent zero-dimensional points. Algorithms validate rectangular shape and finite values; memberships additionally lie in0..1 with unit row sums within1e-12.
/// @par Example
/// @code
/// dm::algorithms::FuzzyCMeansMatrix x{{0},{5},{10}};
/// std::cout << x.size(); // 3
/// @endcode
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
using FuzzyCMeansMatrix = std::vector<std::vector<double>>;
/// @brief Owned centers, updated memberships and final numerical objective.
/// Field order and layout are retained. Centers use the incoming memberships of the last iteration; returned memberships use those centers. No final center refresh. A zero-iteration fit has initialized memberships and zero centers; iterations0 does not mean convergence.
/// @par Example
/// @code
/// dm::algorithms::FuzzyCMeansResult x;
/// std::cout << x.iterations; // 0
/// @endcode
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
struct FuzzyCMeansResult {
    std::vector<std::vector<double>> centers; ///< c x d centers in original cluster order.
    std::vector<std::vector<double>> membership; ///< n x c updated memberships; graded similarities, not probabilities.
    double objective = 0.0; ///< Final sum pow(u,m)*squared distance using the returned pair.
    std::size_t iterations = 0; ///< Actual completed center/membership updates.
};
/// @brief Actual executed fit work, excluding validation, allocation and copying.
/// Draws count uniform distribution calls, not engine words. Ratio powers occur only on rows without a near-coincident center. Diagnostic objectives are additional trace work, distinct from the final objective.
/// @par Example
/// @code
/// dm::algorithms::FuzzyCMeansWork x;
/// std::cout << x.ratio_powers; // 0
/// @endcode
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
struct FuzzyCMeansWork {
    std::size_t initial_draws{}; ///< Uniform(.1,1) calls; supplied starts draw none.
    std::size_t center_weights{}; ///< Actual membership pow(u,m) evaluations in center updates.
    std::size_t weighted_coordinates{}; ///< Actual numerator multiply-accumulate coordinate terms.
    std::size_t center_divisions{}; ///< Actual coordinate divisions for positive-denominator clusters.
    std::size_t membership_distances{}; ///< Actual point-center Euclidean distances in membership updates.
    std::size_t ratio_powers{}; ///< Actual pairwise distance-ratio powers on noncoincident rows.
    std::size_t membership_updates{}; ///< Actual membership assignments.
    std::size_t final_objective_pairs{}; ///< Point-cluster terms in the final objective.
    std::size_t diagnostic_objective_pairs{}; ///< Additional point-cluster terms in trace phase objectives.
};
/// @brief Fit plus actual work and stopping information.
/// Strict max_change<tol controls early stopping; tol0 runs the cap. max_change0 is only a placeholder when no iteration occurred. There is no global-optimum or probability-calibration guarantee.
/// @par Example
/// @code
/// dm::algorithms::FuzzyCMeansRun x;
/// std::cout << x.stopped_by_tolerance; // 0
/// @endcode
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
struct FuzzyCMeansRun {
    FuzzyCMeansResult result{}; ///< Owned numerical fit.
    FuzzyCMeansWork work{}; ///< Actual operation counts.
    bool stopped_by_tolerance{}; ///< Whether the strict membership-change test stopped execution.
    double max_change{}; ///< Last performed maximum change;0 placeholder with iterations0.
};
/// @brief Owned actual center update, distances and membership update for one iteration.
/// Numerators and denominators are the actual ordered accumulations, not reconstructed weighted means. Distances precede membership assignment. Objectives bracket each coordinate block; exact block descent does not promise strict finite-precision descent with the near-coincidence shortcut.
/// @par Example
/// @code
/// dm::algorithms::FuzzyCMeansFrame x;
/// std::cout << x.distances.size(); // 0
/// @endcode
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
struct FuzzyCMeansFrame {
    std::size_t iteration{}; ///< Zero-based actual iteration.
    FuzzyCMeansMatrix centers{}; ///< Actual c x d post-center values.
    FuzzyCMeansMatrix membership{}; ///< Actual n x c post-membership values.
    FuzzyCMeansMatrix numerators{}; ///< Actual c x d weighted sums before division.
    FuzzyCMeansMatrix distances{}; ///< Actual n x c Euclidean distances.
    std::vector<double> denominators{}; ///< Actual c sums of powered incoming memberships.
    double max_change{}; ///< Actual largest membership change.
    double objective_before{}; ///< Objective before both updates.
    double objective_centers{}; ///< Objective after centers with incoming memberships.
    double objective_membership{}; ///< Objective after membership update with new centers.
};
/// @brief Owned inputs, initial state and actual once-only iteration history.
/// No replay, caller mutation, shared state or partial result. Whole conservative admission precedes initialization draws and input copies. Extra phase objectives may overflow even when an ordinary final fit is representable.
/// @par Example
/// @code
/// dm::algorithms::FuzzyCMeansTrace x;
/// std::cout << x.frames.size(); // 0
/// @endcode
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
struct FuzzyCMeansTrace : public FuzzyCMeansRun {
    FuzzyCMeansMatrix data{}; ///< Owned raw points.
    FuzzyCMeansMatrix initial_membership{}; ///< Owned actual normalized random or supplied initial memberships.
    FuzzyCMeansMatrix initial_centers{}; ///< Owned initial zero centers.
    double initial_objective{}; ///< Actual initial objective at zero centers.
    std::vector<FuzzyCMeansFrame> frames{}; ///< Actual performed updates, no padding after early stop.
};
/// @brief Mean partition coefficient and natural-log entropy of membership rows.
/// These describe concentration, not calibrated probabilities or evidence for the number of clusters. Entropy uses0log0=0. No sample correction or label mapping.
/// @par Example
/// @code
/// dm::algorithms::FuzzyCMeansPartition x;
/// std::cout << x.coefficient; // 0
/// @endcode
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
struct FuzzyCMeansPartition {
    double coefficient{}; ///< Mean sum of squared memberships; nominal range1/c..1.
    double entropy_nats{}; ///< Mean negative sum u log u; nominal range0..log c.
};
/// @brief Fit Euclidean fuzzy c-means with the preserved ordered update policy.
/// Validate all rectangular rows then all finite values and fit parameters before copies, RNG, quotas or zero iterations. Empty data or c0/c>n is invalid; zero-dimensional points are supported. Random rows are normalized in double, with standard-library-dependent distribution mapping. Ordered weighted means precede pairwise distance-ratio updates. When several center distances are strictly below1e-12, the last cluster wins one-hot. A zero-weight cluster retains its previous center. Nonpositive tolerance is valid only at0, which disables early stopping. No data mutation, rescaling, relabeling or final center refresh. Finite numerical rounding/individual underflow remain possible; no global convergence certificate.
/// @param data Rectangular finite points, n x d.
/// @param c Cluster count,1..n.
/// @param fuzzifier Finite exponent m>1. Default 2.0.
/// @param max_iterations Maximum completed center/membership updates. Default 100.
/// @param tol Finite nonnegative strict maximum-membership-change tolerance. Default 1e-5.
/// @param seed mt19937_64 seed for row-major uniform(.1,1) initialization. Default 0.
/// @return Owned final centers/memberships/objective and actual iteration count.
/// @throws std::invalid_argument Invalid dimensions, parameters, nonfinite entries or membership rows.
/// @throws std::length_error Count/container overflow or optional quota exceeded.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @throws std::underflow_error All positive memberships of a cluster yield zero powered weight.
/// @par Example
/// @code
/// const auto x=dm::algorithms::fuzzy_c_means({{1},{3}},1);
/// std::cout << x.centers[0][0]; // 2
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
FuzzyCMeansResult fuzzy_c_means(const FuzzyCMeansMatrix& data, std::size_t c, double fuzzifier = 2.0, std::size_t max_iterations = 100, double tol = 1e-5, std::uint64_t seed = 0);
/// @brief Fit and measure actual initialization and update work.
/// Validate all rectangular rows then all finite values and fit parameters before copies, RNG, quotas or zero iterations. Empty data or c0/c>n is invalid; zero-dimensional points are supported. Random rows are normalized in double, with standard-library-dependent distribution mapping. Ordered weighted means precede pairwise distance-ratio updates. When several center distances are strictly below1e-12, the last cluster wins one-hot. A zero-weight cluster retains its previous center. Nonpositive tolerance is valid only at0, which disables early stopping. No data mutation, rescaling, relabeling or final center refresh. Finite numerical rounding/individual underflow remain possible; no global convergence certificate.
/// @param data Rectangular finite points, n x d.
/// @param c Cluster count,1..n.
/// @param fuzzifier Finite exponent m>1. Default 2.0.
/// @param max_iterations Maximum completed center/membership updates. Default 100.
/// @param tol Finite nonnegative strict maximum-membership-change tolerance. Default 1e-5.
/// @param seed mt19937_64 seed for row-major uniform(.1,1) initialization. Default 0.
/// @return Fit, actual operation counts and strict stopping status.
/// @throws std::invalid_argument Invalid dimensions, parameters, nonfinite entries or membership rows.
/// @throws std::length_error Count/container overflow or optional quota exceeded.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @throws std::underflow_error All positive memberships of a cluster yield zero powered weight.
/// @par Example
/// @code
/// const auto x=dm::algorithms::fuzzy_c_means_work({{1},{3}},1,2,1);
/// std::cout << x.work.center_weights; // 2
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
FuzzyCMeansRun fuzzy_c_means_work(const FuzzyCMeansMatrix& data, std::size_t c, double fuzzifier = 2.0, std::size_t max_iterations = 100, double tol = 1e-5, std::uint64_t seed = 0);
/// @brief Forecast conservative whole trace entries from virtual dimensions.
/// For n points,d coordinates,c clusters,T requested updates: base=2n+c+nd+nc+cd; frame=1+3c+2n+2cd+2nc. Return base+T*frame, including matrix row records and cells, not bytes or peak RAM. Bound all requested frames despite early stopping; check c1..n, arithmetic and vector capacities without allocation. Actual trace validates inputs before this forecast.
/// @param n Positive point count.
/// @param d Coordinate count, including0.
/// @param c Cluster count,1..n.
/// @param max_iterations Whole requested update count. Default 100.
/// @return Checked conservative retained-entry bound.
/// @throws std::invalid_argument Invalid dimensions, parameters, nonfinite entries or membership rows.
/// @throws std::length_error Count/container overflow or optional quota exceeded.
/// @par Example
/// @code
/// std::cout << dm::algorithms::fuzzy_c_means_trace_entries(3,1,2,1); // 48
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
std::size_t fuzzy_c_means_trace_entries(std::size_t n, std::size_t d, std::size_t c, std::size_t max_iterations = 100);
/// @brief Capture actual seeded update phases after whole-history admission.
/// Validate all rectangular rows then all finite values and fit parameters before copies, RNG, quotas or zero iterations. Empty data or c0/c>n is invalid; zero-dimensional points are supported. Random rows are normalized in double, with standard-library-dependent distribution mapping. Ordered weighted means precede pairwise distance-ratio updates. When several center distances are strictly below1e-12, the last cluster wins one-hot. A zero-weight cluster retains its previous center. Nonpositive tolerance is valid only at0, which disables early stopping. No data mutation, rescaling, relabeling or final center refresh. Finite numerical rounding/individual underflow remain possible; no global convergence certificate. Includes initial and two per-iteration diagnostic objectives; these extra computations may overflow when an ordinary fit does not. Uses fuzzy_c_means_trace_entries() before any RNG or input copy; no partial history or truncation.
/// @param data Rectangular finite points, n x d.
/// @param c Cluster count,1..n.
/// @param fuzzifier Finite exponent m>1. Default 2.0.
/// @param max_iterations Maximum completed center/membership updates. Default 100.
/// @param tol Finite nonnegative strict maximum-membership-change tolerance. Default 1e-5.
/// @param seed mt19937_64 seed for row-major uniform(.1,1) initialization. Default 0.
/// @param max_entries Conservative whole-history entry cap;0 disables only this cap. Default 1000000.
/// @return Owned actual initial state, phase objectives, sums, distances, updates and fit.
/// @throws std::invalid_argument Invalid dimensions, parameters, nonfinite entries or membership rows.
/// @throws std::length_error Count/container overflow or optional quota exceeded.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @throws std::underflow_error All positive memberships of a cluster yield zero powered weight.
/// @par Example
/// @code
/// const auto x=dm::algorithms::fuzzy_c_means_trace({{1},{3}},1,2,1);
/// std::cout << x.frames.size(); // 1
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
FuzzyCMeansTrace fuzzy_c_means_trace(const FuzzyCMeansMatrix& data, std::size_t c, double fuzzifier = 2.0, std::size_t max_iterations = 100, double tol = 1e-5, std::uint64_t seed = 0, std::size_t max_entries = 1000000);
/// @brief Capture one fit from supplied memberships without RNG draws.
/// Validate all rectangular rows then all finite values and fit parameters before copies, RNG, quotas or zero iterations. Empty data or c0/c>n is invalid; zero-dimensional points are supported. Random rows are normalized in double, with standard-library-dependent distribution mapping. Ordered weighted means precede pairwise distance-ratio updates. When several center distances are strictly below1e-12, the last cluster wins one-hot. A zero-weight cluster retains its previous center. Nonpositive tolerance is valid only at0, which disables early stopping. No data mutation, rescaling, relabeling or final center refresh. Finite numerical rounding/individual underflow remain possible; no global convergence certificate. Supplied rows must match data, be rectangular n x c in0..1 and have long-double row sums within absolute1e-12 of1. No normalization is applied. All supplied values validate before quota/copies. Initial centers remain zero, including zero-weight clusters. Diagnostic objectives have the same additional numeric limits as seeded capture.
/// @param data Rectangular finite points, n x d.
/// @param initial_membership Supplied n x c rows; c inferred and constrained1..n.
/// @param fuzzifier Finite exponent m>1. Default 2.0.
/// @param max_iterations Maximum completed center/membership updates. Default 100.
/// @param tol Finite nonnegative strict maximum-membership-change tolerance. Default 1e-5.
/// @param max_entries Conservative whole-history entry cap;0 disables only this cap. Default 1000000.
/// @return Owned actual supplied start and performed phase history.
/// @throws std::invalid_argument Invalid dimensions, parameters, nonfinite entries or membership rows.
/// @throws std::length_error Count/container overflow or optional quota exceeded.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @throws std::underflow_error All positive memberships of a cluster yield zero powered weight.
/// @par Example
/// @code
/// const auto x=dm::algorithms::fuzzy_c_means_trace_from({{1},{3}},{{1},{1}},2,1);
/// std::cout << x.work.initial_draws; // 0
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
FuzzyCMeansTrace fuzzy_c_means_trace_from(const FuzzyCMeansMatrix& data, const FuzzyCMeansMatrix& initial_membership, double fuzzifier = 2.0, std::size_t max_iterations = 100, double tol = 1e-5, std::size_t max_entries = 1000000);
/// @brief Evaluate query memberships with the same numerical update kernel.
/// Validate every shape then finite data/centers and m>1, even with no queries. Queries may be empty and c may exceed their count. Last distance strictly below1e-12 wins one-hot; otherwise use the original pairwise ratio-power formula. No fitting, rescaling or probability interpretation.
/// @param data Rectangular finite points, n x d.
/// @param centers Nonempty c x d finite centers, preserving caller order.
/// @param fuzzifier Finite exponent m>1. Default 2.0.
/// @return Owned n x c query memberships, or empty for no queries.
/// @throws std::invalid_argument Invalid dimensions, parameters, nonfinite entries or membership rows.
/// @throws std::length_error Count/container overflow or optional quota exceeded.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @par Example
/// @code
/// const auto u=dm::algorithms::fuzzy_c_means_membership({{5}},{{0},{10}});
/// std::cout << u[0][0]; // 0.5
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
FuzzyCMeansMatrix fuzzy_c_means_membership(const FuzzyCMeansMatrix& data, const FuzzyCMeansMatrix& centers, double fuzzifier = 2.0);
/// @brief Evaluate the ordered double fuzzy least-squares objective.
/// All shapes, finite values, exponent and supplied simplex rows validate before arithmetic. Empty queries with empty memberships return0; centers must be nonempty. Does not optimize centers or memberships. Uses sum_i sum_j pow(u_ij,m)*sum_d(x-v)^2 in the original order.
/// @param data Rectangular finite points, n x d.
/// @param centers Nonempty c x d finite centers, preserving caller order.
/// @param membership n x c finite simplex rows; no normalization.
/// @param fuzzifier Finite exponent m>1. Default 2.0.
/// @return Finite ordered objective for the supplied pair.
/// @throws std::invalid_argument Invalid dimensions, parameters, nonfinite entries or membership rows.
/// @throws std::length_error Count/container overflow or optional quota exceeded.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @par Example
/// @code
/// std::cout << dm::algorithms::fuzzy_c_means_objective({{1}},{{0}},{{1}}); // 1
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
double fuzzy_c_means_objective(const FuzzyCMeansMatrix& data, const FuzzyCMeansMatrix& centers, const FuzzyCMeansMatrix& membership, double fuzzifier = 2.0);
/// @brief Summarize membership concentration with coefficient and entropy.
/// Require nonempty rectangular simplex rows, finite values0..1 and row sums within absolute1e-12. No normalization. Zero entries contribute0 to entropy. These measures change with m and c and do not establish clustering quality.
/// @param membership Nonempty n x c simplex rows, c positive.
/// @return Mean sum u² and mean entropy in nats.
/// @throws std::invalid_argument Invalid dimensions, parameters, nonfinite entries or membership rows.
/// @par Example
/// @code
/// const auto x=dm::algorithms::fuzzy_c_means_partition({{.5,.5}});
/// std::cout << x.coefficient; // 0.5
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
FuzzyCMeansPartition fuzzy_c_means_partition(const FuzzyCMeansMatrix& membership);
/// @brief Generate the original160-point one-dimensional overlapping-blob fixture.
/// Uses fresh mt19937_64 seed21;80 N(0,2.2²) draws then80 N(10,2.2²) draws, with separate normal distributions. No shared state; standard-library distribution mapping affects bit portability.
/// @return Owned160 x1 synthetic points.
/// @par Example
/// @code
/// std::cout << dm::algorithms::fuzzy_c_means_example_data().size(); // 160
/// @endcode
/// @par References
/// - J.C. Dunn (1973), A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. doi:10.1080/01969727308546046.
/// - J.C. Bezdek, R. Ehrlich, W. Full (1984), FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences10,191-203. doi:10.1016/0098-3004(84)90020-7. This implementation uses Euclidean distance and a last-near-coincident tie rule.
/// @see fuzzy_c_means(), fuzzy_c_means_trace()
FuzzyCMeansMatrix fuzzy_c_means_example_data();
} // namespace datamunge::algorithms
