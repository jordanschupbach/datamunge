#pragma once
/// @file k_medoids.hpp
/// @brief Checked Euclidean random-initialized best-swap medoids with actual candidate/state evidence.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned selected input identities, cluster assignments and distance objective.
/// Cluster ids index medoids, not original points. Equal computed distances keep
/// the first medoid slot; coincident medoids can therefore have empty clusters.
/// iterations counts accepted swaps, not candidate evaluations or convergence tests.
/// @par Example
/// @code
/// const auto r=dm::algorithms::k_medoids({{0},{2},{10}},1);
/// std::print("{} {}\n",r.medoids[0],r.total_cost); // 1 10
/// @endcode
/// @see k_medoids_trace()
struct KMedoidsResult {
  std::vector<std::size_t> medoids; ///< Owned distinct selected original point indices, slot order.
  std::vector<std::size_t> assignment; ///< Owned nearest-medoid slot for every original point.
  double total_cost{0}; ///< Actual sequential double sum of returned nearest Euclidean distances.
  std::size_t iterations{0}; ///< Number of actual committed improving swaps.
};
/// @brief Reason the actual best-swap loop stopped; neither value certifies global optimality.
/// NoImprovement means the full neighborhood has no candidate beating the running
/// scan incumbent by absolute1e-12; IterationLimit means that test was not completed
/// for the final state. A zero iteration limit returns the random initialization.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_medoids_trace({{0},{2},{10}},1,0);
/// std::print("{}\n",t.stop==dm::algorithms::KMedoidsStop::IterationLimit); // true
/// @endcode
/// @see KMedoidsTrace
enum class KMedoidsStop {
  NoImprovement, ///< Complete final candidate scan accepted no swap.
  IterationLimit ///< Caller iteration ceiling ended the loop without a final neighborhood test.
};
/// @brief One actual proposed replacement and its producing rounded objective.
/// Records follow medoid-slot then original-point order and exclude existing
/// medoids. best_changed is the actual strict trial_cost<incumbent-1e-12 test.
/// These are evaluations, not accepted trajectory states or replayed objectives.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_medoids_trace({{0},{2},{10}},1,1);
/// std::print("{}\n",t.candidates.size()); // 2
/// @endcode
/// @see KMedoidsState
struct KMedoidsCandidate {
  std::size_t iteration{0};        ///< Zero-based actual full scan ordinal.
  std::size_t slot{0};             ///< Medoid slot replaced only in this temporary proposal.
  std::size_t point{0};            ///< Proposed original non-medoid point identity.
  double      cost{0};             ///< Actual full assignment objective of the proposal.
  bool        best_changed{false}; ///< Whether this proposal became the current scan incumbent.
};
/// @brief Owned actual initialization or post-commit assignments and medoids.
/// states[0] precedes candidate scans; later states exist only for accepted swaps.
/// No caller references or filter callbacks are retained.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_medoids_trace({{0},{2},{10}},1,0);
/// std::print("{}\n",t.states.size()); // 1
/// @endcode
/// @see KMedoidsTrace
struct KMedoidsState {
  std::vector<std::size_t> medoids;    ///< Owned actual original medoid identities, slot order.
  std::vector<std::size_t> assignment; ///< Owned actual assignment slots.
  double cost{0}; ///< Actual sequential objective after this initialization/commit.
};
/// @brief Selected producing counts, excluding validation, shuffling, copies, allocation and time.
/// Every assignment pass inspects n*k precomputed distances. Pair distances count
/// only i<j computations; coordinate arithmetic and swap selection tests excluded.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_medoids_trace({{0},{2},{10}},1,0);
/// std::print("{} {}\n",t.work.distance_pairs,t.work.assignment_passes); // 3 1
/// @endcode
/// @see KMedoidsTrace
struct KMedoidsWork {
  std::size_t distance_pairs{0};    ///< Actual unordered distinct point-pair norm evaluations.
  std::size_t assignment_passes{0}; ///< Actual full assignment/objective evaluations.
  std::size_t candidates{0};        ///< Actual evaluated non-medoid replacement proposals.
  std::size_t distance_lookups{0};  ///< Actual point-to-medoid matrix entries inspected.
};
/// @brief Owned inputs, actual full candidate scans, accepted states and final result.
/// Trace admission counts one record per input row, candidate and trajectory state,
/// regardless of its inner vector lengths. It excludes individual coordinates,
/// assignments, medoid identities, output/matrix/runtime copies and capacity/bytes.
/// Separate distance-cell admission bounds n*n before the native matrix allocation.
/// Neither ceiling is a peak-memory guarantee. No second optimization/replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_medoids_trace({{0},{2},{10}},1);
/// std::print("{}\n",t.result.total_cost); // 10
/// @endcode
/// @see k_medoids_trace()
struct KMedoidsTrace {
  std::vector<std::vector<double>> data; ///< Owned complete original finite rectangular points.
  KMedoidsResult result; ///< Actual returned optimum candidate; no global certificate.
  KMedoidsWork   work;   ///< Actual selected work from this execution.
  KMedoidsStop   stop{KMedoidsStop::IterationLimit}; ///< Actual loop termination reason.
  std::vector<KMedoidsCandidate>
                             candidates; ///< Actual chronological complete candidate evaluations.
  std::vector<KMedoidsState> states; ///< Initialization followed by every actual accepted state.
  std::size_t captured_entries{0};   ///< Input-row+candidate+state logical record count.
};
/// @brief Fit Euclidean medoids using random distinct initialization and full best-swap scans.
/// Nonempty positive-dimensional matched finite rows and0<k<=n required. Seeded
/// std::shuffle(mt19937_64) chooses initial identities, not PAM's greedy BUILD.
/// Each scan visits slots then original non-medoids, retaining a proposal only if
/// its rounded cost beats the running incumbent by absolute1e-12. Commit the last
/// retained proposal and recompute its assignment once. Stop on no improvement or
/// max_iterations accepted scans. Zero iterations returns initialization. No
/// global-optimum or outlier-invariance guarantee; distances are Euclidean only.
/// Widened subtraction/hypot avoids raw squared overflow/underflow; final distances
/// and sequential objective are finite doubles or throw. Ordinary last-bit costs
/// can differ from the historical squared-double norm. std::shuffle seed mapping
/// is standard-library-specific. O(n*n*d+I*k*(n-k)*n*k) work and O(n*n+n*d) stored
/// matrix/input geometry (input borrowed); no optional trace copies/shared state.
/// @param data Borrowed complete rectangular finite point rows, unchanged.
/// @param k Number of distinct medoid identities in[1,n].
/// @param max_iterations Scan/commit ceiling, default100, zero permits initialization only.
/// @param seed Actual mt19937_64/std::shuffle seed, default0.
/// @return Owned actual final medoids, assignments, sequential cost and committed count.
/// @throws std::invalid_argument On invalid k, empty dimension, ragged or nonfinite input.
/// @throws std::length_error On matrix/container/work-count representability limits.
/// @throws std::overflow_error On an unrepresentable finite double distance or objective.
/// @throws std::bad_alloc On native matrix/temporary/output allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::k_medoids({{0},{2},{10}},1);
/// std::print("{} {}\n",r.medoids[0],r.total_cost); // 1 10
/// @endcode
/// @par References
/// - Kaufman and Rousseeuw(1990), Finding Groups in Data, https://doi.org/10.1002/9780470316801.
///   This implementation uses random initialization and exhaustive swap evaluation, not greedy BUILD or FastPAM.
/// @see k_medoids_trace()
KMedoidsResult k_medoids(
    const std::vector<std::vector<double>>& data,
    std::size_t                             k,
    std::size_t                             max_iterations = 100,
    std::uint64_t                           seed           = 0);
/// @brief Capture same-execution original points, proposal costs and actual accepted states.
/// Same numerical/ordering/termination contracts as k_medoids(). Complete input and
/// matrix geometry validate before copies. Admit input rows and initial state
/// before copying; admit each proposal before objective work and each accepted
/// state before mutation. No partial trace escapes failure; scalar API has neither
/// optional ceiling. Trace cost adds owned input/state/history copies to kernel work.
/// @param data Borrowed original finite rectangular rows.
/// @param k Distinct medoid count in[1,n].
/// @param max_iterations Actual scan/commit ceiling, default100, zero initialization only.
/// @param seed Actual standard-library shuffle seed, default0.
/// @param max_entries Input-row+candidate+state ceiling, default1000000, zero disables optional cap.
/// @param max_distance_cells Complete n*n matrix cell ceiling, default1000000, zero disables optional cap.
/// @return Owned actual full trace/result; no references or replay.
/// @throws std::invalid_argument On invalid k/shape/nonfinite data.
/// @throws std::length_error On enabled ceilings or count/container representability limits.
/// @throws std::overflow_error On nonfinite double norm/objective output.
/// @throws std::bad_alloc On native/trace allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_medoids_trace({{0},{2},{10}},1,0,0,4,9);
/// std::print("{} {}\n",t.captured_entries,t.states.size()); // 4 1
/// @endcode
/// @par References
/// - Kaufman and Rousseeuw(1990), Finding Groups in Data, https://doi.org/10.1002/9780470316801.
/// @see KMedoidsTrace, k_medoids()
KMedoidsTrace k_medoids_trace(
    const std::vector<std::vector<double>>& data,
    std::size_t                             k,
    std::size_t                             max_iterations     = 100,
    std::uint64_t                           seed               = 0,
    std::size_t                             max_entries        = 1000000,
    std::size_t                             max_distance_cells = 1000000);
} // namespace datamunge::algorithms
