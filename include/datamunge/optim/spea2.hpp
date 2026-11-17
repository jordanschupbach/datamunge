#pragma once
/// @file spea2.hpp
/// @brief Actual SPEA2 fitness, archive selection and seeded generation histories.
#include <cstddef>
#include <cstdint>
#include <datamunge/optim/multi_objective.hpp>
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::optim {
/// @brief SPEA2Options owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::optim::SPEA2Options value{};
/// std::print("{}\n",value.seed); // 42
/// @endcode
struct SPEA2Options {
  std::size_t population_size{100}; ///< Requested population, effective minimum2.

  /// @brief Size of the external archive of best solutions; if 0, defaults to population_size.
  std::size_t archive_size{0};
  std::size_t max_generations{
      250}; ///< Number of actual archive-selection generations; zero retains legacy empty front.
  double crossover_rate{0.9}; ///< Finite SBX probability in[0,1].
  double eta_crossover{15.0}; ///< Finite positive SBX distribution index.

  /// @brief Per-variable polynomial-mutation probability; if < 0, defaults to 1 / n_variables.
  double        mutation_rate{-1.0};
  double        eta_mutation{20.0};  ///< Finite positive polynomial-mutation distribution index.
  std::uint64_t seed{42};            ///< Local mt19937_64 seed per optimize call.
  std::size_t   max_pool_size{2000}; ///< Complete population plus archive bound; zero disables.
  std::size_t   max_evaluations{1000000}; ///< Actual callback evaluations; zero disables.
  std::size_t   max_dominance_tests{
      100000000}; ///< Actual ordered-pair dominance calls; zero disables.
  std::size_t max_distance_calls{
      1000000000}; ///< Actual fitness/truncation distance calls; zero disables.
  std::size_t max_history_values{
      5000000}; ///< Captured records, coordinate/objective values and archive indices; zero disables.
};
/// @brief SPEA2Candidate owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::optim::SPEA2Candidate value{};
/// std::print("{}\n",value.strength); // 0
/// @endcode
struct SPEA2Candidate {
  std::vector<double>
      x; ///< Actual owned decision coordinates, empty for objective-only selection diagnostics.
  std::vector<double> f;          ///< Actual complete finite minimized objective vector.
  double              fitness{};  ///< Actual raw+density selection fitness.
  std::size_t         strength{}; ///< Actual number of other pool points strictly dominated.
  double              raw{};      ///< Actual sum of strengths of dominating pool points.
  double
      density{}; ///< Actual1/(sigma_k+2), with one-based nonself neighbor rank floor(sqrt(pool size)).
  std::size_t pool_index{}; ///< Actual zero-based position in this generation/diagnostic pool.
};
/// @brief SPEA2Generation owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::optim::SPEA2Generation value{};
/// std::print("{}\n",value.evaluations); // 0
/// @endcode
struct SPEA2Generation {
  std::size_t generation{};  ///< Actual zero-based archive-selection generation.
  std::size_t evaluations{}; ///< Actual cumulative objective callbacks before this selection.
  std::vector<SPEA2Candidate>
      pool; ///< Actual scored producing pool, population first then previous archive.
  std::vector<std::size_t> selected;  ///< Actual retained pool indices in resulting archive order.
  std::vector<std::size_t> truncated; ///< Actual crowded pool indices removed in deletion order.
};
/// @brief SPEA2Run owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::optim::SPEA2Run value{};
/// std::print("{}\n",value.evaluations); // 0
/// @endcode
struct SPEA2Run {
  std::vector<ParetoPoint>
      front; ///< Actual nondominated subset of final archive; empty for zero selected generations.
  std::size_t evaluations{}; ///< Actual callback evaluations including initial population.
  std::size_t
      dominance_tests{}; ///< Actual ordered-pair tests during fitness assignment; final front filtering excluded.
  std::size_t distance_calls{}; ///< Actual Euclidean objective distances during fitness/truncation.
  std::size_t archive_removals{}; ///< Actual crowded archive deletions.
  std::vector<SPEA2Generation>
      generations; ///< Optional actual scored pools and selected/truncated identities.
};
/// @brief SPEA2Selection owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::optim::SPEA2Selection value{};
/// std::print("{}\n",value.dominance_tests); // 0
/// @endcode
struct SPEA2Selection {
  std::vector<SPEA2Candidate> pool;      ///< Actual scored complete supplied objective pool.
  std::vector<std::size_t>    selected;  ///< Actual retained original pool indices.
  std::vector<std::size_t>    truncated; ///< Actual crowded original indices removed in order.
  std::size_t                 dominance_tests{}; ///< Actual ordered-pair fitness dominance calls.
  std::size_t                 distance_calls{};  ///< Actual density/truncation distance calls.
};
/// @brief Score and select an archive from a complete objective pool.
/// All objectives are minimized, finite and equal positive dimension. Empty pool returns empty selection. Strength counts strict dominance; raw sums dominator strengths; density uses the one-based kth nonself neighbor with k=floor(sqrt(P)), clamped to available neighbors; singleton distance0 gives density1/2. This corrects the historical off-by-one third-neighbor lookup when k=2. Raw-zero points enter first; dominated points fill by fitness then pool index, or crowded points are removed by lexicographic sorted distance lists with first-index exact ties. Keeps at most min(capacity,P). Same actual fitness/selection helpers as optimizer; no report replay. O(P^2) dominance/distance storage, repeated truncation can require O(P^4) distance work; bounds apply. Owned outputs, no shared state.
/// @param objectives Complete borrowed finite objective vectors.
/// @param capacity Positive desired archive size.
/// @param options Pool, actual work and capture policies; evolutionary controls are ignored.
/// @return Owned actual fitness and selected/truncated indices.
/// @throws std::invalid_argument Zero capacity or malformed objective values/dimensions.
/// @throws std::length_error Pool/work/history/counter capacity exceeded.
/// @throws std::overflow_error Objective distance or fitness not representable.
/// @par Example
/// @code
/// auto r=dm::optim::spea2_select_trace({{0,1},{1,0},{1,1}},2);
/// std::print("{}\n",r.selected); // [0, 1]
/// @endcode
/// @par References
/// - E. Zitzler, M. Laumanns and L. Thiele, SPEA2: Improving the Strength Pareto Evolutionary Algorithm, ETH TIK Report103,2001.
/// @see SPEA2
SPEA2Selection spea2_select_trace(
    const std::vector<std::vector<double>>& objectives,
    std::size_t                             capacity,
    const SPEA2Options&                     options = {});
/// @brief Optimize bounded multiobjective functions by checked actual SPEA2 generations.
/// Owns options only. Each run owns an mt19937_64 and uses canonical SBX/polynomial mutation. Callbacks may mutate their own state and must be synchronized by caller if shared. Default actual density rank is the published one-based nonself rank, correcting the historical off-by-one. No optimality or convergence guarantee from a finite evolutionary archive.
/// @par Example
/// @code
/// const dm::optim::SPEA2 optimizer;
/// std::print("{}\n",optimizer.options().seed); // 42
/// @endcode
/// @par References
/// - Zitzler, Laumanns and Thiele (2001), SPEA2, TIK Report103.
/// @see spea2_select_trace()
class SPEA2 {
 public:
  /// @brief Store per-run evolutionary controls and native policies.
  /// Controls are validated when executing a run; copying the optimizer owns an independent options record.
  /// @param options Complete controls, defaults in SPEA2Options.
  /// @par Example
  /// @code
  /// const dm::optim::SPEA2 optimizer;
  /// std::print("{}\n",optimizer.options().population_size); // 100
  /// @endcode
  /// @see optimize_run()
  explicit SPEA2(SPEA2Options options = {});
  /// @brief Execute actual seeded SPEA2 archive generations.
  /// Finite strict lower<upper bounds with finite span and consistent finite nonempty callback objectives. Population=max(requested,2); archive_size0 uses that population count. Select archive each generation; final generation does not breed unused children. Zero generations still evaluates initial population and returns empty front for legacy compatibility. Rates must be finite in[0,1] (negative mutation selects1/n), finite positive eta controls. Complete pool and actual callback/pair/distance/history policies bound work; zero disables policies. Raw/density fitness and truncation use the exact same helpers as spea2_select_trace. Density-neighbor correction intentionally changes some seeded trajectories. Returned front is finite nondominated archive approximation, not proof of reaching the analytic Pareto set. Inputs borrowed only for call; outputs own all values. Callback exceptions propagate.
  /// @param function Mutable borrowed minimized vector objective callback.
  /// @param lower_bound Complete finite lower coordinate bounds.
  /// @param upper_bound Complete finite upper coordinate bounds.
  /// @return Owned actual final front, callback/selection work and optional real scored generations.
  /// @throws std::invalid_argument Malformed bounds/controls/objective values or dimension change.
  /// @throws std::length_error Complete pool, actual work/history or counter capacity exceeded.
  /// @throws std::overflow_error Actual distance/fitness/offspring not representable.
  /// @par Example
  /// @code
  /// dm::optim::ZDT1Objective f(2);dm::optim::SPEA2Options o;o.population_size=2;o.max_generations=0;
  /// std::print("{}\n",dm::optim::SPEA2(o).optimize_run(f,{0,0},{1,1}).evaluations); // 2
  /// @endcode
  /// @see optimize()
  SPEA2Run optimize_run(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound) const;
  /// @brief Execute actual seeded SPEA2 archive generations with scored pool histories.
  /// Finite strict lower<upper bounds with finite span and consistent finite nonempty callback objectives. Population=max(requested,2); archive_size0 uses that population count. Select archive each generation; final generation does not breed unused children. Zero generations still evaluates initial population and returns empty front for legacy compatibility. Rates must be finite in[0,1] (negative mutation selects1/n), finite positive eta controls. Complete pool and actual callback/pair/distance/history policies bound work; zero disables policies. Raw/density fitness and truncation use the exact same helpers as spea2_select_trace. Density-neighbor correction intentionally changes some seeded trajectories. Returned front is finite nondominated archive approximation, not proof of reaching the analytic Pareto set. Inputs borrowed only for call; outputs own all values. Callback exceptions propagate.
  /// @param function Mutable borrowed minimized vector objective callback.
  /// @param lower_bound Complete finite lower coordinate bounds.
  /// @param upper_bound Complete finite upper coordinate bounds.
  /// @return Owned actual final front, callback/selection work and optional real scored generations.
  /// @throws std::invalid_argument Malformed bounds/controls/objective values or dimension change.
  /// @throws std::length_error Complete pool, actual work/history or counter capacity exceeded.
  /// @throws std::overflow_error Actual distance/fitness/offspring not representable.
  /// @par Example
  /// @code
  /// dm::optim::ZDT1Objective f(2);dm::optim::SPEA2Options o;o.population_size=2;o.max_generations=0;
  /// std::print("{}\n",dm::optim::SPEA2(o).optimize_trace(f,{0,0},{1,1}).evaluations); // 2
  /// @endcode
  /// @see optimize()
  SPEA2Run optimize_trace(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound) const;
  /// @brief Return the preserved final SPEA2 front.
  /// Delegates to the same checked actual producer without captured generations. Published density-rank correction may change seeded trajectories.
  /// @param function Mutable borrowed minimized vector objective.
  /// @param lower_bound Complete finite lower bounds.
  /// @param upper_bound Complete finite upper bounds.
  /// @return Owned nondominated final archive approximation.
  /// @throws std::invalid_argument Malformed controls/bounds/objective.
  /// @throws std::length_error Default work/pool/capacity exceeded.
  /// @throws std::overflow_error Actual arithmetic not representable.
  /// @par Example
  /// @code
  /// dm::optim::ZDT1Objective f(2);dm::optim::SPEA2Options o;o.population_size=2;o.max_generations=0;
  /// std::print("{}\n",dm::optim::SPEA2(o).optimize(f,{0,0},{1,1}).size()); // 0
  /// @endcode
  /// @see optimize_run()
  std::vector<ParetoPoint> optimize(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound) const;
  /// @brief Borrow the immutable stored controls.
  /// Reference lifetime ends with this optimizer; const reads do not mutate it.
  /// @return Borrowed options record.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::optim::SPEA2().options().seed); // 42
  /// @endcode
  /// @see SPEA2Options
  const SPEA2Options& options() const noexcept { return options_; }

 private:
  SPEA2Options options_; ///< Owned immutable-per-call controls.
};
} // namespace datamunge::optim
