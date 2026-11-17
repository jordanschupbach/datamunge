#pragma once
/// @file nsga2.hpp
/// @brief Checked canonical NSGA-II with actual dominance/crowding/tournament/survival histories.
#include <cstddef>
#include <cstdint>
#include <datamunge/optim/multi_objective.hpp>
namespace datamunge::optim {
/// @brief Owned original-order native SBX/polynomial-mutation configuration.
/// @par Example
/// @code
/// dm::optim::NSGA2Options o;std::print("{}\n",o.population_size); // 100
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2Options {
  std::size_t population_size{
      100}; ///< Requested population,0/1 become2; odd values round up; complete effective cap1000000.
  std::size_t max_generations{
      250}; ///< Actual complete offspring/survival generations, zero permits initialization only.
  double crossover_rate{.9}; ///< Finite SBX probability in[0,1].
  double eta_crossover{15};  ///< Finite nonnegative SBX distribution index.
  double mutation_rate{-1};  ///< Finite negative selects1/nvars; otherwise probability in[0,1].
  double eta_mutation{20};   ///< Finite nonnegative polynomial-mutation distribution index.
  std::uint64_t seed{42};    ///< Local MT19937_64 seed, including zero.
};
/// @brief Actual selected producer counters, excluding callback cost, sort comparator counts and final front filtering.
/// @par Example
/// @code
/// dm::optim::NSGA2Work w;std::print("{}\n",w.evaluations); // 0
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2Work {
  std::size_t evaluations{
      0}; ///< Actual objective callbacks, N*(G+1), no discarded dimension probe.
  std::size_t ranking_passes{0}; ///< Actual non-dominated sorting calls,1+G.
  std::size_t ranking_pairs{
      0}; ///< Actual unordered candidate pairs inspected across ranking passes.
  std::size_t ranking_dominance_calls{
      0}; ///< Actual shared dominates() calls,1 or2 per pair; final filter excluded.
  std::size_t crowding_sorts{0}; ///< Actual objective-order sorts on processed nonempty fronts.
  std::size_t crowding_gaps{
      0}; ///< Actual interior normalized-gap additions; boundary infinities skip them.
  std::size_t tournaments{0};     ///< Actual binary tournaments, N*G.
  std::size_t offspring_pairs{0}; ///< Actual parent/SBX/mutation pairs, N*G/2.
};
/// @brief Owned actual ranks, processed-front crowding and complete partition.
/// Full ranking processes every front. Survival processes fronts only until
/// its capacity is filled; later crowding values are explicit placeholders.
/// Constant objective ranges still mark first/last sorted points infinite before
/// skipping gap divisions, matching the original producer. Ties use std::sort
/// without an added ID rule; no cross-standard-library tie ordering promise.
/// @par Example
/// @code
/// auto r=dm::optim::nsga2_rank(dm::optim::nsga2_demo_points());std::print("{}\n",r.fronts[0].size()); // 4
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2Ranking {
  std::vector<ParetoPoint> points; ///< Owned actual input points in original order.
  std::vector<std::size_t> ranks;  ///< Actual zero-based non-dominated rank per point.
  std::vector<double>
      crowding; ///< Actual processed-front crowding;0 placeholder when assigned flag false.
  std::vector<char>
      crowding_assigned; ///< Actual per-point flag1 for processed crowding,0 for unprocessed fronts.
  std::vector<std::vector<std::size_t>>
            fronts; ///< Actual fronts in native pair/decrement discovery order.
  NSGA2Work work;   ///< Actual selected ranking/crowding work only.
};
/// @brief Actual unordered pair relation and number of dominance calls.
/// @par Example
/// @code
/// auto t=dm::optim::nsga2_rank_trace(dm::optim::nsga2_demo_points());std::print("{}\n",t.pairs.size()); // 15
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2DominancePair {
  std::size_t first{0};    ///< Actual original index p.
  std::size_t second{0};   ///< Actual original index q>p.
  int         relation{0}; ///< -1 first dominates second,+1 second dominates first,0 neither.
  std::size_t calls{0};    ///< Actual1 if first dominates,otherwise2 shared dominates calls.
};
/// @brief Actual sorted front order and objective range used by crowding assignment.
/// @par Example
/// @code
/// auto t=dm::optim::nsga2_rank_trace(dm::optim::nsga2_demo_points());std::print("{}\n",t.orders[0].objective); // 0
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2CrowdingOrder {
  std::size_t              front{0};     ///< Actual front index being processed.
  std::size_t              objective{0}; ///< Actual objective index.
  std::vector<std::size_t> order;    ///< Actual std::sort original-point IDs in objective order.
  double                   range{0}; ///< Actual finite maximum-minus-minimum objective value.
};
/// @brief Actual complete pair/crowding-order history and actual ranking snapshot.
/// @par Example
/// @code
/// auto t=dm::optim::nsga2_rank_trace(dm::optim::nsga2_demo_points());std::print("{}\n",t.ranking.ranks[4]); // 1
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2RankingTrace {
  NSGA2Ranking ranking; ///< Actual same-producer ranks, points, partition and processed crowding.
  std::vector<NSGA2DominancePair> pairs;  ///< Actual p-then-q unordered pair decisions, no replay.
  std::vector<NSGA2CrowdingOrder> orders; ///< Actual processed-front objective orders.
  std::size_t                     captured_entries{
      0}; ///< Actual logical units contributed by this complete ranking history.
};
/// @brief Actual rank/crowding snapshot and selected original IDs under a capacity.
/// @par Example
/// @code
/// auto t=dm::optim::nsga2_survival_trace(dm::optim::nsga2_truncation_points(),4);std::print("{}\n",t.selected.size()); // 4
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2SurvivalTrace {
  NSGA2RankingTrace
      combined; ///< Actual input-pool ranking and only the crowding fronts needed for survival.
  std::vector<std::size_t> selected; ///< Actual survivor original IDs in next-population order.
  std::size_t captured_entries{0};   ///< Actual combined history units plus retained survivor IDs.
};
/// @brief Actual two sampled slots, their comparison data and winning parent slot.
/// @par Example
/// @code
/// dm::optim::NSGA2Tournament t;std::print("{}\n",t.winner); // 0
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2Tournament {
  std::size_t first{0};  ///< Actual first uniform-int parent-population slot.
  std::size_t second{0}; ///< Actual second uniform-int slot, sampled with replacement.
  std::size_t winner{
      0}; ///< Actual lower-rank/larger-crowding winner; exact ties choose second draw.
  std::size_t rank_first{0};      ///< Actual first slot rank at tournament time.
  std::size_t rank_second{0};     ///< Actual second slot rank at tournament time.
  double      crowding_first{0};  ///< Actual first slot crowding, possibly positive infinity.
  double      crowding_second{0}; ///< Actual second slot crowding.
};
/// @brief Actual tournament winners and their offspring positions in the combined pool.
/// @par Example
/// @code
/// dm::optim::NSGA2ParentPair p;std::print("{}\n",p.first_parent); // 0
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2ParentPair {
  std::size_t first_parent{0};  ///< Actual first winning parent slot.
  std::size_t second_parent{0}; ///< Actual second winning parent slot; may equal first.
  std::size_t first_child{0};   ///< Actual first child combined-pool index,at leastN.
  std::size_t second_child{0};  ///< Actual second child combined-pool index.
};
/// @brief Actual tournaments, offspring-parent links, combined ranking and survivor IDs.
/// @par Example
/// @code
/// dm::optim::NSGA2GenerationTrace g;std::print("{}\n",g.selected.size()); // 0
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2GenerationTrace {
  std::vector<NSGA2Tournament>
      tournaments; ///< Actual first-parent then second-parent tournament order for each pair.
  std::vector<NSGA2ParentPair>
      parents; ///< Actual child links; SBX/mutation are shared canonical variation operators.
  NSGA2RankingTrace
      combined; ///< Actual parents-then-offspring points, ranks, pair decisions and processed crowding.
  std::vector<std::size_t>
      selected; ///< Actual next-parent original combined-pool IDs, in selected order.
};
/// @brief Actual final population/front, effective dimensions and selected work.
/// @par Example
/// @code
/// dm::optim::NSGA2Run r;std::print("{}\n",r.effective_population); // 0
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2Run {
  std::vector<ParetoPoint> population; ///< Actual final parent population in selected order.
  std::vector<ParetoPoint>
      front; ///< Actual non-dominated subset of final population, duplicate values retained.
  std::size_t effective_population{0}; ///< Actual even population after min2/odd rounding.
  std::size_t objectives{0}; ///< Actual fixed objective count from first initial callback.
  NSGA2Work   work;          ///< Actual callbacks and selected variation/ranking/crowding work.
};
/// @brief Owned original controls and complete actual NSGA-II producer history.
/// Units count original seven options and bound elements; each retained point
/// record plus coordinates/objectives; rank/crowding/assigned entries; each
/// front/order record plus IDs; pair/tournament/parent/generation records;
/// selected IDs and final population/front point records. Fixed work metadata
/// is excluded. Detailed pair histories can be quadratic per generation.
/// @par Example
/// @code
/// dm::optim::NSGA2Trace t;std::print("{}\n",t.captured_entries); // 0
/// @endcode
/// @see NSGA2, nsga2_rank_trace()
struct NSGA2Trace {
  NSGA2Options        options; ///< Owned original seven configuration scalars.
  std::vector<double> lower;   ///< Owned original lower bounds.
  std::vector<double> upper;   ///< Owned original upper bounds.
  NSGA2RankingTrace
      initial; ///< Actual initialized points and complete initial rank/crowding decisions.
  std::vector<NSGA2GenerationTrace>
              generations;         ///< Actual sequential generations; no optimized-prefix replay.
  NSGA2Run    run;                 ///< Actual final result from the same execution.
  std::size_t captured_entries{0}; ///< Actual admitted logical record/numeric-entry units.
};
/// @brief Rank the entire finite objective cohort and assign all native crowding distances.
/// Complete nonempty fixed objective count>=2 and finite objective/coordinate
/// values are validated; empty cohort is accepted. Coordinates are metadata
/// and may have different lengths, including empty, for objective-only studies.
/// Cohort count cap1000000/int geometry guards precede quadratic state allocation.
/// Fast sort inspects each unordered pair once, assigning every rank. Crowding
/// marks objective-order endpoints infinite even for a constant range, then
/// skips zero-range divisions. Finite range/gap accumulation is guarded.
/// No input mutation/retained references; equal-value sorting is toolchain-dependent.
/// O(mP squared+mP log P) work and O(P squared+input elements) state; owned
/// detailed history adds actual pair/order records, not a peak-memory claim.
/// @param points Complete objective cohort; no callback evaluations occur.
/// @return Owned actual NSGA2Ranking, with explicit assigned flags for crowding.
/// @throws std::invalid_argument If complete shapes/values/capacity are invalid.
/// @throws std::length_error If complete geometry is exceeded.
/// @throws std::overflow_error If objective ranges or crowding gaps are nonfinite.
/// @par Example
/// @code
/// auto r=dm::optim::nsga2_rank(dm::optim::nsga2_demo_points());
/// std::print("{}\n",r.fronts[0].size()); // 4
/// @endcode
/// @see NSGA2Ranking, NSGA2RankingTrace
NSGA2Ranking nsga2_rank(const std::vector<ParetoPoint>& points);
/// @brief Capture actual same-producer ranking and all-front crowding decisions.
/// Complete nonempty fixed objective count>=2 and finite objective/coordinate
/// values are validated; empty cohort is accepted. Coordinates are metadata
/// and may have different lengths, including empty, for objective-only studies.
/// Cohort count cap1000000/int geometry guards precede quadratic state allocation.
/// Fast sort inspects each unordered pair once, assigning every rank. Crowding
/// marks objective-order endpoints infinite even for a constant range, then
/// skips zero-range divisions. Finite range/gap accumulation is guarded.
/// No input mutation/retained references; equal-value sorting is toolchain-dependent.
/// O(mP squared+mP log P) work and O(P squared+input elements) state; owned
/// detailed history adds actual pair/order records, not a peak-memory claim.
/// @param points Complete objective cohort; no callback evaluations occur.
/// @param max_entries Actual logical record/numeric-entry quota, default1000000, zero disables.
/// @return Owned actual NSGA2RankingTrace, with explicit assigned flags for crowding.
/// @throws std::invalid_argument If complete shapes/values/capacity are invalid.
/// @throws std::length_error If complete geometry/actual history quota is exceeded.
/// @throws std::overflow_error If objective ranges or crowding gaps are nonfinite.
/// @par Example
/// @code
/// auto r=dm::optim::nsga2_rank_trace(dm::optim::nsga2_demo_points());
/// std::print("{}\n",r.ranking.fronts[0].size()); // 4
/// @endcode
/// @see NSGA2Ranking, NSGA2RankingTrace
NSGA2RankingTrace nsga2_rank_trace(
    const std::vector<ParetoPoint>& points,
    std::size_t                     max_entries = 1000000);
/// @brief Capture actual same-producer ranking and capacity-limited survival.
/// Complete nonempty fixed objective count>=2 and finite objective/coordinate
/// values are validated; empty cohort is accepted. Coordinates are metadata
/// and may have different lengths, including empty, for objective-only studies.
/// Cohort count cap1000000/int geometry guards precede quadratic state allocation.
/// Fast sort inspects each unordered pair once, assigning every rank. Crowding
/// marks objective-order endpoints infinite even for a constant range, then
/// skips zero-range divisions. Finite range/gap accumulation is guarded.
/// No input mutation/retained references; equal-value sorting is toolchain-dependent.
/// O(mP squared+mP log P) work and O(P squared+input elements) state; owned
/// detailed history adds actual pair/order records, not a peak-memory claim.
/// @param points Complete objective cohort; no callback evaluations occur.
/// @param keep Survivor capacity0..P; complete fronts then crowding-descending partial front, exact ties unspecified.
/// @param max_entries Actual logical record/numeric-entry quota, default1000000, zero disables.
/// @return Owned actual NSGA2SurvivalTrace, with explicit assigned flags for crowding.
/// @throws std::invalid_argument If complete shapes/values/capacity are invalid.
/// @throws std::length_error If complete geometry/actual history quota is exceeded.
/// @throws std::overflow_error If objective ranges or crowding gaps are nonfinite.
/// @par Example
/// @code
/// auto r=dm::optim::nsga2_survival_trace(dm::optim::nsga2_demo_points(),4);
/// std::print("{}\n",r.selected.size()); // 4
/// @endcode
/// @see NSGA2Ranking, NSGA2RankingTrace
NSGA2SurvivalTrace nsga2_survival_trace(
    const std::vector<ParetoPoint>& points,
    std::size_t                     keep,
    std::size_t                     max_entries = 1000000);
/// @brief Native NSGA-II with original random/selection/variation order and guarded objective contracts.
/// Every objective is minimized. Actual initial population requires no discarded
/// dimension probe; callbacks=N*(G+1). Rank-first/crowding-second tournaments
/// sample with replacement, and exact ties choose the second draw. Both SBX
/// children mutate/evaluate in original order. Survival processes complete fronts
/// until the final partial front, truncating by descending crowding. Nondominated
/// points CAN be lost when a front exceeds capacity; no global archive is kept.
/// Sorting equal values preserves the standard library's unspecified tie order.
/// Work O((G+1)*mN squared+G*N*nvars), including quadratic final filtering,
/// plus callback/variation and crowding-sort costs. Driver dominated-set storage
/// is O(N squared+N*(nvars+m)); actual detailed traces can cost O(G*N squared).
/// Concurrent const calls own RNG/state, but callbacks own their thread safety.
/// @par Example
/// @code
/// dm::optim::NSGA2 solver;std::print("{}\n",solver.options().population_size); // 100
/// @endcode
/// @par References
/// - K. Deb et al.(2002), A fast and elitist multiobjective genetic algorithm:NSGA-II, IEEE Transactions on Evolutionary Computation6,182–197,doi:10.1109/4235.996017.
/// @see NSGA2Options, NSGA2Trace
class NSGA2 {
 public:
  /// @brief Own configuration; complete validation occurs on run/optimize/trace before callbacks.
  /// @param options Original configuration values, default native settings.
  /// @par Example
  /// @code
  /// dm::optim::NSGA2 solver({.population_size=6});std::print("{}\n",solver.options().population_size); // 6
  /// @endcode
  /// @see options()
  explicit NSGA2(NSGA2Options options = {});
  /// @brief Approximate a bounded multi-objective minimum with the canonical NSGA-II producer.
  /// Validate complete finite nonempty matching strict bounds/finite widths and
  /// all rates/indices/effective even population<=1000000 before callbacks.
  /// First actual initial callback fixes m>=2 (cap1000000); every later callback
  /// must return that many finite objectives. Variation coordinates must remain
  /// finite/in bounds. No discarded probe, input mutation or borrowed outputs.
  /// Callback effects can precede later callback/numeric/quota failure; no partial
  /// result is returned. Zero generations returns initialized/ranked population.
  /// @param function Mutable objective callback, evaluated sequentially in original order.
  /// @param lower_bound Complete finite lower coordinate bounds.
  /// @param upper_bound Matching finite upper bounds, representable positive widths.
  /// @return Owned actual std::vector<ParetoPoint>; a finite approximation, not a convergence certificate.
  /// @throws std::invalid_argument If complete options/bounds are invalid before callbacks.
  /// @throws std::domain_error If objective count/value changes or is invalid.
  /// @throws std::overflow_error If variation/range/crowding arithmetic is nonfinite.
  /// @throws std::length_error If complete count geometry is exceeded.
  /// @par Example
  /// @code
  /// struct F:dm::optim::MultiObjectiveFunction{std::vector<double> evaluate(const std::vector<double>& x) override{return {x[0],1-x[0]};}}f;
  /// dm::optim::NSGA2 solver({.population_size=2,.max_generations=0});auto r=solver.optimize(f,{0},{1});
  /// std::print("{}\n",r.size()); // 2
  /// @endcode
  /// @see NSGA2Run, NSGA2Trace
  [[nodiscard]] std::vector<ParetoPoint> optimize(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound) const;
  /// @brief Approximate a bounded multi-objective minimum with the canonical NSGA-II producer.
  /// Validate complete finite nonempty matching strict bounds/finite widths and
  /// all rates/indices/effective even population<=1000000 before callbacks.
  /// First actual initial callback fixes m>=2 (cap1000000); every later callback
  /// must return that many finite objectives. Variation coordinates must remain
  /// finite/in bounds. No discarded probe, input mutation or borrowed outputs.
  /// Callback effects can precede later callback/numeric/quota failure; no partial
  /// result is returned. Zero generations returns initialized/ranked population.
  /// @param function Mutable objective callback, evaluated sequentially in original order.
  /// @param lower_bound Complete finite lower coordinate bounds.
  /// @param upper_bound Matching finite upper bounds, representable positive widths.
  /// @return Owned actual NSGA2Run; a finite approximation, not a convergence certificate.
  /// @throws std::invalid_argument If complete options/bounds are invalid before callbacks.
  /// @throws std::domain_error If objective count/value changes or is invalid.
  /// @throws std::overflow_error If variation/range/crowding arithmetic is nonfinite.
  /// @throws std::length_error If complete count geometry is exceeded.
  /// @par Example
  /// @code
  /// struct F:dm::optim::MultiObjectiveFunction{std::vector<double> evaluate(const std::vector<double>& x) override{return {x[0],1-x[0]};}}f;
  /// dm::optim::NSGA2 solver({.population_size=2,.max_generations=0});auto r=solver.run(f,{0},{1});
  /// std::print("{}\n",r.work.evaluations); // 2
  /// @endcode
  /// @see NSGA2Run, NSGA2Trace
  [[nodiscard]] NSGA2Run run(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound) const;
  /// @brief Capture every actual native selection/ranking/survival decision.
  /// Validate complete finite nonempty matching strict bounds/finite widths and
  /// all rates/indices/effective even population<=1000000 before callbacks.
  /// First actual initial callback fixes m>=2 (cap1000000); every later callback
  /// must return that many finite objectives. Variation coordinates must remain
  /// finite/in bounds. No discarded probe, input mutation or borrowed outputs.
  /// Callback effects can precede later callback/numeric/quota failure; no partial
  /// result is returned. Zero generations returns initialized/ranked population.
  /// @param function Mutable objective callback, evaluated sequentially in original order.
  /// @param lower_bound Complete finite lower coordinate bounds.
  /// @param upper_bound Matching finite upper bounds, representable positive widths.
  /// @param max_entries Actual logical record/element quota, default1000000, zero disables.
  /// @return Owned actual NSGA2Trace; a finite approximation, not a convergence certificate.
  /// @throws std::invalid_argument If complete options/bounds are invalid before callbacks.
  /// @throws std::domain_error If objective count/value changes or is invalid.
  /// @throws std::overflow_error If variation/range/crowding arithmetic is nonfinite.
  /// @throws std::length_error If complete count geometry or actual capture quota is exceeded.
  /// @par Example
  /// @code
  /// struct F:dm::optim::MultiObjectiveFunction{std::vector<double> evaluate(const std::vector<double>& x) override{return {x[0],1-x[0]};}}f;
  /// dm::optim::NSGA2 solver({.population_size=2,.max_generations=0});auto r=solver.trace(f,{0},{1});
  /// std::print("{}\n",r.run.work.evaluations); // 2
  /// @endcode
  /// @see NSGA2Run, NSGA2Trace
  [[nodiscard]] NSGA2Trace trace(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound,
      std::size_t                max_entries = 1000000) const;
  /// @brief Borrow the owned original configuration, unchanged by const runs.
  /// @return Const reference valid for this optimizer lifetime.
  /// @par Example
  /// @code
  /// dm::optim::NSGA2 solver;std::print("{}\n",solver.options().seed); // 42
  /// @endcode
  /// @see NSGA2Options
  [[nodiscard]] const NSGA2Options& options() const noexcept { return options_; }

 private:
  NSGA2Options options_; ///< Owned original configuration, shared by no execution state.
  /// @brief Execute one original-order optimizer, optionally collecting actual owned history.
  /// @param function Mutable sequential objective callback.
  /// @param lower Complete lower bounds.
  /// @param upper Complete matching upper bounds.
  /// @param history Optional actual root-history sink.
  /// @param cap Actual root-history quota, zero disables.
  /// @return Actual final result and selected work.
  /// @throws std::exception Propagates documented public validation/callback/numeric/history errors.
  NSGA2Run execute(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower,
      const std::vector<double>& upper,
      NSGA2Trace*                history,
      std::size_t                cap) const;
};
/// @brief Return the original six two-objective A..F points, with empty coordinates.
/// @return(.1,.9),(.3,.4),(.6,.2),(.9,.1),(.5,.6),(.8,.5), in original order.
/// @par Example
/// @code
/// std::print("{}\n",dm::optim::nsga2_demo_points().size()); // 6
/// @endcode
/// @see nsga2_rank_trace()
std::vector<ParetoPoint> nsga2_demo_points();
/// @brief Return six mutually nondominated collinear points to expose finite-capacity truncation.
/// @return Ordered objectives(0,1),(.2,.8),(.4,.6),(.6,.4),(.8,.2),(1,0), empty coordinates.
/// @par Example
/// @code
/// std::print("{}\n",dm::optim::nsga2_survival_trace(dm::optim::nsga2_truncation_points(),4).selected.size()); // 4
/// @endcode
/// @see nsga2_survival_trace()
std::vector<ParetoPoint> nsga2_truncation_points();
/// @brief Own a presentation copy sorted by first objective, without refitting/filtering.
/// @param front Points with at least one finite objective, all coordinates finite.
/// @return Copied ascending-first-objective points; equal values use unspecified std::sort order.
/// @throws std::invalid_argument If any point lacks finite objective/coordinate values.
/// @par Example
/// @code
/// auto p=dm::optim::nsga2_sorted_front(dm::optim::nsga2_demo_points());std::print("{}\n",p[0].objectives[0]); // 0.1
/// @endcode
/// @see NSGA2::optimize()
std::vector<ParetoPoint> nsga2_sorted_front(const std::vector<ParetoPoint>& front);
} // namespace datamunge::optim
