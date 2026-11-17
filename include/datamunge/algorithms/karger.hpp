#pragma once
/// @file karger.hpp
/// @brief Checked unweighted multigraph contraction with actual trial and component histories.
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned nontrivial partition and original-edge crossing multiplicity.
/// Undirected parallel edges each count; loops never cross. side_a contains
/// vertex0, both sides sorted. Disconnected graphs may return the exact zero cut.
/// A positive cut is a Monte Carlo candidate, not a proof of optimality.
/// Original legacy field order/types retained.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::karger_min_cut(2,{{0,1},{0,1}},1).cut_size); // 2
/// @endcode
/// @see karger_trace()
struct MinCutResult {
  std::size_t              cut_size{0}; ///< Original crossing edges counted with multiplicity.
  std::vector<std::size_t> side_a;      ///< Sorted owned side containing vertex0.
  std::vector<std::size_t> side_b;      ///< Sorted owned nonempty complementary side.
};
/// @brief Complete input/trial ceilings and online owned-history record admission.
/// Zero disables an optional cap, not size arithmetic. History charges n+1 records
/// per contraction snapshot or completed trial partition; edges copied once are
/// covered by complete n+m admission. Capacity/bytes/peak RAM are excluded.
/// @par Example
/// @code
/// const dm::algorithms::KargerOptions o{10,100,1000};std::print("{}\n",o.max_trials); // 100
/// @endcode
/// @see karger_trace()
struct KargerOptions {
  std::size_t max_input_records{
      1000000};                    ///< Complete vertices+edges ceiling, default million, zero off.
  std::size_t max_trials{1000000}; ///< Effective trial count ceiling, default million, zero off.
  std::size_t max_history_records{
      1000000}; ///< Online contraction/trial partition record ceiling, default million, zero off.
};
/// @brief Selected actual logical work, excluding validation, union-find internals and trace copies.
/// Edge scans include every original record when rebuilding candidates and
/// counting final cuts, including loops. Random draws include rejected modulo-
/// bias words. Counts are operations, not elapsed time or allocation traffic.
/// @par Example
/// @code
/// const auto t=dm::algorithms::karger_trace(2,{{0,1}},3);std::print("{} {}\n",t.work.trials,t.work.random_draws); // 3 0
/// @endcode
/// @see KargerTrace
struct KargerWork {
  std::size_t trials{0};       ///< Actual completed trials.
  std::size_t contractions{0}; ///< Actual successful component merges.
  std::size_t candidate_edge_checks{
      0};                         ///< Original edge inspections rebuilding crossing candidates.
  std::size_t cut_edge_checks{0}; ///< Original edge inspections counting final candidate cuts.
  std::size_t random_draws{0};    ///< Actual mt19937_64 outputs, rejection draws included.
};
/// @brief Actual selected original edge and owned canonical components after its contraction.
/// Labels are minimum original vertices, not union-find root ids. Crossing list
/// preserves original edge order and parallel multiplicity; chosen_rank is the
/// unbiased accepted word modulo crossing_count. No union/search replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::karger_trace(3,{{0,1},{1,2},{2,0}},1);
/// std::print("{}\n",t.steps[0].groups_after); // 2
/// @endcode
/// @see KargerTrace
struct KargerStep {
  std::size_t   trial{0};          ///< Zero-based actual trial id.
  std::size_t   groups_before{0};  ///< Actual components before this merge.
  std::size_t   groups_after{0};   ///< Actual components after this merge.
  std::size_t   crossing_count{0}; ///< Actual candidate list size at selection time.
  std::size_t   chosen_rank{0};    ///< Actual accepted rank in that input-ordered list.
  std::size_t   edge_index{0};     ///< Actual original selected edge id.
  std::uint64_t accepted_word{0};  ///< Actual accepted mt19937_64 output.
  std::size_t   draw_count{0}; ///< Actual words consumed for this selection, including rejections.
  std::vector<std::size_t>
      labels; ///< Owned minimum-original-id label per vertex after actual union.
};
/// @brief Actual completed trial cut/partition and range of contraction records.
/// If no crossing edge remains before two groups, the graph is disconnected;
/// choose component0 against the rest, giving a nontrivial zero cut. First strict
/// best retained on ties; best_changed marks that actual decision.
/// @par Example
/// @code
/// const auto t=dm::algorithms::karger_trace(2,{{0,1}},1);std::print("{}\n",t.trials[0].cut_size); // 1
/// @endcode
/// @see KargerTrace
struct KargerTrial {
  std::size_t cut_size{0};         ///< Actual original-edge crossing multiplicity.
  std::size_t groups{0};           ///< Actual terminal group count, possibly>2 when disconnected.
  std::size_t first_step{0};       ///< First actual contraction-record offset for this trial.
  std::size_t step_count{0};       ///< Actual successful contractions in this trial.
  bool        best_changed{false}; ///< Actual strict best-so-far update, true on first trial.
  std::vector<std::size_t> side_a; ///< Owned sorted partition side containing0.
  std::vector<std::size_t> side_b; ///< Owned sorted complementary side.
};
/// @brief Owned original multigraph, actual contraction histories and every completed trial.
/// No borrowed graph storage or replay. Extra storage O(m+B*n*n) worst case
/// for B trials of n vertices; complete record caps can reject large captures.
/// @par Example
/// @code
/// const auto t=dm::algorithms::karger_trace(2,{{0,1}},2);std::print("{} {}\n",t.effective_trials,t.trials.size()); // 2 2
/// @endcode
/// @see karger_trace()
struct KargerTrace {
  MinCutResult  result;              ///< Actual first best candidate partition.
  std::size_t   vertex_count{0};     ///< Complete original vertices, isolates included.
  std::size_t   effective_trials{0}; ///< Requested positive count, or checked automatic schedule.
  std::uint64_t seed{42};            ///< Original deterministic mt19937_64 seed.
  std::vector<std::pair<std::size_t, std::size_t>>
             edges; ///< Owned original undirected input edge order, loops/parallel included.
  KargerWork work;  ///< Actual selected operations.
  std::vector<KargerStep>  steps;  ///< Actual component merges and snapshots.
  std::vector<KargerTrial> trials; ///< Actual complete trial partitions and best decisions.
};
/// @brief Contract an unweighted undirected multigraph repeatedly and retain the first smallest candidate.
/// Requires n>=2 and in-range endpoints. Loops accepted but never contracted/cut;
/// parallel edges preserve multiplicity. Rebuild crossing list each contraction,
/// sample its rank with mt19937_64 plus rejection to avoid modulo bias, and merge
/// until two groups or no crossing edges. Partition component0 versus others and
/// count original crossings. Independent ideal uniform trials give the classical
/// lower success bound2/(n(n-1)) for connected graphs; a deterministic seeded
/// pseudorandom run is not an observed confidence certificate. Legacy positive
/// trials retained; zero chooses max(1,n*n*ceil(log(n))) with checked arithmetic.
/// Ordinary draw mapping matches the old modulo sampler when no rejection occurs.
/// Expected word work O(B*(n+(n+1)*m*alpha(n))) for B trials using ranked compressed
/// union-find; O(n+m) working/output storage. No input mutation or shared state.
/// @param vertex_count Original vertices, isolates included, at least2.
/// @param edges Borrowed undirected endpoint pairs in deterministic input order.
/// @param trials Positive full trial count, or0(default) for the automatic schedule.
/// @param seed Local mt19937_64 seed, default42.
/// @return Owned smallest observed cut and nontrivial partition; not a guaranteed positive mincut.
/// @throws std::invalid_argument On n<2 or invalid endpoints.
/// @throws std::length_error On schedule/size/work/container overflow; legacy optional caps are off.
/// @throws std::bad_alloc On working/output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::karger_min_cut(4,{{0,1},{1,2},{2,0},{2,3}},100,7).cut_size); // 1
/// @endcode
/// @par References
/// - Karger(1993), Global Min-cuts in RNC, and Other Ramifications of a Simple Min-Cut Algorithm, SODA21-30; author preprint https://people.csail.mit.edu/karger/Papers/mincut.pdf .
/// @see karger_trace()
MinCutResult karger_min_cut(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             trials = 0,
    std::uint64_t                                           seed   = 42);
/// @brief Capture actual sampled contractions and every trial partition from the shared kernel.
/// Complete endpoint/schedule/input admission precedes graph copies. Before a
/// merge/captured partition, admit its n+1 complete records. Failure throws without
/// exposing partial results. Same sampling/tie/zero-cut contracts as scalar API.
/// @param vertex_count Original vertex count, at least2.
/// @param edges Borrowed undirected multigraph edge records.
/// @param trials Positive count, or0(default) for checked automatic schedule.
/// @param seed Local mt19937_64 seed, default42.
/// @param options Complete input/trial and online owned-history ceilings.
/// @return Owned complete original graph, same-execution best result and actual histories.
/// @throws std::invalid_argument On invalid graph input.
/// @throws std::length_error On enabled ceilings or schedule/size/work/container overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::karger_trace(2,{{0,1},{0,1}},1);std::print("{}\n",t.result.cut_size); // 2
/// @endcode
/// @see KargerOptions, KargerStep
KargerTrace karger_trace(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             trials  = 0,
    std::uint64_t                                           seed    = 42,
    KargerOptions                                           options = {});
} // namespace datamunge::algorithms
